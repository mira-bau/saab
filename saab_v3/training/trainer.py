"""Training orchestrator for Transformer models."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from saab_v3.data.batcher import Batcher
from saab_v3.data.constants import PAD_IDX
from saab_v3.data.structures import Batch
from saab_v3.tasks.msm_field import make_msm_field_mask, make_msm_field_mask_balanced
from saab_v3.training.checkpoint import CheckpointManager
from saab_v3.training.diagnostics import compute_attention_entropy, compute_same_field_mass
from saab_v3.training.metrics import MetricsLogger
from saab_v3.training.schedulers import create_lr_scheduler
from saab_v3.utils.device import get_device

if TYPE_CHECKING:
    from saab_v3.training.config import TrainingConfig
    from saab_v3.training.dataset import StructuredDataset


class Trainer:
    """Training orchestrator for Transformer models."""

    def __init__(
        self,
        model: nn.Module,
        config: "TrainingConfig",
        train_dataset: "StructuredDataset",
        val_dataset: "StructuredDataset" | None = None,
        task_head: nn.Module | None = None,
        loss_fn: nn.Module | None = None,
        task_type: str | None = None,
        experiment_name: str = "experiment",
        model_config: dict | None = None,
        task_config: dict | None = None,
        preprocessing_config: dict | None = None,
        dataset_name: str | None = None,
        model_type: str | None = None,
        eval_every_steps: int | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Transformer model (Flat/Scratch/SAAB)
            config: TrainingConfig instance
            train_dataset: Training StructuredDataset
            val_dataset: Optional validation StructuredDataset
            task_head: Optional task head
            loss_fn: Loss function (optional: auto-created if task_type provided)
            task_type: Task type for auto-creating loss function ("classification", "regression", etc.)
            experiment_name: Name of experiment (for checkpoint/logging directories)
            model_config: Optional model configuration dict to save in checkpoint
            task_config: Optional task configuration dict to save in checkpoint
            preprocessing_config: Optional preprocessing configuration dict to save in checkpoint
            dataset_name: Optional dataset name to save in checkpoint
            model_type: Optional model type ('flat', 'scratch', or 'saab') to save in checkpoint
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.task_head = task_head
        self.task_type = task_type
        self.experiment_name = experiment_name
        self.model_config = model_config
        self.task_config = task_config
        self.preprocessing_config = preprocessing_config
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.eval_every_steps = eval_every_steps

        # Set random seeds for reproducibility
        self._set_random_seeds(config.seed)

        # Move model to device
        device = get_device(config.device)
        self.model = self.model.to(device)
        if self.task_head is not None:
            self.task_head = self.task_head.to(device)

        # Create Batcher for manual batching
        max_seq_len = train_dataset.preprocessor.config.max_seq_len
        self.batcher = Batcher(max_seq_len=max_seq_len, pad_token_id=PAD_IDX, device=device)
        
        # Store batching configuration
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle

        # Batching state for step-based training (maintains position across steps)
        self._dataset_indices: list[int] | None = None
        self._dataset_pointer: int = 0
        self._current_permutation_epoch: int = -1

        # Loss function
        if loss_fn is None and task_type is not None:
            # Auto-create loss function from task_type
            from saab_v3.training.loss import create_loss_fn

            # Extract task parameters from task_head if possible
            loss_kwargs = {}
            if task_head is not None:
                if hasattr(task_head, "num_classes"):
                    loss_kwargs["num_classes"] = task_head.num_classes
                if hasattr(task_head, "multi_label"):
                    loss_kwargs["multi_label"] = task_head.multi_label
                if hasattr(task_head, "num_labels"):
                    loss_kwargs["num_labels"] = task_head.num_labels
                if hasattr(task_head, "num_targets"):
                    loss_kwargs["num_targets"] = task_head.num_targets

            self.loss_fn = create_loss_fn(task_type, **loss_kwargs)
        elif loss_fn is None:
            # Default to classification loss (for backward compatibility)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn
        self.loss_fn = self.loss_fn.to(device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Calculate number of training steps
        num_training_steps = self._calculate_training_steps()

        # Create LR scheduler
        self.scheduler = create_lr_scheduler(self.optimizer, config, num_training_steps)
        
        # Check if scheduler is ReduceLROnPlateau (needs special handling)
        self.is_plateau_scheduler = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        # Validate LR schedule configuration
        if config.lr_schedule == "constant":
            if self.scheduler is not None:
                raise ValueError(
                    f"LR schedule is 'constant' but scheduler was created: {type(self.scheduler).__name__}. "
                    f"This indicates a bug in scheduler creation."
                )
        else:
            if self.scheduler is None:
                raise ValueError(
                    f"LR schedule is '{config.lr_schedule}' but no scheduler was created. "
                    f"This indicates a bug in scheduler creation."
                )
            else:
                # Log scheduler details for debugging
                print(f"\n[DIAGNOSTIC] LR Scheduler created:")
                print(f"  - Type: {type(self.scheduler).__name__}")
                print(f"  - LR Schedule: {config.lr_schedule}")
                if hasattr(self.scheduler, 'warmup_steps'):
                    print(f"  - Warmup steps: {self.scheduler.warmup_steps}")
                if hasattr(self.scheduler, 'decay_steps'):
                    print(f"  - Decay steps: {self.scheduler.decay_steps}")
                if self.is_plateau_scheduler:
                    print(f"  - Mode: {self.scheduler.mode}")
                    print(f"  - Patience: {self.scheduler.patience}")
                    print(f"  - Factor: {self.scheduler.factor}")
                    print(f"  - Min LR: {self.scheduler.min_lrs[0]}")
                if self.is_plateau_scheduler:
                    print(f"  - Mode: {self.scheduler.mode}")
                    print(f"  - Patience: {self.scheduler.patience}")
                    print(f"  - Factor: {self.scheduler.factor}")
                    print(f"  - Min LR: {self.scheduler.min_lrs[0]}")
                print(f"  - Total training steps: {num_training_steps}")
                print(f"  - Initial LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        # Checkpoint manager
        save_dir = config.save_dir or Path("checkpoints") / experiment_name
        self.checkpoint_manager = CheckpointManager(
            save_dir=save_dir, keep_checkpoints=config.keep_checkpoints
        )

        # Metrics logger
        self.metrics_logger = MetricsLogger(config, experiment_name)
        
        # Per-step loss tracking (for Loss@k computation)
        self._per_step_losses: list[float] = []  # Indexed by step
        log_dir = Path(config.log_dir) if config.log_dir else Path("logs") / experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self._per_step_loss_file = log_dir / "per_step_loss.jsonl"

        # Setup warning logger (logs to file)
        log_dir = (
            Path(config.log_dir) if config.log_dir else Path("logs") / experiment_name
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        warning_log_file = log_dir / "warnings.log"

        self.warning_logger = logging.getLogger(f"{experiment_name}_warnings")
        self.warning_logger.setLevel(logging.WARNING)
        # Remove existing handlers to avoid duplicates
        self.warning_logger.handlers.clear()
        # File handler (mode='w' to clear file on each execution)
        file_handler = logging.FileHandler(warning_log_file, mode='w')
        file_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.warning_logger.addHandler(file_handler)
        # Prevent propagation to root logger
        self.warning_logger.propagate = False

        # Log initial learning rate for verification (after logger is set up)
        initial_lr = (
            self.scheduler.get_last_lr()[0]
            if self.scheduler is not None and not self.is_plateau_scheduler
            else self.optimizer.param_groups[0]["lr"]
        )
        if abs(initial_lr - config.learning_rate) > 1e-10:
            self.warning_logger.warning(
                f"Learning rate mismatch: config LR={config.learning_rate:.2e}, "
                f"actual optimizer LR={initial_lr:.2e}"
            )

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric_value: float | None = None
        self._zero_loss_streak: int = 0  # Track consecutive zero loss steps
        
        # Track losses across gradient accumulation for proper averaging
        self._accumulated_losses: list[float] = []  # Losses for current accumulation cycle
        self._microbatch_count: int = 0  # Track microbatches within current accumulation cycle
        self._accumulated_batch_indices: list[list[int]] = []  # Batch indices for all microbatches in current cycle
        self._accumulated_labels: list[list[int]] = []  # Labels for all microbatches in current cycle (for diagnostics)
        
        # Store problematic batch for re-analysis (step 400)
        self._step400_batch_indices: list[int] | None = None
        self._step400_batch: Batch | None = None
        self._small_grad_streak: int = 0  # Track consecutive small gradient steps
        
        # Deterministic permutation for label-diverse batching
        # Use numpy RNG for deterministic permutation (same seed = same perm for Scratch/SAAB)
        import numpy as np
        self._np_rng = np.random.default_rng(seed=config.seed)
        self._dataset_permutation: np.ndarray | None = None  # Fixed permutation for entire run
        self._permutation_pointer: int = 0  # Current position in permutation
        
        # Store dataset_name for permutation file path
        self._dataset_name = dataset_name
        
        # Run Identity tracking: permutation and mask hashes (set at step 0)
        self._permutation_hash_16: str | None = None
        self._permutation_len: int | None = None
        self._train_mask_hash_16: str | None = None
        self._train_mask_count: int | None = None
        self._train_mask_nonpad_count: int | None = None
        self._train_mask_ratio: float | None = None
        self._val_mask_hash_16: str | None = None
        self._val_mask_count: int | None = None
        self._val_mask_nonpad_count: int | None = None
        self._val_mask_ratio: float | None = None
        self._val_eval_index: int = 0  # Validation evaluation index (independent of training steps)
        
        # Check if dataset is sorted by labels (diagnostic)
        self._check_dataset_label_order()
        
        # Early stopping state
        self._early_stopping_patience_counter: int = 0
        self._best_early_stopping_metric_value: float | None = None

        # Config for checkpointing
        self._config_dict = {
            "training_config": config.model_dump()
            if hasattr(config, "model_dump")
            else {},
        }
        
        # Add model_config and task_config if provided
        if model_config is not None:
            # If it's a Pydantic model, dump it; otherwise use as-is
            if hasattr(model_config, "model_dump"):
                self._config_dict["model_config"] = model_config.model_dump()
            else:
                self._config_dict["model_config"] = model_config
        
        if task_config is not None:
            self._config_dict["task"] = task_config
        
        # Add preprocessing_config, dataset_name, model_type, and experiment_name
        if preprocessing_config is not None:
            # If it's a Pydantic model, dump it; otherwise use as-is
            if hasattr(preprocessing_config, "model_dump"):
                self._config_dict["preprocessing_config"] = preprocessing_config.model_dump()
            else:
                self._config_dict["preprocessing_config"] = preprocessing_config
        
        if dataset_name is not None:
            self._config_dict["dataset_name"] = dataset_name
        
        if model_type is not None:
            self._config_dict["model_type"] = model_type
        
        # Always save experiment_name for metadata
        self._config_dict["experiment_name"] = experiment_name

    def _set_random_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set deterministic operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _compute_batch_hash_int(self, batch_indices: list[int] | None) -> int:
        """Compute stable batch hash from batch indices.
        
        Args:
            batch_indices: List of dataset indices used in this batch
            
        Returns:
            Integer hash (first 8 bytes of SHA256 as int)
        """
        if batch_indices is None or len(batch_indices) == 0:
            return 0
        
        import hashlib
        # Sort indices for stability (order-independent hash)
        sorted_indices = sorted(batch_indices)
        indices_bytes = str(sorted_indices).encode('utf-8')
        hash_obj = hashlib.sha256(indices_bytes)
        # Take first 8 bytes and convert to int
        hash_bytes = hash_obj.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder='big', signed=False)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config.
        
        Includes all trainable parameters from both model and task head.
        
        Raises:
            AssertionError: If task head parameters are not included in optimizer
        """
        # Collect all trainable parameters (model + task head)
        all_params = list(self.model.parameters())
        
        if self.task_head is not None:
            task_head_params = list(self.task_head.parameters())
            all_params.extend(task_head_params)
        
        # Create optimizer with all parameters
        if self.config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                all_params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                all_params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer_type: {self.config.optimizer_type}")
        
        # Sanity check: Verify all parameters are in optimizer
        optimizer_param_ids = set()
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                optimizer_param_ids.add(id(param))
        
        model_param_ids = set(id(p) for p in self.model.parameters())
        task_head_param_ids = set(id(p) for p in self.task_head.parameters()) if self.task_head is not None else set()
        
        missing_model_params = model_param_ids - optimizer_param_ids
        missing_task_head_params = task_head_param_ids - optimizer_param_ids
        
        if missing_model_params:
            raise AssertionError(
                f"CRITICAL: {len(missing_model_params)} model parameters missing from optimizer!"
            )
        
        if missing_task_head_params:
            raise AssertionError(
                f"CRITICAL: {len(missing_task_head_params)} task head parameters missing from optimizer! "
                f"Task head will not be trained. Missing: {[name for name, p in self.task_head.named_parameters() if id(p) in missing_task_head_params]}"
            )
        
        # Log parameter counts for verification
        # Note: logger may not be initialized yet, so use print for critical info
        print(
            f"[OPTIMIZER] Created with {len(optimizer_param_ids)} parameters "
            f"(model: {len(model_param_ids)}, task_head: {len(task_head_param_ids)})"
        )
        
        return optimizer

    def _calculate_training_steps(self) -> int:
        """Calculate total number of training steps."""
        if self.config.max_steps is not None:
            return self.config.max_steps

        if self.config.num_epochs is None:
            raise ValueError("Either num_epochs or max_steps must be set")

        # Calculate number of batches per epoch
        num_batches = (len(self.train_dataset) + self.batch_size - 1) // self.batch_size
        steps_per_epoch = num_batches // self.config.gradient_accumulation_steps
        return self.config.num_epochs * steps_per_epoch

    def _check_dataset_label_order(self) -> None:
        """Check if dataset is sorted by labels (diagnostic for label-homogeneous batches)."""
        if self.train_dataset is None:
            return
        
        # Extract first 200 labels in dataset order
        labels_first_200 = []
        for idx in range(min(200, len(self.train_dataset))):
            label = self.train_dataset._extract_label(idx)
            if label is not None:
                labels_first_200.append(int(label))
        
        if len(labels_first_200) < 10:
            return  # Not enough labels to check
        
        # Check if labels are sorted
        is_sorted_asc = all(labels_first_200[i] <= labels_first_200[i+1] for i in range(len(labels_first_200)-1))
        is_sorted_desc = all(labels_first_200[i] >= labels_first_200[i+1] for i in range(len(labels_first_200)-1))
        
        print(f"\n[DATASET LABEL ORDER CHECK]")
        print(f"  First 200 labels: {labels_first_200[:50]}... (showing first 50)")
        print(f"  Is sorted ascending: {is_sorted_asc}")
        print(f"  Is sorted descending: {is_sorted_desc}")
        if is_sorted_asc or is_sorted_desc:
            print(f"  ⚠️  WARNING: Dataset appears to be sorted by labels!")
            print(f"  This will cause label-homogeneous batches if batching sequentially.")
        print()

    def _compute_file_fingerprint(self, file_path: Path) -> dict:
        """Compute fingerprint of train.csv file.
        
        Args:
            file_path: Path to train.csv file
            
        Returns:
            Dictionary with file fingerprint (size, mtime, hash)
        """
        if not file_path.exists():
            return {}
        
        stat = file_path.stat()
        file_size = stat.st_size
        mtime = stat.st_mtime
        
        # Compute hash of first and last 1MB (or entire file if smaller)
        chunk_size = 1024 * 1024  # 1MB
        hasher = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Hash first chunk
            first_chunk = f.read(chunk_size)
            hasher.update(first_chunk)
            
            # If file is larger than 2MB, also hash last chunk
            if file_size > 2 * chunk_size:
                f.seek(file_size - chunk_size)
                last_chunk = f.read(chunk_size)
                hasher.update(last_chunk)
            elif file_size > chunk_size:
                # File is between 1MB and 2MB, hash remaining bytes
                remaining = f.read()
                hasher.update(remaining)
        
        file_hash = hasher.hexdigest()[:16]  # First 16 chars
        
        return {
            "size": file_size,
            "mtime": mtime,
            "hash": file_hash,
        }

    def _get_permutation_file_path(self, dataset_size: int) -> tuple[Path, Path]:
        """Get paths to permutation file and fingerprint file.
        
        Args:
            dataset_size: Size of the dataset
            
        Returns:
            Tuple of (permutation_file_path, fingerprint_file_path)
        """
        if self._dataset_name is None:
            # Fallback: use experiment_name or default
            base_name = self.experiment_name if hasattr(self, 'experiment_name') else "default"
        else:
            base_name = self._dataset_name
            
        # Use same artifacts directory structure as preprocessor
        artifacts_dir = Path(__file__).parent.parent.parent / "dataset" / "artifacts" / base_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # File names: train_perm_seed{seed}_N{size}.npy and .json
        base_filename = f"train_perm_seed{self.config.seed}_N{dataset_size}"
        perm_file = artifacts_dir / f"{base_filename}.npy"
        fingerprint_file = artifacts_dir / f"{base_filename}.json"
        
        return perm_file, fingerprint_file

    def _load_or_create_permutation(self, dataset: "StructuredDataset") -> np.ndarray:
        """Load existing permutation file or create and save new one.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Permutation array (guaranteed int64 dtype)
        """
        dataset_size = len(dataset)
        perm_file, fingerprint_file = self._get_permutation_file_path(dataset_size)
        
        # Get train file path for fingerprinting
        train_file_path = None
        if hasattr(dataset, '_data_path') and dataset._data_path is not None:
            train_file_path = dataset._data_path
        elif self._dataset_name is not None:
            # Fallback: construct path from dataset_name
            train_file_path = Path(__file__).parent.parent.parent / "dataset" / "raw" / self._dataset_name / "train.csv"
        
        # Compute current file fingerprint
        current_fingerprint = None
        if train_file_path and train_file_path.exists():
            current_fingerprint = self._compute_file_fingerprint(train_file_path)
        
        if perm_file.exists() and fingerprint_file.exists():
            # Load existing permutation and verify fingerprint
            permutation = np.load(str(perm_file))
            
            # CRITICAL: Verify dtype is int64
            if permutation.dtype != np.int64:
                raise ValueError(
                    f"Permutation file has wrong dtype: {permutation.dtype}, expected int64. "
                    f"Please delete {perm_file} and regenerate."
                )
            if permutation.ndim != 1:
                raise ValueError(
                    f"Permutation file has wrong shape: {permutation.ndim}D, expected 1D. "
                    f"Please delete {perm_file} and regenerate."
                )
            if len(permutation) != dataset_size:
                raise ValueError(
                    f"Permutation file size mismatch: {len(permutation)} != {dataset_size}. "
                    f"Please delete {perm_file} and regenerate."
                )
            
            # Load and verify fingerprint
            with open(fingerprint_file, "r") as f:
                saved_fingerprint = json.load(f)
            
            if current_fingerprint and saved_fingerprint:
                if saved_fingerprint.get("size") != current_fingerprint.get("size") or \
                   saved_fingerprint.get("mtime") != current_fingerprint.get("mtime") or \
                   saved_fingerprint.get("hash") != current_fingerprint.get("hash"):
                    raise ValueError(
                        f"Train file fingerprint mismatch! Permutation was created for a different train.csv.\n"
                        f"  Saved: size={saved_fingerprint.get('size')}, mtime={saved_fingerprint.get('mtime')}, hash={saved_fingerprint.get('hash')}\n"
                        f"  Current: size={current_fingerprint.get('size')}, mtime={current_fingerprint.get('mtime')}, hash={current_fingerprint.get('hash')}\n"
                        f"  Please delete {perm_file} and {fingerprint_file} and regenerate."
                    )
            
            print(f"[BATCHING] Loaded existing permutation from {perm_file.name}")
            print(f"  Verified: dtype=int64, shape=({dataset_size},), fingerprint matches")
        else:
            # Create new permutation
            indices = np.arange(dataset_size, dtype=np.int64)
            permutation = self._np_rng.permutation(indices)
            
            # CRITICAL: Ensure int64 dtype before saving
            permutation = np.asarray(permutation, dtype=np.int64)
            
            # Save permutation
            np.save(str(perm_file), permutation)
            
            # Save fingerprint
            if current_fingerprint:
                with open(fingerprint_file, "w") as f:
                    json.dump(current_fingerprint, f, indent=2)
                print(f"[BATCHING] Created and saved new permutation to {perm_file.name}")
                print(f"  Saved fingerprint: size={current_fingerprint['size']}, hash={current_fingerprint['hash']}")
            else:
                print(f"[BATCHING] Created and saved new permutation to {perm_file.name}")
                print(f"  WARNING: Could not compute file fingerprint (train file not found)")
        
        # Print verification info
        first_20 = permutation[:20].tolist()
        # CRITICAL: Hash from array bytes, not list
        perm_hash = hashlib.sha256(permutation.tobytes()).hexdigest()[:16]
        print(f"[BATCHING] Permutation verification:")
        print(f"  Seed: {self.config.seed}, Dataset size: {dataset_size}")
        print(f"  Dtype: {permutation.dtype} (verified int64)")
        print(f"  First 20 indices: {first_20}")
        print(f"  Hash (first 16 chars): {perm_hash}")
        print(f"  Permutation ensures label diversity across batches")
        
        return permutation

    def _get_next_batch(self, dataset: "StructuredDataset", shuffle: bool = False) -> tuple[Batch, list[int]]:
        """Get next batch with deterministic, label-diverse permutation.
        
        Uses a fixed-seed permutation for the entire run to ensure:
        - Label diversity within batches
        - Deterministic ordering (same seed = same permutation for Scratch/SAAB)
        - Fair comparison between models
        
        Args:
            dataset: StructuredDataset to iterate over
            shuffle: Whether to use permutation (always True for label diversity)
        
        Returns:
            Tuple of (Batch object, list of sample indices used)
        """
        # Initialize fixed permutation once (for entire run)
        if self._dataset_permutation is None:
            # Try to load existing permutation file, otherwise create and save
            permutation = self._load_or_create_permutation(dataset)
            self._dataset_permutation = permutation
            self._permutation_pointer = 0
            
            # Log permutation hash at step 0 for deterministic behavior verification
            if self.current_step == 0:
                import hashlib
                perm_hash = hashlib.sha256(permutation.tobytes()).hexdigest()[:16]
                # Store for Run Identity
                self._permutation_hash_16 = perm_hash
                self._permutation_len = len(permutation)
                print(f"\n[Permutation Hash] Step 0:")
                print(f"  - Permutation hash (first 16 chars): {perm_hash}")
                print(f"  - Permutation length: {len(permutation)}")
                print(f"  - Seed: {self.config.seed}")
                print(f"  - First 20 indices: {permutation[:20].tolist()}")
                print(f"  (Compare this hash with SAAB/Scratch runs to ensure identical permutation)")
        
        # Check if we need to wrap around (restart permutation deterministically)
        if self._permutation_pointer + self.batch_size > len(self._dataset_permutation):
            # Restart permutation (same permutation, reset pointer)
            # This ensures we see all samples before repeating
            self._permutation_pointer = 0
        
        # Get batch indices from permutation
        batch_indices = self._dataset_permutation[
            self._permutation_pointer : self._permutation_pointer + self.batch_size
        ].tolist()
        
        # Advance pointer
        self._permutation_pointer += self.batch_size
        
        # Create batch
        batch_items = [dataset[int(idx)] for idx in batch_indices]
        batch = self.batcher.batch(batch_items, task_type=self.task_type)
        
        return batch, batch_indices

    def train(self) -> dict:
        """Run training loop.

        Returns:
            Dictionary with training history (losses, metrics, etc.)
        """
        history = {
            "train_losses": [],
            "val_losses": [],
            "train_metrics": [],
            "val_metrics": [],
        }

        # Determine training mode: step-based or epoch-based
        use_step_based = self.config.max_steps is not None
        
        if use_step_based:
            # Step-based training: iterate until max_steps
            total_steps = self.config.max_steps
            print(f"Starting step-based training for {total_steps} steps...")
            print(f"Device: {get_device(self.config.device)}")
            
            # Initialize epoch counter (for LR scheduling and validation)
            self.current_epoch = 0
            self.model.train()
            if self.task_head is not None:
                self.task_head.train()
            
            # Track losses for history
            step_losses = []
            
            # Step-based training loop
            while self.current_step < total_steps:
                # Get next batch (maintains state across steps)
                batch, batch_indices = self._get_next_batch(
                    self.train_dataset, shuffle=self.shuffle
                )
                
                # Move batch to device
                batch = batch.to(get_device(self.config.device))
                
                # Compute MSM diagnostics BEFORE step (to use original batch, doesn't consume/advance iterator)
                should_compute_diagnostics = (
                    self.current_step in {0, 100}
                    or self.current_step % self.config.log_steps == 0
                )
                diag_metrics = {}
                if should_compute_diagnostics:
                    # Only compute diagnostics for MSM-Field task
                    task_type_check = self._infer_task_type()
                    if task_type_check == "msm_field":
                        # Use current batch BEFORE masking (doesn't consume/advance iterator)
                        # This ensures diagnostics use original (unmasked) field_ids
                        # Note: original_field_ids will be None here (before _step), so it uses batch.field_ids (pre-mask)
                        diag_metrics = self._compute_msm_diagnostics(batch=batch, field_ids=None)
                
                # Training step (pass batch_indices for diagnostics)
                # NOTE: This processes ONE microbatch. step_metrics is only populated when
                # gradient accumulation completes (i.e., after processing gradient_accumulation_steps microbatches)
                step_metrics = self._step(batch, batch_indices=batch_indices)
                
                # Only process step-level operations when gradient accumulation completes
                # step_metrics is non-empty only when accumulation completes (i.e., at optimizer step)
                if step_metrics:  # This means we just completed an optimizer step
                    # Verification: log batch IDs at specific optimizer steps
                    if self.current_step in {0, 100, 200, 300, 400, 500, 1000}:
                        print(f"[BATCH_VERIFICATION] Optimizer Step {self.current_step}: Last microbatch batch_indices[:20] = {batch_indices[:20]}")
                        print(f"  (Note: This shows only the LAST microbatch. See [DIAGNOSTIC] for all microbatches in this optimizer step)")
                    # At step 0, print full batch indices for fairness verification
                    if self.current_step == 0:
                        print(f"\n[FAIRNESS VERIFICATION] Step 0 batch_indices (full batch): {batch_indices}")
                        print(f"  Model type: {self.model_type if hasattr(self, 'model_type') and self.model_type else 'unknown'}")
                        print(f"  Compare this with SAAB/Scratch runs to ensure identical batching")
                    # Only append loss if step_metrics contains it (after accumulation completes)
                    # Task 1: Write per-step loss ONLY when optimizer step completes
                    # Step index is self.current_step (0 to max_steps-1)
                    if "loss" in step_metrics:
                        step_losses.append(step_metrics["loss"])
                        # Track per-step loss for Loss@k computation
                        # Ensure list is long enough (indexed by step: 0 to max_steps-1)
                        while len(self._per_step_losses) <= self.current_step:
                            self._per_step_losses.append(None)
                        self._per_step_losses[self.current_step] = step_metrics["loss"]
                    
                    # Add diagnostic metrics to step_metrics (computed before step)
                    if diag_metrics:
                        step_metrics.update(diag_metrics)
                        
                        # Print diagnostic block at step 0 and 100
                        if self.current_step in {0, 100}:
                            self._print_msm_diagnostics(self.current_step, diag_metrics)
                    
                    # Log step metrics
                    if self.current_step % self.config.log_steps == 0:
                        self.metrics_logger.log_step(
                            self.current_step, step_metrics, phase="train"
                        )
                    
                    # Save checkpoint (step-based)
                    if (
                        self.config.save_steps is not None
                        and self.current_step % self.config.save_steps == 0
                    ):
                        self._save_checkpoint(self.current_epoch, step_metrics, is_latest=False)
                    
                    # Run evaluation if eval_every_steps is set (for step-based training)
                    if (
                        self.eval_every_steps is not None
                        and self.val_dataset is not None
                        and self.current_step % self.eval_every_steps == 0
                    ):
                        val_metrics = self._validate()
                        # Log validation metrics
                        if val_metrics:
                            self.metrics_logger.log_step(
                                self.current_step, val_metrics, phase="val"
                            )
                            print(
                                f"Step {self.current_step}: Validation loss = {val_metrics.get('loss', 0.0):.6f}"
                            )
                            history["val_losses"].append(val_metrics.get("loss", 0.0))
                            history["val_metrics"].append(val_metrics)
                        
                        # Task 2: Print determinism summary line at step 0 (after validation completes)
                        # _val_eval_index is 1 after first validation (was 0, then incremented)
                        if self.current_step == 0 and self._val_eval_index == 1:
                            perm_hash = self._permutation_hash_16
                            train_mask_hash = self._train_mask_hash_16
                            val_mask_hash = self._val_mask_hash_16
                            if perm_hash and train_mask_hash and val_mask_hash:
                                print(f"\n[RUN_ID_PROOF] perm_hash={perm_hash}, train_mask_hash={train_mask_hash}, val_mask_hash={val_mask_hash}, val_eval_index=0")
                            if self.config.save_best:
                                self._check_and_save_best(val_metrics, self.current_epoch)
                    
                    # Validate (step-based) - legacy eval_steps support
                    if (
                        self.val_dataset is not None
                        and self.config.eval_steps is not None
                        and self.eval_every_steps is None  # Only use if eval_every_steps not set
                        and self.current_step % self.config.eval_steps == 0
                    ):
                        val_metrics = self._validate()
                        history["val_losses"].append(val_metrics.get("loss", 0.0))
                        history["val_metrics"].append(val_metrics)
                        if self.config.save_best:
                            self._check_and_save_best(val_metrics, self.current_epoch)
                    
                    # Increment step (only after gradient accumulation completes)
                    self.current_step += 1
                
                # Update epoch counter (for LR scheduling and permutation reset)
                # Calculate epoch based on samples seen
                samples_seen = self.current_step * self.batch_size * self.config.gradient_accumulation_steps
                new_epoch = samples_seen // len(self.train_dataset)
                if new_epoch > self.current_epoch:
                    self.current_epoch = new_epoch
                    # Reset permutation for new epoch (deterministic)
                    self._current_permutation_epoch = -1
                
                # Check early stopping (step-based)
                if (
                    self.config.early_stop_zero_loss_steps is not None
                    and hasattr(self, "_zero_loss_streak")
                    and self._zero_loss_streak >= self.config.early_stop_zero_loss_steps
                ):
                    print(f"\nEarly stopping: Zero loss for {self._zero_loss_streak} consecutive steps.")
                    break
            
            # For step-based training, store both average and final step loss
            avg_train_loss = sum(step_losses) / len(step_losses) if step_losses else 0.0
            # Final step loss (for summary display)
            final_step_loss = step_losses[-1] if step_losses else 0.0
            history["train_losses"].append(avg_train_loss)
            history["train_metrics"].append({"loss": avg_train_loss})
            # Store final step loss separately for accurate summary
            history["final_step_loss"] = final_step_loss
            
            # Final validation
            if self.val_dataset is not None:
                val_metrics = self._validate()
                history["val_losses"].append(val_metrics.get("loss", 0.0))
                history["val_metrics"].append(val_metrics)
            
            print(f"\nReached max_steps ({total_steps}), stopping training.")
            
        else:
            # Epoch-based training (original logic)
            total_epochs = self.config.num_epochs
            print(f"Starting training for {total_epochs} epochs...")
            print(f"Total steps: {self._calculate_training_steps()}")
            print(f"Device: {get_device(self.config.device)}")

            for epoch in range(total_epochs):
                self.current_epoch = epoch

                # Train epoch
                train_metrics = self._train_epoch(epoch)
                train_loss = train_metrics.get("loss", 0.0)
                history["train_losses"].append(train_loss)
                history["train_metrics"].append(train_metrics)
                # Store for validation comparison
                self._last_train_loss = train_loss

                # Validate
                if (
                    self.val_dataset is not None
                    and (epoch + 1) % self.config.eval_epochs == 0
                ):
                    val_metrics = self._validate()
                    history["val_losses"].append(val_metrics.get("loss", 0.0))
                    history["val_metrics"].append(val_metrics)

                    # Calculate and log train/val loss gap
                    train_loss = train_metrics.get("loss", 0.0)
                    val_loss = val_metrics.get("loss", 0.0)
                    if train_loss > 0 and val_loss > 0:
                        train_val_gap = abs(train_loss - val_loss)
                        train_val_gap_ratio = train_val_gap / train_loss if train_loss > 0 else 0.0
                        val_metrics["train_val_gap"] = train_val_gap
                        val_metrics["train_val_gap_ratio"] = train_val_gap_ratio
                        
                        # Warn if gap is too large (indicates overfitting)
                        if train_val_gap_ratio > 0.5:  # More than 50% gap
                            self.warning_logger.warning(
                                f"Large train/val loss gap detected at epoch {epoch}: "
                                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                                f"gap_ratio={train_val_gap_ratio:.2%}. This may indicate overfitting."
                            )
                        elif val_loss < train_loss * 0.1:  # Val loss much lower than train
                            self.warning_logger.warning(
                                f"Validation loss is much lower than training loss at epoch {epoch}: "
                                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}. "
                                f"This may indicate a validation computation issue."
                            )

                    # Check if this is the best model
                    if self.config.save_best:
                        self._check_and_save_best(val_metrics, epoch)
                    
                    # Step ReduceLROnPlateau scheduler with validation metric
                    if self.is_plateau_scheduler:
                        metric_value = val_metrics.get(self.config.early_stopping_metric, val_metrics.get("loss", 0.0))
                        self.scheduler.step(metric_value)
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        val_metrics["learning_rate"] = current_lr
                        print(f"  LR after plateau check: {current_lr:.2e}")

                    # Check early stopping
                    if self.config.early_stopping_patience is not None:
                        should_stop = self._check_early_stopping(val_metrics)
                        if should_stop:
                            print(
                                f"\nEarly stopping triggered: Validation {self.config.early_stopping_metric} "
                                f"did not improve for {self.config.early_stopping_patience} epochs "
                                f"(min_delta={self.config.early_stopping_min_delta})."
                            )
                            break

                # Save checkpoint (epoch-based)
                if (
                    self.config.save_epochs is not None
                    and (epoch + 1) % self.config.save_epochs == 0
                ):
                    self._save_checkpoint(epoch, train_metrics, is_latest=True)

        # Save final checkpoint
        self._save_checkpoint(self.current_epoch, train_metrics if not use_step_based else {}, is_latest=True)

        # Cleanup
        self.metrics_logger.close()
        self.checkpoint_manager.cleanup_old_checkpoints()
        
        # Write per-step losses to file (for Loss@k computation)
        # Task 1: Write exactly one record per optimizer step (steps 0 to max_steps-1)
        import json
        with open(self._per_step_loss_file, "w") as f:
            for step, loss in enumerate(self._per_step_losses):
                if loss is not None:
                    f.write(json.dumps({"step": step, "loss": loss}) + "\n")
        
        # Task 1: Hard assertion - exactly max_steps records
        num_records = sum(1 for l in self._per_step_losses if l is not None)
        max_steps = self.config.max_steps
        if max_steps is not None:
            use_step_based = True
            gradient_accumulation_steps = self.config.gradient_accumulation_steps
        else:
            use_step_based = False
            gradient_accumulation_steps = None
        
        if max_steps is not None and num_records != max_steps:
            raise RuntimeError(
                f"Per-step loss series length mismatch!\n"
                f"  Expected: {max_steps} records (steps 0 to {max_steps-1})\n"
                f"  Found: {num_records} records\n"
                f"  max_steps: {max_steps}\n"
                f"  gradient_accumulation_steps: {gradient_accumulation_steps}\n"
                f"  step_based_mode: {use_step_based}\n"
                f"  This indicates a bug in per-step loss tracking."
            )
        
        return history
    
    def get_run_identity_data(self) -> dict[str, Any]:
        """Get Run Identity data (permutation and mask hashes) for persistence.
        
        Returns:
            Dictionary with permutation and mask hash data
        """
        data = {}
        
        if self._permutation_hash_16 is not None:
            data["train_permutation_hash_16"] = self._permutation_hash_16
            data["train_permutation_len"] = self._permutation_len
            data["train_permutation_seed"] = self.config.seed
        
        if self._train_mask_hash_16 is not None:
            data["train_mask_step0"] = {
                "mask_count": self._train_mask_count,
                "nonpad_count": self._train_mask_nonpad_count,
                "mask_ratio": self._train_mask_ratio,
                "mask_hash_16": self._train_mask_hash_16,
            }
        
        if self._val_mask_hash_16 is not None:
            data["val_mask_step0"] = {
                "mask_count": self._val_mask_count,
                "nonpad_count": self._val_mask_nonpad_count,
                "mask_ratio": self._val_mask_ratio,
                "mask_hash_16": self._val_mask_hash_16,
            }
            # Task 2: Store val_eval_index at step 0 (should be 0)
            data["val_eval_index_step0"] = 0
        
        return data

    def _train_epoch(self, epoch: int) -> dict:
        """Train for one epoch (used only for epoch-based training)."""
        self.model.train()
        if self.task_head is not None:
            self.task_head.train()

        epoch_losses = []
        epoch_metrics = {}

        # Reset permutation for new epoch
        self._current_permutation_epoch = -1

        # Calculate batches per epoch
        num_batches = (len(self.train_dataset) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            # Get next batch (maintains state across batches)
            batch, batch_indices = self._get_next_batch(
                self.train_dataset, shuffle=self.shuffle
            )
            
            # Verification: log batch IDs at specific steps (expanded for diagnostics)
            if self.current_step in {0, 100, 200, 300, 400, 500, 1000}:
                print(f"[BATCH_VERIFICATION] Step {self.current_step}: batch_indices[:20] = {batch_indices[:20]}")
            # At step 0, print full batch indices for fairness verification
            if self.current_step == 0:
                print(f"\n[FAIRNESS VERIFICATION] Step 0 batch_indices (full batch): {batch_indices}")
                print(f"  Model type: {self.model_type if hasattr(self, 'model_type') and self.model_type else 'unknown'}")
                print(f"  Compare this with SAAB/Scratch runs to ensure identical batching")
            
            # Move batch to device
            batch = batch.to(get_device(self.config.device))

            # Training step (pass batch_indices for diagnostics)
            step_metrics = self._step(batch, batch_indices=batch_indices)

            # Only append loss if step_metrics contains it (after accumulation completes)
            if "loss" in step_metrics:
                epoch_losses.append(step_metrics["loss"])

            # Log step metrics (only when gradient accumulation completes)
            if step_metrics and self.current_step % self.config.log_steps == 0:
                self.metrics_logger.log_step(
                    self.current_step, step_metrics, phase="train"
                )

            # Save checkpoint (step-based)
            if (
                self.config.save_steps is not None
                and self.current_step % self.config.save_steps == 0
            ):
                self._save_checkpoint(epoch, step_metrics, is_latest=False)

            # Validate (step-based)
            if (
                self.val_dataset is not None
                and self.config.eval_steps is not None
                and self.current_step % self.config.eval_steps == 0
            ):
                val_metrics = self._validate()
                if self.config.save_best:
                    self._check_and_save_best(val_metrics, epoch)

            self.current_step += 1

            # Check if we've reached max_steps (shouldn't happen in epoch-based, but safety check)
            if (
                self.config.max_steps is not None
                and self.current_step >= self.config.max_steps
            ):
                break

        # Calculate epoch averages
        epoch_metrics["loss"] = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        )

        # Log epoch metrics
        if self.config.log_epochs:
            self.metrics_logger.log_epoch(epoch, epoch_metrics, phase="train")

        return epoch_metrics

    def _step(self, batch: Batch, batch_indices: list[int] | None = None) -> dict:
        """Single training step (may be a microbatch in gradient accumulation).
        
        Args:
            batch: Batch to process
            batch_indices: Optional list of sample indices (for diagnostics)
        
        Returns:
            Dictionary with step metrics. Loss is only logged when gradient accumulation completes.
        """
        # Initialize step_metrics dict to collect warnings and metrics
        step_metrics = {}

        # Apply masking for MSM-Field task (before forward pass)
        task_type = self._infer_task_type()
        msm_field_mask = None
        original_field_ids = None
        
        if task_type == "msm_field":
            # Get mask_prob from task config
            mask_prob = 0.15  # default
            if self.task_config is not None:
                if isinstance(self.task_config, dict):
                    # Direct dict with mask_prob (from model_dump())
                    mask_prob = self.task_config.get("mask_prob", 0.15)
                elif hasattr(self.task_config, "mask_prob"):
                    # Pydantic model
                    mask_prob = self.task_config.mask_prob
            
            # Get MASK_FIELD_ID from model's field embedding size
            # MASK_FIELD_ID = field_vocab_size (the last index in embedding table)
            # field_vocab_size = embedding_table_size - 1 (since we added +1 for MASK_FIELD_ID)
            # So MASK_FIELD_ID = embedding_table_size - 1
            device = batch.field_ids.device
            if hasattr(self.model, "embeddings") and hasattr(self.model.embeddings, "field_embedding"):
                field_embedding_size = self.model.embeddings.field_embedding.embedding.num_embeddings
                mask_field_id = field_embedding_size - 1  # Last index is MASK_FIELD_ID
            else:
                # Fallback: use num_fields from task_head (original vocab size)
                # MASK_FIELD_ID = num_fields (since embedding table is num_fields + 1)
                if hasattr(self.task_head, "num_fields"):
                    mask_field_id = self.task_head.num_fields
                else:
                    raise ValueError("Cannot determine MASK_FIELD_ID: model has no field embedding")
            
            # Create deterministic field-balanced mask
            # Task 2: Use separate RNG stream with batch hash for train masking
            # seed_train = base_seed + 10_000 * optimizer_step + batch_hash_int
            pad_field_id = 0  # PAD_FIELD_ID from constants
            batch_hash_int = self._compute_batch_hash_int(batch_indices)
            train_seed = self.config.seed + 10_000 * self.current_step + batch_hash_int
            msm_field_mask = make_msm_field_mask_balanced(
                field_ids=batch.field_ids,
                mask_prob=mask_prob,
                mask_field_id=mask_field_id,
                pad_field_id=pad_field_id,
                seed=train_seed,
                step=0,  # Seed already includes step, so use step=0
            )
            
            # Internal check: masking must never select padding positions
            # Padding has attention_mask == 0, so msm_field_mask should be False for those positions
            padding_mask = batch.attention_mask == 0
            if (msm_field_mask & padding_mask).any():
                raise ValueError(
                    "CRITICAL: Training mask selected padding positions! "
                    "This should never happen. Check make_msm_field_mask_balanced implementation."
                )
            
            # Store original field_ids for loss computation
            original_field_ids = batch.field_ids.clone()
            
            # Task 3: MASK_FIELD_ID consistency assertions (first MSM step only)
            if not hasattr(self, '_msm_field_assertions_done'):
                self._msm_field_assertions_done = True
                # Get num_fields from task config
                num_fields = None
                if self.task_config is not None:
                    if isinstance(self.task_config, dict):
                        num_fields = self.task_config.get("num_fields")
                    elif hasattr(self.task_config, "num_fields"):
                        num_fields = self.task_config.num_fields
                
                # Task 3: Get field embedding size from model (single source of truth)
                field_emb_size = None
                if hasattr(self.model, "embeddings") and hasattr(self.model.embeddings, "field_embedding"):
                    field_emb_size = self.model.embeddings.field_embedding.embedding.num_embeddings
                
                # Fail-fast assertions
                if num_fields is None:
                    raise ValueError("Cannot verify MASK_FIELD_ID: num_fields not found in task_config")
                if field_emb_size is None:
                    raise ValueError("Cannot verify MASK_FIELD_ID: field_embedding not found in model")
                
                # Task 3: Verify field_emb_size consistency
                # The model's field_embedding includes ALL tokens from vocab + MASK_FIELD_ID
                # MASK_FIELD_TOKEN is NOT in vocab (excluded in TagEncoder), so:
                #   field_emb_size = field_vocab_size_raw + 1
                #   num_fields = field_vocab_size_raw (includes all field IDs, PAD handled by ignore_index)
                #   Therefore: field_emb_size = num_fields + 1
                expected_field_emb_size = num_fields + 1
                if field_emb_size != expected_field_emb_size:
                    raise ValueError(
                        f"MASK_FIELD_ID consistency check FAILED:\n"
                        f"  num_fields (from task_config): {num_fields}\n"
                        f"  field_emb_size (from model): {field_emb_size}\n"
                        f"  expected_field_emb_size: {expected_field_emb_size} (num_fields + 2)\n"
                        f"  This indicates a mismatch between task config and model embedding size.\n"
                        f"  Check that MASK_FIELD_TOKEN is excluded from field vocabulary."
                    )
                
                # Task 3: Assert MASK_FIELD_ID = field_emb_size - 1
                expected_mask_field_id = field_emb_size - 1
                if mask_field_id != expected_mask_field_id:
                    raise ValueError(
                        f"MASK_FIELD_ID computation mismatch:\n"
                        f"  computed MASK_FIELD_ID: {mask_field_id}\n"
                        f"  expected MASK_FIELD_ID: {expected_mask_field_id} (field_emb_size - 1)\n"
                        f"  field_emb_size: {field_emb_size}"
                    )
                
                # Task 3: Assert original_field_ids.max() < MASK_FIELD_ID
                original_max = original_field_ids.max().item()
                if original_max >= mask_field_id:
                    raise ValueError(
                        f"Field IDs out of range:\n"
                        f"  original_field_ids.max(): {original_max}\n"
                        f"  MASK_FIELD_ID: {mask_field_id}\n"
                        f"  All field IDs must be < MASK_FIELD_ID (field IDs are 0-indexed, MASK is last)\n"
                        f"  This indicates field IDs are using indices >= MASK_FIELD_ID, which is invalid."
                    )
                
                # Task 3: Assert MASK_FIELD_ID != any vocab-mapped field id (optional but good)
                # Verify that MASK_FIELD_ID is not in the field vocabulary range
                # field_vocab_size_raw should be <= mask_field_id (since mask is last)
                # This ensures MASK_FIELD_ID is truly separate from vocab tokens
                field_vocab_size_raw = field_emb_size - 1  # Reverse: vocab_size = emb_size - 1 (for MASK_FIELD_ID)
                if mask_field_id < field_vocab_size_raw:
                    raise ValueError(
                        f"MASK_FIELD_ID overlaps with vocabulary:\n"
                        f"  MASK_FIELD_ID: {mask_field_id}\n"
                        f"  field_vocab_size_raw: {field_vocab_size_raw}\n"
                        f"  MASK_FIELD_ID must be >= field_vocab_size_raw (MASK is last index)"
                    )
                # Also verify that MASK_FIELD_ID is exactly field_vocab_size_raw (last index)
                if mask_field_id != field_vocab_size_raw:
                    raise ValueError(
                        f"MASK_FIELD_ID must be exactly field_vocab_size_raw (last index):\n"
                        f"  MASK_FIELD_ID: {mask_field_id}\n"
                        f"  field_vocab_size_raw: {field_vocab_size_raw}\n"
                        f"  field_emb_size: {field_emb_size}"
                    )
                
                # Task 3: Print consistency check
                print(f"\n[MSM CONSISTENCY] num_fields={num_fields}, field_emb_size={field_emb_size}, MASK_FIELD_ID={mask_field_id}, original_max={original_max}")
                print(f"  ✓ All consistency checks passed")
            
            # Replace masked positions with MASK_FIELD_ID
            batch.field_ids = torch.where(
                msm_field_mask,
                torch.full_like(batch.field_ids, mask_field_id),
                batch.field_ids,
            )
            
            # Log masking info at step 0 (only once per optimizer step, not per microbatch)
            if self.current_step == 0 and self._microbatch_count == 0:
                num_masked = msm_field_mask.sum().item()
                total_non_padding = (batch.attention_mask == 1).sum().item()
                mask_ratio = num_masked / total_non_padding if total_non_padding > 0 else 0.0
                
                # Compute mask hash for Run Identity
                import hashlib
                mask_hash = hashlib.sha256(msm_field_mask.cpu().numpy().tobytes()).hexdigest()[:16]
                # Store for Run Identity
                self._train_mask_hash_16 = mask_hash
                self._train_mask_count = num_masked
                self._train_mask_nonpad_count = total_non_padding
                self._train_mask_ratio = mask_ratio
                
                print(f"\n[MSM-Field Masking] Step 0 (Optimizer Step, not microbatch):")
                print(f"  - mask_prob: {mask_prob}")
                print(f"  - MASK_FIELD_ID: {mask_field_id}")
                print(f"  - Total non-padding tokens: {total_non_padding}")
                print(f"  - Masked tokens: {num_masked} ({100.0 * num_masked / total_non_padding:.2f}%)")
                print(f"  - Train mask hash (first 16 chars): {mask_hash}")
                
                # Debug: Verify batch.labels is not used for MSM-Field
                has_labels = batch.labels is not None
                print(f"  - batch.labels exists: {has_labels} (should be ignored for MSM-Field)")
                print(f"  - Using batch.field_ids for targets: True")
                print(f"  - Loss computation path: _compute_msm_field_loss (uses field_ids, not labels)")
                
                # Sample masked positions (first 10)
                masked_positions = torch.nonzero(msm_field_mask, as_tuple=False)
                if len(masked_positions) > 0:
                    sample_size = min(10, len(masked_positions))
                    sample_positions = masked_positions[:sample_size]
                    print(f"  - Sample masked positions (first {sample_size}):")
                    for pos in sample_positions:
                        b, l = pos[0].item(), pos[1].item()
                        orig_field = original_field_ids[b, l].item()
                        print(f"    Batch {b}, Position {l}: field_id {orig_field} -> MASK_FIELD_ID {mask_field_id}")
                
                # Masked target distribution (histogram of original field_ids at masked positions)
                masked_field_ids = original_field_ids[msm_field_mask]  # Get original field_ids at masked positions
                # Exclude padding (field_id == 0)
                non_pad_masked = masked_field_ids[masked_field_ids > 0]
                if len(non_pad_masked) > 0:
                    unique_fields, counts = torch.unique(non_pad_masked, return_counts=True)
                    print(f"\n[Masked Target Distribution] Step 0:")
                    print(f"  Total masked (non-padding): {len(non_pad_masked)}")
                    print(f"  Unique field IDs in masked positions: {len(unique_fields)}")
                    print(f"  Histogram:")
                    for field_id, count in zip(unique_fields, counts):
                        pct = 100.0 * count.item() / len(non_pad_masked)
                        print(f"    Field {field_id.item()}: {count.item()} tokens ({pct:.1f}%)")

        # Forward pass
        if task_type == "ranking":
            outputs = self._forward_ranking(batch)
        else:
            outputs = self._forward_single(batch)

        # Model output validation
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            self.warning_logger.warning(
                f"NaN/Inf detected in model outputs at step {self.current_step}"
            )
            # Will fail in loss computation, but log warning first

        # Check for extreme logit values
        max_logit = outputs.abs().max().item()
        if max_logit > 100:
            self.warning_logger.warning(
                f"Extreme logit values detected: max={max_logit:.2f} at step {self.current_step}"
            )

        # Step 0: Compare pre-mask vs post-mask same_field_mass (only once per optimizer step)
        # This must be after forward pass to get attention weights, but before loss computation
        if task_type == "msm_field" and self.current_step == 0 and original_field_ids is not None:
            # Check if this is the first microbatch of step 0 (to avoid repeating)
            if not hasattr(self, '_step0_diag_comparison_done'):
                self._step0_diag_comparison_done = True
                # Temporarily set to eval mode for stable diagnostics
                was_training = self.model.training
                self.model.eval()
                
                with torch.no_grad():
                    # Re-run forward pass to get attention weights (we need them for diagnostics)
                    # Note: This is a second forward pass, but only at step 0
                    if task_type == "ranking":
                        _, attention_weights_list = self.model(batch, return_attention_weights=True)
                    else:
                        _, attention_weights_list = self.model(batch, return_attention_weights=True)
                    
                    # Compute diagnostics with post-mask field_ids (current batch.field_ids)
                    diag_post_mask = {}
                    # Compute diagnostics with pre-mask field_ids (original_field_ids)
                    diag_pre_mask = {}
                    
                    num_layers = len(attention_weights_list)
                    for layer_idx, attn_weights in enumerate(attention_weights_list):
                        # Post-mask (current batch.field_ids)
                        post_mass = compute_same_field_mass(
                            attn_weights=attn_weights,
                            field_ids=batch.field_ids,
                            attention_mask=batch.attention_mask,
                        )
                        diag_post_mask[f"attn/same_field_mass_layer_{layer_idx}"] = post_mass
                        
                        # Pre-mask (original_field_ids)
                        pre_mass = compute_same_field_mass(
                            attn_weights=attn_weights,
                            field_ids=original_field_ids,
                            attention_mask=batch.attention_mask,
                        )
                        diag_pre_mask[f"attn/same_field_mass_layer_{layer_idx}"] = pre_mass
                
                # Restore training mode
                if was_training:
                    self.model.train()
                
                print(f"\n[MSM Diagnostics Comparison] Step 0 (Pre-Mask vs Post-Mask):")
                for layer_idx in range(num_layers):
                    pre_mass = diag_pre_mask.get(f"attn/same_field_mass_layer_{layer_idx}", 0.0)
                    post_mass = diag_post_mask.get(f"attn/same_field_mass_layer_{layer_idx}", 0.0)
                    diff = post_mass - pre_mass
                    pct_change = 100.0 * diff / max(pre_mass, 1e-8) if pre_mass > 1e-8 else 0.0
                    print(f"  Layer {layer_idx}:")
                    print(f"    Pre-mask same_field_mass:  {pre_mass:.6f}")
                    print(f"    Post-mask same_field_mass: {post_mass:.6f}")
                    print(f"    Difference: {diff:+.6f} ({pct_change:+.2f}% change)")

        # Compute loss
        # For MSM-Field, pass mask and original field_ids
        if task_type == "msm_field" and msm_field_mask is not None and original_field_ids is not None:
            loss = self._compute_loss(batch, outputs, msm_field_mask=msm_field_mask, original_field_ids=original_field_ids)
            # Compute masked-field accuracy for MSM-Field
            accuracy = self._compute_msm_field_accuracy(outputs, original_field_ids, msm_field_mask)
            if accuracy is not None:
                step_metrics["msm_field/accuracy"] = accuracy
        else:
            loss = self._compute_loss(batch, outputs)

        # Loss validation
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(
                f"NaN/Inf loss detected at step {self.current_step}. "
                f"Loss value: {loss.item()}. Stopping training."
            )

        # Track if loss was negative (before clamping)
        if loss < 0:
            self.warning_logger.warning(
                f"Negative loss detected: {loss.item():.6f} at step {self.current_step}"
            )
            # Clamp to small positive value to prevent further issues
            loss = torch.clamp(loss, min=1e-8)

        # Store raw loss (before division) for accumulation tracking
        raw_loss_value = loss.item()
        self._accumulated_losses.append(raw_loss_value)
        self._microbatch_count += 1

        # Check if gradient accumulation is complete
        is_accumulation_complete = self._microbatch_count % self.config.gradient_accumulation_steps == 0
        
        # Label diversity diagnostics (cheap, per optimizer step)
        # Only run for classification tasks (not for MSM-Field or other self-supervised tasks)
        if is_accumulation_complete:
            task_type_check = self._infer_task_type()
            if task_type_check == "classification":
                self._log_label_diversity_diagnostics(batch, batch_indices)
            
            # Masked target distribution at step 100 (MSM-Field only)
            if task_type_check == "msm_field" and self.current_step == 100 and msm_field_mask is not None and original_field_ids is not None:
                masked_field_ids = original_field_ids[msm_field_mask]
                non_pad_masked = masked_field_ids[masked_field_ids > 0]
                if len(non_pad_masked) > 0:
                    unique_fields, counts = torch.unique(non_pad_masked, return_counts=True)
                    print(f"\n[Masked Target Distribution] Step 100:")
                    print(f"  Total masked (non-padding): {len(non_pad_masked)}")
                    print(f"  Unique field IDs in masked positions: {len(unique_fields)}")
                    print(f"  Histogram:")
                    for field_id, count in zip(unique_fields, counts):
                        pct = 100.0 * count.item() / len(non_pad_masked)
                        print(f"    Field {field_id.item()}: {count.item()} tokens ({pct:.1f}%)")
        
        # CRITICAL DIAGNOSTICS: Catastrophic instability investigation
        # Log diagnostics at steps 200, 300, 400, 500 (when accumulation completes)
        if is_accumulation_complete and self.current_step in {200, 300, 400, 500}:
            self._log_catastrophic_instability_diagnostics(
                batch, outputs, loss, batch_indices, raw_loss_value
            )
        
        # Diagnostic logging at specific optimizer steps (only when accumulation completes)
        if is_accumulation_complete and self.current_step in {100, 200, 400}:
            print(f"\n[DIAGNOSTIC] Optimizer Step {self.current_step} (after {self.config.gradient_accumulation_steps} microbatches):")
            print(f"  gradient_accumulation_steps = {self.config.gradient_accumulation_steps}")
            print(f"  microbatch_count = {self._microbatch_count}")
            
            # Show all microbatches in this accumulation cycle
            print(f"  All microbatches in this optimizer step:")
            for mb_idx in range(len(self._accumulated_batch_indices)):
                mb_batch_ids = self._accumulated_batch_indices[mb_idx]
                mb_labels = self._accumulated_labels[mb_idx] if mb_idx < len(self._accumulated_labels) else []
                mb_loss = self._accumulated_losses[mb_idx]
                print(f"    Microbatch {mb_idx + 1}/{self.config.gradient_accumulation_steps}:")
                print(f"      batch_indices[:20] = {mb_batch_ids[:20]}")
                if mb_labels:
                    print(f"      labels[:20] = {mb_labels[:20]}")
                print(f"      loss = {mb_loss:.6f}")
            
            # Show last microbatch details (logits, attention mask)
            print(f"  Last microbatch details:")
            print(f"    logits.shape = {outputs.shape}")
            if outputs.numel() > 0:
                probs = torch.softmax(outputs[0], dim=-1)
                print(f"    logits.softmax(-1)[0][:5] = {probs[:5].cpu().tolist()}")
            if hasattr(batch, 'attention_mask') and batch.attention_mask is not None:
                avg_seq_len = batch.attention_mask.sum(dim=1).float().mean().item()
                print(f"    attention_mask.sum(dim=1).mean() = {avg_seq_len:.2f}")
            
            # Summary
            print(f"  Summary:")
            print(f"    accumulated_losses (all {len(self._accumulated_losses)} microbatches) = {self._accumulated_losses}")
            avg_loss = sum(self._accumulated_losses) / len(self._accumulated_losses)
            print(f"    average_loss (logged) = {avg_loss:.6f}")
            print(f"    model.training = {self.model.training}")
            
            # Check for batch repetition within accumulation cycle
            all_batch_ids_flat = []
            for mb_ids in self._accumulated_batch_indices:
                all_batch_ids_flat.extend(mb_ids)
            unique_ids = set(all_batch_ids_flat)
            if len(all_batch_ids_flat) != len(unique_ids):
                print(f"    ⚠️  WARNING: Batch repetition detected! {len(all_batch_ids_flat)} total IDs, {len(unique_ids)} unique")
            else:
                print(f"    ✓ No batch repetition ({len(unique_ids)} unique IDs across all microbatches)")
            print()

        # Early stopping: Check for zero loss collapse (use raw loss)
        if raw_loss_value < 1e-8:  # Consider loss as zero if very small
            self._zero_loss_streak += 1
            if self.config.early_stop_zero_loss_steps is not None:
                # Warn at 50% threshold
                if (
                    self._zero_loss_streak
                    == self.config.early_stop_zero_loss_steps // 2
                ):
                    self.warning_logger.warning(
                        f"Zero loss streak at {self._zero_loss_streak} steps "
                        f"(50% of threshold {self.config.early_stop_zero_loss_steps})"
                    )
                # Stop if threshold reached
                if self._zero_loss_streak >= self.config.early_stop_zero_loss_steps:
                    raise RuntimeError(
                        f"Training stopped: Loss has been zero for "
                        f"{self._zero_loss_streak} consecutive steps "
                        f"(threshold: {self.config.early_stop_zero_loss_steps}). "
                        f"Model has stopped learning."
                    )
        else:
            # Reset streak when loss is positive
            self._zero_loss_streak = 0

        # Backward pass (with gradient accumulation)
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        # Clear MPS cache after backward pass to free memory (only for MPS device)
        if self.config.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Update weights (if gradient accumulation is complete)
        if is_accumulation_complete:
            # Clear MPS cache before gradient validation to free memory (only for MPS device)
            if self.config.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Gradient validation
            has_nan_grad = False
            nan_grad_params = []
            param_count = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Check for NaN/Inf more memory-efficiently
                    grad_data = param.grad.data
                    if torch.isnan(grad_data).any() or torch.isinf(grad_data).any():
                        nan_grad_params.append(name)
                        has_nan_grad = True
                    # Clear cache periodically during validation for large models (every 50 params)
                    param_count += 1
                    if (self.config.device == "mps" and torch.backends.mps.is_available() 
                        and param_count % 50 == 0):
                        torch.mps.empty_cache()

            if has_nan_grad:
                self.warning_logger.warning(
                    f"NaN/Inf gradients detected in {len(nan_grad_params)} parameters "
                    f"at step {self.current_step}: {', '.join(nan_grad_params[:5])}"
                    + (
                        f" and {len(nan_grad_params) - 5} more"
                        if len(nan_grad_params) > 5
                        else ""
                    )
                )
                # Zero out gradients to prevent optimizer step with invalid gradients
                self.optimizer.zero_grad()
                # Skip this step but still log the warning
                current_lr = (
                    self.scheduler.get_last_lr()[0]
                    if self.scheduler is not None and not self.is_plateau_scheduler
                    else self.optimizer.param_groups[0]["lr"]
                )
                step_metrics["grad_norm"] = float("inf")
                step_metrics["learning_rate"] = current_lr
                return step_metrics

            # CRITICAL DIAGNOSTIC: Gradient norm measurement verification
            # Compute grad_norm BEFORE clipping to verify measurement correctness
            if self.current_step in {200, 300, 400, 500}:
                # Compute grad_norm before clipping
                all_params = list(self.model.parameters())
                if self.task_head is not None:
                    all_params.extend(list(self.task_head.parameters()))
                
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    all_params, float('inf')  # No clipping, just compute norm
                ).item()
                print(f"\n[GRAD_NORM_DIAGNOSTIC] Step {self.current_step}:")
                print(f"  grad_norm_before_clipping = {grad_norm_before_clip:.6e}")
                print(f"  max_grad_norm (clip threshold) = {self.config.max_grad_norm}")
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                grad_norm_value = grad_norm.item()
                step_metrics["grad_norm"] = grad_norm_value
                
                if self.current_step in {200, 300, 400, 500}:
                    print(f"  grad_norm_after_clipping = {grad_norm_value:.6e}")
                    print(f"  (Note: This is the logged grad_norm value)")
                    print()

                # Gradient collapse detection
                if grad_norm_value < 1e-6:
                    self._small_grad_streak += 1
                    if self._small_grad_streak == 1 or self._small_grad_streak % 100 == 0:
                        self.warning_logger.warning(
                            f"Gradient collapse detected: grad_norm={grad_norm_value:.2e} "
                            f"at step {self.current_step} (streak: {self._small_grad_streak} steps)"
                        )
                else:
                    # Reset streak when gradients are meaningful
                    if self._small_grad_streak > 0:
                        self._small_grad_streak = 0

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Clear MPS cache to prevent memory issues (only for MPS device)
            if self.config.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # LR scheduler step (only for time-based schedulers, not ReduceLROnPlateau)
            if self.scheduler is not None and not self.is_plateau_scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                step_metrics["learning_rate"] = current_lr
                # Log LR changes for first few steps and periodically
                if self.current_step < 10 or (self.current_step % 1000 == 0):
                    if hasattr(self.scheduler, 'last_epoch'):
                        print(f"[DIAGNOSTIC] Step {self.current_step}: LR = {current_lr:.8f}, scheduler.last_epoch = {self.scheduler.last_epoch}")
            else:
                # Constant LR or ReduceLROnPlateau - log from optimizer
                # Verify it matches config (should be constant)
                actual_lr = self.optimizer.param_groups[0]["lr"]
                step_metrics["learning_rate"] = actual_lr
                # Log warning if LR changed unexpectedly (shouldn't happen with constant schedule)
                if self.current_step == 0:
                    if abs(actual_lr - self.config.learning_rate) > 1e-10:
                        self.warning_logger.warning(
                            f"Step 0 LR mismatch: config={self.config.learning_rate:.2e}, "
                            f"actual={actual_lr:.2e}"
                        )
            
            # Only add loss to metrics when gradient accumulation is complete
            # Use average loss across all microbatches
            avg_loss = sum(self._accumulated_losses) / len(self._accumulated_losses)
            step_metrics["loss"] = avg_loss
            # Reset accumulated losses, batch indices, labels, and microbatch count for next cycle
            self._accumulated_losses = []
            self._accumulated_batch_indices = []
            self._accumulated_labels = []
            self._microbatch_count = 0
        else:
            # Return empty dict - loss will be logged when accumulation completes
            step_metrics = {}

        return step_metrics

    def _log_label_diversity_diagnostics(self, batch: Batch, batch_indices: list[int] | None) -> None:
        """Log label diversity diagnostics per optimizer step (cheap check).
        
        Checks:
        - Unique label count in last microbatch and across accumulation cycle
        - Label histogram for the optimizer step
        - Asserts labels are in valid range
        
        NOTE: This should only be called for classification tasks.
        """
        # Guard: Only run for classification tasks
        task_type = self._infer_task_type()
        if task_type != "classification":
            return
        
        if batch.labels is None:
            return
        
        # Get labels from last microbatch
        labels_last_mb = batch.labels.cpu().tolist()
        unique_labels_last_mb = set(labels_last_mb)
        
        # Get labels from all microbatches in accumulation cycle
        all_labels_cycle = []
        for mb_labels in self._accumulated_labels:
            all_labels_cycle.extend(mb_labels)
        unique_labels_cycle = set(all_labels_cycle) if all_labels_cycle else set()
        
        # Log at specific steps or if label diversity is low
        should_log = (
            self.current_step in {0, 50, 100, 200, 300, 400, 500}
            or len(unique_labels_last_mb) == 1  # Warning: homogeneous batch
            or len(unique_labels_cycle) <= 2  # Warning: very low diversity
        )
        
        if should_log:
            print(f"\n[LABEL DIVERSITY] Step {self.current_step}:")
            print(f"  Last microbatch: {len(unique_labels_last_mb)} unique labels out of {len(labels_last_mb)} samples")
            print(f"    Unique labels: {sorted(unique_labels_last_mb)}")
            
            # CRITICAL: Verify logging is truthful - print actual labels for steps 0 and 1
            if self.current_step in {0, 1}:
                print(f"    [VERIFICATION] Actual labels[:32] = {labels_last_mb[:32]}")
                # Histogram for this microbatch
                mb_counts = Counter(labels_last_mb)
                print(f"    [VERIFICATION] Label counts in this microbatch: {dict(sorted(mb_counts.items()))}")
                # Check if it's stratified (approximately balanced)
                if len(unique_labels_last_mb) == num_classes if (num_classes := getattr(self.task_head, 'num_classes', None) if self.task_head else None) else False:
                    expected_per_class = len(labels_last_mb) // num_classes
                    is_stratified = all(abs(count - expected_per_class) <= 2 for count in mb_counts.values())
                    if is_stratified:
                        print(f"    [VERIFICATION] ⚠️  Batch appears STRATIFIED (balanced)! ~{expected_per_class} samples per class")
                        print(f"    This suggests permutation may be interleaving classes, not purely random.")
                    else:
                        print(f"    [VERIFICATION] Batch is NOT stratified (random distribution)")
            
            if len(unique_labels_last_mb) == 1:
                print(f"    ⚠️  WARNING: Batch is label-homogeneous! All samples have label {labels_last_mb[0]}")
            
            if all_labels_cycle:
                print(f"  Accumulation cycle: {len(unique_labels_cycle)} unique labels across {len(all_labels_cycle)} samples")
                print(f"    Unique labels: {sorted(unique_labels_cycle)}")
                
                # Label histogram
                label_counts = Counter(all_labels_cycle)
                print(f"    Label histogram: {dict(sorted(label_counts.items()))}")
                
                # For steps 0 and 1, also verify accumulation cycle labels
                if self.current_step in {0, 1}:
                    print(f"    [VERIFICATION] All labels in accumulation cycle: {all_labels_cycle[:64]}... (showing first 64)")
                    cycle_counts = Counter(all_labels_cycle)
                    print(f"    [VERIFICATION] Accumulation cycle label counts: {dict(sorted(cycle_counts.items()))}")
                
                # For steps 0 and 1, also verify accumulation cycle labels
                if self.current_step in {0, 1}:
                    print(f"    [VERIFICATION] All labels in accumulation cycle: {all_labels_cycle[:64]}... (showing first 64)")
                    cycle_counts = Counter(all_labels_cycle)
                    print(f"    [VERIFICATION] Accumulation cycle label counts: {dict(sorted(cycle_counts.items()))}")
            
            # Assert labels are in valid range
            if self.task_type == "classification":
                num_classes = getattr(self.task_head, 'num_classes', None) if self.task_head else None
                if num_classes:
                    invalid_labels = [label_val for label_val in unique_labels_cycle if label_val < 0 or label_val >= num_classes]
                    if invalid_labels:
                        print(f"    ⚠️  ERROR: Invalid labels detected! {invalid_labels} (expected [0, {num_classes-1}])")
                    else:
                        print(f"    ✓ All labels in valid range [0, {num_classes-1}]")
            print()

    def _log_catastrophic_instability_diagnostics(
        self,
        batch: Batch,
        outputs: torch.Tensor,
        loss: torch.Tensor,
        batch_indices: list[int] | None,
        raw_loss_value: float,
    ) -> None:
        """Comprehensive diagnostic logging for catastrophic instability investigation.
        
        Logs all critical information at steps 200, 300, 400, 500 to identify root cause.
        """
        print(f"\n{'='*80}")
        print(f"[CATASTROPHIC INSTABILITY DIAGNOSTIC] Optimizer Step {self.current_step}")
        print(f"{'='*80}")
        
        # 1. Inspect labels at step 400
        if batch.labels is not None:
            labels = batch.labels
            print(f"\n[1] LABELS INSPECTION:")
            print(f"  labels.shape = {labels.shape}")
            print(f"  labels.dtype = {labels.dtype}")
            print(f"  labels.device = {labels.device}")
            print(f"  labels.min() = {labels.min().item()}")
            print(f"  labels.max() = {labels.max().item()}")
            unique_labels = torch.unique(labels)
            print(f"  torch.unique(labels) = {unique_labels.cpu().tolist()}")
            print(f"  num_unique_labels = {len(unique_labels)}")
            
            # Assert all labels are integers in [0, 13] for 14-class classification
            if self.task_type == "classification":
                num_classes = getattr(self.task_head, 'num_classes', None) if self.task_head else None
                if num_classes:
                    if labels.min() < 0 or labels.max() >= num_classes:
                        print(f"  ⚠️  ERROR: Labels out of range! Expected [0, {num_classes-1}], got [{labels.min().item()}, {labels.max().item()}]")
                    else:
                        print(f"  ✓ Labels in valid range [0, {num_classes-1}]")
            
            # Check for dtype/device mismatches
            if labels.dtype != torch.long:
                print(f"  ⚠️  WARNING: Labels dtype is {labels.dtype}, expected torch.long")
            if labels.device != outputs.device:
                print(f"  ⚠️  WARNING: Labels device ({labels.device}) != outputs device ({outputs.device})")
        
        # 2. Inspect logits before softmax at steps 400 and 500
        print(f"\n[2] LOGITS INSPECTION (before softmax/clamping):")
        print(f"  outputs.shape = {outputs.shape}")
        print(f"  outputs.dtype = {outputs.dtype}")
        print(f"  outputs.device = {outputs.device}")
        print(f"  outputs.min() = {outputs.min().item():.6f}")
        print(f"  outputs.max() = {outputs.max().item():.6f}")
        print(f"  outputs.mean() = {outputs.mean().item():.6f}")
        print(f"  outputs.std() = {outputs.std().item():.6f}")
        
        # Check for NaN/Inf
        is_finite = torch.isfinite(outputs).all()
        has_nan = torch.isnan(outputs).any()
        has_inf = torch.isinf(outputs).any()
        print(f"  torch.isfinite(outputs).all() = {is_finite.item()}")
        print(f"  torch.isnan(outputs).any() = {has_nan.item()}")
        print(f"  torch.isinf(outputs).any() = {has_inf.item()}")
        
        if not is_finite:
            nan_count = torch.isnan(outputs).sum().item()
            inf_count = torch.isinf(outputs).sum().item()
            print(f"  ⚠️  ERROR: Non-finite values detected! NaN count: {nan_count}, Inf count: {inf_count}")
        
        # Check for extremely large logits
        if outputs.abs().max() > 50:
            print(f"  ⚠️  WARNING: Extreme logit values detected! |logit| > 50")
            print(f"  Max absolute logit: {outputs.abs().max().item():.2f}")
        
        # Show softmax output for step 400 (where it was exactly [0,0,...,1.0])
        if self.current_step == 400:
            task_type = self._infer_task_type()
            probs = torch.softmax(outputs, dim=-1)
            print(f"\n  Softmax output (first sample):")
            
            if task_type == "classification":
                # Classification: outputs are [batch, num_classes]
                print(f"    probs[0] = {probs[0].cpu().tolist()}")
                print(f"    probs[0].max() = {probs[0].max().item():.6f}")
                print(f"    probs[0].argmax() = {probs[0].argmax().item()}")
                if len(probs.shape) == 2 and probs.shape[1] > 0:
                    expected_one_hot = torch.tensor([0.0] * (probs.shape[1] - 1) + [1.0], device=probs.device)
                    if (probs[0] == expected_one_hot).all():
                        print(f"    ⚠️  WARNING: Softmax is exactly [0,0,...,1.0] - extreme confidence!")
            elif task_type == "msm_field":
                # MSM-Field: outputs are [batch, seq_len, num_fields]
                # Check if any token position has extreme confidence
                print(f"    probs.shape = {probs.shape}")
                print(f"    probs[0].max() = {probs[0].max().item():.6f}")
                print(f"    probs[0].mean() = {probs[0].mean().item():.6f}")
                # Check for extreme confidence at any position
                max_probs_per_token = probs[0].max(dim=-1).values
                if (max_probs_per_token > 0.99).any():
                    extreme_count = (max_probs_per_token > 0.99).sum().item()
                    print(f"    ⚠️  WARNING: {extreme_count} token positions have >99% confidence!")
            else:
                # Other tasks: just show basic info
                print(f"    probs.shape = {probs.shape}")
                print(f"    probs[0].max() = {probs[0].max().item():.6f}")
        
        # 3. Inspect token IDs for the problematic batch
        print(f"\n[3] TOKEN IDs INSPECTION:")
        if hasattr(batch, 'token_ids') and batch.token_ids is not None:
            token_ids = batch.token_ids
            print(f"  token_ids.shape = {token_ids.shape}")
            print(f"  token_ids.min() = {token_ids.min().item()}")
            print(f"  token_ids.max() = {token_ids.max().item()}")
            
            # Get vocab size from model
            vocab_size = None
            if hasattr(self.model, 'token_embedding'):
                if hasattr(self.model.token_embedding, 'num_embeddings'):
                    vocab_size = self.model.token_embedding.num_embeddings
            elif hasattr(self.model, 'embeddings'):
                if hasattr(self.model.embeddings, 'token_embedding'):
                    if hasattr(self.model.embeddings.token_embedding, 'num_embeddings'):
                        vocab_size = self.model.embeddings.token_embedding.num_embeddings
            
            if vocab_size:
                print(f"  model vocab_size = {vocab_size}")
                if token_ids.max() >= vocab_size:
                    print(f"  ⚠️  ERROR: token_ids.max() ({token_ids.max().item()}) >= vocab_size ({vocab_size})")
                else:
                    print(f"  ✓ All token IDs < vocab_size")
            
            # Check for negative or invalid IDs
            if token_ids.min() < 0:
                print(f"  ⚠️  ERROR: Negative token IDs detected! min = {token_ids.min().item()}")
            
            # Check padding ID (usually 0)
            pad_id = getattr(batch, 'pad_token_id', 0)
            non_pad_mask = token_ids != pad_id
            print(f"  Non-padding tokens: {non_pad_mask.sum().item()} / {token_ids.numel()}")
        else:
            print(f"  ⚠️  WARNING: token_ids not available in batch")
        
        # 4. Verify gradient norm measurement (will be logged after backward pass)
        print(f"\n[4] GRADIENT NORM MEASUREMENT:")
        print(f"  (Gradient norm will be logged after backward pass)")
        print(f"  Measurement location: after loss.backward(), before gradient clipping")
        
        # 5. Check AMP / mixed precision
        print(f"\n[5] AMP / MIXED PRECISION STATUS:")
        try:
            # Check if torch.cuda.amp is available (without shadowing torch module)
            has_amp = hasattr(torch.cuda, 'amp')
            if has_amp:
                print(f"  torch.cuda.amp available: True")
                print(f"  Note: Check if autocast is used in forward pass")
            # Check for scaler
            if hasattr(self, 'scaler'):
                print(f"  GradScaler present: True")
                if hasattr(self.scaler, 'get_scale'):
                    print(f"  Current scale: {self.scaler.get_scale()}")
            else:
                print(f"  GradScaler present: False (no AMP scaler found)")
        except (ImportError, AttributeError):
            print(f"  torch.cuda.amp not available (CPU/MPS device)")
        
        # Check device and dtype
        print(f"  Device: {self.config.device}")
        print(f"  Model dtype: {next(self.model.parameters()).dtype}")
        if outputs.dtype == torch.float16:
            print(f"  ⚠️  WARNING: Outputs are float16 - potential precision issues!")
        elif outputs.dtype == torch.bfloat16:
            print(f"  Outputs are bfloat16 (mixed precision)")
        else:
            print(f"  Outputs are float32 (full precision)")
        
        # 6. Inspect loss computation
        print(f"\n[6] LOSS COMPUTATION INSPECTION:")
        print(f"  loss.shape = {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
        print(f"  loss.item() = {raw_loss_value:.6f}")
        print(f"  loss.dtype = {loss.dtype}")
        
        # Verify loss is per-example (classification should be [batch] -> scalar after mean)
        if self.task_type == "classification":
            print(f"  Task type: classification")
            print(f"  Expected: loss computed from [batch, 14] logits -> scalar (mean reduction)")
            if outputs.dim() == 2 and outputs.shape[1] == 14:
                print(f"  ✓ Logits shape correct: [batch={outputs.shape[0]}, num_classes=14]")
            else:
                print(f"  ⚠️  WARNING: Unexpected logits shape: {outputs.shape}")
        
        # Check if loss has .detach() or graph break
        if not loss.requires_grad:
            print(f"  ⚠️  ERROR: loss.requires_grad = False (graph broken!)")
        else:
            print(f"  ✓ loss.requires_grad = True (graph intact)")
        
        # 7. Save step 400 batch for re-analysis
        if self.current_step == 400 and batch_indices is not None:
            print(f"\n[7] SAVING STEP 400 BATCH FOR RE-ANALYSIS:")
            self._step400_batch_indices = batch_indices.copy()
            # Create a deep copy of batch (move to CPU to avoid device issues)
            import copy
            self._step400_batch = copy.deepcopy(batch)
            # Move to CPU for storage
            if hasattr(self._step400_batch, 'token_ids') and self._step400_batch.token_ids is not None:
                self._step400_batch.token_ids = self._step400_batch.token_ids.cpu()
            if hasattr(self._step400_batch, 'labels') and self._step400_batch.labels is not None:
                self._step400_batch.labels = self._step400_batch.labels.cpu()
            print(f"  Saved batch_indices: {batch_indices[:20]}... (total: {len(batch_indices)})")
            print(f"  Batch saved to self._step400_batch (on CPU)")
            print(f"  To re-run: model.eval(); with torch.no_grad(): outputs = model(step400_batch.to(device))")
        
        print(f"{'='*80}\n")

    def _compute_loss(
        self,
        batch: Batch,
        outputs: torch.Tensor,
        msm_field_mask: torch.Tensor | None = None,
        original_field_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss for batch.

        Args:
            batch: Batch object (may contain labels)
            outputs: Task head outputs (logits, predictions, etc.)
            msm_field_mask: Optional mask for MSM-Field task [batch, seq_len] (True = masked)
            original_field_ids: Optional original field_ids for MSM-Field task [batch, seq_len]
                                (before masking was applied)

        Returns:
            Loss tensor

        Raises:
            ValueError: If labels are required but missing
        """
        if self.task_head is None:
            # No task head: dummy loss (for backward compatibility)
            return outputs.mean()

        # Get task type
        task_type = self._infer_task_type()

        # MSM-Field uses field_ids from batch, not labels
        if task_type == "msm_field":
            # Use original_field_ids if provided (before masking), otherwise use batch.field_ids
            field_ids = original_field_ids if original_field_ids is not None else batch.field_ids
            return self._compute_msm_field_loss(outputs, field_ids, mask=msm_field_mask)
        
        # Other tasks require labels
        if batch.labels is None:
            raise ValueError(
                "Labels are required for supervised learning. "
                "Batch must contain 'labels' field."
            )

        if task_type == "classification":
            return self._compute_classification_loss(outputs, batch.labels)
        elif task_type == "regression":
            return self._compute_regression_loss(outputs, batch.labels)
        elif task_type == "token_classification":
            return self._compute_token_classification_loss(
                outputs, batch.labels, batch.attention_mask
            )
        elif task_type == "ranking":
            return self._compute_ranking_loss(outputs, batch.labels)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _infer_task_type(self) -> str:
        """Infer task type from task_head or loss_fn.

        Returns:
            Task type string
        """
        if self.task_type is not None:
            return self.task_type

        # Try to infer from task_head class name
        if self.task_head is not None:
            class_name = self.task_head.__class__.__name__
            if "MSMField" in class_name:
                return "msm_field"
            elif "Classification" in class_name:
                return "classification"
            elif "Regression" in class_name:
                return "regression"
            elif "TokenClassification" in class_name:
                return "token_classification"
            elif "Ranking" in class_name:
                return "ranking"

        # Try to infer from loss function type
        if self.loss_fn is not None:
            loss_class_name = self.loss_fn.__class__.__name__
            if "CrossEntropy" in loss_class_name or "BCEWithLogits" in loss_class_name:
                return "classification"
            elif "MSELoss" in loss_class_name or "L1Loss" in loss_class_name:
                return "regression"
            elif "Ranking" in loss_class_name:
                return "ranking"

        # Default to classification
        return "classification"

    def _compute_classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification loss with numerical stability checks.

        Args:
            logits: Task head outputs [batch, num_classes] or [batch, num_classes] for multi-label
            labels: Labels [batch] (class indices) or [batch, num_classes] (multi-label binary vectors)

        Returns:
            Loss tensor
        """
        # NOTE: Logit clamping removed - rely on LayerNorm normalization instead
        # LayerNorm before classifier should prevent logits explosion
        # If numerical issues occur, they indicate a deeper problem that should be fixed
        # rather than masked with clamping.

        loss = self.loss_fn(logits, labels)

        # Safety clamp: Additional protection against numerical issues
        # Clamp to [0, 100] range (100 is reasonable upper bound for 14 classes)
        loss = torch.clamp(loss, min=0.0, max=100.0)

        return loss

    def _compute_regression_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute regression loss.

        Args:
            predictions: Task head outputs [batch, num_targets]
            labels: Labels [batch, num_targets] (continuous values)

        Returns:
            Loss tensor
        """
        return self.loss_fn(predictions, labels)

    def _compute_token_classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute token classification loss.

        Args:
            logits: Task head outputs [batch, seq_len, num_labels]
            labels: Labels [batch, seq_len] (label indices per token)
            attention_mask: Attention mask [batch, seq_len] (1 = valid, 0 = pad)

        Returns:
            Loss tensor
        """
        # Reshape logits: [batch, seq_len, num_labels] -> [batch * seq_len, num_labels]
        batch_size, seq_len, num_labels = logits.shape
        logits_flat = logits.view(-1, num_labels)

        # Reshape labels: [batch, seq_len] -> [batch * seq_len]
        labels_flat = labels.view(-1)

        # Mask padding positions (set to -100 for ignore_index in CrossEntropyLoss)
        # Padding positions have attention_mask = 0
        mask_flat = attention_mask.view(-1)
        ignore_index = torch.tensor(-100, dtype=labels.dtype, device=labels.device)
        labels_flat = torch.where(mask_flat == 1, labels_flat, ignore_index)

        return self.loss_fn(logits_flat, labels_flat)

    def _compute_msm_field_loss(
        self,
        logits: torch.Tensor,
        field_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MSM-Field loss.

        Args:
            logits: Task head outputs [batch, seq_len, num_fields]
            field_ids: Field IDs [batch, seq_len] (field indices per token, 0 = padding)
            mask: Optional mask [batch, seq_len] (True = masked, compute loss only on these)

        Returns:
            Loss tensor
        """
        batch_size, seq_len, num_fields = logits.shape
        
        if mask is not None:
            # Only compute loss on masked tokens
            # Reshape to [batch * seq_len, ...]
            logits_flat = logits.view(-1, num_fields)  # [B*L, num_fields]
            field_ids_flat = field_ids.view(-1)  # [B*L]
            mask_flat = mask.view(-1)  # [B*L]
            
            # Filter to only masked positions
            masked_logits = logits_flat[mask_flat]  # [num_masked, num_fields]
            masked_field_ids = field_ids_flat[mask_flat]  # [num_masked]
            
            if masked_logits.shape[0] == 0:
                # No masked tokens, return zero loss
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            # Compute loss only on masked tokens
            loss = self.loss_fn(masked_logits, masked_field_ids)
        else:
            # Fallback: compute on all non-padding tokens (backward compatibility)
            # Reshape logits: [batch, seq_len, num_fields] -> [batch * seq_len, num_fields]
            logits_flat = logits.view(-1, num_fields)
            
            # Reshape field_ids: [batch, seq_len] -> [batch * seq_len]
            field_ids_flat = field_ids.view(-1)
            
            # Loss function already has ignore_index=0 (PAD_FIELD_ID)
            # Padding positions in field_ids have value 0, which will be ignored
            loss = self.loss_fn(logits_flat, field_ids_flat)

        return loss

    def _compute_msm_field_accuracy(
        self,
        logits: torch.Tensor,
        field_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> float | None:
        """Compute masked-field accuracy for MSM-Field task.
        
        Accuracy is computed over masked positions only:
        accuracy = (argmax(logits_masked) == field_id_masked).float().mean()
        
        Args:
            logits: Task head outputs [batch, seq_len, num_fields]
            field_ids: Field IDs [batch, seq_len] (field indices per token, 0 = padding)
            mask: Mask [batch, seq_len] (True = masked, compute accuracy only on these)
        
        Returns:
            Accuracy as float (0.0 to 1.0), or None if no masked tokens
        """
        batch_size, seq_len, num_fields = logits.shape
        
        # Reshape to [batch * seq_len, ...]
        logits_flat = logits.view(-1, num_fields)  # [B*L, num_fields]
        field_ids_flat = field_ids.view(-1)  # [B*L]
        mask_flat = mask.view(-1)  # [B*L]
        
        # Filter to only masked positions
        masked_logits = logits_flat[mask_flat]  # [num_masked, num_fields]
        masked_field_ids = field_ids_flat[mask_flat]  # [num_masked]
        
        if masked_logits.shape[0] == 0:
            # No masked tokens
            return None
        
        # Compute predictions: argmax over num_fields dimension
        predictions = masked_logits.argmax(dim=-1)  # [num_masked]
        
        # Compute accuracy: (predictions == targets).float().mean()
        correct = (predictions == masked_field_ids).float()
        accuracy = correct.mean().item()
        
        return accuracy

    def _forward_single(self, batch: Batch) -> torch.Tensor:
        """Forward pass for single-sequence tasks.

        Args:
            batch: Batch object

        Returns:
            Task head outputs
        """
        outputs = self.model(batch)
        if self.task_head is not None:
            # Pass attention_mask to task head (required for mean pooling)
            outputs = self.task_head(outputs, attention_mask=batch.attention_mask)
        return outputs

    def _forward_ranking(self, batch: Batch) -> torch.Tensor:
        """Forward pass for ranking task.

        Args:
            batch: Batch with token_ids_b (ranking batch)

        Returns:
            Ranking scores [batch_size] where higher score means seq_a is better than seq_b
        """
        # Extract batch_a and batch_b
        batch_a = self._extract_batch_a(batch)
        batch_b = self._extract_batch_b(batch)

        # Encode both sequences
        outputs_a = self.model(batch_a)  # [batch, seq_len, d_model]
        outputs_b = self.model(batch_b)  # [batch, seq_len, d_model]

        # Pool to get representations (use mean pooling over all tokens)
        # Mean pool both sequences using attention masks
        from saab_v3.tasks.pooling import MeanPooling
        mean_pooling = MeanPooling()
        repr_a = mean_pooling(outputs_a, batch_a.attention_mask)  # [batch, d_model]
        repr_b = mean_pooling(outputs_b, batch_b.attention_mask)  # [batch, d_model]

        # Apply ranking head (needs TWO representations)
        scores = self.task_head(repr_a, repr_b)  # [batch]

        return scores

    def _extract_batch_a(self, batch: Batch) -> Batch:
        """Extract batch A from ranking batch.

        Args:
            batch: Ranking batch with _b fields

        Returns:
            Batch object with only batch A fields (non-_b fields)
        """
        return Batch(
            token_ids=batch.token_ids,
            attention_mask=batch.attention_mask,
            field_ids=batch.field_ids,
            entity_ids=batch.entity_ids,
            time_ids=batch.time_ids,
            edge_ids=batch.edge_ids,
            role_ids=batch.role_ids,
            token_type_ids=batch.token_type_ids,
            sequence_lengths=batch.sequence_lengths,
            sequence_ids=batch.sequence_ids,
            labels=None,  # Labels are ranking-specific, not per-sequence
        )

    def _extract_batch_b(self, batch: Batch) -> Batch:
        """Extract batch B from ranking batch.

        Args:
            batch: Ranking batch with _b fields

        Returns:
            Batch object with batch B fields extracted from _b fields

        Raises:
            ValueError: If batch B fields are not present
        """
        if batch.token_ids_b is None:
            raise ValueError("Batch B not available (not a ranking batch)")

        return Batch(
            token_ids=batch.token_ids_b,
            attention_mask=batch.attention_mask_b,
            field_ids=batch.field_ids_b,
            entity_ids=batch.entity_ids_b,
            time_ids=batch.time_ids_b,
            edge_ids=batch.edge_ids_b,
            role_ids=batch.role_ids_b,
            token_type_ids=batch.token_type_ids_b,
            sequence_lengths=batch.sequence_lengths_b,
            sequence_ids=None,  # sequence_ids_b not stored, use None
            labels=None,  # Labels are ranking-specific, not per-sequence
        )

    def _compute_ranking_loss(
        self, scores: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute ranking loss.

        Args:
            scores: Ranking scores [batch_size] (higher = seq_a better)
            labels: Ranking labels [batch_size] (1 = a better, -1 = b better, or binary 0/1)

        Returns:
            Loss tensor
        """
        return self.loss_fn(scores, labels)

    def _validate(self) -> dict:
        """Run validation."""
        if self.val_dataset is None:
            return {}
        
        # Task 2: Increment validation eval_index (independent of training steps)
        eval_index = self._val_eval_index
        self._val_eval_index += 1
        
        self.model.eval()
        if self.task_head is not None:
            self.task_head.eval()

        val_losses = []
        val_metrics = {}
        batch_count = 0

        with torch.no_grad():
            # For validation, iterate through dataset sequentially (no state needed)
            val_indices = list(range(len(self.val_dataset)))
            for i in range(0, len(val_indices), self.batch_size):
                batch_indices = val_indices[i:i + self.batch_size]
                batch_items = [self.val_dataset[idx] for idx in batch_indices]
                batch = self.batcher.batch(batch_items, task_type=self.task_type)
                # Move batch to device
                batch = batch.to(get_device(self.config.device))

                # Apply MSM-Field masking BEFORE forward pass (same as training)
                task_type = self._infer_task_type()
                val_mask = None
                original_field_ids = None
                if task_type == "msm_field":
                    # Get mask_prob from task config
                    mask_prob = 0.15  # default
                    if self.task_config is not None:
                        if isinstance(self.task_config, dict):
                            mask_prob = self.task_config.get("mask_prob", 0.15)
                        elif hasattr(self.task_config, "mask_prob"):
                            mask_prob = self.task_config.mask_prob
                    
                    # Get MASK_FIELD_ID and PAD_FIELD_ID
                    if hasattr(self.model, "embeddings") and hasattr(self.model.embeddings, "field_embedding"):
                        field_embedding_size = self.model.embeddings.field_embedding.embedding.num_embeddings
                        val_mask_field_id = field_embedding_size - 1
                    else:
                        if hasattr(self.task_head, "num_fields"):
                            val_mask_field_id = self.task_head.num_fields
                        else:
                            raise ValueError("Cannot determine MASK_FIELD_ID for validation")
                    pad_field_id = 0  # PAD_FIELD_ID from constants
                    
                    # Store original field_ids BEFORE masking
                    original_field_ids = batch.field_ids.clone()
                    
                    # Task 2: Create deterministic field-balanced mask for validation
                    # Use separate RNG stream with eval_index and batch hash
                    # seed_val = base_seed + 1_000_000 + 10_000 * eval_index + batch_hash_int
                    # eval_index is independent of training steps
                    batch_hash_int = self._compute_batch_hash_int(batch_indices)
                    val_seed = self.config.seed + 1_000_000 + 10_000 * eval_index + batch_hash_int
                    val_mask = make_msm_field_mask_balanced(
                        field_ids=batch.field_ids,
                        mask_prob=mask_prob,
                        mask_field_id=val_mask_field_id,
                        pad_field_id=pad_field_id,
                        seed=val_seed,
                        step=0,  # Seed already includes eval_index, so use step=0
                    )
                    
                    # Internal check: masking must never select padding positions
                    # Padding has attention_mask == 0, so val_mask should be False for those positions
                    padding_mask = batch.attention_mask == 0
                    if (val_mask & padding_mask).any():
                        raise ValueError(
                            "CRITICAL: Validation mask selected padding positions! "
                            "This should never happen."
                        )
                    
                    # Replace masked positions with MASK_FIELD_ID (same as training)
                    batch.field_ids = torch.where(
                        val_mask,
                        torch.full_like(batch.field_ids, val_mask_field_id),
                        batch.field_ids,
                    )
                    
                    # Log validation mask stats (first batch only, once per validation)
                    # Also capture mask hash at step 0 for Run Identity
                    if batch_count == 0 and not hasattr(self, '_val_mask_logged'):
                        self._val_mask_logged = True
                        num_masked = val_mask.sum().item()
                        total_non_padding = (batch.attention_mask == 1).sum().item()
                        mask_ratio = num_masked / total_non_padding if total_non_padding > 0 else 0.0
                        
                        # Compute mask hash for Run Identity (only at step 0, eval_index=0)
                        if self.current_step == 0 and eval_index == 0:
                            import hashlib
                            val_mask_hash = hashlib.sha256(val_mask.cpu().numpy().tobytes()).hexdigest()[:16]
                            # Store for Run Identity
                            self._val_mask_hash_16 = val_mask_hash
                            self._val_mask_count = num_masked
                            self._val_mask_nonpad_count = total_non_padding
                            self._val_mask_ratio = mask_ratio
                        
                        print(f"\n[VAL MSM-Field Masking] First validation batch (eval_index={eval_index}):")
                        print(f"  - mask_prob: {mask_prob}")
                        print(f"  - MASK_FIELD_ID: {val_mask_field_id}")
                        print(f"  - Total non-padding tokens: {total_non_padding}")
                        print(f"  - Masked tokens: {num_masked} ({100.0 * num_masked / total_non_padding:.2f}%)")
                        if self.current_step == 0 and eval_index == 0:
                            print(f"  - Val mask hash (first 16 chars): {val_mask_hash}")
                        print(f"  - Using deterministic masking with eval_index={eval_index} (independent of training steps)")

                # Forward pass
                if task_type == "ranking":
                    outputs = self._forward_ranking(batch)
                else:
                    outputs = self._forward_single(batch)

                # Compute loss (for MSM-Field, use mask and original_field_ids like training)
                if task_type == "msm_field" and val_mask is not None and original_field_ids is not None:
                    loss = self._compute_loss(batch, outputs, msm_field_mask=val_mask, original_field_ids=original_field_ids)
                else:
                    loss = self._compute_loss(batch, outputs)
                val_losses.append(loss.item())
                
                # Compute masked-field accuracy for MSM-Field during validation
                if task_type == "msm_field" and val_mask is not None and original_field_ids is not None:
                    # Compute accuracy on masked tokens using original_field_ids
                    accuracy = self._compute_msm_field_accuracy(outputs, original_field_ids, val_mask)
                    if accuracy is not None:
                        # Initialize accuracy list if not exists
                        if "msm_field/accuracy" not in val_metrics:
                            val_metrics["msm_field/accuracy"] = []
                        val_metrics["msm_field/accuracy"].append(accuracy)
                
                batch_count += 1

        # Calculate validation metrics
        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        val_metrics["loss"] = val_loss
        
        # Average accuracy if computed for MSM-Field
        if "msm_field/accuracy" in val_metrics and isinstance(val_metrics["msm_field/accuracy"], list):
            accuracies = val_metrics["msm_field/accuracy"]
            if accuracies:
                val_metrics["msm_field/accuracy"] = sum(accuracies) / len(accuracies)
            else:
                val_metrics.pop("msm_field/accuracy", None)
        
        # Diagnostic: Log individual batch losses for first validation
        if self.current_epoch == 0 and len(val_losses) > 0:
            print(f"\n[DIAGNOSTIC] Validation batch losses (first 10 batches):")
            for i, batch_loss in enumerate(val_losses[:10]):
                print(f"  Batch {i}: loss = {batch_loss:.6f}")
            if len(val_losses) > 10:
                print(f"  ... ({len(val_losses) - 10} more batches)")

        # Validation loss sanity checks
        if val_loss < 1e-5:
            self.warning_logger.warning(
                f"Validation loss is suspiciously low: {val_loss:.2e} at epoch {self.current_epoch}. "
                f"This may indicate overfitting or a computation issue."
            )
        
        # Log validation dataset info
        if batch_count == 0:
            self.warning_logger.warning(
                f"Validation dataset is empty at epoch {self.current_epoch}"
            )
        else:
            # Log validation dataset size (approximate)
            val_dataset_size = batch_count * self.batch_size
            self.warning_logger.info(
                f"Validation: {batch_count} batches, approximate size: {val_dataset_size}"
            )
            # Print diagnostic info
            print(f"\n[DIAGNOSTIC] Validation at epoch {self.current_epoch}:")
            print(f"  - Batches: {batch_count}")
            print(f"  - Approximate size: {val_dataset_size}")
            print(f"  - Validation loss: {val_loss:.6f}")
            # Compare with training loss if available
            if hasattr(self, '_last_train_loss'):
                train_loss = self._last_train_loss
                print(f"  - Training loss (epoch avg): {train_loss:.6f}")
                gap = abs(train_loss - val_loss)
                gap_ratio = gap / train_loss if train_loss > 0 else 0.0
                print(f"  - Loss gap: {gap:.6f} ({gap_ratio:.2%})")
                if val_loss < train_loss * 0.5:
                    print(f"  - ⚠️  WARNING: Val loss is much lower than train loss!")

        # Log validation metrics
        self.metrics_logger.log_epoch(self.current_epoch, val_metrics, phase="val")

        return val_metrics

    def _compute_msm_diagnostics(self, batch: Batch | None = None, field_ids: torch.Tensor | None = None) -> dict:
        """Compute MSM training diagnostics (attention entropy and same-field mass).

        Args:
            batch: Optional batch to use for diagnostics. If None, samples a fresh batch.
            field_ids: Optional field_ids to use (should be PRE-MASK for MSM-Field). 
                      If None, uses batch.field_ids.

        Returns:
            Dictionary with diagnostic metrics per layer:
            {
                "attn/entropy_mean_layer_0": float,
                "attn/same_field_mass_layer_0": float,
                "attn/entropy_mean_layer_1": float,
                ...
            }
        """
        # Use provided batch or sample a fresh one
        if batch is None:
            # Sample a fresh batch from training dataset
            sample_indices = list(range(min(self.batch_size, len(self.train_dataset))))
            batch_items = [self.train_dataset[idx] for idx in sample_indices]
            batch = self.batcher.batch(batch_items, task_type=self.task_type)
            batch = batch.to(get_device(self.config.device))

        # Store original training mode
        was_training = self.model.training
        was_task_head_training = self.task_head.training if self.task_head is not None else None

        # Temporarily set to eval mode for stable diagnostics
        self.model.eval()
        if self.task_head is not None:
            self.task_head.eval()

        diagnostics = {}

        with torch.no_grad():
            # Forward pass with attention weights
            encoder_output, attention_weights_list = self.model(
                batch, return_attention_weights=True
            )

            # Compute diagnostics for each layer
            num_layers = len(attention_weights_list)
            for layer_idx, attn_weights in enumerate(attention_weights_list):
                # Compute attention entropy
                entropy = compute_attention_entropy(
                    attn_weights=attn_weights,
                    attention_mask=batch.attention_mask,
                )
                diagnostics[f"attn/entropy_mean_layer_{layer_idx}"] = entropy

                # Compute same-field mass
                # Use provided field_ids (pre-mask) or fall back to batch.field_ids
                field_ids_for_diag = field_ids if field_ids is not None else batch.field_ids
                same_field_mass = compute_same_field_mass(
                    attn_weights=attn_weights,
                    field_ids=field_ids_for_diag,
                    attention_mask=batch.attention_mask,
                )
                diagnostics[f"attn/same_field_mass_layer_{layer_idx}"] = same_field_mass

        # Restore original training mode
        if was_training:
            self.model.train()
        if was_task_head_training is not None and was_task_head_training:
            if self.task_head is not None:
                self.task_head.train()

        return diagnostics

    def _print_msm_diagnostics(self, step: int, diagnostics: dict) -> None:
        """Print formatted MSM diagnostic block.

        Args:
            step: Current training step
            diagnostics: Dictionary with diagnostic metrics per layer
        """
        print(f"\n[MSM Diagnostics] Step {step}:")
        
        # Extract layer indices from metric keys
        layer_indices = set()
        for key in diagnostics.keys():
            if "layer_" in key:
                # Extract layer index from key like "attn/entropy_mean_layer_0"
                parts = key.split("_layer_")
                if len(parts) == 2:
                    try:
                        layer_idx = int(parts[1])
                        layer_indices.add(layer_idx)
                    except ValueError:
                        continue
        
        # Sort layer indices
        sorted_layers = sorted(layer_indices)
        
        # Print metrics for each layer
        for layer_idx in sorted_layers:
            entropy_key = f"attn/entropy_mean_layer_{layer_idx}"
            mass_key = f"attn/same_field_mass_layer_{layer_idx}"
            
            entropy = diagnostics.get(entropy_key, 0.0)
            mass = diagnostics.get(mass_key, 0.0)
            
            print(f"  Layer {layer_idx}:")
            print(f"    - entropy_mean: {entropy:.6f}")
            print(f"    - same_field_mass: {mass:.6f}")

    def _check_and_save_best(self, val_metrics: dict, epoch: int):
        """Check if current model is best and save if so."""
        metric_value = val_metrics.get(self.config.best_metric, float("inf"))

        is_best = False
        if self.best_metric_value is None:
            is_best = True
            self.best_metric_value = metric_value
        elif self.config.best_mode == "min":
            if metric_value < self.best_metric_value:
                is_best = True
                self.best_metric_value = metric_value
        elif self.config.best_mode == "max":
            if metric_value > self.best_metric_value:
                is_best = True
                self.best_metric_value = metric_value

        if is_best:
            self._save_checkpoint(epoch, val_metrics, is_best=True)

    def _check_early_stopping(self, val_metrics: dict) -> bool:
        """Check if early stopping should be triggered.
        
        Args:
            val_metrics: Validation metrics dictionary
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.config.early_stopping_patience is None:
            return False
        
        metric_name = self.config.early_stopping_metric
        metric_value = val_metrics.get(metric_name, None)
        
        if metric_value is None:
            # Metric not found, can't do early stopping
            return False
        
        # Determine if metric improved (assuming "min" mode for loss, "max" for others)
        # For now, we'll use "min" mode (lower is better) for loss, "max" for others
        is_min_mode = metric_name == "loss" or self.config.best_mode == "min"
        
        if self._best_early_stopping_metric_value is None:
            # First validation, record as best
            self._best_early_stopping_metric_value = metric_value
            self._early_stopping_patience_counter = 0
            return False
        
        # Check if metric improved
        improved = False
        if is_min_mode:
            # Lower is better
            if metric_value < (self._best_early_stopping_metric_value - self.config.early_stopping_min_delta):
                improved = True
        else:
            # Higher is better
            if metric_value > (self._best_early_stopping_metric_value + self.config.early_stopping_min_delta):
                improved = True
        
        if improved:
            # Metric improved, reset counter and update best
            self._best_early_stopping_metric_value = metric_value
            self._early_stopping_patience_counter = 0
        else:
            # No improvement, increment counter
            self._early_stopping_patience_counter += 1
        
        # Check if patience exceeded
        if self._early_stopping_patience_counter >= self.config.early_stopping_patience:
            return True
        
        return False

    def _save_checkpoint(
        self, epoch: int, metrics: dict, is_best: bool = False, is_latest: bool = False
    ):
        """Save checkpoint."""
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            step=self.current_step,
            metrics=metrics,
            is_best=is_best,
            is_latest=is_latest,
            config=self._config_dict,
            task_head=self.task_head,
        )

    def load_checkpoint(
        self, checkpoint_path: str | None = None, resume: bool = True
    ) -> dict:
        """Load checkpoint and optionally resume training.

        Args:
            checkpoint_path: Path to checkpoint file (None = load latest)
            resume: If True, resume training from checkpoint

        Returns:
            Dictionary with loaded state
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoint found to load")

        checkpoint_path = Path(checkpoint_path)

        state = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer if resume else None,
            scheduler=self.scheduler if resume else None,
            task_head=self.task_head,
        )

        if resume:
            self.current_epoch = state["epoch"]
            self.current_step = state["step"]
            if "metrics" in state and self.config.best_metric in state["metrics"]:
                self.best_metric_value = state["metrics"][self.config.best_metric]

        return state
