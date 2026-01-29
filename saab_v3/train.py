"""
Training entrypoint script for Transformer models.

Usage:
    poetry run python -m saab_v3.train --dataset-name <dataset_name> --model-type <model_type>

Required Arguments:
    --dataset-name, --dataset: Name of the dataset directory in dataset/raw/
    --model-type, --model: Model type ('flat', 'scratch', or 'saab')
    --device: Device to use for training ('cuda', 'cpu', etc.)

Optional Arguments:
    --resume: Path to checkpoint to resume training from
    --experiment-name: Name of experiment (default: {dataset_name}_{model_type})

Examples:
    # Basic training without task head (unsupervised/pretraining)
    poetry run python -m saab_v3.train --dataset-name mydataset --model-type saab --device cuda

    # Resume training from checkpoint
    poetry run python -m saab_v3.train \\
        --dataset-name mydataset \\
        --model-type saab \\
        --device cuda \\
        --resume checkpoints/mydataset_saab/checkpoint_epoch_5.pt

Configuration:
    All configuration uses Pydantic defaults (single source of truth).
    To customize, modify the Pydantic config objects in the code:
    - PreprocessingConfig: preprocessing settings
    - ModelConfig: model architecture settings
    - TrainingConfig: training hyperparameters
    - TaskConfig: task head settings (optional, for supervised learning)

Dataset Requirements:
    The dataset must be located in dataset/raw/{dataset_name}/ directory
    and should contain train.csv and val.csv files.

    For supervised learning, CSV files should include a 'label' or 'target' column.
    Supported label formats:
    - Classification: integer class indices or binary vectors (multi-label)
    - Regression: continuous float values
    - Token Classification: JSON array of label indices per token

Data Locations:
    - Raw data: dataset/raw/{dataset_name}/
    - Artifacts: dataset/artifacts/{dataset_name}/
    - Checkpoints: checkpoints/{experiment_name}/
    - Logs: logs/{experiment_name}/
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from saab_v3.models import (
    ModelConfig,
    create_flat_transformer,
    create_scratch_transformer,
    create_saab_transformer,
)
from saab_v3.training import (
    PreprocessingConfig,
    Preprocessor,
    StructuredDataset,
    TrainingConfig,
)
from saab_v3.training.trainer import Trainer
from saab_v3.training.loss import create_loss_fn
from saab_v3.training.run_identity import (
    create_run_identity,
    save_run_identity,
    update_run_identity,
    print_run_identity,
    compute_array_hash,
)
from saab_v3.tasks.config import ClassificationTaskConfig, MSMFieldTaskConfig
from saab_v3.tasks import create_task_head_from_config

# ============================================================================
# Command-Line Arguments
# ============================================================================

parser = argparse.ArgumentParser(
    description="Train Transformer models on structured data"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    dest="dataset_name",
    help="Name of the dataset directory in dataset/raw/",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["flat", "scratch", "saab"],
    dest="model_type",
    help="Model type: 'flat', 'scratch', or 'saab'",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume from (optional)",
)
parser.add_argument(
    "--experiment-name",
    type=str,
    default=None,
    help="Experiment name (default: {dataset_name}_{model_type})",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",  # Default to CPU to avoid MPS memory issues
    help="Device to use for training (e.g., 'cuda', 'cpu', 'mps', 'auto')",
)
parser.add_argument(
    "--task",
    type=str,
    default="classification",
    choices=["classification", "ranking", "regression", "token_classification", "msm_field"],
    help="Task type: 'classification', 'ranking', 'regression', 'token_classification', or 'msm_field'",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    dest="max_steps",
    help="Maximum number of training steps (overrides config, sets num_epochs=None)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=None,
    dest="batch_size",
    help="Batch size (overrides config)",
)
parser.add_argument(
    "--mask-prob",
    type=float,
    default=None,
    dest="mask_prob",
    help="Mask probability for MSM-Field task (overrides config, default: 0.15)",
)
parser.add_argument(
    "--log-every",
    type=int,
    default=None,
    dest="log_every",
    help="Log every N steps (overrides config log_steps)",
)
parser.add_argument(
    "--eval-every",
    type=int,
    default=None,
    dest="eval_every",
    help="Evaluate on validation set every N steps (for step-based training)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    dest="seed",
    help="Random seed for reproducibility (overrides config, default: 42)",
)
parser.add_argument(
    "--refit-preprocessor",
    action="store_true",
    dest="refit_preprocessor",
    help="Force refit preprocessor even if artifacts exist (default: False, will error if artifacts incomplete)",
)
parser.add_argument(
    "--determinism-strict",
    action="store_true",
    dest="determinism_strict",
    help="Enable strict determinism (torch.manual_seed, cudnn.deterministic=True, etc.)",
)
args = parser.parse_args()

dataset_name = args.dataset_name
model_type = args.model_type
resume_checkpoint = args.resume
experiment_name = args.experiment_name or f"{dataset_name}_{model_type}"
device = args.device or "cpu"  # Fallback to CPU if not provided
task_type = args.task  # Task type from command line

# ============================================================================
# Configuration - Single Source of Truth: Pydantic Defaults
# ============================================================================

print("\nUsing Pydantic defaults for configuration...")

# Simple constructor calls with values
# Enable text tokenization for DBpedia dataset
if dataset_name == "dbpedia":
    preprocessing_config = PreprocessingConfig(
        device=device,
        text_fields=["title", "content"],  # Mark these fields as text for subword tokenization
        use_text_tokenizer=True,  # Enable BPE tokenizer for text fields
        text_tokenizer_type="bpe",
        text_tokenizer_vocab_size=30000,  # Vocabulary size for text tokenizer
        field_boundary_token=True,  # Add [FIELD_START] tokens before each field
        max_seq_len=256,  # Lower max_seq_len for DBpedia
    )
else:
    preprocessing_config = PreprocessingConfig(
        device=device,
    )
model_config = ModelConfig(
    device=device,
    dropout=0.2,  # Increased for regularization
    num_layers=4,
    num_heads=6,
)
# Calculate training steps and warmup for DBpedia
# For subset mode, use epochs; otherwise use steps
if dataset_name == "dbpedia":
    # Subset mode configuration for fast local experiments
    training_config = TrainingConfig(
        device=device,
        num_epochs=10,  # Use epochs when subset mode enabled
        max_steps=None,  # Disable steps when using epochs
        learning_rate=1e-4,  # Reduced to 1e-4 for better stability
        batch_size=128,  # Reduced from 512 to prevent CUDA OOM
        gradient_accumulation_steps=4,  # Effective batch size = 128 × 4 = 512
        lr_schedule="linear_warmup_cosine",  # Warmup + cosine decay for better convergence
        warmup_ratio=0.1,  # 10% warmup ratio when using epochs
        warmup_steps=None,  # Disabled when using warmup_ratio
        min_lr_ratio=0.1,  # Decay to 10% of initial LR by end of training
        max_grad_norm=1.0,  # FIXED: Increased from 0.1 (was clipping gradients by 19,000x!)
        early_stop_zero_loss_steps=100,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        early_stopping_metric="loss",
        subset_size_train=10000,  # Subset mode: 10k train samples
        subset_size_val=2000,  # Subset mode: 2k val samples
        subset_size_test=2000,  # Subset mode: 2k test samples
    )
else:
    # Full dataset mode (original configuration)
    max_steps = 200  # Reduced from 8000 for faster experiments
    warmup_steps = int(0.1 * max_steps)  # 10% warmup = 20 steps
    decay_steps = max_steps - warmup_steps  # 180 steps
    
    training_config = TrainingConfig(
        device=device,
        num_epochs=None,  # Use steps instead of epochs
        max_steps=max_steps,  # Train for 200 steps
        learning_rate=1e-4,  # Reduced to 1e-4 for better stability
        batch_size=128,  # Reduced from 512 to prevent CUDA OOM
        gradient_accumulation_steps=4,  # Effective batch size = 128 × 4 = 512
        lr_schedule="linear_warmup_cosine",  # Warmup + cosine decay for better convergence
        warmup_steps=warmup_steps,  # 10% of max_steps = 20 steps
        warmup_ratio=None,  # Disabled ratio-based warmup in favor of fixed steps
        min_lr_ratio=0.1,  # Decay to 10% of initial LR by end of training
        max_grad_norm=1.0,  # FIXED: Increased from 0.1 (was clipping gradients by 19,000x!)
        early_stop_zero_loss_steps=100,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        early_stopping_metric="loss",
    )

# Override training config with command-line arguments if provided
# Use model_copy to properly update Pydantic model with validation
override_dict = {}
if args.max_steps is not None:
    override_dict["max_steps"] = args.max_steps
    override_dict["num_epochs"] = None  # Step-based mode when max_steps is set
    # Adjust warmup for step-based mode if needed
    if training_config.warmup_ratio is not None and training_config.warmup_steps is None:
        # Convert warmup_ratio to warmup_steps for step-based training
        override_dict["warmup_steps"] = int(args.max_steps * training_config.warmup_ratio)
        override_dict["warmup_ratio"] = None

if args.batch_size is not None:
    override_dict["batch_size"] = args.batch_size

if args.log_every is not None:
    override_dict["log_steps"] = args.log_every

if args.seed is not None:
    override_dict["seed"] = args.seed

# Apply overrides using model_copy (proper Pydantic way)
if override_dict:
    training_config = training_config.model_copy(update=override_dict)

# Task config will be created after preprocessor is fitted (needed for msm_field num_fields)
task_config = None  # Will be set after preprocessor fitting

# Print configuration summary
print(f"\n{'=' * 60}")
print("=" * 60)
print("Experiment Configuration:")
print(f"  - model_type: {model_type}")
print(f"  - task_type: {task_type}")
if hasattr(task_config, "pooling"):
    print(f"  - pooling: {task_config.pooling}")
if training_config.max_steps is not None:
    print(f"  - max_steps: {training_config.max_steps}")
if training_config.warmup_steps is not None:
    print(f"  - warmup_steps: {training_config.warmup_steps}")
if training_config.warmup_ratio is not None:
    print(f"  - warmup_ratio: {training_config.warmup_ratio}")
print("\nPreprocessing:")
print(f"  - vocab_size: {preprocessing_config.vocab_size}")
print(f"  - max_seq_len: {preprocessing_config.max_seq_len}")
print(f"  - device: {preprocessing_config.device}")
if preprocessing_config.use_text_tokenizer:
    print(f"  - text_fields: {preprocessing_config.text_fields}")
    print(f"  - text_tokenizer_type: {preprocessing_config.text_tokenizer_type}")
    print(f"  - text_tokenizer_vocab_size: {preprocessing_config.text_tokenizer_vocab_size}")
    print(f"  - field_boundary_token: {preprocessing_config.field_boundary_token}")
print("\nModel:")
print(f"  - d_model: {model_config.d_model}")
print(f"  - num_layers: {model_config.num_layers}")
print(f"  - num_heads: {model_config.num_heads}")
print(f"  - dropout: {model_config.dropout}")
print(f"  - device: {model_config.device}")
print("\nTraining:")
print(f"  - learning_rate: {training_config.learning_rate}")
print(f"  - batch_size: {training_config.batch_size}")
print(f"  - num_epochs: {training_config.num_epochs}")
print(f"  - max_steps: {training_config.max_steps}")
print(f"  - lr_schedule: {training_config.lr_schedule}")
print(f"  - max_grad_norm: {training_config.max_grad_norm}")
print(f"  - early_stopping_patience: {training_config.early_stopping_patience}")
print(f"  - device: {training_config.device}")
if training_config.subset_size_train is not None:
    print(f"  - subset_size_train: {training_config.subset_size_train}")
if training_config.subset_size_val is not None:
    print(f"  - subset_size_val: {training_config.subset_size_val}")
if training_config.subset_size_test is not None:
    print(f"  - subset_size_test: {training_config.subset_size_test}")
if training_config.subset_size_train is not None:
    print(f"  - fit_preprocessor_on_subset: {training_config.fit_preprocessor_on_subset}")
print("=" * 60)
print()

# ============================================================================
# Helper Functions
# ============================================================================

def _load_or_create_subset_indices(
    split: str,
    dataset_size: int,
    subset_size: int,
    base_seed: int,
    dataset_name: str,
    artifacts_dir: Path,
) -> np.ndarray:
    """Load existing subset indices or create and save new ones.
    
    Args:
        split: Dataset split name ("train", "val", "test")
        dataset_size: Full dataset size
        subset_size: Desired subset size
        base_seed: Base random seed (split-specific seed will be computed)
        dataset_name: Name of dataset
        artifacts_dir: Path to artifacts directory
        
    Returns:
        Subset indices array (int64 dtype)
    """
    # Use split-specific seed to avoid accidental coupling between splits
    # train = seed + 0, val = seed + 1, test = seed + 2
    split_seed_offsets = {"train": 0, "val": 1, "test": 2}
    split_seed = base_seed + split_seed_offsets.get(split, 0)
    
    # Ensure subset_size doesn't exceed dataset_size
    subset_size = min(subset_size, dataset_size)
    
    # File name: {split}_subset_seed{split_seed}_K{K}.npy
    subset_file = artifacts_dir / f"{split}_subset_seed{split_seed}_K{subset_size}.npy"
    
    if subset_file.exists():
        # Load existing subset indices
        indices = np.load(str(subset_file))
        
        # Verify dtype and shape
        if indices.dtype != np.int64:
            raise ValueError(
                f"Subset file has wrong dtype: {indices.dtype}, expected int64. "
                f"Please delete {subset_file} and regenerate."
            )
        if indices.ndim != 1:
            raise ValueError(
                f"Subset file has wrong shape: {indices.ndim}D, expected 1D. "
                f"Please delete {subset_file} and regenerate."
            )
        if len(indices) != subset_size:
            raise ValueError(
                f"Subset file size mismatch: {len(indices)} != {subset_size}. "
                f"Please delete {subset_file} and regenerate."
            )
        
        return indices
    else:
        # Create new subset indices using deterministic permutation with split-specific seed
        rng = np.random.default_rng(seed=split_seed)
        indices = np.arange(dataset_size, dtype=np.int64)
        permuted_indices = rng.permutation(indices)
        subset_indices = permuted_indices[:subset_size]
        
        # Ensure int64 dtype before saving
        subset_indices = np.asarray(subset_indices, dtype=np.int64)
        
        # Save subset indices
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(subset_file), subset_indices)
        
        return subset_indices


# ============================================================================
# Dataset Paths
# ============================================================================

# Dataset directory (all datasets must be in dataset/raw/)
DATASET_DIR = Path(__file__).parent.parent / "dataset" / "raw" / dataset_name

# File paths
train_path = DATASET_DIR / "train.csv"
val_path = DATASET_DIR / "val.csv"
# NOTE: test dataset will be used for evaluation

# ============================================================================
# Preprocessing
# ============================================================================

# Check if dataset directory exists
if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}\n"
        f"Please ensure the dataset is located in dataset/raw/{dataset_name}/"
    )

# Check if files exist
if not train_path.exists():
    raise FileNotFoundError(f"Training file not found: {train_path}")
if not val_path.exists():
    raise FileNotFoundError(f"Validation file not found: {val_path}")

# Check if artifacts exist and load them, otherwise fit and save
artifacts_dir = Path(__file__).parent.parent / "dataset" / "artifacts" / dataset_name
config_path = artifacts_dir / "config.json"
text_tokenizer_path = artifacts_dir / "vocabularies" / "text_tokenizer.json"

# Determine if we should fit preprocessor on subset
# Load train data early to check if subset fitting is needed
train_df_full = pd.read_csv(train_path)
fit_on_subset = (
    training_config.fit_preprocessor_on_subset
    and training_config.subset_size_train is not None
    and len(train_df_full) > training_config.subset_size_train
)

# Store train subset indices if we're fitting on subset (for reuse later)
train_subset_indices_for_fitting = None

if fit_on_subset:
    # Get subset indices for preprocessor fitting
    train_subset_indices_for_fitting = _load_or_create_subset_indices(
        split="train",
        dataset_size=len(train_df_full),
        subset_size=training_config.subset_size_train,
        base_seed=training_config.seed,
        dataset_name=dataset_name,
        artifacts_dir=artifacts_dir,
    )
    train_df_for_fitting = train_df_full.iloc[train_subset_indices_for_fitting].reset_index(drop=True)
    # Compute hash of indices for verification
    indices_hash = hashlib.sha256(train_subset_indices_for_fitting.tobytes()).hexdigest()[:16]
    print(f"[PREPROCESSOR] Fitting preprocessor on train subset: K={len(train_subset_indices_for_fitting)}, seed={training_config.seed}, indices_hash={indices_hash}")
else:
    train_df_for_fitting = train_df_full
    print(f"[PREPROCESSOR] Fitting preprocessor on full train: N={len(train_df_for_fitting)}")

# Artifact loading with strict checking
if args.refit_preprocessor:
    print("[ARTIFACTS] Refitting preprocessor (--refit-preprocessor flag set)")
    preprocessor = Preprocessor(preprocessing_config)
    preprocessor.fit(train_df_for_fitting)
    preprocessor.save_artifacts(dataset_name)
    print(f"[ARTIFACTS] refit: {artifacts_dir}")
elif artifacts_dir.exists() and config_path.exists():
    # Check if text tokenizer exists (if needed)
    if preprocessing_config.use_text_tokenizer:
        if not text_tokenizer_path.exists():
            raise FileNotFoundError(
                f"Artifacts exist but text tokenizer missing: {text_tokenizer_path}\n"
                f"To refit preprocessor, use --refit-preprocessor flag.\n"
                f"This ensures fair comparison between Scratch and SAAB runs."
            )
    
    print(f"[ARTIFACTS] loaded: {artifacts_dir}")
    preprocessor = Preprocessor.load_artifacts(dataset_name)
    print("✓ Preprocessor loaded from artifacts (reusing for fair comparison)")
else:
    # Initialize preprocessor
    print(f"[ARTIFACTS] refit: {artifacts_dir}")
    preprocessor = Preprocessor(preprocessing_config)
    
    # Fit on training data (subset or full, based on config)
    print("Fitting preprocessor on training data...")
    preprocessor.fit(train_df_for_fitting)
    
    # Save artifacts for later use
    print(f"Saving preprocessing artifacts for '{dataset_name}'...")
    preprocessor.save_artifacts(dataset_name)

# ============================================================================
# Determinism Setup
# ============================================================================
if args.determinism_strict:
    import torch.backends.cudnn as cudnn
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # Disable TF32 for strict determinism
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"[DETERMINISM] Strict determinism enabled (seed={training_config.seed})")
else:
    print(f"[DETERMINISM] Strict determinism disabled (use --determinism-strict to enable)")

# Create task config (after preprocessor is fitted so we can access vocab sizes)
print(f"\nCreating task config for task: {task_type}...")
if task_type == "msm_field":
    # Task 2: Compute num_fields correctly (single source of truth)
    # num_fields = count of REAL field IDs (excluding PAD=0 only)
    # MASK_FIELD_TOKEN should NEVER be in the field vocabulary (it's added as MASK_FIELD_ID in embedding)
    from saab_v3.data.constants import PAD_TOKEN, MASK_FIELD_TOKEN
    
    field_vocab = preprocessor.tag_encoder.tag_vocabs["field"]
    field_vocab_size_raw = len(field_vocab)
    
    # Fail-fast: MASK_FIELD_TOKEN must NOT be in field vocabulary
    if MASK_FIELD_TOKEN in field_vocab.token_to_idx:
        raise ValueError(
            f"MASK_FIELD_TOKEN must not be in field vocabulary! "
            f"Found MASK_FIELD_TOKEN at index {field_vocab.token_to_idx[MASK_FIELD_TOKEN]}. "
            f"This indicates a bug in TagEncoder initialization. "
            f"MASK_FIELD_TOKEN should be excluded from field vocab special tokens."
        )
    
    # Count real field IDs: num_fields should be field_vocab_size_raw
    # Field IDs in vocab are 0 (PAD), 1, 2, ..., field_vocab_size_raw-1
    # Classifier needs to output logits for all field IDs: 0, 1, 2, ..., field_vocab_size_raw-1
    # So num_fields = field_vocab_size_raw (to cover all field ID values)
    # PAD (field_id=0) is handled by ignore_index=0 in the loss function
    num_fields = field_vocab_size_raw  # Classifier outputs: 0, 1, ..., field_vocab_size_raw-1
    
    # Ensure we have at least one real field (vocab must have at least PAD + one real field)
    if num_fields <= 1:
        raise ValueError(
            f"Invalid num_fields computation: {num_fields} (field_vocab_size_raw={field_vocab_size_raw}). "
            f"Field vocabulary must contain at least PAD (0) and one real field ID."
        )
    
    mask_prob = args.mask_prob if args.mask_prob is not None else 0.15
    task_config = MSMFieldTaskConfig(
        num_fields=num_fields,
        loss_weight=1.0,
        mask_prob=mask_prob,
    )
    print(f"[MSM CONFIG] derived_num_fields={num_fields}, field_vocab_size_raw={field_vocab_size_raw}, pad_id=0")
    print(f"  - num_fields: {num_fields} (real field IDs, excluding PAD only)")
    print(f"  - mask_prob: {mask_prob}")
elif task_type == "classification":
    task_config = ClassificationTaskConfig(
        num_classes=14,  # Hardcoded for mydataset (14 classes)
        multi_label=False,
        pooling="mean",  # Use mean pooling instead of CLS for DBpedia
        label_smoothing=0.0,  # Temporarily disabled for training stability
    )
else:
    raise ValueError(f"Task type '{task_type}' not yet implemented in train.py. "
                     f"Supported: 'classification', 'msm_field'")

# Create datasets
print("Creating datasets...")
# CRITICAL: Only train dataset is used for fitting preprocessor
# val/test are NEVER used for fitting (only for evaluation)

# Load raw CSV files into DataFrames (reuse train_df_full if already loaded)
if 'train_df_full' not in locals():
    train_df = pd.read_csv(train_path)
else:
    train_df = train_df_full.copy()
val_df = pd.read_csv(val_path)
test_path = DATASET_DIR / "test.csv"
test_df = pd.read_csv(test_path) if test_path.exists() else None

# Apply subset filtering if configured
subset_enabled = (
    training_config.subset_size_train is not None
    or training_config.subset_size_val is not None
    or training_config.subset_size_test is not None
)

# Track subset hashes for Run Identity
subset_hashes = {"train": None, "val": None, "test": None}
subset_sizes = {
    "train": training_config.subset_size_train,
    "val": training_config.subset_size_val,
    "test": training_config.subset_size_test,
}

if subset_enabled:
    print("\n[SUBSET MODE] Enabled")
    
    # Process train split
    if training_config.subset_size_train is not None:
        original_train_size = len(train_df)
        # Reuse subset indices if we already loaded them for preprocessor fitting
        if train_subset_indices_for_fitting is not None:
            subset_indices = train_subset_indices_for_fitting
        else:
            subset_indices = _load_or_create_subset_indices(
                split="train",
                dataset_size=original_train_size,
                subset_size=training_config.subset_size_train,
                base_seed=training_config.seed,
                dataset_name=dataset_name,
                artifacts_dir=artifacts_dir,
            )
        train_df = train_df.iloc[subset_indices].reset_index(drop=True)
        # Use split-specific seed for filename
        split_seed = training_config.seed + 0  # train = seed + 0
        subset_file = artifacts_dir / f"train_subset_seed{split_seed}_K{training_config.subset_size_train}.npy"
        # Compute hash and show first 20 indices for verification across runs
        indices_hash = hashlib.sha256(subset_indices.tobytes()).hexdigest()[:16]
        subset_hashes["train"] = indices_hash
        first_20_indices = subset_indices[:20].tolist()
        print(f"  Train: {original_train_size} -> {len(train_df)} (using {subset_file.name})")
        print(f"    Subset identity: first_20_indices={first_20_indices}, indices_hash={indices_hash}")
    
    # Process val split
    if training_config.subset_size_val is not None:
        original_val_size = len(val_df)
        subset_indices = _load_or_create_subset_indices(
            split="val",
            dataset_size=original_val_size,
            subset_size=training_config.subset_size_val,
            base_seed=training_config.seed,
            dataset_name=dataset_name,
            artifacts_dir=artifacts_dir,
        )
        val_df = val_df.iloc[subset_indices].reset_index(drop=True)
        # Use split-specific seed for filename
        split_seed = training_config.seed + 1  # val = seed + 1
        subset_file = artifacts_dir / f"val_subset_seed{split_seed}_K{training_config.subset_size_val}.npy"
        indices_hash = hashlib.sha256(subset_indices.tobytes()).hexdigest()[:16]
        subset_hashes["val"] = indices_hash
        first_20_indices = subset_indices[:20].tolist()
        print(f"  Val: {original_val_size} -> {len(val_df)} (using {subset_file.name})")
        print(f"    Subset identity: first_20_indices={first_20_indices}, indices_hash={indices_hash}")
    
    # Process test split (if exists and configured)
    if test_df is not None and training_config.subset_size_test is not None:
        original_test_size = len(test_df)
        subset_indices = _load_or_create_subset_indices(
            split="test",
            dataset_size=original_test_size,
            subset_size=training_config.subset_size_test,
            base_seed=training_config.seed,
            dataset_name=dataset_name,
            artifacts_dir=artifacts_dir,
        )
        test_df = test_df.iloc[subset_indices].reset_index(drop=True)
        # Use split-specific seed for filename
        split_seed = training_config.seed + 2  # test = seed + 2
        subset_file = artifacts_dir / f"test_subset_seed{split_seed}_K{training_config.subset_size_test}.npy"
        indices_hash = hashlib.sha256(subset_indices.tobytes()).hexdigest()[:16]
        subset_hashes["test"] = indices_hash
        first_20_indices = subset_indices[:20].tolist()
        print(f"  Test: {original_test_size} -> {len(test_df)} (using {subset_file.name})")
        print(f"    Subset identity: first_20_indices={first_20_indices}, indices_hash={indices_hash}")
else:
    print("[SUBSET MODE] Disabled (using full dataset)")

# Create StructuredDataset from DataFrames (filtered if subset mode enabled)
train_dataset = StructuredDataset(
    train_df, preprocessor, split="train", task_type=task_type
)
val_dataset = StructuredDataset(
    val_df, preprocessor, split="val", task_type=task_type
)
print("✓ Datasets created")
print("  - Train dataset: used for training only (preprocessor was fitted on train.csv)")
print("  - Val dataset: used for evaluation only (NEVER used for fitting)")

# Datasets are ready for manual batching (no DataLoader needed)
print("Datasets ready for training...")

# ============================================================================
# Model Creation
# ============================================================================

print(f"\nCreating {model_type.upper()} model...")

if model_type == "flat":
    model = create_flat_transformer(preprocessor, model_config)
elif model_type == "scratch":
    model = create_scratch_transformer(preprocessor, model_config)
elif model_type == "saab":
    model = create_saab_transformer(preprocessor, model_config)
else:
    raise ValueError(f"Unknown model_type: {model_type}")

print(f"Model created: {model.__class__.__name__}")
print(f"  - d_model: {model.d_model}")
print(f"  - num_layers: {model.num_layers}")
print(f"  - num_heads: {model.num_heads}")

# ============================================================================
# Task Head and Loss Function
# ============================================================================

# Create task head from Pydantic config
task_head = create_task_head_from_config(task_config, d_model=model_config.d_model)
print(f"✓ Task head created: {task_head.__class__.__name__}")

# Extract loss params from task config (filters out task head params)
loss_params = task_config.get_loss_params()

# Create loss function from task type and loss params
loss_fn = create_loss_fn(task_type, **loss_params)
print(f"✓ Loss function created for task: {task_type}")

# ============================================================================
# Training
# ============================================================================

print(f"\nInitializing trainer for experiment: {experiment_name}...")

# Prepare configs for checkpointing
model_config_dict = (
    model_config.model_dump()
    if hasattr(model_config, "model_dump")
    else model_config.__dict__
)

# Convert task_config to dict for checkpointing if it's a Pydantic model
task_config_dict = (
    task_config.model_dump() if hasattr(task_config, "model_dump") else task_config
)

# Convert preprocessing_config to dict for checkpointing if it's a Pydantic model
preprocessing_config_dict = (
    preprocessing_config.model_dump()
    if hasattr(preprocessing_config, "model_dump")
    else preprocessing_config.__dict__
)

# Pass eval_every_steps if provided (for step-based training)
eval_every_steps = args.eval_every if args.eval_every is not None else None

trainer = Trainer(
    model=model,
    config=training_config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    task_head=task_head,
    loss_fn=loss_fn,
    task_type=task_type,
    experiment_name=experiment_name,
    model_config=model_config_dict,
    task_config=task_config_dict,
    preprocessing_config=preprocessing_config_dict,
    dataset_name=dataset_name,
    model_type=model_type,
    eval_every_steps=eval_every_steps,
)

# Resume from checkpoint if provided
if resume_checkpoint:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.load_checkpoint(checkpoint_path=resume_checkpoint, resume=True)

# ============================================================================
# Run Identity Creation
# ============================================================================
# Get field_emb_size and mask_field_id from model (for MSM-Field)
field_emb_size = None
mask_field_id = None
if task_type == "msm_field" and hasattr(model, "embeddings") and hasattr(model.embeddings, "field_embedding"):
    field_emb_size = model.embeddings.field_embedding.embedding.num_embeddings
    mask_field_id = field_emb_size - 1  # MASK_FIELD_ID is last index

# Create Run Identity
run_identity = create_run_identity(
    model_type=model_type,
    task=task_type,
    seed=training_config.seed,
    dataset_name=dataset_name,
    artifacts_dir=artifacts_dir,
    subset_sizes=subset_sizes,
    subset_hashes=subset_hashes,
    permutation_hash=None,  # Will be set by trainer at step 0
    num_fields=num_fields if task_type == "msm_field" else None,
    field_emb_size=field_emb_size,
    mask_field_id=mask_field_id,
    mask_prob=task_config.mask_prob if task_type == "msm_field" else None,
    determinism_strict=args.determinism_strict,
)

# Save Run Identity to log directory
log_dir = Path("logs") / experiment_name
identity_path = save_run_identity(run_identity, log_dir)
print_run_identity(run_identity)
print(f"Run Identity saved to: {identity_path}\n")

# Run training
print("\nStarting training...")
print("=" * 60)
history = trainer.train()
print("=" * 60)
print("\nTraining complete!")

# Update Run Identity with trainer data (permutation and mask hashes from step 0)
# Do this in a try-except to ensure it happens even if checkpoint saving fails later
try:
    trainer_data = trainer.get_run_identity_data()
    if trainer_data:
        update_run_identity(log_dir, trainer_data)
        print(f"Run Identity updated with permutation and mask hashes")
        
        # Print determinism summary line (Task D: one line at step 0)
        if trainer_data.get("train_permutation_hash_16") and trainer_data.get("train_mask_step0") and trainer_data.get("val_mask_step0"):
            perm_hash = trainer_data["train_permutation_hash_16"]
            train_mask_hash = trainer_data["train_mask_step0"]["mask_hash_16"]
            val_mask_hash = trainer_data["val_mask_step0"]["mask_hash_16"]
            print(f"\n[DETERMINISM] Step 0: perm_hash={perm_hash}, train_mask_hash={train_mask_hash}, val_mask_hash={val_mask_hash}")
except Exception as e:
    print(f"\nWARNING: Failed to update Run Identity: {e}")
    import traceback
    traceback.print_exc()

# Print summary
print("\nTraining Summary:")
print(f"  - Total epochs: {len(history['train_losses'])}")
# For step-based training, show final step loss instead of average
if "final_step_loss" in history:
    print(f"  - Final train loss (step {trainer.config.max_steps-1}): {history['final_step_loss']:.6f}")
    print(f"  - Average train loss: {history['train_losses'][-1]:.6f}")
else:
    print(f"  - Final train loss: {history['train_losses'][-1]:.6f}")
if history["val_losses"]:
    print(f"  - Final val loss: {history['val_losses'][-1]:.6f}")
print(f"  - Checkpoints saved to: {trainer.checkpoint_manager.save_dir}")
print(f"  - Logs saved to: {trainer.metrics_logger.log_dir}")


if __name__ == "__main__":
    # Script is executed when run directly
    pass
