"""Training orchestrator for Transformer models."""

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from saab_v3.data.structures import Batch
from saab_v3.training.checkpoint import CheckpointManager
from saab_v3.training.loss import create_loss_fn
from saab_v3.training.metrics import MetricsLogger
from saab_v3.training.schedulers import create_lr_scheduler
from saab_v3.utils.device import get_device

if TYPE_CHECKING:
    from saab_v3.training.config import TrainingConfig


class Trainer:
    """Training orchestrator for Transformer models."""

    def __init__(
        self,
        model: nn.Module,
        config: "TrainingConfig",
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        task_head: nn.Module | None = None,
        loss_fn: nn.Module | None = None,
        task_type: str | None = None,
        experiment_name: str = "experiment",
    ):
        """Initialize trainer.

        Args:
            model: Transformer model (Flat/Scratch/SAAB)
            config: TrainingConfig instance
            train_loader: Training DataLoader
            val_loader: Optional validation DataLoader
            task_head: Optional task head
            loss_fn: Loss function (optional: auto-created if task_type provided)
            task_type: Task type for auto-creating loss function ("classification", "regression", etc.)
            experiment_name: Name of experiment (for checkpoint/logging directories)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_head = task_head
        self.task_type = task_type
        self.experiment_name = experiment_name

        # Set random seeds for reproducibility
        self._set_random_seeds(config.seed)

        # Move model to device
        device = get_device(config.device)
        self.model = self.model.to(device)
        if self.task_head is not None:
            self.task_head = self.task_head.to(device)

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
        self.scheduler = create_lr_scheduler(
            self.optimizer, config, num_training_steps
        )

        # Checkpoint manager
        save_dir = config.save_dir or Path("checkpoints") / experiment_name
        self.checkpoint_manager = CheckpointManager(
            save_dir=save_dir, keep_checkpoints=config.keep_checkpoints
        )

        # Metrics logger
        self.metrics_logger = MetricsLogger(config, experiment_name)

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric_value: float | None = None

        # Config for checkpointing
        self._config_dict = {
            "training_config": config.model_dump() if hasattr(config, "model_dump") else {},
        }

    def _set_random_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set deterministic operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        if self.config.optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer_type: {self.config.optimizer_type}")

    def _calculate_training_steps(self) -> int:
        """Calculate total number of training steps."""
        if self.config.max_steps is not None:
            return self.config.max_steps

        if self.config.num_epochs is None:
            raise ValueError("Either num_epochs or max_steps must be set")

        # Handle case where train_loader might be None (for testing)
        if self.train_loader is None:
            # Return a default value for testing scenarios
            return self.config.num_epochs * 100  # Default to 100 steps per epoch

        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        return self.config.num_epochs * steps_per_epoch

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

        # Determine training duration
        if self.config.num_epochs is not None:
            total_epochs = self.config.num_epochs
        else:
            # Calculate epochs from max_steps
            steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
            total_epochs = (self.config.max_steps // steps_per_epoch) + 1

        print(f"Starting training for {total_epochs} epochs...")
        print(f"Total steps: {self._calculate_training_steps()}")
        print(f"Device: {get_device(self.config.device)}")

        for epoch in range(total_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self._train_epoch(epoch)
            history["train_losses"].append(train_metrics.get("loss", 0.0))
            history["train_metrics"].append(train_metrics)

            # Validate
            if self.val_loader is not None and (epoch + 1) % self.config.eval_epochs == 0:
                val_metrics = self._validate()
                history["val_losses"].append(val_metrics.get("loss", 0.0))
                history["val_metrics"].append(val_metrics)

                # Check if this is the best model
                if self.config.save_best:
                    self._check_and_save_best(val_metrics, epoch)

            # Save checkpoint (epoch-based)
            if (
                self.config.save_epochs is not None
                and (epoch + 1) % self.config.save_epochs == 0
            ):
                self._save_checkpoint(epoch, train_metrics, is_latest=True)

            # Check if we've reached max_steps
            if self.config.max_steps is not None and self.current_step >= self.config.max_steps:
                print(f"Reached max_steps ({self.config.max_steps}), stopping training.")
                break

        # Save final checkpoint
        self._save_checkpoint(self.current_epoch, train_metrics, is_latest=True)

        # Cleanup
        self.metrics_logger.close()
        self.checkpoint_manager.cleanup_old_checkpoints()

        return history

    def _train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        if self.task_head is not None:
            self.task_head.train()

        epoch_losses = []
        epoch_metrics = {}

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = batch.to(get_device(self.config.device))

            # Training step
            step_metrics = self._step(batch)

            epoch_losses.append(step_metrics["loss"])

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
                self._save_checkpoint(epoch, step_metrics, is_latest=False)

            # Validate (step-based)
            if (
                self.val_loader is not None
                and self.config.eval_steps is not None
                and self.current_step % self.config.eval_steps == 0
            ):
                val_metrics = self._validate()
                if self.config.save_best:
                    self._check_and_save_best(val_metrics, epoch)

            self.current_step += 1

            # Check if we've reached max_steps
            if self.config.max_steps is not None and self.current_step >= self.config.max_steps:
                break

        # Calculate epoch averages
        epoch_metrics["loss"] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        # Log epoch metrics
        if self.config.log_epochs:
            self.metrics_logger.log_epoch(epoch, epoch_metrics, phase="train")

        return epoch_metrics

    def _step(self, batch: Batch) -> dict:
        """Single training step."""
        # Forward pass
        task_type = self._infer_task_type()
        if task_type == "ranking":
            outputs = self._forward_ranking(batch)
        else:
            outputs = self._forward_single(batch)

        # Compute loss
        loss = self._compute_loss(batch, outputs)

        # Backward pass (with gradient accumulation)
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        step_metrics = {"loss": loss.item() * self.config.gradient_accumulation_steps}

        # Update weights (if gradient accumulation is complete)
        if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                step_metrics["grad_norm"] = grad_norm.item()

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # LR scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                step_metrics["learning_rate"] = self.scheduler.get_last_lr()[0]

        return step_metrics

    def _compute_loss(self, batch: Batch, outputs: torch.Tensor) -> torch.Tensor:
        """Compute loss for batch.

        Args:
            batch: Batch object (may contain labels)
            outputs: Task head outputs (logits, predictions, etc.)

        Returns:
            Loss tensor

        Raises:
            ValueError: If labels are required but missing
        """
        if self.task_head is None:
            # No task head: dummy loss (for backward compatibility)
            return outputs.mean()

        if batch.labels is None:
            raise ValueError(
                "Labels are required for supervised learning. "
                "Batch must contain 'labels' field."
            )

        # Get task type
        task_type = self._infer_task_type()

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
            if "Classification" in class_name:
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
        """Compute classification loss.

        Args:
            logits: Task head outputs [batch, num_classes] or [batch, num_classes] for multi-label
            labels: Labels [batch] (class indices) or [batch, num_classes] (multi-label binary vectors)

        Returns:
            Loss tensor
        """
        return self.loss_fn(logits, labels)

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

    def _forward_single(self, batch: Batch) -> torch.Tensor:
        """Forward pass for single-sequence tasks.

        Args:
            batch: Batch object

        Returns:
            Task head outputs
        """
        outputs = self.model(batch)
        if self.task_head is not None:
            outputs = self.task_head(outputs)
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

        # Pool to get representations (use CLS token at position 0)
        repr_a = outputs_a[:, 0, :]  # [batch, d_model]
        repr_b = outputs_b[:, 0, :]  # [batch, d_model]

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
        self.model.eval()
        if self.task_head is not None:
            self.task_head.eval()

        val_losses = []
        val_metrics = {}

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = batch.to(get_device(self.config.device))

                # Forward pass
                task_type = self._infer_task_type()
                if task_type == "ranking":
                    outputs = self._forward_ranking(batch)
                else:
                    outputs = self._forward_single(batch)

                # Compute loss
                loss = self._compute_loss(batch, outputs)
                val_losses.append(loss.item())

        # Calculate validation metrics
        val_metrics["loss"] = sum(val_losses) / len(val_losses) if val_losses else 0.0

        # Log validation metrics
        self.metrics_logger.log_epoch(self.current_epoch, val_metrics, phase="val")

        return val_metrics

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
        )

        if resume:
            self.current_epoch = state["epoch"]
            self.current_step = state["step"]
            if "metrics" in state and self.config.best_metric in state["metrics"]:
                self.best_metric_value = state["metrics"][self.config.best_metric]

        return state

