"""Training orchestrator for Transformer models."""

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from saab_v3.data.structures import Batch
from saab_v3.training.checkpoint import CheckpointManager
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
        model_config: dict | None = None,
        task_config: dict | None = None,
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
            model_config: Optional model configuration dict to save in checkpoint
            task_config: Optional task configuration dict to save in checkpoint
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_head = task_head
        self.task_type = task_type
        self.experiment_name = experiment_name
        self.model_config = model_config
        self.task_config = task_config

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
        # File handler
        file_handler = logging.FileHandler(warning_log_file)
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
        self._small_grad_streak: int = 0  # Track consecutive small gradient steps
        
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

        steps_per_epoch = (
            len(self.train_loader) // self.config.gradient_accumulation_steps
        )
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
            steps_per_epoch = (
                len(self.train_loader) // self.config.gradient_accumulation_steps
            )
            total_epochs = (self.config.max_steps // steps_per_epoch) + 1

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
                self.val_loader is not None
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

            # Check if we've reached max_steps
            if (
                self.config.max_steps is not None
                and self.current_step >= self.config.max_steps
            ):
                print(
                    f"Reached max_steps ({self.config.max_steps}), stopping training."
                )
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

    def _step(self, batch: Batch) -> dict:
        """Single training step."""
        # Initialize step_metrics dict to collect warnings and metrics
        step_metrics = {}

        # Forward pass
        task_type = self._infer_task_type()
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

        # Compute loss
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

        # Early stopping: Check for zero loss collapse
        loss_value = loss.item() * self.config.gradient_accumulation_steps
        if loss_value < 1e-8:  # Consider loss as zero if very small
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

        # Add loss to metrics
        step_metrics["loss"] = loss_value

        # Update weights (if gradient accumulation is complete)
        if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient validation
            has_nan_grad = False
            nan_grad_params = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_grad_params.append(name)
                        has_nan_grad = True

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

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                grad_norm_value = grad_norm.item()
                step_metrics["grad_norm"] = grad_norm_value

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
        """Compute classification loss with numerical stability checks.

        Args:
            logits: Task head outputs [batch, num_classes] or [batch, num_classes] for multi-label
            labels: Labels [batch] (class indices) or [batch, num_classes] (multi-label binary vectors)

        Returns:
            Loss tensor
        """
        # Clamp logits to prevent numerical overflow in softmax/log-softmax
        # CrossEntropyLoss uses log-softmax internally. Extreme logits (>50 or <-50)
        # cause numerical overflow, leading to negative loss values.
        # Range [-50.0, 50.0] is safe for float32:
        # - exp(50) ≈ 5.18e21 (close to float32 max)
        # - exp(-50) ≈ 1.93e-22 (close to float32 min)
        # This prevents overflow while preserving gradient information.
        logits = torch.clamp(logits, min=-50.0, max=50.0)

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
        batch_count = 0

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

                # Compute loss (no clamping applied during validation)
                loss = self._compute_loss(batch, outputs)
                val_losses.append(loss.item())
                batch_count += 1

        # Calculate validation metrics
        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        val_metrics["loss"] = val_loss
        
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
            val_dataset_size = batch_count * (self.val_loader.batch_size if hasattr(self.val_loader, 'batch_size') else 1)
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
