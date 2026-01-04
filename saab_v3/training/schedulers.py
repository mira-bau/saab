"""Learning rate scheduler factory functions."""

import math
from typing import TYPE_CHECKING

import torch
import torch.optim.lr_scheduler as lr_scheduler

if TYPE_CHECKING:
    from saab_v3.training.config import TrainingConfig


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: "TrainingConfig",
    num_training_steps: int,
) -> lr_scheduler._LRScheduler | lr_scheduler.ReduceLROnPlateau | None:
    """Create learning rate scheduler from config.

    Supports:
    - constant: No scheduling
    - linear_warmup: Linear warmup then constant
    - linear_warmup_cosine: Linear warmup then cosine decay
    - linear_warmup_polynomial: Linear warmup then polynomial decay
    - reduce_on_plateau: Adaptive LR reduction when validation metric plateaus

    Args:
        optimizer: Optimizer instance
        config: TrainingConfig instance
        num_training_steps: Total number of training steps

    Returns:
        LR scheduler instance, or None if constant schedule

    Raises:
        ValueError: If config is invalid
    """
    if config.lr_schedule == "constant":
        return None

    if config.lr_schedule == "reduce_on_plateau":
        # Adaptive scheduler based on validation metric
        # Note: verbose parameter removed - PyTorch's ReduceLROnPlateau doesn't support it
        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=config.lr_mode,
            factor=config.lr_factor,
            patience=config.lr_patience,
            threshold=config.lr_threshold,
            min_lr=config.lr_min,
            cooldown=config.lr_cooldown,
        )

    # Calculate warmup steps for time-based schedulers
    if config.warmup_steps is not None:
        warmup_steps = config.warmup_steps
    elif config.warmup_ratio is not None:
        warmup_steps = int(num_training_steps * config.warmup_ratio)
    else:
        raise ValueError("Either warmup_steps or warmup_ratio must be set")

    # Calculate decay steps
    if config.max_steps is not None:
        decay_steps = config.max_steps - warmup_steps
    else:
        decay_steps = num_training_steps - warmup_steps

    if decay_steps <= 0:
        raise ValueError(
            f"Decay steps must be > 0. Got decay_steps={decay_steps}, "
            f"warmup_steps={warmup_steps}, num_training_steps={num_training_steps}"
        )

    if config.lr_schedule == "linear_warmup":
        # Linear warmup then constant
        return _LinearWarmupConstantScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

    elif config.lr_schedule == "linear_warmup_cosine":
        # Linear warmup then cosine decay
        return _LinearWarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

    elif config.lr_schedule == "linear_warmup_polynomial":
        # Linear warmup then polynomial decay
        return _LinearWarmupPolynomialScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")


class _LinearWarmupConstantScheduler(lr_scheduler._LRScheduler):
    """Linear warmup then constant learning rate."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            min_lr_ratio: Minimum LR as ratio of initial LR
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Get learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [
                base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * warmup_factor)
                for base_lr in self.base_lrs
            ]
        else:
            # Constant after warmup
            return [base_lr for base_lr in self.base_lrs]


class _LinearWarmupCosineScheduler(lr_scheduler._LRScheduler):
    """Linear warmup then cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        decay_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            decay_steps: Number of decay steps after warmup
            min_lr_ratio: Minimum LR as ratio of initial LR
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Get learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [
                base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * warmup_factor)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay after warmup
            progress = (self.last_epoch - self.warmup_steps) / self.decay_steps
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                base_lr
                * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor)
                for base_lr in self.base_lrs
            ]


class _LinearWarmupPolynomialScheduler(lr_scheduler._LRScheduler):
    """Linear warmup then polynomial decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        decay_steps: int,
        min_lr_ratio: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            decay_steps: Number of decay steps after warmup
            min_lr_ratio: Minimum LR as ratio of initial LR
            power: Polynomial power (default: 1.0 for linear decay)
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Get learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [
                base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * warmup_factor)
                for base_lr in self.base_lrs
            ]
        else:
            # Polynomial decay after warmup
            progress = (self.last_epoch - self.warmup_steps) / self.decay_steps
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            poly_factor = (1 - progress) ** self.power
            return [
                base_lr
                * (self.min_lr_ratio + (1 - self.min_lr_ratio) * poly_factor)
                for base_lr in self.base_lrs
            ]

