"""Preprocessing and training configuration models."""

from pathlib import Path
from pydantic import ConfigDict, field_validator, model_validator

from saab_v3.config.base import BaseConfig


class PreprocessingConfig(BaseConfig):
    """Configuration for preprocessing pipeline.

    Inherits device field from BaseConfig.
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    vocab_size: int = 30000
    max_seq_len: int = 512
    extractor_type: str | None = None  # "table", "json", "graph", or None (auto-detect)
    extractor_schema: dict | None = None  # Optional schema for extractors
    # device inherited from BaseConfig
    data_dir: Path | None = None  # Override default data/ directory (optional)
    
    # Text tokenization settings
    text_fields: list[str] | None = None  # Explicit list of text field names (e.g., ["title", "content"])
    use_text_tokenizer: bool = False  # Enable subword tokenization for text fields
    text_tokenizer_type: str = "bpe"  # "bpe" or "wordpiece"
    text_tokenizer_vocab_size: int = 30000  # Vocabulary size for text tokenizer
    field_boundary_token: bool = True  # Add [FIELD_START] tokens before each field

    @field_validator("vocab_size")
    @classmethod
    def validate_vocab_size(cls, v: int) -> int:
        """Validate that vocab_size is positive."""
        if v <= 0:
            raise ValueError(f"vocab_size must be > 0, got {v}")
        return v

    @field_validator("max_seq_len")
    @classmethod
    def validate_max_seq_len(cls, v: int) -> int:
        """Validate that max_seq_len is positive."""
        if v <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {v}")
        return v

    @field_validator("extractor_type")
    @classmethod
    def validate_extractor_type(cls, v: str | None) -> str | None:
        """Validate that extractor_type is valid."""
        if v is not None and v not in ["table", "json", "graph"]:
            raise ValueError(
                f"extractor_type must be one of ['table', 'json', 'graph', None], got {v}"
            )
        return v

    @field_validator("data_dir")
    @classmethod
    def validate_data_dir(cls, v: Path | str | None) -> Path | None:
        """Convert string to Path if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v
    
    @field_validator("text_tokenizer_type")
    @classmethod
    def validate_text_tokenizer_type(cls, v: str) -> str:
        """Validate that text_tokenizer_type is valid."""
        if v not in ["bpe", "wordpiece"]:
            raise ValueError(f"text_tokenizer_type must be one of ['bpe', 'wordpiece'], got {v}")
        return v
    
    @field_validator("text_tokenizer_vocab_size")
    @classmethod
    def validate_text_tokenizer_vocab_size(cls, v: int) -> int:
        """Validate that text_tokenizer_vocab_size is positive."""
        if v <= 0:
            raise ValueError(f"text_tokenizer_vocab_size must be > 0, got {v}")
        return v


class TrainingConfig(BaseConfig):
    """Configuration for training pipeline.

    Inherits device field from BaseConfig.
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    # Optimizer settings
    optimizer_type: str = "adamw"  # "adam" or "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Learning rate schedule
    lr_schedule: str = "linear_warmup_cosine"  # "constant", "linear_warmup", "linear_warmup_cosine", "linear_warmup_polynomial", "reduce_on_plateau"
    warmup_steps: int | None = None
    warmup_ratio: float | None = None  # Alternative to warmup_steps
    max_steps: int | None = None  # For LR schedule
    min_lr_ratio: float = 0.0  # Minimum LR as ratio of initial LR
    
    # ReduceLROnPlateau parameters (only used when lr_schedule="reduce_on_plateau")
    lr_mode: str = "min"  # "min" or "max" - whether to minimize or maximize the metric
    lr_factor: float = 0.5  # Factor by which LR is reduced (new_lr = old_lr * factor)
    lr_patience: int = 3  # Number of epochs with no improvement before reducing LR
    lr_threshold: float = 1e-4  # Minimum change to qualify as improvement
    lr_min: float = 1e-8  # Minimum learning rate threshold
    lr_cooldown: int = 0  # Number of epochs to wait before resuming normal operation after LR reduction

    # Training settings
    batch_size: int = 32
    num_epochs: int | None = None
    max_steps: int | None = None  # Alternative to num_epochs
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0  # Gradient clipping
    seed: int = 42  # For reproducibility

    # Checkpointing
    save_dir: Path | str | None = None  # Default: checkpoints/{experiment_name}/
    save_steps: int | None = None  # Save every N steps
    save_epochs: int | None = None  # Save every N epochs
    keep_checkpoints: int = 3  # Keep last N checkpoints
    save_best: bool = True  # Save best model based on validation metric
    best_metric: str = "loss"  # Metric to track for best model
    best_mode: str = "min"  # "min" or "max"

    # Logging
    log_dir: Path | str | None = None  # Default: logs/{experiment_name}/
    log_steps: int = 100  # Log every N steps
    log_epochs: bool = True  # Log at end of each epoch
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # Validation
    eval_steps: int | None = None  # Evaluate every N steps
    eval_epochs: int = 1  # Evaluate every N epochs
    eval_metrics: list[str] = ["loss"]

    # Early stopping
    early_stop_zero_loss_steps: int | None = None  # Stop if loss is zero for N consecutive steps
    early_stopping_patience: int | None = None  # Stop if validation metric doesn't improve for N epochs
    early_stopping_min_delta: float = 0.0  # Minimum change to qualify as improvement
    early_stopping_metric: str = "loss"  # Metric to monitor for early stopping

    @field_validator("optimizer_type")
    @classmethod
    def validate_optimizer_type(cls, v: str) -> str:
        """Validate optimizer type."""
        if v not in ["adam", "adamw"]:
            raise ValueError(f"optimizer_type must be one of ['adam', 'adamw'], got {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate is positive."""
        if v <= 0:
            raise ValueError(f"learning_rate must be > 0, got {v}")
        return v

    @field_validator("weight_decay")
    @classmethod
    def validate_weight_decay(cls, v: float) -> float:
        """Validate weight decay is non-negative."""
        if v < 0:
            raise ValueError(f"weight_decay must be >= 0, got {v}")
        return v

    @field_validator("lr_schedule")
    @classmethod
    def validate_lr_schedule(cls, v: str) -> str:
        """Validate LR schedule type."""
        valid_schedules = ["constant", "linear_warmup", "linear_warmup_cosine", "linear_warmup_polynomial", "reduce_on_plateau"]
        if v not in valid_schedules:
            raise ValueError(f"lr_schedule must be one of {valid_schedules}, got {v}")
        return v
    
    @field_validator("lr_mode")
    @classmethod
    def validate_lr_mode(cls, v: str) -> str:
        """Validate LR mode for ReduceLROnPlateau."""
        if v not in ["min", "max"]:
            raise ValueError(f"lr_mode must be one of ['min', 'max'], got {v}")
        return v
    
    @field_validator("lr_factor")
    @classmethod
    def validate_lr_factor(cls, v: float) -> float:
        """Validate LR factor is in (0, 1)."""
        if not 0 < v < 1:
            raise ValueError(f"lr_factor must be in (0, 1), got {v}")
        return v
    
    @field_validator("lr_patience")
    @classmethod
    def validate_lr_patience(cls, v: int) -> int:
        """Validate LR patience is positive."""
        if v <= 0:
            raise ValueError(f"lr_patience must be > 0, got {v}")
        return v
    
    @field_validator("lr_min")
    @classmethod
    def validate_lr_min(cls, v: float) -> float:
        """Validate minimum LR is positive."""
        if v <= 0:
            raise ValueError(f"lr_min must be > 0, got {v}")
        return v
    
    @field_validator("lr_cooldown")
    @classmethod
    def validate_lr_cooldown(cls, v: int) -> int:
        """Validate LR cooldown is non-negative."""
        if v < 0:
            raise ValueError(f"lr_cooldown must be >= 0, got {v}")
        return v

    @field_validator("warmup_steps")
    @classmethod
    def validate_warmup_steps(cls, v: int | None) -> int | None:
        """Validate warmup steps is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError(f"warmup_steps must be > 0, got {v}")
        return v

    @field_validator("warmup_ratio")
    @classmethod
    def validate_warmup_ratio(cls, v: float | None) -> float:
        """Validate warmup ratio is in [0, 1) if provided."""
        if v is not None:
            if not (0 <= v < 1):
                raise ValueError(f"warmup_ratio must be in [0, 1), got {v}")
        return v

    @field_validator("min_lr_ratio")
    @classmethod
    def validate_min_lr_ratio(cls, v: float) -> float:
        """Validate min LR ratio is in [0, 1]."""
        if not (0 <= v <= 1):
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError(f"batch_size must be > 0, got {v}")
        return v

    @field_validator("num_epochs")
    @classmethod
    def validate_num_epochs(cls, v: int | None) -> int | None:
        """Validate num_epochs is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError(f"num_epochs must be > 0, got {v}")
        return v

    @field_validator("max_steps")
    @classmethod
    def validate_max_steps(cls, v: int | None) -> int | None:
        """Validate max_steps is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError(f"max_steps must be > 0, got {v}")
        return v

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def validate_gradient_accumulation_steps(cls, v: int) -> int:
        """Validate gradient accumulation steps is positive."""
        if v <= 0:
            raise ValueError(f"gradient_accumulation_steps must be > 0, got {v}")
        return v

    @field_validator("max_grad_norm")
    @classmethod
    def validate_max_grad_norm(cls, v: float) -> float:
        """Validate max grad norm is positive."""
        if v <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {v}")
        return v

    @field_validator("best_mode")
    @classmethod
    def validate_best_mode(cls, v: str) -> str:
        """Validate best mode."""
        if v not in ["min", "max"]:
            raise ValueError(f"best_mode must be one of ['min', 'max'], got {v}")
        return v

    @field_validator("save_dir")
    @classmethod
    def validate_save_dir(cls, v: Path | str | None) -> Path | None:
        """Convert string to Path if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("log_dir")
    @classmethod
    def validate_log_dir(cls, v: Path | str | None) -> Path | None:
        """Convert string to Path if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("log_steps")
    @classmethod
    def validate_log_steps(cls, v: int) -> int:
        """Validate log_steps is positive."""
        if v <= 0:
            raise ValueError(f"log_steps must be > 0, got {v}")
        return v

    @field_validator("eval_epochs")
    @classmethod
    def validate_eval_epochs(cls, v: int) -> int:
        """Validate eval_epochs is positive."""
        if v <= 0:
            raise ValueError(f"eval_epochs must be > 0, got {v}")
        return v

    @field_validator("early_stop_zero_loss_steps")
    @classmethod
    def validate_early_stop_zero_loss_steps(cls, v: int | None) -> int | None:
        """Validate early_stop_zero_loss_steps is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError(f"early_stop_zero_loss_steps must be > 0, got {v}")
        return v

    @field_validator("early_stopping_patience")
    @classmethod
    def validate_early_stopping_patience(cls, v: int | None) -> int | None:
        """Validate early_stopping_patience is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError(f"early_stopping_patience must be > 0, got {v}")
        return v

    @field_validator("early_stopping_min_delta")
    @classmethod
    def validate_early_stopping_min_delta(cls, v: float) -> float:
        """Validate early_stopping_min_delta is non-negative."""
        if v < 0:
            raise ValueError(f"early_stopping_min_delta must be >= 0, got {v}")
        return v

    @field_validator("early_stopping_metric")
    @classmethod
    def validate_early_stopping_metric(cls, v: str) -> str:
        """Validate early_stopping_metric is a valid metric name."""
        if not v or not isinstance(v, str):
            raise ValueError(f"early_stopping_metric must be a non-empty string, got {v}")
        return v

    @model_validator(mode="after")
    def validate_training_duration(self) -> "TrainingConfig":
        """Ensure either num_epochs or max_steps is set."""
        if self.num_epochs is None and self.max_steps is None:
            raise ValueError("Either num_epochs or max_steps must be set")
        return self

    @model_validator(mode="after")
    def validate_warmup(self) -> "TrainingConfig":
        """Ensure warmup is properly configured if warmup is enabled."""
        if self.lr_schedule not in ["constant", "reduce_on_plateau"]:
            if self.warmup_steps is None and self.warmup_ratio is None:
                raise ValueError(
                    "Either warmup_steps or warmup_ratio must be set when lr_schedule != 'constant' and != 'reduce_on_plateau'"
                )
        return self
