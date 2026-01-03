"""Specs for PreprocessingConfig and TrainingConfig - happy path only."""

import pytest

from saab_v3.config.base import BaseConfig
from saab_v3.training.config import PreprocessingConfig, TrainingConfig


# ============================================================================
# PreprocessingConfig Device Inheritance Tests
# ============================================================================


def spec_preprocessing_config_device_inheritance():
    """Verify PreprocessingConfig inherits device field from BaseConfig."""
    # Arrange & Act
    config = PreprocessingConfig()

    # Assert
    assert hasattr(config, "device")
    assert config.device == "cpu"  # Default from BaseConfig
    assert issubclass(PreprocessingConfig, BaseConfig)

    # Test device field is accessible without explicit definition
    assert (
        "device" in PreprocessingConfig.model_fields
        or "device" in BaseConfig.model_fields
    )
    # Verify device is inherited from BaseConfig
    assert hasattr(config, "device")

    # Test device validation works (inherited from BaseConfig)
    with pytest.raises(ValueError, match="device must be one of"):
        PreprocessingConfig(device="invalid", vocab_size=1000)


def spec_preprocessing_config_device_with_other_fields():
    """Verify device can be set alongside other PreprocessingConfig fields."""
    # Arrange & Act
    config = PreprocessingConfig(
        device="auto",
        vocab_size=5000,
        max_seq_len=256,
    )

    # Assert
    assert config.device == "auto"
    assert config.vocab_size == 5000
    assert config.max_seq_len == 256

    # Test device doesn't interfere with other config fields
    config2 = PreprocessingConfig(
        device="cpu",
        vocab_size=10000,
        max_seq_len=512,
    )
    assert config2.device == "cpu"
    assert config2.vocab_size == 10000
    assert config2.max_seq_len == 512

    # Test config serialization includes device field
    config_dict = config.model_dump()
    assert "device" in config_dict
    assert config_dict["device"] == "auto"
    assert "vocab_size" in config_dict
    assert config_dict["vocab_size"] == 5000


# ============================================================================
# TrainingConfig Tests
# ============================================================================


def spec_training_config_defaults():
    """Verify TrainingConfig has correct default values."""
    # Arrange & Act (must provide num_epochs or max_steps, and warmup for non-constant schedule)
    config = TrainingConfig(num_epochs=10, warmup_steps=1000)

    # Assert
    assert config.optimizer_type == "adamw"
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 0.01
    assert config.batch_size == 32
    assert config.num_epochs == 10
    assert config.lr_schedule == "linear_warmup_cosine"
    assert config.warmup_steps == 1000
    assert config.seed == 42
    assert config.device == "cpu"  # Inherited from BaseConfig


def spec_training_config_validation():
    """Verify TrainingConfig validates fields correctly."""
    # Test optimizer_type validation
    with pytest.raises(ValueError, match="optimizer_type must be one of"):
        TrainingConfig(optimizer_type="invalid", num_epochs=1)

    # Test learning_rate validation
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        TrainingConfig(learning_rate=0, num_epochs=1)

    # Test batch_size validation
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        TrainingConfig(batch_size=0, num_epochs=1)

    # Test num_epochs or max_steps required
    with pytest.raises(ValueError, match="Either num_epochs or max_steps must be set"):
        TrainingConfig()

    # Test warmup required for non-constant schedule
    with pytest.raises(ValueError, match="Either warmup_steps or warmup_ratio must be set"):
        TrainingConfig(num_epochs=1, lr_schedule="linear_warmup_cosine")


def spec_training_config_device_inheritance():
    """Verify TrainingConfig inherits device field from BaseConfig."""
    # Arrange & Act (use constant schedule to avoid warmup requirement)
    config = TrainingConfig(num_epochs=1, lr_schedule="constant")

    # Assert
    assert hasattr(config, "device")
    assert config.device == "cpu"  # Default from BaseConfig
    assert issubclass(TrainingConfig, BaseConfig)


def spec_training_config_custom_values():
    """Verify TrainingConfig can be initialized with custom values."""
    # Arrange & Act
    config = TrainingConfig(
        optimizer_type="adam",
        learning_rate=2e-4,
        batch_size=64,
        num_epochs=5,
        lr_schedule="linear_warmup",
        warmup_steps=500,
        seed=123,
        device="auto",
    )

    # Assert
    assert config.optimizer_type == "adam"
    assert config.learning_rate == 2e-4
    assert config.batch_size == 64
    assert config.num_epochs == 5
    assert config.lr_schedule == "linear_warmup"
    assert config.warmup_steps == 500
    assert config.seed == 123
    assert config.device == "auto"
