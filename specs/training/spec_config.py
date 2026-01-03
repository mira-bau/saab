"""Specs for PreprocessingConfig device inheritance - happy path only."""

import pytest

from saab_v3.config.base import BaseConfig
from saab_v3.training.config import PreprocessingConfig


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
        preserve_original_tags=True,
    )

    # Assert
    assert config.device == "auto"
    assert config.vocab_size == 5000
    assert config.max_seq_len == 256
    assert config.preserve_original_tags is True

    # Test device doesn't interfere with other config fields
    config2 = PreprocessingConfig(
        device="cpu",
        vocab_size=10000,
        max_seq_len=512,
        preserve_original_tags=False,
    )
    assert config2.device == "cpu"
    assert config2.vocab_size == 10000
    assert config2.max_seq_len == 512
    assert config2.preserve_original_tags is False

    # Test config serialization includes device field
    config_dict = config.model_dump()
    assert "device" in config_dict
    assert config_dict["device"] == "auto"
    assert "vocab_size" in config_dict
    assert config_dict["vocab_size"] == 5000
