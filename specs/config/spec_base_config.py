"""Specs for BaseConfig - happy path only."""

import warnings

import pytest
import torch

from saab_v3.config.base import BaseConfig
from saab_v3.training.config import PreprocessingConfig


# ============================================================================
# BaseConfig Device Field Tests
# ============================================================================


def spec_base_config_device_default():
    """Verify BaseConfig has default device field."""
    # Arrange & Act
    config = BaseConfig()

    # Assert
    assert hasattr(config, "device")
    assert config.device == "cpu"


def spec_base_config_device_validation():
    """Verify BaseConfig validates device strings correctly."""
    # Test valid device strings that are always available
    always_available = ["cpu", "auto"]
    for device in always_available:
        config = BaseConfig(device=device)
        assert config.device == device

    # Test CUDA - if available, should be set; if not, should fallback to CPU
    if torch.cuda.is_available():
        config = BaseConfig(device="cuda")
        assert config.device == "cuda"
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BaseConfig(device="cuda")
            # Should fallback to CPU
            assert config.device == "cpu"
            assert len(w) > 0
            assert "Device 'cuda' is not available" in str(w[0].message)

    # Test MPS - if available, should be set; if not, should fallback to CPU
    if torch.backends.mps.is_available():
        config = BaseConfig(device="mps")
        assert config.device == "mps"
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BaseConfig(device="mps")
            # Should fallback to CPU
            assert config.device == "cpu"
            assert len(w) > 0
            assert "Device 'mps' is not available" in str(w[0].message)

    # Test invalid device string
    with pytest.raises(ValueError, match="device must be one of"):
        BaseConfig(device="invalid")


def spec_base_config_device_inheritance():
    """Verify PreprocessingConfig inherits device field from BaseConfig."""
    # Arrange & Act
    config = PreprocessingConfig()

    # Assert
    assert hasattr(config, "device")
    assert config.device == "cpu"  # Default from BaseConfig
    assert issubclass(PreprocessingConfig, BaseConfig)

    # Test device can be set in PreprocessingConfig constructor
    config_with_device = PreprocessingConfig(device="auto", vocab_size=1000)
    assert config_with_device.device == "auto"
    assert config_with_device.vocab_size == 1000

    # Test device validation works in inherited configs
    with pytest.raises(ValueError, match="device must be one of"):
        PreprocessingConfig(device="invalid", vocab_size=1000)
