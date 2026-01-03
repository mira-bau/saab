"""Specs for ModelConfig - happy path only."""

import pytest

from saab_v3.config.base import BaseConfig
from saab_v3.models.config import ModelConfig


# ============================================================================
# ModelConfig Specs
# ============================================================================


def spec_model_config_defaults():
    """Verify ModelConfig has correct default values."""
    # Arrange & Act
    config = ModelConfig()

    # Assert
    assert config.d_model == 768
    assert config.num_layers == 12
    assert config.num_heads == 12
    assert config.ffn_dim == 3072
    assert config.max_seq_len == 512
    assert config.dropout == 0.1
    assert config.layer_norm_eps == 1e-5
    assert config.positional_learned is True
    assert config.lambda_bias == 1.0
    assert config.learnable_lambda is False
    assert config.bias_normalization == 1.0
    assert config.device == "cpu"  # Inherited from BaseConfig


def spec_model_config_custom_values():
    """Verify ModelConfig can be initialized with custom values."""
    # Arrange & Act
    config = ModelConfig(
        d_model=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        max_seq_len=256,
        dropout=0.2,
        layer_norm_eps=1e-6,
        positional_learned=False,
        lambda_bias=0.5,
        learnable_lambda=True,
        bias_normalization=0.5,
        device="auto",
    )

    # Assert
    assert config.d_model == 256
    assert config.num_layers == 6
    assert config.num_heads == 8
    assert config.ffn_dim == 1024
    assert config.max_seq_len == 256
    assert config.dropout == 0.2
    assert config.layer_norm_eps == 1e-6
    assert config.positional_learned is False
    assert config.lambda_bias == 0.5
    assert config.learnable_lambda is True
    assert config.bias_normalization == 0.5
    assert config.device == "auto"


def spec_model_config_validation_d_model():
    """Verify ModelConfig validates d_model is positive."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="d_model must be > 0"):
        ModelConfig(d_model=0)

    with pytest.raises(ValueError, match="d_model must be > 0"):
        ModelConfig(d_model=-1)


def spec_model_config_validation_num_layers():
    """Verify ModelConfig validates num_layers is positive."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="num_layers must be > 0"):
        ModelConfig(num_layers=0)

    with pytest.raises(ValueError, match="num_layers must be > 0"):
        ModelConfig(num_layers=-1)


def spec_model_config_validation_num_heads():
    """Verify ModelConfig validates num_heads is positive."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="num_heads must be > 0"):
        ModelConfig(num_heads=0)

    with pytest.raises(ValueError, match="num_heads must be > 0"):
        ModelConfig(num_heads=-1)


def spec_model_config_validation_ffn_dim():
    """Verify ModelConfig validates ffn_dim is positive."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="ffn_dim must be > 0"):
        ModelConfig(ffn_dim=0)

    with pytest.raises(ValueError, match="ffn_dim must be > 0"):
        ModelConfig(ffn_dim=-1)


def spec_model_config_validation_max_seq_len():
    """Verify ModelConfig validates max_seq_len is positive."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="max_seq_len must be > 0"):
        ModelConfig(max_seq_len=0)

    with pytest.raises(ValueError, match="max_seq_len must be > 0"):
        ModelConfig(max_seq_len=-1)


def spec_model_config_validation_dropout():
    """Verify ModelConfig validates dropout is in [0, 1)."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="dropout must be in"):
        ModelConfig(dropout=-0.1)

    with pytest.raises(ValueError, match="dropout must be in"):
        ModelConfig(dropout=1.0)

    # Valid values should work
    config1 = ModelConfig(dropout=0.0)
    assert config1.dropout == 0.0

    config2 = ModelConfig(dropout=0.99)
    assert config2.dropout == 0.99


def spec_model_config_validation_layer_norm_eps():
    """Verify ModelConfig validates layer_norm_eps is positive."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="layer_norm_eps must be > 0"):
        ModelConfig(layer_norm_eps=0)

    with pytest.raises(ValueError, match="layer_norm_eps must be > 0"):
        ModelConfig(layer_norm_eps=-1e-6)


def spec_model_config_validation_lambda_bias():
    """Verify ModelConfig validates lambda_bias is non-negative."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="lambda_bias must be >= 0"):
        ModelConfig(lambda_bias=-0.1)

    # Valid values should work
    config1 = ModelConfig(lambda_bias=0.0)
    assert config1.lambda_bias == 0.0

    config2 = ModelConfig(lambda_bias=2.0)
    assert config2.lambda_bias == 2.0


def spec_model_config_validation_bias_normalization():
    """Verify ModelConfig validates bias_normalization is positive."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="bias_normalization must be > 0"):
        ModelConfig(bias_normalization=0)

    with pytest.raises(ValueError, match="bias_normalization must be > 0"):
        ModelConfig(bias_normalization=-0.5)


def spec_model_config_device_inheritance():
    """Verify ModelConfig inherits device field from BaseConfig."""
    # Arrange & Act
    config = ModelConfig()

    # Assert
    assert hasattr(config, "device")
    assert config.device == "cpu"  # Default from BaseConfig
    assert issubclass(ModelConfig, BaseConfig)

    # Test device validation works (inherited from BaseConfig)
    with pytest.raises(ValueError, match="device must be one of"):
        ModelConfig(device="invalid", d_model=128)


def spec_model_config_device_with_other_fields():
    """Verify device can be set alongside other ModelConfig fields."""
    # Arrange & Act
    config = ModelConfig(
        device="auto",
        d_model=256,
        num_layers=6,
        lambda_bias=1.5,
    )

    # Assert
    assert config.device == "auto"
    assert config.d_model == 256
    assert config.num_layers == 6
    assert config.lambda_bias == 1.5


def spec_model_config_serialization():
    """Verify ModelConfig can be serialized with model_dump()."""
    # Arrange
    config = ModelConfig(
        d_model=256,
        num_layers=6,
        lambda_bias=1.5,
        device="auto",
    )

    # Act
    config_dict = config.model_dump()

    # Assert
    assert isinstance(config_dict, dict)
    assert config_dict["d_model"] == 256
    assert config_dict["num_layers"] == 6
    assert config_dict["lambda_bias"] == 1.5
    assert config_dict["device"] == "auto"
    assert "dropout" in config_dict
    assert "max_seq_len" in config_dict


def spec_model_config_saab_parameters():
    """Verify SAAB-specific parameters work correctly."""
    # Arrange & Act
    config = ModelConfig(
        lambda_bias=2.0,
        learnable_lambda=True,
        bias_normalization=0.5,
    )

    # Assert
    assert config.lambda_bias == 2.0
    assert config.learnable_lambda is True
    assert config.bias_normalization == 0.5

    # Test lambda_bias=0 (should be valid)
    config_zero = ModelConfig(lambda_bias=0.0)
    assert config_zero.lambda_bias == 0.0
