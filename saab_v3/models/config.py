"""Model configuration classes."""

from pydantic import ConfigDict, field_validator

from saab_v3.config.base import BaseConfig


class ModelConfig(BaseConfig):
    """Configuration for Transformer models.

    Shared across Flat, Scratc
    h, and SAAB for fair comparison.
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    # Architecture hyperparameters
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    max_seq_len: int = 512

    # Regularization
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Embeddings
    positional_learned: bool = True

    # SAAB-specific (only used for SAABTransformer)
    lambda_bias: float = 1.0
    learnable_lambda: bool = False
    bias_normalization: float = 1.0

    @field_validator("d_model")
    @classmethod
    def validate_d_model(cls, v: int) -> int:
        """Validate that d_model is positive."""
        if v <= 0:
            raise ValueError(f"d_model must be > 0, got {v}")
        return v

    @field_validator("num_layers")
    @classmethod
    def validate_num_layers(cls, v: int) -> int:
        """Validate that num_layers is positive."""
        if v <= 0:
            raise ValueError(f"num_layers must be > 0, got {v}")
        return v

    @field_validator("num_heads")
    @classmethod
    def validate_num_heads(cls, v: int) -> int:
        """Validate that num_heads is positive."""
        if v <= 0:
            raise ValueError(f"num_heads must be > 0, got {v}")
        return v

    @field_validator("ffn_dim")
    @classmethod
    def validate_ffn_dim(cls, v: int) -> int:
        """Validate that ffn_dim is positive."""
        if v <= 0:
            raise ValueError(f"ffn_dim must be > 0, got {v}")
        return v

    @field_validator("max_seq_len")
    @classmethod
    def validate_max_seq_len(cls, v: int) -> int:
        """Validate that max_seq_len is positive."""
        if v <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {v}")
        return v

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        """Validate that dropout is in [0, 1)."""
        if not (0 <= v < 1):
            raise ValueError(f"dropout must be in [0, 1), got {v}")
        return v

    @field_validator("layer_norm_eps")
    @classmethod
    def validate_layer_norm_eps(cls, v: float) -> float:
        """Validate that layer_norm_eps is positive."""
        if v <= 0:
            raise ValueError(f"layer_norm_eps must be > 0, got {v}")
        return v

    @field_validator("lambda_bias")
    @classmethod
    def validate_lambda_bias(cls, v: float) -> float:
        """Validate that lambda_bias is non-negative."""
        if v < 0:
            raise ValueError(f"lambda_bias must be >= 0, got {v}")
        return v

    @field_validator("bias_normalization")
    @classmethod
    def validate_bias_normalization(cls, v: float) -> float:
        """Validate that bias_normalization is positive."""
        if v <= 0:
            raise ValueError(f"bias_normalization must be > 0, got {v}")
        return v
