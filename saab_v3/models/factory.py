"""Factory functions for creating models from preprocessor.

All factory functions automatically move models to the device specified in
config.device (using get_device utility). This ensures consistency with the
existing pattern where dataloader uses preprocessor config device.

Example usage:
    >>> from saab_v3.models import ModelConfig, create_flat_transformer
    >>> config = ModelConfig(device="mps")  # or "cuda", "cpu", "auto"
    >>> model = create_flat_transformer(preprocessor, config)
    >>> # Model is automatically on MPS device
    >>> # Batches from dataloader (using same preprocessor config) are also on MPS
    >>> output = model(batch)  # Everything matches!
"""

from saab_v3.training.preprocessor import Preprocessor
from saab_v3.models.config import ModelConfig
from saab_v3.models.flat_transformer import FlatTransformer
from saab_v3.models.scratch_transformer import ScratchTransformer
from saab_v3.models.saab_transformer import SAABTransformer
from saab_v3.utils.device import get_device


def get_vocab_sizes(preprocessor: Preprocessor) -> dict[str, int]:
    """Extract vocabulary sizes from preprocessor.

    Args:
        preprocessor: Fitted Preprocessor instance

    Returns:
        Dictionary with vocabulary sizes:
        - token_vocab_size
        - token_type_vocab_size
        - field_vocab_size
        - entity_vocab_size
        - time_vocab_size
    """
    if not preprocessor._is_fitted:
        raise ValueError("Preprocessor must be fitted before extracting vocab sizes")

    vocab_sizes = {
        "token_vocab_size": len(preprocessor.tokenizer.vocab),
        "token_type_vocab_size": len(preprocessor.tag_encoder.tag_vocabs["token_type"]),
        "field_vocab_size": len(preprocessor.tag_encoder.tag_vocabs["field"]),
        "entity_vocab_size": len(preprocessor.tag_encoder.tag_vocabs["entity"]),
        "time_vocab_size": len(preprocessor.tag_encoder.tag_vocabs["time"]),
    }

    return vocab_sizes


def create_flat_transformer(
    preprocessor: Preprocessor,
    config: ModelConfig,
) -> FlatTransformer:
    """Create FlatTransformer from preprocessor and config.

    The model is automatically moved to the device specified in config.device
    (using get_device utility). Ensure batches are on the same device.

    Args:
        preprocessor: Fitted Preprocessor instance
        config: ModelConfig instance with device specified

    Returns:
        FlatTransformer instance on the device specified in config.device

    Example:
        >>> config = ModelConfig(device="mps")
        >>> model = create_flat_transformer(preprocessor, config)
        >>> # Model is now on MPS device
    """
    vocab_sizes = get_vocab_sizes(preprocessor)

    model = FlatTransformer(
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        vocab_sizes=vocab_sizes,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        layer_norm_eps=config.layer_norm_eps,
        positional_learned=config.positional_learned,
    )

    # Automatically move model to device specified in config
    device = get_device(config.device)
    model = model.to(device)

    return model


def create_scratch_transformer(
    preprocessor: Preprocessor,
    config: ModelConfig,
) -> ScratchTransformer:
    """Create ScratchTransformer from preprocessor and config.

    The model is automatically moved to the device specified in config.device
    (using get_device utility). Ensure batches are on the same device.

    Args:
        preprocessor: Fitted Preprocessor instance
        config: ModelConfig instance with device specified

    Returns:
        ScratchTransformer instance on the device specified in config.device

    Example:
        >>> config = ModelConfig(device="mps")
        >>> model = create_scratch_transformer(preprocessor, config)
        >>> # Model is now on MPS device
    """
    vocab_sizes = get_vocab_sizes(preprocessor)

    model = ScratchTransformer(
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        vocab_sizes=vocab_sizes,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        layer_norm_eps=config.layer_norm_eps,
        positional_learned=config.positional_learned,
    )

    # Automatically move model to device specified in config
    device = get_device(config.device)
    model = model.to(device)

    return model


def create_saab_transformer(
    preprocessor: Preprocessor,
    config: ModelConfig,
    lambda_bias: float | None = None,
) -> SAABTransformer:
    """Create SAABTransformer from preprocessor and config.

    The model is automatically moved to the device specified in config.device
    (using get_device utility). Ensure batches are on the same device.

    Args:
        preprocessor: Fitted Preprocessor instance
        config: ModelConfig instance with device specified
        lambda_bias: Optional override for lambda_bias (uses config.lambda_bias if None)

    Returns:
        SAABTransformer instance on the device specified in config.device

    Example:
        >>> config = ModelConfig(device="mps")
        >>> model = create_saab_transformer(preprocessor, config)
        >>> # Model is now on MPS device
    """
    vocab_sizes = get_vocab_sizes(preprocessor)

    model = SAABTransformer(
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        vocab_sizes=vocab_sizes,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        layer_norm_eps=config.layer_norm_eps,
        positional_learned=config.positional_learned,
        lambda_bias=lambda_bias if lambda_bias is not None else config.lambda_bias,
        learnable_lambda=config.learnable_lambda,
        bias_normalization=config.bias_normalization,
    )

    # Automatically move model to device specified in config
    device = get_device(config.device)
    model = model.to(device)

    return model
