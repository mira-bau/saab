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

    When text tokenization is enabled, uses the text tokenizer's vocabulary size
    instead of the categorical vocabulary size, since text tokenizer IDs are
    what actually get passed to the model.

    Args:
        preprocessor: Fitted Preprocessor instance

    Returns:
        Dictionary with vocabulary sizes:
        - token_vocab_size (from text tokenizer if enabled, else categorical vocab)
        - token_type_vocab_size
        - field_vocab_size
        - entity_vocab_size
        - time_vocab_size
    """
    if not preprocessor._is_fitted:
        raise ValueError("Preprocessor must be fitted before extracting vocab sizes")

    # Determine token vocabulary size
    # If text tokenizer is enabled, use its vocab size (that's what produces the token IDs)
    # Otherwise, use categorical vocab size
    if preprocessor.tokenizer.text_tokenizer is not None:
        # Get vocab size from text tokenizer
        # The tokenizer's vocab size is stored in the tokenizer object
        try:
            # Try to get vocab size from the underlying tokenizer
            text_tokenizer = preprocessor.tokenizer.text_tokenizer.tok
            # HuggingFace tokenizers have get_vocab_size() method
            if hasattr(text_tokenizer, "get_vocab_size"):
                token_vocab_size = text_tokenizer.get_vocab_size()
            elif hasattr(text_tokenizer, "get_vocab"):
                # Fallback: get vocab dict and check size
                vocab = text_tokenizer.get_vocab()
                token_vocab_size = len(vocab)
            else:
                # Fallback: use the config value
                token_vocab_size = preprocessor.config.text_tokenizer_vocab_size
        except (AttributeError, TypeError) as e:
            # Fallback to config value
            token_vocab_size = preprocessor.config.text_tokenizer_vocab_size
        
        # Also need to account for categorical vocab tokens (like [FIELD_START])
        # Use max of both to ensure all token IDs fit
        categorical_vocab_size = len(preprocessor.tokenizer.vocab)
        token_vocab_size = max(token_vocab_size, categorical_vocab_size)
    else:
        # Use categorical vocabulary size
        token_vocab_size = len(preprocessor.tokenizer.vocab)

    # Task 4: Field vocab size computation
    # field_vocab_size_raw = total vocab size (includes PAD, NONE, UNK, plus real fields)
    # MASK_FIELD_TOKEN is NOT in the vocab (excluded in TagEncoder.__init__)
    # field_vocab_size for embedding = field_vocab_size_raw + 1 (the +1 is MASK_FIELD_ID, which is NOT in vocab)
    # Note: num_fields (computed in train.py) = field_vocab_size_raw (includes all field IDs, PAD handled by ignore_index)
    #   - field_emb_size = field_vocab_size_raw + 1 (all vocab tokens + MASK_FIELD_ID)
    #   - Therefore: field_emb_size = num_fields + 1
    field_vocab_size_raw = len(preprocessor.tag_encoder.tag_vocabs["field"])
    # Add exactly one +1 for MASK_FIELD_ID (which is added as last index in embedding table, not in vocab)
    field_vocab_size = field_vocab_size_raw + 1
    
    vocab_sizes = {
        "token_vocab_size": token_vocab_size,
        "token_type_vocab_size": len(preprocessor.tag_encoder.tag_vocabs["token_type"]),
        "field_vocab_size": field_vocab_size,  # Includes +1 for MASK_FIELD_ID
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
