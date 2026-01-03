"""Factory functions for creating models from preprocessor."""

from saab_v3.training.preprocessor import Preprocessor
from saab_v3.models.config import ModelConfig
from saab_v3.models.flat_transformer import FlatTransformer
from saab_v3.models.scratch_transformer import ScratchTransformer
from saab_v3.models.saab_transformer import SAABTransformer


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
        "token_vocab_size": preprocessor.tokenizer.vocab.size(),
        "token_type_vocab_size": preprocessor.tag_encoder.tag_vocabs[
            "token_type"
        ].size(),
        "field_vocab_size": preprocessor.tag_encoder.tag_vocabs["field"].size(),
        "entity_vocab_size": preprocessor.tag_encoder.tag_vocabs["entity"].size(),
        "time_vocab_size": preprocessor.tag_encoder.tag_vocabs["time"].size(),
    }

    return vocab_sizes


def create_flat_transformer(
    preprocessor: Preprocessor,
    config: ModelConfig,
) -> FlatTransformer:
    """Create FlatTransformer from preprocessor and config.

    Args:
        preprocessor: Fitted Preprocessor instance
        config: ModelConfig instance

    Returns:
        FlatTransformer instance
    """
    vocab_sizes = get_vocab_sizes(preprocessor)

    return FlatTransformer(
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


def create_scratch_transformer(
    preprocessor: Preprocessor,
    config: ModelConfig,
) -> ScratchTransformer:
    """Create ScratchTransformer from preprocessor and config.

    Args:
        preprocessor: Fitted Preprocessor instance
        config: ModelConfig instance

    Returns:
        ScratchTransformer instance
    """
    vocab_sizes = get_vocab_sizes(preprocessor)

    return ScratchTransformer(
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


def create_saab_transformer(
    preprocessor: Preprocessor,
    config: ModelConfig,
    lambda_bias: float | None = None,
) -> SAABTransformer:
    """Create SAABTransformer from preprocessor and config.

    Args:
        preprocessor: Fitted Preprocessor instance
        config: ModelConfig instance
        lambda_bias: Optional override for lambda_bias (uses config.lambda_bias if None)

    Returns:
        SAABTransformer instance
    """
    vocab_sizes = get_vocab_sizes(preprocessor)

    return SAABTransformer(
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
