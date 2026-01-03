"""Model implementations for SAAB Transformer project."""

from saab_v3.models.components import (
    Dropout,
    FeedForward,
    LayerNorm,
    MultiHeadAttention,
    SAABAttention,
    SAABEncoderLayer,
    TransformerEncoderLayer,
)
from saab_v3.models.flat_transformer import FlatTransformer
from saab_v3.models.scratch_transformer import ScratchTransformer
from saab_v3.models.saab_transformer import SAABTransformer
from saab_v3.models.config import ModelConfig
from saab_v3.models.factory import (
    create_flat_transformer,
    create_scratch_transformer,
    create_saab_transformer,
    get_vocab_sizes,
)

__all__ = [
    # Components
    "MultiHeadAttention",
    "SAABAttention",
    "FeedForward",
    "LayerNorm",
    "TransformerEncoderLayer",
    "SAABEncoderLayer",
    "Dropout",
    # Models
    "FlatTransformer",
    "ScratchTransformer",
    "SAABTransformer",
    # Config
    "ModelConfig",
    # Factory functions
    "create_flat_transformer",
    "create_scratch_transformer",
    "create_saab_transformer",
    "get_vocab_sizes",
]
