"""Fixtures for model component specs."""

import torch
import pytest

from saab_v3.models.components.attention import MultiHeadAttention
from saab_v3.models.components.ffn import FeedForward
from saab_v3.models.components.normalization import LayerNorm
from saab_v3.models.components.encoder_layer import TransformerEncoderLayer
from saab_v3.models.components.dropout import Dropout


@pytest.fixture
def d_model():
    """Default model dimension."""
    return 128


@pytest.fixture
def num_heads():
    """Default number of attention heads."""
    return 4


@pytest.fixture
def ffn_dim(d_model):
    """Default feed-forward network dimension."""
    return 4 * d_model


@pytest.fixture
def batch_size():
    """Default batch size."""
    return 2


@pytest.fixture
def seq_len():
    """Default sequence length."""
    return 10


@pytest.fixture
def sample_tensor(batch_size, seq_len, d_model):
    """Sample input tensor for testing."""
    return torch.randn(batch_size, seq_len, d_model)


@pytest.fixture
def sample_attention_mask(batch_size, seq_len):
    """Sample attention mask (1 = valid, 0 = pad)."""
    # Create mask where first 7 positions are valid, last 3 are padding
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    mask[:, 7:] = 0
    return mask


@pytest.fixture
def multi_head_attention(d_model, num_heads):
    """MultiHeadAttention instance."""
    return MultiHeadAttention(d_model=d_model, num_heads=num_heads)


@pytest.fixture
def feed_forward(d_model, ffn_dim):
    """FeedForward instance."""
    return FeedForward(d_model=d_model, ffn_dim=ffn_dim)


@pytest.fixture
def layer_norm(d_model):
    """LayerNorm instance."""
    return LayerNorm(d_model=d_model)


@pytest.fixture
def encoder_layer(d_model, num_heads, ffn_dim):
    """TransformerEncoderLayer instance."""
    return TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
    )


@pytest.fixture
def dropout():
    """Dropout instance."""
    return Dropout(p=0.1)
