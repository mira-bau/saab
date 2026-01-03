"""Fixtures for model component specs."""

import torch
import pytest

from saab_v3.data.constants import PAD_IDX
from saab_v3.data.structures import Batch
from saab_v3.models.components.attention import MultiHeadAttention
from saab_v3.models.components.ffn import FeedForward
from saab_v3.models.components.normalization import LayerNorm
from saab_v3.models.components.encoder_layer import TransformerEncoderLayer
from saab_v3.models.components.dropout import Dropout
from saab_v3.models.components.saab_attention import SAABAttention


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


@pytest.fixture
def sample_tag_indices(batch_size, seq_len):
    """Sample tag index tensors for SAAB attention testing."""
    # Create tag indices with some structure:
    # Batch 0: field_ids = [1, 1, 2, 2, 0, 0, 0, 0, 0, 0] (first 4 valid, rest padding)
    # Batch 1: field_ids = [1, 2, 1, 2, 0, 0, 0, 0, 0, 0] (first 4 valid, rest padding)
    # Use non-zero values for valid tokens, 0 (PAD_IDX) for padding
    field_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    field_ids[0, :4] = torch.tensor([1, 1, 2, 2])
    field_ids[1, :4] = torch.tensor([1, 2, 1, 2])
    # Rest are already 0 (PAD_IDX)

    # Entity IDs: same pattern but different values
    entity_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    entity_ids[0, :4] = torch.tensor([10, 10, 20, 20])
    entity_ids[1, :4] = torch.tensor([10, 20, 10, 20])

    # Time IDs: all same for first batch, different for second
    time_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    time_ids[0, :4] = torch.tensor([100, 100, 100, 100])
    time_ids[1, :4] = torch.tensor([100, 101, 100, 101])

    # Token type IDs: alternating pattern
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    token_type_ids[0, :4] = torch.tensor([0, 0, 1, 1])
    token_type_ids[1, :4] = torch.tensor([0, 1, 0, 1])

    return {
        "field_ids": field_ids,
        "entity_ids": entity_ids,
        "time_ids": time_ids,
        "token_type_ids": token_type_ids,
    }


@pytest.fixture
def sample_edge_role_indices(batch_size, seq_len):
    """Sample optional edge and role index tensors."""
    edge_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    edge_ids[0, :2] = torch.tensor([5, 5])  # First two tokens have edge
    edge_ids[1, 1] = 6  # Second token has edge

    role_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    role_ids[0, :2] = torch.tensor([1, 1])  # First two tokens have same role
    role_ids[1, :2] = torch.tensor([1, 2])  # Different roles

    return {"edge_ids": edge_ids, "role_ids": role_ids}


@pytest.fixture
def saab_attention(d_model, num_heads):
    """SAABAttention instance with default lambda=1.0."""
    return SAABAttention(d_model=d_model, num_heads=num_heads, lambda_bias=1.0)


@pytest.fixture
def sample_batch(batch_size, seq_len):
    """Sample Batch object for transformer testing."""
    # Create minimal valid batch
    token_ids = torch.randint(1, 100, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[:, 7:] = 0  # Last 3 positions are padding

    field_ids = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.long)
    field_ids[:, 7:] = PAD_IDX

    entity_ids = torch.randint(1, 20, (batch_size, seq_len), dtype=torch.long)
    entity_ids[:, 7:] = PAD_IDX

    time_ids = torch.randint(1, 5, (batch_size, seq_len), dtype=torch.long)
    time_ids[:, 7:] = PAD_IDX

    token_type_ids = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long)
    token_type_ids[:, 7:] = PAD_IDX

    return Batch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        field_ids=field_ids,
        entity_ids=entity_ids,
        time_ids=time_ids,
        token_type_ids=token_type_ids,
        sequence_lengths=[7, 7],
    )


@pytest.fixture
def sample_vocab_sizes():
    """Sample vocabulary sizes for transformer models."""
    return {
        "token_vocab_size": 1000,
        "token_type_vocab_size": 5,
        "field_vocab_size": 20,
        "entity_vocab_size": 50,
        "time_vocab_size": 10,
    }


@pytest.fixture
def sample_model_config(d_model):
    """Sample ModelConfig for testing."""
    from saab_v3.models.config import ModelConfig

    return ModelConfig(
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        ffn_dim=4 * d_model,
        max_seq_len=512,
        device="cpu",
    )
