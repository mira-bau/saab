"""Fixtures for task head specs."""

import pytest
import torch

from saab_v3.tasks.pooling import CLSPooling


@pytest.fixture
def sample_d_model():
    """Default model dimension for testing."""
    return 128


@pytest.fixture
def sample_batch_size():
    """Default batch size for testing."""
    return 4


@pytest.fixture
def sample_seq_len():
    """Default sequence length for testing."""
    return 32


@pytest.fixture
def sample_encoder_output(sample_batch_size, sample_seq_len, sample_d_model):
    """Sample encoder output tensor.

    Shape: [batch_size, seq_len, d_model]
    """
    return torch.randn(sample_batch_size, sample_seq_len, sample_d_model)


@pytest.fixture
def sample_attention_mask(sample_batch_size, sample_seq_len):
    """Sample attention mask tensor.

    Shape: [batch_size, seq_len]
    Values: 1 for valid tokens, 0 for padding
    """
    # Create mask with some padding at the end
    mask = torch.ones(sample_batch_size, sample_seq_len, dtype=torch.long)
    # Add some padding: last 25% of sequence is padding
    padding_start = int(sample_seq_len * 0.75)
    mask[:, padding_start:] = 0
    return mask


@pytest.fixture
def sample_attention_mask_full(sample_batch_size, sample_seq_len):
    """Sample attention mask with no padding (all valid tokens)."""
    return torch.ones(sample_batch_size, sample_seq_len, dtype=torch.long)


@pytest.fixture
def sample_attention_mask_all_padding(sample_batch_size, sample_seq_len):
    """Sample attention mask with all padding (edge case)."""
    return torch.zeros(sample_batch_size, sample_seq_len, dtype=torch.long)


@pytest.fixture
def default_pooling():
    """Default pooling strategy (CLS pooling)."""
    return CLSPooling()
