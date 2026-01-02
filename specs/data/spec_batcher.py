"""Specs for Batcher - happy path only."""

import torch

from saab_v3.data.batcher import Batcher
from saab_v3.data.constants import PAD_IDX


# ============================================================================
# Batcher Basic Functionality
# ============================================================================


def spec_batcher_basic_batching(sample_encoded_sequences):
    """Verify Batcher creates batches with same-length sequences."""
    # Arrange
    batcher = Batcher(max_seq_len=512)
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences, device="cpu")

    # Assert
    assert batch.token_ids.shape[0] == len(sequences)  # batch_size
    assert batch.token_ids.shape[1] == len(sequences[0][1])  # seq_len (same for all)
    # Verify all sequences have same length (no padding needed)
    assert all(seq_len == len(sequences[0][1]) for seq_len in batch.sequence_lengths)
    # Verify attention masks are all 1s (no padding)
    assert torch.all(batch.attention_mask == 1)


def spec_batcher_dynamic_padding(sample_encoded_sequences_different_lengths):
    """Verify Batcher pads sequences to max length in batch."""
    # Arrange
    batcher = Batcher(max_seq_len=512)
    sequences = sample_encoded_sequences_different_lengths

    # Act
    batch = batcher.batch(sequences, device="cpu")

    # Assert
    # Find max length in original sequences
    max_len = max(len(token_ids) for _, token_ids, _ in sequences)
    # Verify batch is padded to max length
    assert batch.token_ids.shape[1] == max_len
    # Verify sequence lengths are preserved
    assert batch.sequence_lengths[0] == len(sequences[0][1])
    assert batch.sequence_lengths[1] == len(sequences[1][1])
    # Verify attention masks: first sequence should have padding
    assert batch.attention_mask[0].sum().item() == batch.sequence_lengths[0]
    assert batch.attention_mask[1].sum().item() == batch.sequence_lengths[1]
    # Verify padding tokens are PAD_IDX
    first_seq_padding_start = batch.sequence_lengths[0]
    assert torch.all(batch.token_ids[0, first_seq_padding_start:] == PAD_IDX)


def spec_batcher_truncation(sample_encoded_sequences):
    """Verify Batcher truncates sequences longer than max_seq_len."""
    # Arrange
    batcher = Batcher(max_seq_len=1)  # Very short max length
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences, device="cpu")

    # Assert
    # Verify all sequences are truncated to max_seq_len
    assert batch.token_ids.shape[1] == 1
    # Verify sequence lengths are capped at max_seq_len
    assert all(length <= 1 for length in batch.sequence_lengths)
    # Verify attention masks are correct
    assert torch.all(
        batch.attention_mask == 1
    )  # All valid (no padding after truncation)


def spec_batcher_attention_masks(sample_encoded_sequences_different_lengths):
    """Verify Batcher creates correct attention masks."""
    # Arrange
    batcher = Batcher(max_seq_len=512)
    sequences = sample_encoded_sequences_different_lengths

    # Act
    batch = batcher.batch(sequences, device="cpu")

    # Assert
    # Verify attention masks are binary (0 or 1)
    assert torch.all((batch.attention_mask == 0) | (batch.attention_mask == 1))
    # Verify mask sums match sequence lengths
    for i, seq_len in enumerate(batch.sequence_lengths):
        assert batch.attention_mask[i].sum().item() == seq_len
    # Verify padding positions are 0
    for i, seq_len in enumerate(batch.sequence_lengths):
        if seq_len < batch.token_ids.shape[1]:
            assert torch.all(batch.attention_mask[i, seq_len:] == 0)
            assert torch.all(batch.attention_mask[i, :seq_len] == 1)


def spec_batcher_optional_tags(
    sample_encoded_sequences, sample_encoded_sequences_with_edges_roles
):
    """Verify Batcher handles optional tags (edge_ids, role_ids) correctly."""
    # Arrange
    batcher = Batcher(max_seq_len=512)

    # Test sequences with edges/roles
    sequences_with_edges = sample_encoded_sequences_with_edges_roles

    # Act
    batch_with_edges = batcher.batch(sequences_with_edges, device="cpu")

    # Assert
    # Sequences with edges/roles should have tensors
    assert batch_with_edges.edge_ids is not None
    assert batch_with_edges.role_ids is not None
    # Verify tensor shapes
    assert batch_with_edges.edge_ids.shape == batch_with_edges.token_ids.shape
    assert batch_with_edges.role_ids.shape == batch_with_edges.token_ids.shape
    # Verify tensors contain valid indices (not all PAD)
    # Since edges/roles are present, they should have non-PAD values
    assert torch.any(batch_with_edges.edge_ids != PAD_IDX)
    assert torch.any(batch_with_edges.role_ids != PAD_IDX)


def spec_batcher_tensor_shapes(sample_encoded_sequences):
    """Verify all tensors have consistent [batch_size, seq_len] shapes."""
    # Arrange
    batcher = Batcher(max_seq_len=512)
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences, device="cpu")

    # Assert
    batch_size = batch.token_ids.shape[0]
    seq_len = batch.token_ids.shape[1]

    # Verify all required tensors have correct shape
    assert batch.token_ids.shape == (batch_size, seq_len)
    assert batch.attention_mask.shape == (batch_size, seq_len)
    assert batch.field_ids.shape == (batch_size, seq_len)
    assert batch.entity_ids.shape == (batch_size, seq_len)
    assert batch.time_ids.shape == (batch_size, seq_len)
    assert batch.token_type_ids.shape == (batch_size, seq_len)

    # Verify sequence_lengths matches batch_size
    assert len(batch.sequence_lengths) == batch_size


def spec_batcher_sequence_metadata(sample_encoded_sequences):
    """Verify Batcher preserves sequence metadata (lengths, IDs)."""
    # Arrange
    batcher = Batcher(max_seq_len=512)
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences, device="cpu")

    # Assert
    # Verify sequence_lengths are correct
    assert len(batch.sequence_lengths) == len(sequences)
    for i, (_, token_ids, _) in enumerate(sequences):
        assert batch.sequence_lengths[i] == len(token_ids)

    # Verify sequence_ids are preserved
    assert batch.sequence_ids is not None
    assert len(batch.sequence_ids) == len(sequences)
    for i, (seq, _, _) in enumerate(sequences):
        assert batch.sequence_ids[i] == seq.sequence_id
