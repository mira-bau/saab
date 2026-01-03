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
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences)

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
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    sequences = sample_encoded_sequences_different_lengths

    # Act
    batch = batcher.batch(sequences)

    # Assert
    # Find max length in original sequences
    max_len = max(len(token_ids) for _, token_ids, _, _ in sequences)
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
    batcher = Batcher(
        max_seq_len=1, device=torch.device("cpu")
    )  # Very short max length
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences)

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
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    sequences = sample_encoded_sequences_different_lengths

    # Act
    batch = batcher.batch(sequences)

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
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))

    # Test sequences with edges/roles
    sequences_with_edges = sample_encoded_sequences_with_edges_roles

    # Act
    batch_with_edges = batcher.batch(sequences_with_edges)

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
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences)

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
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences)

    # Assert
    # Verify sequence_lengths are correct
    assert len(batch.sequence_lengths) == len(sequences)
    for i, (_, token_ids, _, _) in enumerate(sequences):
        assert batch.sequence_lengths[i] == len(token_ids)

    # Verify sequence_ids are preserved
    assert batch.sequence_ids is not None
    assert len(batch.sequence_ids) == len(sequences)
    for i, (seq, _, _, _) in enumerate(sequences):
        assert batch.sequence_ids[i] == seq.sequence_id


# ============================================================================
# Batcher Device Tests
# ============================================================================


def spec_batcher_device_from_init(sample_encoded_sequences):
    """Verify Batcher stores device from __init__() and uses it correctly."""
    # Arrange
    cpu_device = torch.device("cpu")
    batcher = Batcher(max_seq_len=512, device=cpu_device)
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences)

    # Assert
    # Verify Batcher stores device
    assert batcher.device == cpu_device
    # Verify tensors are placed on correct device
    assert batch.token_ids.device == cpu_device
    assert batch.attention_mask.device == cpu_device
    assert batch.field_ids.device == cpu_device
    assert batch.entity_ids.device == cpu_device
    assert batch.time_ids.device == cpu_device
    assert batch.token_type_ids.device == cpu_device

    # Test with MPS if available
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        batcher_mps = Batcher(max_seq_len=512, device=mps_device)
        batch_mps = batcher_mps.batch(sequences)
        assert batcher_mps.device.type == mps_device.type
        assert batch_mps.token_ids.device.type == mps_device.type


def spec_batcher_device_default(sample_encoded_sequences):
    """Verify Batcher defaults to CPU if device not provided."""
    # Arrange
    batcher = Batcher(max_seq_len=512)  # No device parameter
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences)

    # Assert
    # Should default to CPU
    assert batcher.device == torch.device("cpu")
    assert batch.token_ids.device.type == "cpu"

    # Test with device=None explicitly
    batcher_none = Batcher(max_seq_len=512, device=None)
    batch_none = batcher_none.batch(sequences)
    assert batcher_none.device == torch.device("cpu")
    assert batch_none.token_ids.device.type == "cpu"


def spec_batcher_device_consistency(sample_encoded_sequences):
    """Verify all tensors in batch are on same device."""
    # Arrange
    cpu_device = torch.device("cpu")
    batcher = Batcher(max_seq_len=512, device=cpu_device)
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences)

    # Assert
    # All tensors should be on the same device
    expected_device = batcher.device
    assert batch.token_ids.device == expected_device
    assert batch.attention_mask.device == expected_device
    assert batch.field_ids.device == expected_device
    assert batch.entity_ids.device == expected_device
    assert batch.time_ids.device == expected_device
    assert batch.token_type_ids.device == expected_device

    # Verify device matches Batcher's stored device
    assert batch.token_ids.device == batcher.device


# ============================================================================
# Batcher Label Tests
# ============================================================================


def spec_batcher_labels_classification(sample_encoded_sequences):
    """Verify Batcher batches classification labels correctly."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    # Add labels to sequences (classification: class indices)
    # Fixtures now return (seq, token_ids, encoded_tags, label) tuples
    sequences_with_labels = [
        (seq, token_ids, encoded_tags, label)
        for (seq, token_ids, encoded_tags, _), label in zip(
            sample_encoded_sequences, [0, 1]
        )
    ]

    # Act
    batch = batcher.batch(sequences_with_labels)

    # Assert
    assert batch.labels is not None
    assert batch.labels.shape == (2,)  # [batch]
    assert batch.labels.dtype == torch.long
    assert torch.equal(batch.labels, torch.tensor([0, 1], dtype=torch.long))


def spec_batcher_labels_multi_label(sample_encoded_sequences):
    """Verify Batcher batches multi-label classification labels correctly."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    # Add multi-label labels (binary vectors)
    sequences_with_labels = [
        (seq, token_ids, encoded_tags, label)
        for (seq, token_ids, encoded_tags, _), label in zip(
            sample_encoded_sequences, [[1, 0, 1], [0, 1, 0]]
        )
    ]

    # Act
    batch = batcher.batch(sequences_with_labels)

    # Assert
    assert batch.labels is not None
    assert batch.labels.shape == (2, 3)  # [batch, num_classes]
    assert batch.labels.dtype == torch.float
    assert torch.equal(
        batch.labels, torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float)
    )


def spec_batcher_labels_regression(sample_encoded_sequences):
    """Verify Batcher batches regression labels correctly."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    # Add regression labels (continuous values)
    sequences_with_labels = [
        (seq, token_ids, encoded_tags, label)
        for (seq, token_ids, encoded_tags, _), label in zip(
            sample_encoded_sequences, [3.5, 2.1]
        )
    ]

    # Act
    batch = batcher.batch(sequences_with_labels)

    # Assert
    assert batch.labels is not None
    # Regression labels should be [batch, 1] for single-target
    assert batch.labels.shape == (2, 1)
    assert batch.labels.dtype == torch.float
    assert torch.allclose(batch.labels, torch.tensor([[3.5], [2.1]], dtype=torch.float))


def spec_batcher_labels_token_classification(sample_encoded_sequences_different_lengths):
    """Verify Batcher batches token classification labels correctly."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    # Add token classification labels (per-token labels)
    sequences_with_labels = [
        (seq, token_ids, encoded_tags, label)
        for (seq, token_ids, encoded_tags, _), label in zip(
            sample_encoded_sequences_different_lengths,
            [[0, 1], [1, 1, 0, 0]],  # Different lengths
        )
    ]

    # Act
    batch = batcher.batch(sequences_with_labels)

    # Assert
    assert batch.labels is not None
    # Token classification labels should be padded to max_seq_len
    max_len = batch.seq_len
    assert batch.labels.shape == (2, max_len)
    assert batch.labels.dtype == torch.long
    # First sequence: [0, 1, -100, -100] (padded with -100)
    # Second sequence: [1, 1, 0, 0]
    assert batch.labels[0, 0] == 0
    assert batch.labels[0, 1] == 1
    assert batch.labels[0, 2] == -100  # Padding
    assert batch.labels[1, 0] == 1
    assert batch.labels[1, 1] == 1


def spec_batcher_labels_none(sample_encoded_sequences):
    """Verify Batcher handles None labels correctly."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    # Add None labels (fixtures already have None, but we can explicitly set them)
    sequences_with_none_labels = [
        (seq, token_ids, encoded_tags, None)
        for seq, token_ids, encoded_tags, _ in sample_encoded_sequences
    ]

    # Act
    batch = batcher.batch(sequences_with_none_labels)

    # Assert
    assert batch.labels is None


def spec_batcher_labels_mixed_none(sample_encoded_sequences):
    """Verify Batcher handles mixed None and valid labels correctly."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    # Mix of None and valid labels
    sequences_with_mixed_labels = [
        (sample_encoded_sequences[0][0], sample_encoded_sequences[0][1], sample_encoded_sequences[0][2], 0),
        (sample_encoded_sequences[1][0], sample_encoded_sequences[1][1], sample_encoded_sequences[1][2], None),
    ]

    # Act
    batch = batcher.batch(sequences_with_mixed_labels)

    # Assert
    # If any label is None, labels should be None (or handled gracefully)
    # For now, we expect labels to be None if all are None, otherwise batched
    # This depends on implementation - let's check if it handles it
    # The current implementation should batch valid labels and use 0 for None
    assert batch.labels is not None
    assert batch.labels.shape == (2,)


def spec_batcher_labels_device(sample_encoded_sequences):
    """Verify labels are placed on correct device."""
    # Arrange
    cpu_device = torch.device("cpu")
    batcher = Batcher(max_seq_len=512, device=cpu_device)
    sequences_with_labels = [
        (seq, token_ids, encoded_tags, label)
        for (seq, token_ids, encoded_tags, _), label in zip(
            sample_encoded_sequences, [0, 1]
        )
    ]

    # Act
    batch = batcher.batch(sequences_with_labels)

    # Assert
    assert batch.labels is not None
    assert batch.labels.device == cpu_device
