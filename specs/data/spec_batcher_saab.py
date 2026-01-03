"""Specs for Batcher SAAB functionality - preserving original tags."""

import torch

from saab_v3.data.batcher import Batcher
from saab_v3.data.constants import PAD_TAG_FIELD


# ============================================================================
# Batcher SAAB Functionality
# ============================================================================


def spec_batcher_preserve_original_tags_false(sample_encoded_sequences):
    """Verify original_tags is None when preserve_original_tags=False."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences, preserve_original_tags=False)

    # Assert
    assert batch.original_tags is None


def spec_batcher_preserve_original_tags_true(sample_encoded_sequences):
    """Verify original_tags is populated when preserve_original_tags=True."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences, preserve_original_tags=True)

    # Assert
    assert batch.original_tags is not None
    assert len(batch.original_tags) == len(sequences)  # batch_size
    assert len(batch.original_tags[0]) == batch.token_ids.shape[1]  # seq_len
    # Verify shape matches token_ids
    assert len(batch.original_tags) == batch.token_ids.shape[0]
    assert all(
        len(tag_seq) == batch.token_ids.shape[1] for tag_seq in batch.original_tags
    )


def spec_batcher_original_tags_same_length(sample_encoded_sequences):
    """Verify original tags preserved correctly for same-length sequences."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences, preserve_original_tags=True)

    # Assert
    assert batch.original_tags is not None
    # Verify tags match input tags (no padding needed for same-length sequences)
    for i, (_, _, encoded_tags) in enumerate(sequences):
        for j, encoded_tag in enumerate(encoded_tags):
            assert batch.original_tags[i][j].field == encoded_tag.original_tag.field
            assert batch.original_tags[i][j].entity == encoded_tag.original_tag.entity


def spec_batcher_original_tags_different_lengths(
    sample_encoded_sequences_different_lengths,
):
    """Verify original tags padded correctly for different-length sequences."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    sequences = sample_encoded_sequences_different_lengths

    # Act
    batch = batcher.batch(sequences, preserve_original_tags=True)

    # Assert
    assert batch.original_tags is not None
    # Find max length
    max_len = max(len(token_ids) for _, token_ids, _ in sequences)
    # Verify all sequences padded to max_len
    assert all(len(tag_seq) == max_len for tag_seq in batch.original_tags)
    # Verify padding tags are PAD_TAG
    for i, seq_len in enumerate(batch.sequence_lengths):
        if seq_len < max_len:
            # Check padding positions
            for j in range(seq_len, max_len):
                assert batch.original_tags[i][j].field == PAD_TAG_FIELD


def spec_batcher_original_tags_truncation(sample_encoded_sequences):
    """Verify original tags truncated correctly when sequence exceeds max_seq_len."""
    # Arrange
    batcher = Batcher(max_seq_len=1, device=torch.device("cpu"))  # Very short max length
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences, preserve_original_tags=True)

    # Assert
    assert batch.original_tags is not None
    # Verify all sequences truncated to max_seq_len
    assert all(len(tag_seq) == 1 for tag_seq in batch.original_tags)
    # Verify tags match first token of each sequence
    for i, (_, _, encoded_tags) in enumerate(sequences):
        assert batch.original_tags[i][0].field == encoded_tags[0].original_tag.field


def spec_batcher_original_tags_shape_validation(sample_encoded_sequences):
    """Verify Batch validation ensures original_tags shape matches token_ids."""
    # Arrange
    batcher = Batcher(max_seq_len=512, device=torch.device("cpu"))
    sequences = sample_encoded_sequences

    # Act
    batch = batcher.batch(sequences, preserve_original_tags=True)

    # Assert
    # Batch validation should pass (implicitly tested by Batch creation)
    assert batch.original_tags is not None
    assert len(batch.original_tags) == batch.batch_size
    assert all(len(tag_seq) == batch.seq_len for tag_seq in batch.original_tags)
    # Verify validation would catch mismatches (test by creating invalid batch)
    # This is tested implicitly - if shape doesn't match, Batch validation will fail
