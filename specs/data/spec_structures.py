"""Specs for core data structures - happy path only."""

import torch

from saab_v3.data.structures import (
    Batch,
    EncodedTag,
    TokenizedSequence,
)


# ============================================================================
# StructureTag Specs
# ============================================================================


def spec_structure_tag_creation_minimal(sample_structure_tag):
    """Verify StructureTag can be created with single field."""
    # Arrange: sample_structure_tag fixture provides minimal StructureTag

    # Act: Already created by fixture

    # Assert
    assert sample_structure_tag.field == "name"
    assert sample_structure_tag.entity is None
    assert sample_structure_tag.is_empty() is False


def spec_structure_tag_creation_multiple_fields(sample_structure_tag_multiple):
    """Verify StructureTag can be created with multiple fields."""
    # Arrange: sample_structure_tag_multiple fixture provides StructureTag with multiple fields

    # Act: Already created by fixture

    # Assert
    assert sample_structure_tag_multiple.field == "name"
    assert sample_structure_tag_multiple.entity == "user_123"
    assert sample_structure_tag_multiple.token_type == "text"
    assert sample_structure_tag_multiple.is_empty() is False


def spec_structure_tag_creation_all_fields(sample_structure_tag_all_fields):
    """Verify StructureTag can be created with all fields."""
    # Arrange: sample_structure_tag_all_fields fixture provides complete StructureTag

    # Act: Already created by fixture

    # Assert
    assert sample_structure_tag_all_fields.field == "name"
    assert sample_structure_tag_all_fields.entity == "user_123"
    assert sample_structure_tag_all_fields.time == "2023-Q1"
    assert sample_structure_tag_all_fields.edge == "parent_of"
    assert sample_structure_tag_all_fields.role == "primary_key"
    assert sample_structure_tag_all_fields.token_type == "text"
    assert sample_structure_tag_all_fields.is_empty() is False


def spec_structure_tag_model_dump(sample_structure_tag):
    """Verify StructureTag model_dump() serialization works."""
    # Arrange: sample_structure_tag fixture

    # Act
    dumped = sample_structure_tag.model_dump()

    # Assert
    assert isinstance(dumped, dict)
    assert dumped["field"] == "name"
    assert dumped["entity"] is None


# ============================================================================
# Token Specs
# ============================================================================


def spec_token_creation(sample_token):
    """Verify Token can be created with valid value, structure_tag, and position."""
    # Arrange: sample_token fixture provides valid Token

    # Act: Already created by fixture

    # Assert
    assert sample_token.value == "John"
    assert sample_token.structure_tag.field == "name"
    assert sample_token.position == 0


def spec_token_model_dump(sample_token):
    """Verify Token model_dump() serialization works."""
    # Arrange: sample_token fixture

    # Act
    dumped = sample_token.model_dump()

    # Assert
    assert isinstance(dumped, dict)
    assert dumped["value"] == "John"
    assert dumped["position"] == 0
    assert "structure_tag" in dumped


# ============================================================================
# TokenizedSequence Specs
# ============================================================================


def spec_tokenized_sequence_creation(sample_tokens):
    """Verify TokenizedSequence can be created with valid tokens list."""
    # Arrange: sample_tokens fixture provides list of tokens

    # Act
    sequence = TokenizedSequence(tokens=sample_tokens)

    # Assert
    assert len(sequence.tokens) == 3
    assert sequence.sequence_id is None


def spec_tokenized_sequence_creation_with_id(sample_tokens):
    """Verify TokenizedSequence can be created with sequence_id."""
    # Arrange: sample_tokens fixture

    # Act
    sequence = TokenizedSequence(tokens=sample_tokens, sequence_id="seq_001")

    # Assert
    assert sequence.sequence_id == "seq_001"


def spec_tokenized_sequence_len(sample_tokens):
    """Verify __len__() returns correct count."""
    # Arrange
    sequence = TokenizedSequence(tokens=sample_tokens)

    # Act
    length = len(sequence)

    # Assert
    assert length == 3
    assert length == len(sequence.tokens)


def spec_tokenized_sequence_get_tokens_by_field_with_matches(sample_tokens):
    """Verify get_tokens_by_field() filters correctly when matches exist."""
    # Arrange
    sequence = TokenizedSequence(tokens=sample_tokens)

    # Act
    name_tokens = sequence.get_tokens_by_field("name")

    # Assert
    assert len(name_tokens) == 2
    assert all(token.structure_tag.field == "name" for token in name_tokens)


def spec_tokenized_sequence_get_tokens_by_field_no_matches(sample_tokens):
    """Verify get_tokens_by_field() returns empty list when no matches."""
    # Arrange
    sequence = TokenizedSequence(tokens=sample_tokens)

    # Act
    result = sequence.get_tokens_by_field("nonexistent")

    # Assert
    assert result == []
    assert isinstance(result, list)


def spec_tokenized_sequence_model_dump(sample_tokens):
    """Verify TokenizedSequence model_dump() serialization works."""
    # Arrange
    sequence = TokenizedSequence(tokens=sample_tokens)

    # Act
    dumped = sequence.model_dump()

    # Assert
    assert isinstance(dumped, dict)
    assert "tokens" in dumped
    assert len(dumped["tokens"]) == 3


# ============================================================================
# EncodedTag Specs
# ============================================================================


def spec_encoded_tag_creation_full(sample_structure_tag):
    """Verify EncodedTag can be created with all indices."""
    # Arrange
    # Act
    encoded = EncodedTag(
        field_idx=0,
        entity_idx=1,
        time_idx=2,
        edge_idx=3,
        role_idx=4,
        token_type_idx=5,
    )

    # Assert
    assert encoded.field_idx == 0
    assert encoded.entity_idx == 1
    assert encoded.time_idx == 2
    assert encoded.edge_idx == 3
    assert encoded.role_idx == 4
    assert encoded.token_type_idx == 5


def spec_encoded_tag_creation_partial(sample_structure_tag):
    """Verify EncodedTag can be created with partial indices (some None)."""
    # Arrange
    # Act
    encoded = EncodedTag(
        field_idx=0,
        entity_idx=None,
        time_idx=None,
        edge_idx=None,
        role_idx=None,
        token_type_idx=1,
    )

    # Assert
    assert encoded.field_idx == 0
    assert encoded.entity_idx is None
    assert encoded.token_type_idx == 1


def spec_encoded_tag_get_indices(sample_structure_tag):
    """Verify get_indices() returns correct dictionary."""
    # Arrange
    encoded = EncodedTag(
        field_idx=0,
        entity_idx=1,
        time_idx=None,
        edge_idx=2,
        role_idx=None,
        token_type_idx=3,
    )

    # Act
    indices = encoded.get_indices()

    # Assert
    assert isinstance(indices, dict)
    assert indices["field_idx"] == 0
    assert indices["entity_idx"] == 1
    assert indices["time_idx"] is None
    assert indices["edge_idx"] == 2
    assert indices["role_idx"] is None
    assert indices["token_type_idx"] == 3


def spec_encoded_tag_model_dump(sample_structure_tag):
    """Verify EncodedTag model_dump() serialization works."""
    # Arrange
    encoded = EncodedTag(field_idx=0)

    # Act
    dumped = encoded.model_dump()

    # Assert
    assert isinstance(dumped, dict)
    assert dumped["field_idx"] == 0


# ============================================================================
# Batch Specs
# ============================================================================


def spec_batch_creation_minimal(sample_batch_tensors):
    """Verify Batch can be created with required tensors only."""
    # Arrange: sample_batch_tensors fixture provides required tensors

    # Act
    batch = Batch(**sample_batch_tensors)

    # Assert
    assert batch.edge_ids is None
    assert batch.role_ids is None
    assert batch.sequence_ids is None
    assert batch.batch_size == 2
    assert batch.seq_len == 5


def spec_batch_creation_with_optional_tensors(sample_batch_tensors):
    """Verify Batch can be created with optional tensors (edge_ids, role_ids)."""
    # Arrange
    batch_size, seq_len = 2, 5
    sample_batch_tensors["edge_ids"] = torch.zeros(
        batch_size, seq_len, dtype=torch.long
    )
    sample_batch_tensors["role_ids"] = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Act
    batch = Batch(**sample_batch_tensors)

    # Assert
    assert batch.edge_ids is not None
    assert batch.role_ids is not None
    assert batch.edge_ids.shape == (batch_size, seq_len)
    assert batch.role_ids.shape == (batch_size, seq_len)


def spec_batch_creation_with_sequence_ids(sample_batch_tensors):
    """Verify Batch can be created with sequence_ids."""
    # Arrange
    sample_batch_tensors["sequence_ids"] = ["seq_001", "seq_002"]

    # Act
    batch = Batch(**sample_batch_tensors)

    # Assert
    assert batch.sequence_ids == ["seq_001", "seq_002"]
    assert len(batch.sequence_ids) == batch.batch_size


def spec_batch_properties(sample_batch_tensors):
    """Verify batch_size and seq_len properties return correct values."""
    # Arrange
    batch = Batch(**sample_batch_tensors)

    # Act
    batch_size = batch.batch_size
    seq_len = batch.seq_len

    # Assert
    assert batch_size == 2
    assert seq_len == 5
    assert batch_size == batch.token_ids.shape[0]
    assert seq_len == batch.token_ids.shape[1]


def spec_batch_to_device(sample_batch_tensors):
    """Verify to(device) moves all tensors to device correctly."""
    # Arrange
    batch = Batch(**sample_batch_tensors)
    device = torch.device("cpu")

    # Act
    moved_batch = batch.to(device)

    # Assert
    assert moved_batch.token_ids.device == device
    assert moved_batch.attention_mask.device == device
    assert moved_batch.field_ids.device == device
    assert moved_batch.entity_ids.device == device
    assert moved_batch.time_ids.device == device
    assert moved_batch.token_type_ids.device == device


def spec_batch_to_device_preserves_optional_tensors(sample_batch_tensors):
    """Verify to(device) preserves optional tensors correctly."""
    # Arrange
    batch_size, seq_len = 2, 5
    sample_batch_tensors["edge_ids"] = torch.zeros(
        batch_size, seq_len, dtype=torch.long
    )
    sample_batch_tensors["role_ids"] = torch.ones(batch_size, seq_len, dtype=torch.long)
    batch = Batch(**sample_batch_tensors)
    device = torch.device("cpu")

    # Act
    moved_batch = batch.to(device)

    # Assert
    assert moved_batch.edge_ids is not None
    assert moved_batch.role_ids is not None
    assert moved_batch.edge_ids.device == device
    assert moved_batch.role_ids.device == device
