"""Specs for SAAB utilities - happy path only."""

from saab_v3.data.saab_utils import (
    extract_original_tags,
    compute_structural_relationship,
    same_field,
    same_entity,
    has_edge,
    is_pad_tag,
)
from saab_v3.data.structures import StructureTag
from saab_v3.data.constants import PAD_TAG_FIELD


# ============================================================================
# SAAB Utilities Basic Functionality
# ============================================================================


def spec_extract_original_tags(sample_encoded_sequences):
    """Verify extraction from encoded sequences."""
    # Arrange
    sequences = sample_encoded_sequences

    # Act
    original_tags = extract_original_tags(sequences)

    # Assert
    assert len(original_tags) == len(sequences)
    for i, (_, _, encoded_tags) in enumerate(sequences):
        assert len(original_tags[i]) == len(encoded_tags)
        for j, encoded_tag in enumerate(encoded_tags):
            assert original_tags[i][j] == encoded_tag.original_tag


def spec_compute_structural_relationship_same_field():
    """Verify relationship detection for same field."""
    # Arrange
    tag1 = StructureTag(field="name", entity="user_1")
    tag2 = StructureTag(field="name", entity="user_2")

    # Act
    relationship = compute_structural_relationship(tag1, tag2)

    # Assert
    assert relationship["same_field"] is True
    assert relationship["same_entity"] is False
    assert relationship["has_edge"] is False


def spec_compute_structural_relationship_same_entity():
    """Verify relationship detection for same entity."""
    # Arrange
    tag1 = StructureTag(field="name", entity="user_1")
    tag2 = StructureTag(field="age", entity="user_1")

    # Act
    relationship = compute_structural_relationship(tag1, tag2)

    # Assert
    assert relationship["same_field"] is False
    assert relationship["same_entity"] is True
    assert relationship["has_edge"] is False


def spec_compute_structural_relationship_has_edge():
    """Verify edge relationship detection."""
    # Arrange
    tag1 = StructureTag(field="id", entity="graph_1", edge="connects")
    tag2 = StructureTag(field="id", entity="graph_1")

    # Act
    relationship = compute_structural_relationship(tag1, tag2)

    # Assert
    assert relationship["has_edge"] is True
    assert relationship["edge_type"] == "connects"
    assert relationship["same_field"] is True
    assert relationship["same_entity"] is True


def spec_compute_structural_relationship_mixed():
    """Verify multiple relationship types detected correctly."""
    # Arrange
    tag1 = StructureTag(
        field="name", entity="user_1", time="2023-Q1", role="primary_key"
    )
    tag2 = StructureTag(
        field="name", entity="user_1", time="2023-Q1", role="primary_key"
    )

    # Act
    relationship = compute_structural_relationship(tag1, tag2)

    # Assert
    assert relationship["same_field"] is True
    assert relationship["same_entity"] is True
    assert relationship["same_time"] is True
    assert relationship["same_role"] is True
    assert relationship["has_edge"] is False


def spec_compute_structural_relationship_pad_tags():
    """Verify padding tags are handled correctly."""
    # Arrange
    pad_tag = StructureTag(field=PAD_TAG_FIELD)
    normal_tag = StructureTag(field="name", entity="user_1")

    # Act
    relationship = compute_structural_relationship(pad_tag, normal_tag)

    # Assert
    assert relationship["same_field"] is False
    assert relationship["same_entity"] is False
    assert relationship["has_edge"] is False


def spec_same_field_helper():
    """Verify same_field helper function."""
    # Arrange
    tag1 = StructureTag(field="name")
    tag2 = StructureTag(field="name")
    tag3 = StructureTag(field="age")

    # Act & Assert
    assert same_field(tag1, tag2) is True
    assert same_field(tag1, tag3) is False
    # Test with None fields
    tag4 = StructureTag(field="name")
    tag5 = StructureTag(entity="user_1")  # No field
    assert same_field(tag4, tag5) is False


def spec_same_entity_helper():
    """Verify same_entity helper function."""
    # Arrange
    tag1 = StructureTag(field="name", entity="user_1")
    tag2 = StructureTag(field="age", entity="user_1")
    tag3 = StructureTag(field="name", entity="user_2")

    # Act & Assert
    assert same_entity(tag1, tag2) is True
    assert same_entity(tag1, tag3) is False


def spec_has_edge_helper():
    """Verify has_edge helper function."""
    # Arrange
    tag1 = StructureTag(field="id", edge="connects")
    tag2 = StructureTag(field="id")
    tag3 = StructureTag(field="id", edge="parent_of")

    # Act & Assert
    assert has_edge(tag1, tag2) is True
    assert has_edge(tag2, tag3) is True
    assert has_edge(tag2, StructureTag(field="id")) is False


def spec_is_pad_tag_helper():
    """Verify is_pad_tag helper function."""
    # Arrange
    pad_tag = StructureTag(field=PAD_TAG_FIELD)
    normal_tag = StructureTag(field="name")

    # Act & Assert
    assert is_pad_tag(pad_tag) is True
    assert is_pad_tag(normal_tag) is False
