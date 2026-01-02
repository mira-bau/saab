"""Specs for TagEncoder - happy path only."""

from saab_v3.data.tag_encoder import TagEncoder
from saab_v3.data.structures import StructureTag
from saab_v3.data.constants import NONE_TOKEN


# ============================================================================
# TagEncoder Basic Functionality
# ============================================================================


def spec_tag_encoder_build_vocabs(sample_tokenized_sequences):
    """Verify TagEncoder builds all tag vocabularies."""
    # Arrange
    encoder = TagEncoder()

    # Act
    encoder.build_vocabs(sample_tokenized_sequences)

    # Assert
    assert encoder._is_built is True
    # Verify all tag types have vocabularies built
    for tag_type in ["field", "entity", "time", "edge", "role", "token_type"]:
        assert tag_type in encoder.tag_vocabs
        assert len(encoder.tag_vocabs[tag_type]) > 0  # At least special tokens
    # Verify field vocabulary contains expected fields
    assert "name" in encoder.tag_vocabs["field"]
    assert "age" in encoder.tag_vocabs["field"]
    # Verify entity vocabulary contains expected entities
    assert "user_1" in encoder.tag_vocabs["entity"]
    assert "user_2" in encoder.tag_vocabs["entity"]


def spec_tag_encoder_encode(sample_tokenized_sequences):
    """Verify TagEncoder encodes structure tags correctly."""
    # Arrange
    encoder = TagEncoder()
    encoder.build_vocabs(sample_tokenized_sequences)
    tag = StructureTag(field="name", entity="user_1", token_type="text")

    # Act
    encoded = encoder.encode(tag)

    # Assert
    assert encoded.field_idx is not None
    assert encoded.entity_idx is not None
    assert encoded.token_type_idx is not None
    # Verify original tag preserved
    assert encoded.original_tag.field == "name"
    assert encoded.original_tag.entity == "user_1"
    assert encoded.original_tag.token_type == "text"


def spec_tag_encoder_encode_sequence(sample_tokenized_sequences):
    """Verify TagEncoder encodes sequence tags correctly."""
    # Arrange
    encoder = TagEncoder()
    encoder.build_vocabs(sample_tokenized_sequences)
    sequence = sample_tokenized_sequences[0]

    # Act
    encoded_seq, encoded_tags = encoder.encode_sequence(sequence)

    # Assert
    assert len(encoded_tags) == len(sequence.tokens)
    assert all(isinstance(tag, type(encoded_tags[0])) for tag in encoded_tags)
    # Verify original sequence structure preserved
    assert encoded_seq.sequence_id == sequence.sequence_id
    assert len(encoded_seq.tokens) == len(sequence.tokens)
    # Verify all tags encoded
    for i, encoded_tag in enumerate(encoded_tags):
        assert encoded_tag.field_idx is not None
        assert encoded_tag.original_tag.field == sequence.tokens[i].structure_tag.field


def spec_tag_encoder_missing_tags(sample_tokenized_sequences):
    """Verify TagEncoder handles missing tags with NONE token."""
    # Arrange
    encoder = TagEncoder()
    encoder.build_vocabs(sample_tokenized_sequences)
    tag = StructureTag(field="name")  # Missing entity, time, edge, role, token_type

    # Act
    encoded = encoder.encode(tag)

    # Assert
    assert encoded.field_idx is not None
    # Missing fields should use NONE token
    none_idx = encoder.tag_vocabs["entity"].encode(NONE_TOKEN)
    assert encoded.entity_idx == none_idx
    # Verify original tag preserved
    assert encoded.original_tag.field == "name"
    assert encoded.original_tag.entity is None


def spec_tag_encoder_oov_tags(sample_tokenized_sequences):
    """Verify TagEncoder handles OOV tags with UNK."""
    # Arrange
    encoder = TagEncoder()
    encoder.build_vocabs(sample_tokenized_sequences)
    # Tag with unknown field/entity
    tag = StructureTag(field="unknown_field", entity="unknown_entity")

    # Act
    encoded = encoder.encode(tag)

    # Assert
    # OOV tags should get UNK index
    unk_idx = encoder.tag_vocabs["field"].encode("[UNK]")
    assert encoded.field_idx == unk_idx
    # Verify original tag preserved
    assert encoded.original_tag.field == "unknown_field"
    assert encoded.original_tag.entity == "unknown_entity"
