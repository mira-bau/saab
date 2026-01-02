"""Specs for ValueTokenizer - happy path only."""

from saab_v3.data.tokenizer import ValueTokenizer
from saab_v3.data.structures import TokenizedSequence, Token, StructureTag


# ============================================================================
# ValueTokenizer Basic Functionality
# ============================================================================


def spec_value_tokenizer_build_vocab(sample_tokenized_sequences):
    """Verify ValueTokenizer builds vocabulary from sequences."""
    # Arrange
    tokenizer = ValueTokenizer(vocab_size=100)

    # Act
    tokenizer.build_vocab(sample_tokenized_sequences)

    # Assert
    assert tokenizer._is_built is True
    # Verify vocabulary contains expected values
    assert "Alice" in tokenizer.vocab
    assert "25" in tokenizer.vocab
    assert "Bob" in tokenizer.vocab
    assert "30" in tokenizer.vocab


def spec_value_tokenizer_encode(sample_tokenized_sequences):
    """Verify ValueTokenizer encodes single values correctly."""
    # Arrange
    tokenizer = ValueTokenizer(vocab_size=100)
    tokenizer.build_vocab(sample_tokenized_sequences)

    # Act
    alice_id = tokenizer.encode("Alice")
    unknown_id = tokenizer.encode("unknown_value")

    # Assert
    assert isinstance(alice_id, int)
    assert alice_id >= 5  # After special tokens
    assert unknown_id == 1  # Should return UNK index


def spec_value_tokenizer_encode_sequence(sample_tokenized_sequences):
    """Verify ValueTokenizer encodes sequence correctly."""
    # Arrange
    tokenizer = ValueTokenizer(vocab_size=100)
    tokenizer.build_vocab(sample_tokenized_sequences)
    sequence = sample_tokenized_sequences[0]

    # Act
    encoded_seq, token_ids = tokenizer.encode_sequence(sequence)

    # Assert
    assert len(token_ids) == len(sequence.tokens)
    assert isinstance(token_ids, list)
    assert all(isinstance(tid, int) for tid in token_ids)
    # Verify original sequence structure preserved
    assert encoded_seq.sequence_id == sequence.sequence_id
    assert len(encoded_seq.tokens) == len(sequence.tokens)
    # Verify positions maintained
    for i, token in enumerate(encoded_seq.tokens):
        assert token.position == sequence.tokens[i].position
        assert token.structure_tag.field == sequence.tokens[i].structure_tag.field


def spec_value_tokenizer_decode(sample_tokenized_sequences):
    """Verify ValueTokenizer decodes token IDs correctly."""
    # Arrange
    tokenizer = ValueTokenizer(vocab_size=100)
    tokenizer.build_vocab(sample_tokenized_sequences)

    # Act
    alice_id = tokenizer.encode("Alice")
    decoded = tokenizer.decode(alice_id)

    # Assert
    assert decoded == "Alice"


def spec_value_tokenizer_vocab_size_limit():
    """Verify ValueTokenizer respects vocabulary size limit."""
    # Arrange
    tokenizer = ValueTokenizer(vocab_size=7)  # 5 special + 2 regular tokens
    sequences = [
        TokenizedSequence(
            tokens=[
                Token(
                    value="a",
                    structure_tag=StructureTag(field="f1"),
                    position=0,
                ),
                Token(
                    value="b",
                    structure_tag=StructureTag(field="f2"),
                    position=1,
                ),
                Token(
                    value="c",
                    structure_tag=StructureTag(field="f3"),
                    position=2,
                ),
            ]
        )
    ]

    # Act
    tokenizer.build_vocab(sequences)

    # Assert
    # Should have 5 special tokens + at most 2 regular tokens
    assert len(tokenizer.vocab) <= 7
