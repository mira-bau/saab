"""Specs for Vocabulary - happy path only."""

from saab_v3.data.vocabulary import Vocabulary
from saab_v3.data.constants import UNK_TOKEN


# ============================================================================
# Vocabulary Basic Functionality
# ============================================================================


def spec_vocabulary_initialization():
    """Verify Vocabulary initializes with special tokens correctly."""
    # Arrange
    special_tokens = ["[PAD]", "[UNK]", "[MASK]"]

    # Act
    vocab = Vocabulary(special_tokens=special_tokens)
    # Build empty vocabulary to enable encoding
    vocab.build_from_tokens([])

    # Assert
    assert len(vocab) == 3
    assert vocab.encode("[PAD]") == 0
    assert vocab.encode("[UNK]") == 1
    assert vocab.encode("[MASK]") == 2


def spec_vocabulary_build_from_tokens():
    """Verify Vocabulary builds from tokens with frequency counting."""
    # Arrange
    vocab = Vocabulary(special_tokens=["[PAD]", "[UNK]"])
    tokens = ["hello", "world", "hello", "test", "world", "python", "hello"]

    # Act
    vocab.build_from_tokens(tokens, vocab_size=5)

    # Assert
    # Should have 2 special tokens + 3 most frequent (hello=3, world=2, test=1)
    assert len(vocab) == 5
    assert "hello" in vocab
    assert "world" in vocab
    assert "test" in vocab
    # Verify most frequent tokens are included
    assert vocab.encode("hello") >= 2  # After special tokens
    assert vocab.encode("world") >= 2


def spec_vocabulary_encode():
    """Verify Vocabulary encodes tokens correctly."""
    # Arrange
    vocab = Vocabulary(special_tokens=["[PAD]", "[UNK]"])
    vocab.build_from_tokens(["hello", "world"], vocab_size=10)

    # Act & Assert
    hello_idx = vocab.encode("hello")
    world_idx = vocab.encode("world")
    unk_idx = vocab.encode("unknown_token")

    # Assert
    assert hello_idx >= 2  # After special tokens
    assert world_idx >= 2
    assert unk_idx == 1  # Should return UNK index
    assert hello_idx != world_idx


def spec_vocabulary_decode():
    """Verify Vocabulary decodes indices correctly."""
    # Arrange
    vocab = Vocabulary(special_tokens=["[PAD]", "[UNK]"])
    vocab.build_from_tokens(["hello", "world"], vocab_size=10)

    # Act
    pad_token = vocab.decode(0)
    unk_token = vocab.decode(1)
    hello_token = vocab.decode(vocab.encode("hello"))
    out_of_range = vocab.decode(999)

    # Assert
    assert pad_token == "[PAD]"
    assert unk_token == "[UNK]"
    assert hello_token == "hello"
    assert out_of_range == UNK_TOKEN  # Should return UNK for out of range


def spec_vocabulary_encode_batch():
    """Verify Vocabulary batch encoding works correctly."""
    # Arrange
    vocab = Vocabulary(special_tokens=["[PAD]", "[UNK]"])
    vocab.build_from_tokens(["hello", "world"], vocab_size=10)

    # Act
    tokens = ["hello", "world", "unknown", "hello"]
    indices = vocab.encode_batch(tokens)

    # Assert
    assert len(indices) == 4
    assert indices[0] == vocab.encode("hello")
    assert indices[1] == vocab.encode("world")
    assert indices[2] == 1  # UNK index
    assert indices[3] == vocab.encode("hello")


def spec_vocabulary_length():
    """Verify Vocabulary length includes special tokens."""
    # Arrange
    vocab = Vocabulary(special_tokens=["[PAD]", "[UNK]"])

    # Act
    initial_len = len(vocab)
    vocab.build_from_tokens(["hello", "world"], vocab_size=10)
    final_len = len(vocab)

    # Assert
    assert initial_len == 2  # Only special tokens
    assert final_len == 4  # 2 special + 2 regular tokens
    assert len(vocab) == len(vocab.idx_to_token)
