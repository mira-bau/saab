"""Specs for individual embedding modules - happy path only."""

import torch

from saab_v3.data.constants import PAD_IDX
from saab_v3.models.embeddings.token_embedding import TokenEmbedding
from saab_v3.models.embeddings.positional_embedding import PositionalEmbedding
from saab_v3.models.embeddings.token_type_embedding import TokenTypeEmbedding
from saab_v3.models.embeddings.field_embedding import FieldEmbedding
from saab_v3.models.embeddings.entity_embedding import EntityEmbedding
from saab_v3.models.embeddings.time_embedding import TimeEmbedding


# ============================================================================
# TokenEmbedding Specs
# ============================================================================


def spec_token_embedding_forward_pass(batch_size, seq_len, d_model):
    """Verify TokenEmbedding performs forward pass correctly."""
    # Arrange
    vocab_size = 1000
    embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
    token_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Act
    output = embedding(token_ids)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_token_embedding_padding_handling(batch_size, seq_len, d_model):
    """Verify TokenEmbedding handles padding correctly."""
    # Arrange
    vocab_size = 1000
    embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
    token_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    token_ids[:, -3:] = PAD_IDX  # Last 3 positions are padding

    # Act
    output = embedding(token_ids)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Padding positions should have zero embeddings
    assert (output[:, -3:, :] == 0).all()


# ============================================================================
# PositionalEmbedding Specs
# ============================================================================


def spec_positional_embedding_learned(batch_size, seq_len, d_model):
    """Verify PositionalEmbedding with learned embeddings works correctly."""
    # Arrange
    max_seq_len = 512
    embedding = PositionalEmbedding(
        max_seq_len=max_seq_len, d_model=d_model, learned=True
    )
    # Create dummy input to infer sequence length
    x = torch.randn(batch_size, seq_len, d_model)

    # Act
    output = embedding(x)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_positional_embedding_sinusoidal(batch_size, seq_len, d_model):
    """Verify PositionalEmbedding with sinusoidal embeddings works correctly."""
    # Arrange
    max_seq_len = 512
    embedding = PositionalEmbedding(
        max_seq_len=max_seq_len, d_model=d_model, learned=False
    )
    # Create dummy input to infer sequence length
    x = torch.randn(batch_size, seq_len, d_model)

    # Act
    output = embedding(x)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_positional_embedding_different_lengths(batch_size, d_model):
    """Verify PositionalEmbedding works with different sequence lengths."""
    # Arrange
    max_seq_len = 512
    embedding = PositionalEmbedding(
        max_seq_len=max_seq_len, d_model=d_model, learned=True
    )
    test_lengths = [10, 50, 100, 256]

    # Act & Assert
    for seq_len in test_lengths:
        x = torch.randn(batch_size, seq_len, d_model)
        output = embedding(x)
        assert output.shape == (batch_size, seq_len, d_model)


# ============================================================================
# TokenTypeEmbedding Specs
# ============================================================================


def spec_token_type_embedding_forward_pass(batch_size, seq_len, d_model):
    """Verify TokenTypeEmbedding performs forward pass correctly."""
    # Arrange
    vocab_size = 5
    embedding = TokenTypeEmbedding(vocab_size=vocab_size, d_model=d_model)
    token_type_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), dtype=torch.long
    )

    # Act
    output = embedding(token_type_ids)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_token_type_embedding_padding_handling(batch_size, seq_len, d_model):
    """Verify TokenTypeEmbedding handles padding correctly."""
    # Arrange
    vocab_size = 5
    embedding = TokenTypeEmbedding(vocab_size=vocab_size, d_model=d_model)
    token_type_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), dtype=torch.long
    )
    token_type_ids[:, -3:] = PAD_IDX  # Last 3 positions are padding

    # Act
    output = embedding(token_type_ids)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Padding positions should have zero embeddings
    assert (output[:, -3:, :] == 0).all()


# ============================================================================
# FieldEmbedding Specs
# ============================================================================


def spec_field_embedding_forward_pass(batch_size, seq_len, d_model):
    """Verify FieldEmbedding performs forward pass correctly."""
    # Arrange
    vocab_size = 20
    embedding = FieldEmbedding(vocab_size=vocab_size, d_model=d_model)
    field_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Act
    output = embedding(field_ids)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_field_embedding_padding_handling(batch_size, seq_len, d_model):
    """Verify FieldEmbedding handles padding correctly."""
    # Arrange
    vocab_size = 20
    embedding = FieldEmbedding(vocab_size=vocab_size, d_model=d_model)
    field_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    field_ids[:, -3:] = PAD_IDX  # Last 3 positions are padding

    # Act
    output = embedding(field_ids)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Padding positions should have zero embeddings
    assert (output[:, -3:, :] == 0).all()


# ============================================================================
# EntityEmbedding Specs
# ============================================================================


def spec_entity_embedding_forward_pass(batch_size, seq_len, d_model):
    """Verify EntityEmbedding performs forward pass correctly."""
    # Arrange
    vocab_size = 50
    embedding = EntityEmbedding(vocab_size=vocab_size, d_model=d_model)
    entity_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Act
    output = embedding(entity_ids)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_entity_embedding_padding_handling(batch_size, seq_len, d_model):
    """Verify EntityEmbedding handles padding correctly."""
    # Arrange
    vocab_size = 50
    embedding = EntityEmbedding(vocab_size=vocab_size, d_model=d_model)
    entity_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    entity_ids[:, -3:] = PAD_IDX  # Last 3 positions are padding

    # Act
    output = embedding(entity_ids)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Padding positions should have zero embeddings
    assert (output[:, -3:, :] == 0).all()


# ============================================================================
# TimeEmbedding Specs
# ============================================================================


def spec_time_embedding_forward_pass(batch_size, seq_len, d_model):
    """Verify TimeEmbedding performs forward pass correctly."""
    # Arrange
    vocab_size = 10
    embedding = TimeEmbedding(vocab_size=vocab_size, d_model=d_model)
    time_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Act
    output = embedding(time_ids)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_time_embedding_padding_handling(batch_size, seq_len, d_model):
    """Verify TimeEmbedding handles padding correctly."""
    # Arrange
    vocab_size = 10
    embedding = TimeEmbedding(vocab_size=vocab_size, d_model=d_model)
    time_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    time_ids[:, -3:] = PAD_IDX  # Last 3 positions are padding

    # Act
    output = embedding(time_ids)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Padding positions should have zero embeddings
    assert (output[:, -3:, :] == 0).all()


# ============================================================================
# Common Embedding Tests
# ============================================================================


def spec_embeddings_different_d_model(batch_size, seq_len):
    """Verify all embeddings work with different d_model values."""
    # Arrange
    test_d_models = [64, 128, 256, 512]
    vocab_size = 100

    # Act & Assert
    for d_model in test_d_models:
        # Test TokenEmbedding
        token_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        token_ids = torch.randint(
            1, vocab_size, (batch_size, seq_len), dtype=torch.long
        )
        output = token_emb(token_ids)
        assert output.shape == (batch_size, seq_len, d_model)

        # Test FieldEmbedding
        field_emb = FieldEmbedding(vocab_size=vocab_size, d_model=d_model)
        field_ids = torch.randint(
            1, vocab_size, (batch_size, seq_len), dtype=torch.long
        )
        output = field_emb(field_ids)
        assert output.shape == (batch_size, seq_len, d_model)

        # Test EntityEmbedding
        entity_emb = EntityEmbedding(vocab_size=vocab_size, d_model=d_model)
        entity_ids = torch.randint(
            1, vocab_size, (batch_size, seq_len), dtype=torch.long
        )
        output = entity_emb(entity_ids)
        assert output.shape == (batch_size, seq_len, d_model)

        # Test TimeEmbedding
        time_emb = TimeEmbedding(vocab_size=vocab_size, d_model=d_model)
        time_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
        output = time_emb(time_ids)
        assert output.shape == (batch_size, seq_len, d_model)

        # Test TokenTypeEmbedding
        token_type_emb = TokenTypeEmbedding(vocab_size=vocab_size, d_model=d_model)
        token_type_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), dtype=torch.long
        )
        output = token_type_emb(token_type_ids)
        assert output.shape == (batch_size, seq_len, d_model)

        # Test PositionalEmbedding
        pos_emb = PositionalEmbedding(max_seq_len=512, d_model=d_model, learned=True)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pos_emb(x)
        assert output.shape == (batch_size, seq_len, d_model)
