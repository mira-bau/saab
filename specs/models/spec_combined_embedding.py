"""Specs for CombinedEmbedding component - happy path only."""

import torch
import pytest

from saab_v3.models.embeddings.combined_embedding import CombinedEmbedding


# ============================================================================
# CombinedEmbedding Specs
# ============================================================================


def spec_combined_embedding_flat_config(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify CombinedEmbedding works with Flat configuration (token + positional only)."""
    # Arrange
    # Flat config only needs token_vocab_size
    flat_vocab_sizes = {"token_vocab_size": sample_vocab_sizes["token_vocab_size"]}
    embedding = CombinedEmbedding(
        d_model=d_model,
        vocab_sizes=flat_vocab_sizes,
        max_seq_len=512,
        use_token_type=False,
        use_field=False,
        use_entity=False,
        use_time=False,
    )

    # Act
    output = embedding(sample_batch)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_combined_embedding_scratch_saab_config(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify CombinedEmbedding works with Scratch/SAAB configuration (all embeddings)."""
    # Arrange
    embedding = CombinedEmbedding(
        d_model=d_model,
        vocab_sizes=sample_vocab_sizes,
        max_seq_len=512,
        use_token_type=True,
        use_field=True,
        use_entity=True,
        use_time=True,
    )

    # Act
    output = embedding(sample_batch)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_combined_embedding_missing_vocab_size_error(d_model):
    """Verify CombinedEmbedding raises error when required vocab size is missing."""
    # Arrange: Missing token_vocab_size (required)
    vocab_sizes = {}

    # Act & Assert
    with pytest.raises(ValueError, match="token_vocab_size"):
        CombinedEmbedding(
            d_model=d_model,
            vocab_sizes=vocab_sizes,
            max_seq_len=512,
        )


def spec_combined_embedding_missing_field_vocab_error(d_model, sample_vocab_sizes):
    """Verify CombinedEmbedding raises error when field vocab size missing but use_field=True."""
    # Arrange: Missing field_vocab_size but use_field=True
    vocab_sizes = {
        "token_vocab_size": sample_vocab_sizes["token_vocab_size"],
    }

    # Act & Assert
    with pytest.raises(ValueError, match="field_vocab_size"):
        CombinedEmbedding(
            d_model=d_model,
            vocab_sizes=vocab_sizes,
            max_seq_len=512,
            use_field=True,
        )


def spec_combined_embedding_missing_entity_vocab_error(d_model, sample_vocab_sizes):
    """Verify CombinedEmbedding raises error when entity vocab size missing but use_entity=True."""
    # Arrange: Missing entity_vocab_size but use_entity=True
    vocab_sizes = {
        "token_vocab_size": sample_vocab_sizes["token_vocab_size"],
    }

    # Act & Assert
    with pytest.raises(ValueError, match="entity_vocab_size"):
        CombinedEmbedding(
            d_model=d_model,
            vocab_sizes=vocab_sizes,
            max_seq_len=512,
            use_entity=True,
        )


def spec_combined_embedding_missing_time_vocab_error(d_model, sample_vocab_sizes):
    """Verify CombinedEmbedding raises error when time vocab size missing but use_time=True."""
    # Arrange: Missing time_vocab_size but use_time=True
    vocab_sizes = {
        "token_vocab_size": sample_vocab_sizes["token_vocab_size"],
    }

    # Act & Assert
    with pytest.raises(ValueError, match="time_vocab_size"):
        CombinedEmbedding(
            d_model=d_model,
            vocab_sizes=vocab_sizes,
            max_seq_len=512,
            use_time=True,
        )


def spec_combined_embedding_missing_token_type_vocab_error(d_model, sample_vocab_sizes):
    """Verify CombinedEmbedding raises error when token_type vocab size missing but use_token_type=True."""
    # Arrange: Missing token_type_vocab_size but use_token_type=True
    vocab_sizes = {
        "token_vocab_size": sample_vocab_sizes["token_vocab_size"],
    }

    # Act & Assert
    with pytest.raises(ValueError, match="token_type_vocab_size"):
        CombinedEmbedding(
            d_model=d_model,
            vocab_sizes=vocab_sizes,
            max_seq_len=512,
            use_token_type=True,
        )


def spec_combined_embedding_different_d_model(sample_batch, sample_vocab_sizes):
    """Verify CombinedEmbedding works with different d_model values."""
    # Arrange
    test_d_models = [64, 128, 256, 512]

    # Act & Assert
    for d_model in test_d_models:
        embedding = CombinedEmbedding(
            d_model=d_model,
            vocab_sizes=sample_vocab_sizes,
            max_seq_len=512,
            use_token_type=True,
            use_field=True,
            use_entity=True,
            use_time=True,
        )
        output = embedding(sample_batch)
        assert output.shape == (sample_batch.batch_size, sample_batch.seq_len, d_model)


def spec_combined_embedding_partial_embeddings(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify CombinedEmbedding works with partial embedding combinations."""
    # Arrange: Only token_type and field, not entity and time
    partial_vocab_sizes = {
        "token_vocab_size": sample_vocab_sizes["token_vocab_size"],
        "token_type_vocab_size": sample_vocab_sizes["token_type_vocab_size"],
        "field_vocab_size": sample_vocab_sizes["field_vocab_size"],
    }
    embedding = CombinedEmbedding(
        d_model=d_model,
        vocab_sizes=partial_vocab_sizes,
        max_seq_len=512,
        use_token_type=True,
        use_field=True,
        use_entity=False,
        use_time=False,
    )

    # Act
    output = embedding(sample_batch)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_combined_embedding_learned_vs_sinusoidal(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify CombinedEmbedding works with both learned and sinusoidal positional embeddings."""
    # Arrange
    flat_vocab_sizes = {"token_vocab_size": sample_vocab_sizes["token_vocab_size"]}

    embedding_learned = CombinedEmbedding(
        d_model=d_model,
        vocab_sizes=flat_vocab_sizes,
        max_seq_len=512,
        positional_learned=True,
    )

    embedding_sinusoidal = CombinedEmbedding(
        d_model=d_model,
        vocab_sizes=flat_vocab_sizes,
        max_seq_len=512,
        positional_learned=False,
    )

    # Act
    output_learned = embedding_learned(sample_batch)
    output_sinusoidal = embedding_sinusoidal(sample_batch)

    # Assert
    assert output_learned.shape == (batch_size, seq_len, d_model)
    assert output_sinusoidal.shape == (batch_size, seq_len, d_model)
    # They should be different (learned vs fixed)
    assert not torch.allclose(output_learned, output_sinusoidal, atol=1e-5)
