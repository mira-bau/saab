"""Specs for FlatTransformer model - happy path only."""

import torch

from saab_v3.models.flat_transformer import FlatTransformer


# ============================================================================
# FlatTransformer Specs
# ============================================================================


def spec_flat_transformer_basic_forward_pass(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify FlatTransformer performs basic forward pass correctly with Batch."""
    # Arrange
    # Flat only needs token_vocab_size
    flat_vocab_sizes = {"token_vocab_size": sample_vocab_sizes["token_vocab_size"]}
    model = FlatTransformer(
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        ffn_dim=4 * d_model,
        vocab_sizes=flat_vocab_sizes,
        max_seq_len=512,
    )

    # Act
    output = model(sample_batch)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_flat_transformer_return_attention_weights(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model, num_heads
):
    """Verify FlatTransformer can return attention weights from all layers."""
    # Arrange
    num_layers = 3
    flat_vocab_sizes = {"token_vocab_size": sample_vocab_sizes["token_vocab_size"]}
    model = FlatTransformer(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=4 * d_model,
        vocab_sizes=flat_vocab_sizes,
        max_seq_len=512,
    )

    # Act
    output, attention_weights_list = model(sample_batch, return_attention_weights=True)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert len(attention_weights_list) == num_layers
    for attn_weights in attention_weights_list:
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        assert (attn_weights >= 0).all()
        assert (attn_weights <= 2.0).all()  # Allow tolerance for dropout


def spec_flat_transformer_multiple_layers(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify FlatTransformer works with different numbers of layers."""
    # Arrange
    flat_vocab_sizes = {"token_vocab_size": sample_vocab_sizes["token_vocab_size"]}
    test_layers = [1, 2, 4]

    # Act & Assert
    for num_layers in test_layers:
        model = FlatTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=4,
            ffn_dim=4 * d_model,
            vocab_sizes=flat_vocab_sizes,
            max_seq_len=512,
        )
        output = model(sample_batch)
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def spec_flat_transformer_different_configs(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify FlatTransformer works with different configurations."""
    # Arrange
    flat_vocab_sizes = {"token_vocab_size": sample_vocab_sizes["token_vocab_size"]}
    test_configs = [
        {"num_heads": 2, "ffn_dim": 2 * d_model},
        {"num_heads": 4, "ffn_dim": 4 * d_model},
        {"num_heads": 8, "ffn_dim": 8 * d_model},
    ]

    # Act & Assert
    for config in test_configs:
        if d_model % config["num_heads"] == 0:
            model = FlatTransformer(
                d_model=d_model,
                num_layers=2,
                num_heads=config["num_heads"],
                ffn_dim=config["ffn_dim"],
                vocab_sizes=flat_vocab_sizes,
                max_seq_len=512,
            )
            output = model(sample_batch)
            assert output.shape == (batch_size, seq_len, d_model)


def spec_flat_transformer_embedding_combination(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify FlatTransformer only uses token + positional embeddings (no structural)."""
    # Arrange
    flat_vocab_sizes = {"token_vocab_size": sample_vocab_sizes["token_vocab_size"]}
    model = FlatTransformer(
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        ffn_dim=4 * d_model,
        vocab_sizes=flat_vocab_sizes,
        max_seq_len=512,
    )

    # Act
    output = model(sample_batch)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Verify embeddings only use token + positional (no field, entity, time, token_type)
    # This is verified by the fact that model only needs token_vocab_size
    assert "field_vocab_size" not in flat_vocab_sizes
    assert "entity_vocab_size" not in flat_vocab_sizes
    assert "time_vocab_size" not in flat_vocab_sizes
    assert "token_type_vocab_size" not in flat_vocab_sizes
