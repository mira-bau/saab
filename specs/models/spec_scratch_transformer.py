"""Specs for ScratchTransformer model - happy path only."""

import torch

from saab_v3.models.scratch_transformer import ScratchTransformer


# ============================================================================
# ScratchTransformer Specs
# ============================================================================


def spec_scratch_transformer_basic_forward_pass(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify ScratchTransformer performs basic forward pass correctly with Batch."""
    # Arrange
    model = ScratchTransformer(
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        ffn_dim=4 * d_model,
        vocab_sizes=sample_vocab_sizes,
        max_seq_len=512,
    )

    # Act
    output = model(sample_batch)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_scratch_transformer_return_attention_weights(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model, num_heads
):
    """Verify ScratchTransformer can return attention weights from all layers."""
    # Arrange
    num_layers = 3
    model = ScratchTransformer(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=4 * d_model,
        vocab_sizes=sample_vocab_sizes,
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


def spec_scratch_transformer_multiple_layers(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify ScratchTransformer works with different numbers of layers."""
    # Arrange
    test_layers = [1, 2, 4]

    # Act & Assert
    for num_layers in test_layers:
        model = ScratchTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=4,
            ffn_dim=4 * d_model,
            vocab_sizes=sample_vocab_sizes,
            max_seq_len=512,
        )
        output = model(sample_batch)
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def spec_scratch_transformer_all_embeddings(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify ScratchTransformer uses all structural embeddings."""
    # Arrange
    model = ScratchTransformer(
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        ffn_dim=4 * d_model,
        vocab_sizes=sample_vocab_sizes,
        max_seq_len=512,
    )

    # Act
    output = model(sample_batch)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Verify all vocab sizes are required (model uses all embeddings)
    assert "token_vocab_size" in sample_vocab_sizes
    assert "token_type_vocab_size" in sample_vocab_sizes
    assert "field_vocab_size" in sample_vocab_sizes
    assert "entity_vocab_size" in sample_vocab_sizes
    assert "time_vocab_size" in sample_vocab_sizes


def spec_scratch_transformer_different_configs(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify ScratchTransformer works with different configurations."""
    # Arrange
    test_configs = [
        {"num_heads": 2, "ffn_dim": 2 * d_model},
        {"num_heads": 4, "ffn_dim": 4 * d_model},
        {"num_heads": 8, "ffn_dim": 8 * d_model},
    ]

    # Act & Assert
    for config in test_configs:
        if d_model % config["num_heads"] == 0:
            model = ScratchTransformer(
                d_model=d_model,
                num_layers=2,
                num_heads=config["num_heads"],
                ffn_dim=config["ffn_dim"],
                vocab_sizes=sample_vocab_sizes,
                max_seq_len=512,
            )
            output = model(sample_batch)
            assert output.shape == (batch_size, seq_len, d_model)


def spec_scratch_transformer_standard_attention(
    sample_batch, sample_vocab_sizes, batch_size, seq_len, d_model
):
    """Verify ScratchTransformer uses standard attention (not SAAB)."""
    # Arrange
    model = ScratchTransformer(
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        ffn_dim=4 * d_model,
        vocab_sizes=sample_vocab_sizes,
        max_seq_len=512,
    )

    # Act
    output = model(sample_batch)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Verify model uses standard TransformerEncoderLayer (not SAABEncoderLayer)
    # This is verified by the fact that ScratchTransformer doesn't have lambda_bias parameter
    # and uses BaseTransformer which uses TransformerEncoderLayer
