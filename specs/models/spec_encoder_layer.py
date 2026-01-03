"""Specs for TransformerEncoderLayer component - happy path only."""

import torch

from saab_v3.models.components.encoder_layer import TransformerEncoderLayer


# ============================================================================
# TransformerEncoderLayer Specs
# ============================================================================


def spec_encoder_layer_basic_forward_pass(
    encoder_layer, sample_tensor, batch_size, seq_len, d_model
):
    """Verify TransformerEncoderLayer performs basic forward pass correctly."""
    # Arrange
    x = sample_tensor

    # Act
    output = encoder_layer(x)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_encoder_layer_with_mask(
    encoder_layer, sample_tensor, sample_attention_mask, batch_size, seq_len, d_model
):
    """Verify TransformerEncoderLayer handles attention mask correctly."""
    # Arrange
    x = sample_tensor

    # Act
    output = encoder_layer(x, attention_mask=sample_attention_mask)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_encoder_layer_return_weights(
    encoder_layer, sample_tensor, batch_size, seq_len, d_model, num_heads
):
    """Verify TransformerEncoderLayer can return attention weights."""
    # Arrange
    x = sample_tensor

    # Act
    output, attn_weights = encoder_layer(x, return_attention_weights=True)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    # Verify attention weights are non-negative and have reasonable values
    # Note: Dropout is applied after softmax, so sum may not be exactly 1.0
    assert (attn_weights >= 0).all()
    assert (attn_weights <= 2.0).all()  # Allow some tolerance for dropout scaling


def spec_encoder_layer_residual_connection(
    encoder_layer, sample_tensor, batch_size, seq_len, d_model
):
    """Verify TransformerEncoderLayer applies residual connections."""
    # Arrange
    x = sample_tensor

    # Act
    output = encoder_layer(x)

    # Assert: Output should be different from input (transformation occurred)
    assert not torch.allclose(output, x, atol=1e-5)


def spec_encoder_layer_different_configs(d_model, sample_tensor, batch_size, seq_len):
    """Verify TransformerEncoderLayer works with different configurations."""
    # Arrange
    x = sample_tensor
    test_configs = [
        {"num_heads": 2, "ffn_dim": 2 * d_model, "dropout": 0.0},
        {"num_heads": 4, "ffn_dim": 4 * d_model, "dropout": 0.1},
        {"num_heads": 8, "ffn_dim": 8 * d_model, "dropout": 0.2},
    ]

    # Act & Assert
    for config in test_configs:
        if d_model % config["num_heads"] == 0:
            layer = TransformerEncoderLayer(
                d_model=d_model,
                num_heads=config["num_heads"],
                ffn_dim=config["ffn_dim"],
                dropout=config["dropout"],
            )
            output = layer(x)
            assert output.shape == (batch_size, seq_len, d_model)


def spec_encoder_layer_different_shapes(d_model, num_heads, ffn_dim):
    """Verify TransformerEncoderLayer works with different batch sizes and sequence lengths."""
    # Arrange
    test_cases = [
        (1, 10),
        (2, 10),
        (4, 10),
        (2, 50),
        (2, 100),
    ]

    # Act & Assert
    for batch_size, seq_len in test_cases:
        x = torch.randn(batch_size, seq_len, d_model)
        layer = TransformerEncoderLayer(
            d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim
        )
        output = layer(x)
        assert output.shape == (batch_size, seq_len, d_model)


def spec_encoder_layer_with_mask_and_weights(
    encoder_layer,
    sample_tensor,
    sample_attention_mask,
    batch_size,
    seq_len,
    d_model,
    num_heads,
):
    """Verify TransformerEncoderLayer works with both mask and return_attention_weights."""
    # Arrange
    x = sample_tensor

    # Act
    output, attn_weights = encoder_layer(
        x, attention_mask=sample_attention_mask, return_attention_weights=True
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
