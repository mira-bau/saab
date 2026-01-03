"""Specs for FeedForward component - happy path only."""

import torch
import torch.nn as nn

from saab_v3.models.components.ffn import FeedForward


# ============================================================================
# FeedForward Specs
# ============================================================================


def spec_ffn_basic_forward_pass(
    feed_forward, sample_tensor, batch_size, seq_len, d_model
):
    """Verify FeedForward performs basic forward pass correctly."""
    # Arrange
    x = sample_tensor

    # Act
    output = feed_forward(x)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_ffn_different_activations(
    d_model, ffn_dim, sample_tensor, batch_size, seq_len
):
    """Verify FeedForward works with different activation functions."""
    # Arrange
    x = sample_tensor
    activations = [nn.GELU(), nn.ReLU(), nn.Tanh()]

    # Act & Assert
    for activation in activations:
        ffn = FeedForward(d_model=d_model, ffn_dim=ffn_dim, activation=activation)
        output = ffn(x)
        assert output.shape == (batch_size, seq_len, d_model)


def spec_ffn_different_ffn_dims(d_model, sample_tensor, batch_size, seq_len):
    """Verify FeedForward works with different ffn_dim values."""
    # Arrange
    x = sample_tensor
    ffn_dims = [2 * d_model, 4 * d_model, 8 * d_model]

    # Act & Assert
    for ffn_dim in ffn_dims:
        ffn = FeedForward(d_model=d_model, ffn_dim=ffn_dim)
        output = ffn(x)
        assert output.shape == (batch_size, seq_len, d_model)


def spec_ffn_with_without_bias(d_model, ffn_dim, sample_tensor, batch_size, seq_len):
    """Verify FeedForward works with and without bias."""
    # Arrange
    x = sample_tensor

    # Act & Assert: With bias (default)
    ffn_with_bias = FeedForward(d_model=d_model, ffn_dim=ffn_dim, bias=True)
    output_with_bias = ffn_with_bias(x)
    assert output_with_bias.shape == (batch_size, seq_len, d_model)

    # Act & Assert: Without bias
    ffn_without_bias = FeedForward(d_model=d_model, ffn_dim=ffn_dim, bias=False)
    output_without_bias = ffn_without_bias(x)
    assert output_without_bias.shape == (batch_size, seq_len, d_model)


def spec_ffn_different_shapes(d_model, ffn_dim):
    """Verify FeedForward works with different batch sizes and sequence lengths."""
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
        ffn = FeedForward(d_model=d_model, ffn_dim=ffn_dim)
        output = ffn(x)
        assert output.shape == (batch_size, seq_len, d_model)
