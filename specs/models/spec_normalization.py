"""Specs for LayerNorm component - happy path only."""

import torch

from saab_v3.models.components.normalization import LayerNorm


# ============================================================================
# LayerNorm Specs
# ============================================================================


def spec_layer_norm_basic_forward_pass(
    layer_norm, sample_tensor, batch_size, seq_len, d_model
):
    """Verify LayerNorm performs basic forward pass correctly."""
    # Arrange
    x = sample_tensor

    # Act
    output = layer_norm(x)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_layer_norm_different_shapes(d_model):
    """Verify LayerNorm works with different tensor shapes."""
    # Arrange
    test_cases = [
        (2, d_model),  # [batch_size, d_model]
        (2, 10, d_model),  # [batch_size, seq_len, d_model]
        (2, 10, 5, d_model),  # [batch_size, seq_len, features, d_model]
    ]

    # Act & Assert
    for shape in test_cases:
        x = torch.randn(*shape)
        norm = LayerNorm(d_model=d_model)
        output = norm(x)
        assert output.shape == x.shape


def spec_layer_norm_different_eps(d_model, sample_tensor):
    """Verify LayerNorm works with different eps values."""
    # Arrange
    x = sample_tensor

    # Act & Assert: Default eps
    norm_default = LayerNorm(d_model=d_model, eps=1e-5)
    output_default = norm_default(x)
    assert output_default.shape == x.shape

    # Act & Assert: Custom eps
    norm_custom = LayerNorm(d_model=d_model, eps=1e-6)
    output_custom = norm_custom(x)
    assert output_custom.shape == x.shape


def spec_layer_norm_normalization_effect(d_model):
    """Verify LayerNorm normalizes the last dimension."""
    # Arrange
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, d_model) * 10 + 5  # Shift and scale
    norm = LayerNorm(d_model=d_model)

    # Act
    output = norm(x)

    # Assert: Mean should be approximately 0, std approximately 1 along last dimension
    mean = output.mean(dim=-1)
    std = output.std(dim=-1)
    # Allow some tolerance due to learnable parameters
    assert torch.allclose(mean, torch.zeros_like(mean), atol=0.1)
    assert torch.allclose(std, torch.ones_like(std), atol=0.5)
