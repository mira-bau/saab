"""Specs for MultiHeadAttention component - happy path only."""

import torch

from saab_v3.models.components.attention import MultiHeadAttention


# ============================================================================
# MultiHeadAttention Specs
# ============================================================================


def spec_attention_basic_forward_pass(
    multi_head_attention, sample_tensor, batch_size, seq_len, d_model
):
    """Verify MultiHeadAttention performs basic forward pass correctly."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Act
    output = multi_head_attention(query, key, value)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_attention_with_mask(
    multi_head_attention,
    sample_tensor,
    sample_attention_mask,
    batch_size,
    seq_len,
    d_model,
):
    """Verify MultiHeadAttention handles attention mask correctly."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Act
    output = multi_head_attention(
        query, key, value, attention_mask=sample_attention_mask
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def spec_attention_return_weights(
    multi_head_attention, sample_tensor, batch_size, seq_len, d_model, num_heads
):
    """Verify MultiHeadAttention can return attention weights."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Act
    output, attn_weights = multi_head_attention(
        query, key, value, return_attention_weights=True
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    # Verify attention weights are non-negative and have reasonable values
    # Note: Dropout is applied after softmax, so sum may not be exactly 1.0
    assert (attn_weights >= 0).all()
    assert (attn_weights <= 2.0).all()  # Allow some tolerance for dropout scaling


def spec_attention_multiple_heads(d_model, sample_tensor, batch_size, seq_len):
    """Verify MultiHeadAttention works with different numbers of heads."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Act & Assert: Test with different num_heads
    for num_heads in [1, 2, 4, 8]:
        if d_model % num_heads == 0:
            attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
            output = attn(query, key, value)
            assert output.shape == (batch_size, seq_len, d_model)


def spec_attention_different_shapes(d_model, num_heads):
    """Verify MultiHeadAttention works with different batch sizes and sequence lengths."""
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
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        output = attn(query, key, value)

        assert output.shape == (batch_size, seq_len, d_model)


def spec_attention_with_mask_and_weights(
    multi_head_attention,
    sample_tensor,
    sample_attention_mask,
    batch_size,
    seq_len,
    d_model,
    num_heads,
):
    """Verify MultiHeadAttention works with both mask and return_attention_weights."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Act
    output, attn_weights = multi_head_attention(
        query,
        key,
        value,
        attention_mask=sample_attention_mask,
        return_attention_weights=True,
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    # Verify masked positions have attention weights close to 0
    # (masked positions should be -inf before softmax, resulting in ~0 after softmax)
    masked_positions = sample_attention_mask == 0
    if masked_positions.any():
        # Check that attention to masked positions is very small
        for b in range(batch_size):
            for h in range(num_heads):
                masked_attn = attn_weights[b, h, :, masked_positions[b]]
                assert masked_attn.sum() < 0.1  # Should be very small
