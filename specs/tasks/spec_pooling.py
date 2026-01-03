"""Specs for pooling strategies - happy path only."""

import pytest
import torch

from saab_v3.tasks.pooling import CLSPooling, MaxPooling, MeanPooling


# ============================================================================
# CLSPooling Specs
# ============================================================================


def spec_cls_pooling_extract_first_token(sample_encoder_output, sample_attention_mask):
    """Verify CLSPooling extracts the first token (position 0) correctly."""
    # Arrange
    pooling = CLSPooling()

    # Act
    output = pooling(sample_encoder_output, sample_attention_mask)

    # Assert
    assert output.shape == (
        sample_encoder_output.shape[0],
        sample_encoder_output.shape[2],
    )
    # Output should be the first token of each sequence
    assert torch.allclose(output, sample_encoder_output[:, 0, :])


def spec_cls_pooling_without_attention_mask(sample_encoder_output):
    """Verify CLSPooling works without attention mask."""
    # Arrange
    pooling = CLSPooling()

    # Act
    output = pooling(sample_encoder_output, attention_mask=None)

    # Assert
    assert output.shape == (
        sample_encoder_output.shape[0],
        sample_encoder_output.shape[2],
    )
    assert torch.allclose(output, sample_encoder_output[:, 0, :])


def spec_cls_pooling_different_shapes():
    """Verify CLSPooling handles different batch/sequence shapes."""
    # Arrange
    pooling = CLSPooling()

    # Test different shapes
    test_cases = [
        (1, 10, 128),  # Single sample
        (8, 32, 256),  # Larger batch
        (2, 512, 768),  # Long sequence
    ]

    for batch_size, seq_len, d_model in test_cases:
        encoder_output = torch.randn(batch_size, seq_len, d_model)

        # Act
        output = pooling(encoder_output)

        # Assert
        assert output.shape == (batch_size, d_model)
        assert torch.allclose(output, encoder_output[:, 0, :])


def spec_cls_pooling_device_consistency(sample_encoder_output):
    """Verify CLSPooling works on different devices."""
    # Arrange
    pooling = CLSPooling()

    # Test CPU
    output_cpu = pooling(sample_encoder_output)
    assert output_cpu.device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        encoder_output_cuda = sample_encoder_output.cuda()
        pooling_cuda = pooling.cuda()
        output_cuda = pooling_cuda(encoder_output_cuda)
        assert output_cuda.device.type == "cuda"
        assert torch.allclose(output_cpu, output_cuda.cpu(), atol=1e-6)

    # Test MPS if available
    if torch.backends.mps.is_available():
        encoder_output_mps = sample_encoder_output.to("mps")
        pooling_mps = pooling.to("mps")
        output_mps = pooling_mps(encoder_output_mps)
        assert output_mps.device.type == "mps"
        assert torch.allclose(output_cpu, output_mps.cpu(), atol=1e-6)


# ============================================================================
# MeanPooling Specs
# ============================================================================


def spec_mean_pooling_excludes_padding(sample_encoder_output, sample_attention_mask):
    """Verify MeanPooling correctly excludes padding tokens."""
    # Arrange
    pooling = MeanPooling()
    batch_size, seq_len, d_model = sample_encoder_output.shape

    # Act
    output = pooling(sample_encoder_output, sample_attention_mask)

    # Assert
    assert output.shape == (batch_size, d_model)

    # Manually verify for first batch
    mask = sample_attention_mask[0]  # [seq_len]
    valid_tokens = sample_encoder_output[0][mask == 1]  # [num_valid, d_model]
    expected_mean = valid_tokens.mean(dim=0)

    assert torch.allclose(output[0], expected_mean, atol=1e-5)


def spec_mean_pooling_requires_attention_mask(sample_encoder_output):
    """Verify MeanPooling raises error if attention_mask is not provided."""
    # Arrange
    pooling = MeanPooling()

    # Act & Assert
    with pytest.raises(ValueError, match="attention_mask is required"):
        pooling(sample_encoder_output, attention_mask=None)


def spec_mean_pooling_all_padding(
    sample_encoder_output, sample_attention_mask_all_padding
):
    """Verify MeanPooling handles edge case where all tokens are padding."""
    # Arrange
    pooling = MeanPooling()

    # Act
    output = pooling(sample_encoder_output, sample_attention_mask_all_padding)

    # Assert
    # Should return zeros (since no valid tokens)
    assert output.shape == (
        sample_encoder_output.shape[0],
        sample_encoder_output.shape[2],
    )
    assert torch.allclose(output, torch.zeros_like(output))


def spec_mean_pooling_single_token():
    """Verify MeanPooling handles single valid token."""
    # Arrange
    pooling = MeanPooling()
    batch_size, d_model = 2, 128
    encoder_output = torch.randn(batch_size, 1, d_model)
    attention_mask = torch.ones(batch_size, 1, dtype=torch.long)

    # Act
    output = pooling(encoder_output, attention_mask)

    # Assert
    assert output.shape == (batch_size, d_model)
    assert torch.allclose(output, encoder_output.squeeze(1))


def spec_mean_pooling_different_shapes():
    """Verify MeanPooling handles different tensor shapes."""
    # Arrange
    pooling = MeanPooling()

    test_cases = [
        (1, 10, 128),
        (8, 32, 256),
        (2, 512, 768),
    ]

    for batch_size, seq_len, d_model in test_cases:
        encoder_output = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        # Add some padding
        attention_mask[:, seq_len // 2 :] = 0

        # Act
        output = pooling(encoder_output, attention_mask)

        # Assert
        assert output.shape == (batch_size, d_model)


# ============================================================================
# MaxPooling Specs
# ============================================================================


def spec_max_pooling_excludes_padding(sample_encoder_output, sample_attention_mask):
    """Verify MaxPooling correctly excludes padding tokens."""
    # Arrange
    pooling = MaxPooling()
    batch_size, seq_len, d_model = sample_encoder_output.shape

    # Act
    output = pooling(sample_encoder_output, sample_attention_mask)

    # Assert
    assert output.shape == (batch_size, d_model)

    # Manually verify for first batch
    mask = sample_attention_mask[0]  # [seq_len]
    valid_tokens = sample_encoder_output[0][mask == 1]  # [num_valid, d_model]
    expected_max, _ = valid_tokens.max(dim=0)

    assert torch.allclose(output[0], expected_max, atol=1e-5)


def spec_max_pooling_requires_attention_mask(sample_encoder_output):
    """Verify MaxPooling raises error if attention_mask is not provided."""
    # Arrange
    pooling = MaxPooling()

    # Act & Assert
    with pytest.raises(ValueError, match="attention_mask is required"):
        pooling(sample_encoder_output, attention_mask=None)


def spec_max_pooling_all_padding(
    sample_encoder_output, sample_attention_mask_all_padding
):
    """Verify MaxPooling handles edge case where all tokens are padding."""
    # Arrange
    pooling = MaxPooling()

    # Act
    output = pooling(sample_encoder_output, sample_attention_mask_all_padding)

    # Assert
    # Should return minimum value for dtype (since all tokens are masked)
    assert output.shape == (
        sample_encoder_output.shape[0],
        sample_encoder_output.shape[2],
    )
    # All values should be the minimum value for the dtype
    min_value = torch.finfo(sample_encoder_output.dtype).min
    assert torch.allclose(output, torch.full_like(output, min_value))


def spec_max_pooling_single_token():
    """Verify MaxPooling handles single valid token."""
    # Arrange
    pooling = MaxPooling()
    batch_size, d_model = 2, 128
    encoder_output = torch.randn(batch_size, 1, d_model)
    attention_mask = torch.ones(batch_size, 1, dtype=torch.long)

    # Act
    output = pooling(encoder_output, attention_mask)

    # Assert
    assert output.shape == (batch_size, d_model)
    assert torch.allclose(output, encoder_output.squeeze(1))


def spec_max_pooling_different_shapes():
    """Verify MaxPooling handles different tensor shapes."""
    # Arrange
    pooling = MaxPooling()

    test_cases = [
        (1, 10, 128),
        (8, 32, 256),
        (2, 512, 768),
    ]

    for batch_size, seq_len, d_model in test_cases:
        encoder_output = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        # Add some padding
        attention_mask[:, seq_len // 2 :] = 0

        # Act
        output = pooling(encoder_output, attention_mask)

        # Assert
        assert output.shape == (batch_size, d_model)


# ============================================================================
# Common Pooling Specs
# ============================================================================


def spec_pooling_output_shape_validation(sample_encoder_output, sample_attention_mask):
    """Verify all pooling strategies produce correct output shapes."""
    # Arrange
    cls_pooling = CLSPooling()
    mean_pooling = MeanPooling()
    max_pooling = MaxPooling()

    batch_size, seq_len, d_model = sample_encoder_output.shape
    expected_shape = (batch_size, d_model)

    # Act
    cls_output = cls_pooling(sample_encoder_output, sample_attention_mask)
    mean_output = mean_pooling(sample_encoder_output, sample_attention_mask)
    max_output = max_pooling(sample_encoder_output, sample_attention_mask)

    # Assert
    assert cls_output.shape == expected_shape
    assert mean_output.shape == expected_shape
    assert max_output.shape == expected_shape


def spec_pooling_no_nan_or_inf(sample_encoder_output, sample_attention_mask_full):
    """Verify pooling strategies don't produce NaN or Inf values for valid inputs."""
    # Arrange
    cls_pooling = CLSPooling()
    mean_pooling = MeanPooling()
    max_pooling = MaxPooling()

    # Act
    cls_output = cls_pooling(sample_encoder_output, sample_attention_mask_full)
    mean_output = mean_pooling(sample_encoder_output, sample_attention_mask_full)
    max_output = max_pooling(sample_encoder_output, sample_attention_mask_full)

    # Assert
    assert not torch.any(torch.isnan(cls_output))
    assert not torch.any(torch.isinf(cls_output))
    assert not torch.any(torch.isnan(mean_output))
    assert not torch.any(torch.isinf(mean_output))
    assert not torch.any(torch.isnan(max_output))
    # Max pooling can produce -inf for all-padding case, but not here (all valid)
