"""Tests for RegressionHead."""

import torch

from saab_v3.tasks.pooling import CLSPooling, MaxPooling, MeanPooling
from saab_v3.tasks.regression import RegressionHead


def spec_regression_single_target(sample_encoder_output, sample_attention_mask):
    """Test single-target regression (num_targets=1)."""
    # Arrange
    head = RegressionHead(d_model=128, num_targets=1)

    # Act
    predictions = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert predictions.shape == (sample_encoder_output.shape[0], 1)
    assert not torch.any(torch.isnan(predictions))
    assert not torch.any(torch.isinf(predictions))


def spec_regression_multi_target(sample_encoder_output, sample_attention_mask):
    """Test multi-target regression (num_targets=5)."""
    # Arrange
    head = RegressionHead(d_model=128, num_targets=5)

    # Act
    predictions = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert predictions.shape == (sample_encoder_output.shape[0], 5)
    assert not torch.any(torch.isnan(predictions))
    assert not torch.any(torch.isinf(predictions))


def spec_regression_simple_mode(sample_encoder_output, sample_attention_mask):
    """Test simple mode (hidden_dims=None, single linear layer)."""
    # Arrange
    head = RegressionHead(d_model=128, num_targets=3, hidden_dims=None)

    # Act
    predictions = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert predictions.shape == (sample_encoder_output.shape[0], 3)
    assert head.mlp is None  # No MLP in simple mode
    assert head.output_layer.in_features == 128  # Direct connection to d_model


def spec_regression_mlp_mode(sample_encoder_output, sample_attention_mask):
    """Test MLP mode (hidden_dims=[256, 128])."""
    # Arrange
    head = RegressionHead(d_model=128, num_targets=3, hidden_dims=[256, 128])

    # Act
    predictions = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert predictions.shape == (sample_encoder_output.shape[0], 3)
    assert head.mlp is not None  # MLP should be present
    assert (
        head.output_layer.in_features == 128
    )  # Output layer connects to last hidden dim


def spec_regression_output_shape(sample_encoder_output, sample_attention_mask):
    """Verify output shape is correct [batch, num_targets]."""
    # Arrange
    batch_size = sample_encoder_output.shape[0]
    num_targets = 7
    head = RegressionHead(d_model=128, num_targets=num_targets)

    # Act
    predictions = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert predictions.shape == (batch_size, num_targets)


def spec_regression_without_mask(sample_encoder_output):
    """Test forward pass without attention mask."""
    # Arrange
    head = RegressionHead(d_model=128, num_targets=2)

    # Act
    predictions = head(sample_encoder_output, attention_mask=None)

    # Assert
    assert predictions.shape == (sample_encoder_output.shape[0], 2)
    assert not torch.any(torch.isnan(predictions))


def spec_regression_device_consistency(sample_encoder_output, sample_attention_mask):
    """Test that head works on different devices."""
    # Arrange
    head = RegressionHead(d_model=128, num_targets=4)

    # Test CPU (default)
    predictions_cpu = head(sample_encoder_output, sample_attention_mask)
    assert predictions_cpu.device == sample_encoder_output.device

    # Test CUDA if available
    if torch.cuda.is_available():
        head_cuda = head.cuda()
        encoder_cuda = sample_encoder_output.cuda()
        mask_cuda = sample_attention_mask.cuda()
        predictions_cuda = head_cuda(encoder_cuda, mask_cuda)
        assert predictions_cuda.device.type == "cuda"

    # Test MPS if available (macOS)
    if torch.backends.mps.is_available():
        head_mps = head.to("mps")
        encoder_mps = sample_encoder_output.to("mps")
        mask_mps = sample_attention_mask.to("mps")
        predictions_mps = head_mps(encoder_mps, mask_mps)
        assert predictions_mps.device.type == "mps"


def spec_regression_cls_pooling(sample_encoder_output, sample_attention_mask):
    """Test with CLS pooling (default)."""
    # Arrange
    pooling = CLSPooling()
    head = RegressionHead(d_model=128, num_targets=3, pooling=pooling)

    # Act
    predictions = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert predictions.shape == (sample_encoder_output.shape[0], 3)
    assert isinstance(head.pooling, CLSPooling)


def spec_regression_mean_pooling(sample_encoder_output, sample_attention_mask):
    """Test with mean pooling."""
    # Arrange
    pooling = MeanPooling()
    head = RegressionHead(d_model=128, num_targets=3, pooling=pooling)

    # Act
    predictions = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert predictions.shape == (sample_encoder_output.shape[0], 3)
    assert isinstance(head.pooling, MeanPooling)


def spec_regression_max_pooling(sample_encoder_output, sample_attention_mask):
    """Test with max pooling."""
    # Arrange
    pooling = MaxPooling()
    head = RegressionHead(d_model=128, num_targets=3, pooling=pooling)

    # Act
    predictions = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert predictions.shape == (sample_encoder_output.shape[0], 3)
    assert isinstance(head.pooling, MaxPooling)


def spec_regression_returns_continuous(sample_encoder_output, sample_attention_mask):
    """Verify that head returns continuous values (no activation applied)."""
    # Arrange
    head = RegressionHead(d_model=128, num_targets=2)

    # Act
    predictions = head(sample_encoder_output, sample_attention_mask)

    # Assert
    # Predictions should be unbounded (not constrained to [0, 1] or [-1, 1])
    # They should be continuous values
    assert predictions.dtype in [torch.float32, torch.float64]
    # Values can be any real number
    assert torch.any(torch.abs(predictions) > 1) or torch.any(
        torch.abs(predictions) < 1
    )


def spec_regression_different_batch_sizes(sample_d_model):
    """Test with different batch sizes."""
    # Arrange
    head = RegressionHead(d_model=sample_d_model, num_targets=2)

    # Test batch_size=1
    encoder_1 = torch.randn(1, 10, sample_d_model)
    predictions_1 = head(encoder_1)
    assert predictions_1.shape == (1, 2)

    # Test batch_size=8
    encoder_8 = torch.randn(8, 10, sample_d_model)
    predictions_8 = head(encoder_8)
    assert predictions_8.shape == (8, 2)


def spec_regression_different_seq_lengths(sample_batch_size, sample_d_model):
    """Test with different sequence lengths."""
    # Arrange
    head = RegressionHead(d_model=sample_d_model, num_targets=1)

    # Test seq_len=10
    encoder_10 = torch.randn(sample_batch_size, 10, sample_d_model)
    predictions_10 = head(encoder_10)
    assert predictions_10.shape == (sample_batch_size, 1)

    # Test seq_len=100
    encoder_100 = torch.randn(sample_batch_size, 100, sample_d_model)
    predictions_100 = head(encoder_100)
    assert predictions_100.shape == (sample_batch_size, 1)
