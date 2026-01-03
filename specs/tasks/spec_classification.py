"""Tests for ClassificationHead."""

import torch

from saab_v3.tasks.classification import ClassificationHead
from saab_v3.tasks.pooling import CLSPooling, MaxPooling, MeanPooling


def spec_classification_binary(sample_encoder_output, sample_attention_mask):
    """Test binary classification (num_classes=2, multi_label=False)."""
    # Arrange
    head = ClassificationHead(d_model=128, num_classes=2, multi_label=False)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (sample_encoder_output.shape[0], 2)
    assert not torch.any(torch.isnan(logits))
    assert not torch.any(torch.isinf(logits))


def spec_classification_multiclass(sample_encoder_output, sample_attention_mask):
    """Test multi-class classification (num_classes=10, multi_label=False)."""
    # Arrange
    head = ClassificationHead(d_model=128, num_classes=10, multi_label=False)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (sample_encoder_output.shape[0], 10)
    assert not torch.any(torch.isnan(logits))
    assert not torch.any(torch.isinf(logits))


def spec_classification_multilabel(sample_encoder_output, sample_attention_mask):
    """Test multi-label classification (num_classes=10, multi_label=True)."""
    # Arrange
    head = ClassificationHead(d_model=128, num_classes=10, multi_label=True)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (sample_encoder_output.shape[0], 10)
    assert not torch.any(torch.isnan(logits))
    assert not torch.any(torch.isinf(logits))


def spec_classification_simple_mode(sample_encoder_output, sample_attention_mask):
    """Test simple mode (hidden_dims=None, single linear layer)."""
    # Arrange
    head = ClassificationHead(d_model=128, num_classes=5, hidden_dims=None)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (sample_encoder_output.shape[0], 5)
    assert head.mlp is None  # No MLP in simple mode
    assert head.output_layer.in_features == 128  # Direct connection to d_model


def spec_classification_mlp_mode(sample_encoder_output, sample_attention_mask):
    """Test MLP mode (hidden_dims=[256, 128])."""
    # Arrange
    head = ClassificationHead(d_model=128, num_classes=5, hidden_dims=[256, 128])

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (sample_encoder_output.shape[0], 5)
    assert head.mlp is not None  # MLP should be present
    assert (
        head.output_layer.in_features == 128
    )  # Output layer connects to last hidden dim


def spec_classification_output_shape(sample_encoder_output, sample_attention_mask):
    """Verify output shape is correct [batch, num_classes]."""
    # Arrange
    batch_size = sample_encoder_output.shape[0]
    num_classes = 7
    head = ClassificationHead(d_model=128, num_classes=num_classes)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (batch_size, num_classes)


def spec_classification_without_mask(sample_encoder_output):
    """Test forward pass without attention mask."""
    # Arrange
    head = ClassificationHead(d_model=128, num_classes=3)

    # Act
    logits = head(sample_encoder_output, attention_mask=None)

    # Assert
    assert logits.shape == (sample_encoder_output.shape[0], 3)
    assert not torch.any(torch.isnan(logits))


def spec_classification_device_consistency(
    sample_encoder_output, sample_attention_mask
):
    """Test that head works on different devices."""
    # Arrange
    head = ClassificationHead(d_model=128, num_classes=5)

    # Test CPU (default)
    logits_cpu = head(sample_encoder_output, sample_attention_mask)
    assert logits_cpu.device == sample_encoder_output.device

    # Test CUDA if available
    if torch.cuda.is_available():
        head_cuda = head.cuda()
        encoder_cuda = sample_encoder_output.cuda()
        mask_cuda = sample_attention_mask.cuda()
        logits_cuda = head_cuda(encoder_cuda, mask_cuda)
        assert logits_cuda.device.type == "cuda"

    # Test MPS if available (macOS)
    if torch.backends.mps.is_available():
        head_mps = head.to("mps")
        encoder_mps = sample_encoder_output.to("mps")
        mask_mps = sample_attention_mask.to("mps")
        logits_mps = head_mps(encoder_mps, mask_mps)
        assert logits_mps.device.type == "mps"


def spec_classification_cls_pooling(sample_encoder_output, sample_attention_mask):
    """Test with CLS pooling (default)."""
    # Arrange
    pooling = CLSPooling()
    head = ClassificationHead(d_model=128, num_classes=5, pooling=pooling)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (sample_encoder_output.shape[0], 5)
    # CLS pooling should extract first token
    assert isinstance(head.pooling, CLSPooling)


def spec_classification_mean_pooling(sample_encoder_output, sample_attention_mask):
    """Test with mean pooling."""
    # Arrange
    pooling = MeanPooling()
    head = ClassificationHead(d_model=128, num_classes=5, pooling=pooling)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (sample_encoder_output.shape[0], 5)
    assert isinstance(head.pooling, MeanPooling)


def spec_classification_max_pooling(sample_encoder_output, sample_attention_mask):
    """Test with max pooling."""
    # Arrange
    pooling = MaxPooling()
    head = ClassificationHead(d_model=128, num_classes=5, pooling=pooling)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (sample_encoder_output.shape[0], 5)
    assert isinstance(head.pooling, MaxPooling)


def spec_classification_returns_logits(sample_encoder_output, sample_attention_mask):
    """Verify that head returns logits (no activation applied)."""
    # Arrange
    head = ClassificationHead(d_model=128, num_classes=3)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    # Logits should not be in [0, 1] range (would indicate sigmoid/softmax applied)
    # They should be unbounded
    assert torch.any(logits < 0) or torch.any(
        logits > 1
    )  # At least some values outside [0,1]
    # Apply softmax to verify it works
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]))


def spec_classification_different_batch_sizes(sample_d_model):
    """Test with different batch sizes."""
    # Arrange
    head = ClassificationHead(d_model=sample_d_model, num_classes=4)

    # Test batch_size=1
    encoder_1 = torch.randn(1, 10, sample_d_model)
    logits_1 = head(encoder_1)
    assert logits_1.shape == (1, 4)

    # Test batch_size=8
    encoder_8 = torch.randn(8, 10, sample_d_model)
    logits_8 = head(encoder_8)
    assert logits_8.shape == (8, 4)


def spec_classification_different_seq_lengths(sample_batch_size, sample_d_model):
    """Test with different sequence lengths."""
    # Arrange
    head = ClassificationHead(d_model=sample_d_model, num_classes=3)

    # Test seq_len=10
    encoder_10 = torch.randn(sample_batch_size, 10, sample_d_model)
    logits_10 = head(encoder_10)
    assert logits_10.shape == (sample_batch_size, 3)

    # Test seq_len=100
    encoder_100 = torch.randn(sample_batch_size, 100, sample_d_model)
    logits_100 = head(encoder_100)
    assert logits_100.shape == (sample_batch_size, 3)
