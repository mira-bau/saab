"""Tests for TokenClassificationHead."""

import torch

from saab_v3.tasks.token_classification import TokenClassificationHead


def spec_token_classification_basic(sample_encoder_output, sample_attention_mask):
    """Test basic forward pass."""
    # Arrange
    head = TokenClassificationHead(d_model=128, num_labels=10)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (
        sample_encoder_output.shape[0],
        sample_encoder_output.shape[1],
        10,
    )
    assert not torch.any(torch.isnan(logits))
    assert not torch.any(torch.isinf(logits))


def spec_token_classification_output_shape(
    sample_encoder_output, sample_attention_mask
):
    """Verify output shape is correct [batch, seq_len, num_labels]."""
    # Arrange
    batch_size = sample_encoder_output.shape[0]
    seq_len = sample_encoder_output.shape[1]
    num_labels = 7
    head = TokenClassificationHead(d_model=128, num_labels=num_labels)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (batch_size, seq_len, num_labels)


def spec_token_classification_simple_mode(sample_encoder_output, sample_attention_mask):
    """Test simple mode (hidden_dims=None, single linear layer)."""
    # Arrange
    head = TokenClassificationHead(d_model=128, num_labels=5, hidden_dims=None)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (
        sample_encoder_output.shape[0],
        sample_encoder_output.shape[1],
        5,
    )
    assert head.mlp is None  # No MLP in simple mode
    assert head.output_layer.in_features == 128  # Direct connection to d_model


def spec_token_classification_mlp_mode(sample_encoder_output, sample_attention_mask):
    """Test MLP mode (hidden_dims=[256, 128]) - per-token MLP."""
    # Arrange
    head = TokenClassificationHead(d_model=128, num_labels=5, hidden_dims=[256, 128])

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    assert logits.shape == (
        sample_encoder_output.shape[0],
        sample_encoder_output.shape[1],
        5,
    )
    assert head.mlp is not None  # MLP should be present
    assert (
        head.output_layer.in_features == 128
    )  # Output layer connects to last hidden dim


def spec_token_classification_without_mask(sample_encoder_output):
    """Test forward pass without attention mask."""
    # Arrange
    head = TokenClassificationHead(d_model=128, num_labels=3)

    # Act
    logits = head(sample_encoder_output, attention_mask=None)

    # Assert
    assert logits.shape == (
        sample_encoder_output.shape[0],
        sample_encoder_output.shape[1],
        3,
    )
    assert not torch.any(torch.isnan(logits))


def spec_token_classification_device_consistency(
    sample_encoder_output, sample_attention_mask
):
    """Test that head works on different devices."""
    # Arrange
    head = TokenClassificationHead(d_model=128, num_labels=4)

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


def spec_token_classification_different_seq_lengths(sample_batch_size, sample_d_model):
    """Test with different sequence lengths."""
    # Arrange
    head = TokenClassificationHead(d_model=sample_d_model, num_labels=3)

    # Test seq_len=10
    encoder_10 = torch.randn(sample_batch_size, 10, sample_d_model)
    logits_10 = head(encoder_10)
    assert logits_10.shape == (sample_batch_size, 10, 3)

    # Test seq_len=100
    encoder_100 = torch.randn(sample_batch_size, 100, sample_d_model)
    logits_100 = head(encoder_100)
    assert logits_100.shape == (sample_batch_size, 100, 3)


def spec_token_classification_different_batch_sizes(sample_d_model):
    """Test with different batch sizes."""
    # Arrange
    head = TokenClassificationHead(d_model=sample_d_model, num_labels=4)

    # Test batch_size=1
    encoder_1 = torch.randn(1, 10, sample_d_model)
    logits_1 = head(encoder_1)
    assert logits_1.shape == (1, 10, 4)

    # Test batch_size=8
    encoder_8 = torch.randn(8, 10, sample_d_model)
    logits_8 = head(encoder_8)
    assert logits_8.shape == (8, 10, 4)


def spec_token_classification_returns_logits(
    sample_encoder_output, sample_attention_mask
):
    """Verify that head returns logits per token (no activation applied)."""
    # Arrange
    head = TokenClassificationHead(d_model=128, num_labels=3)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    # Logits should not be in [0, 1] range (would indicate softmax applied)
    # They should be unbounded
    assert torch.any(logits < 0) or torch.any(
        logits > 1
    )  # At least some values outside [0,1]
    # Apply softmax per token to verify it works
    probs = torch.softmax(logits, dim=-1)  # [batch, seq_len, num_labels]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0], probs.shape[1]))


def spec_token_classification_no_pooling(sample_encoder_output, sample_attention_mask):
    """Verify that token classification doesn't use pooling (uses all tokens)."""
    # Arrange
    head = TokenClassificationHead(d_model=128, num_labels=5)

    # Act
    logits = head(sample_encoder_output, sample_attention_mask)

    # Assert
    # Output should have same sequence length as input (no pooling applied)
    assert logits.shape[1] == sample_encoder_output.shape[1]
    # Verify that all tokens are processed (not just first token like CLS pooling)
    # If pooling was used, output would be [batch, num_labels], not [batch, seq_len, num_labels]
    assert len(logits.shape) == 3  # [batch, seq_len, num_labels]


def spec_token_classification_per_token_processing(sample_encoder_output):
    """Verify that MLP is applied per-token correctly."""
    # Arrange
    head = TokenClassificationHead(d_model=128, num_labels=3, hidden_dims=[256, 128])

    # Act
    logits = head(sample_encoder_output)

    # Assert
    # Each token should have independent logits
    assert logits.shape == (
        sample_encoder_output.shape[0],
        sample_encoder_output.shape[1],
        3,
    )
    # All tokens should produce valid logits
    assert not torch.any(torch.isnan(logits))
    assert not torch.any(torch.isinf(logits))
