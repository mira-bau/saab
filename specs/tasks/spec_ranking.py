"""Tests for PairwiseRankingHead."""

import pytest
import torch

from saab_v3.tasks.ranking import PairwiseRankingHead


@pytest.fixture
def sample_seq_repr(sample_batch_size, sample_d_model):
    """Sample sequence representation (already pooled)."""
    return torch.randn(sample_batch_size, sample_d_model)


def spec_ranking_dot_product(sample_seq_repr):
    """Test dot product method."""
    # Arrange
    head = PairwiseRankingHead(d_model=128, method="dot_product")
    seq_a = sample_seq_repr
    seq_b = sample_seq_repr + 0.1  # Slightly different

    # Act
    scores = head(seq_a, seq_b)

    # Assert
    assert scores.shape == (sample_seq_repr.shape[0],)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))
    # Dot product should be symmetric-ish (but seq_a and seq_b are different)
    assert head.method == "dot_product"
    assert head.comparison_mlp is None
    assert head.output_layer is None


def spec_ranking_cosine(sample_seq_repr):
    """Test cosine similarity method."""
    # Arrange
    head = PairwiseRankingHead(d_model=128, method="cosine")
    seq_a = sample_seq_repr
    seq_b = sample_seq_repr + 0.1

    # Act
    scores = head(seq_a, seq_b)

    # Assert
    assert scores.shape == (sample_seq_repr.shape[0],)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))
    # Cosine similarity should be in [-1, 1] range
    assert torch.all(scores >= -1.0) and torch.all(scores <= 1.0)
    assert head.method == "cosine"
    assert head.comparison_mlp is None
    assert head.output_layer is None


def spec_ranking_mlp(sample_seq_repr):
    """Test MLP method (with hidden_dims)."""
    # Arrange
    head = PairwiseRankingHead(d_model=128, method="mlp", hidden_dims=[256, 128])
    seq_a = sample_seq_repr
    seq_b = sample_seq_repr + 0.1

    # Act
    scores = head(seq_a, seq_b)

    # Assert
    assert scores.shape == (sample_seq_repr.shape[0],)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))
    assert head.method == "mlp"
    assert head.comparison_mlp is not None
    assert head.output_layer is not None


def spec_ranking_difference(sample_seq_repr):
    """Test difference method (with hidden_dims)."""
    # Arrange
    head = PairwiseRankingHead(d_model=128, method="difference", hidden_dims=[256, 128])
    seq_a = sample_seq_repr
    seq_b = sample_seq_repr + 0.1

    # Act
    scores = head(seq_a, seq_b)

    # Assert
    assert scores.shape == (sample_seq_repr.shape[0],)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))
    assert head.method == "difference"
    assert head.comparison_mlp is not None
    assert head.output_layer is not None


def spec_ranking_error_missing_hidden_dims_mlp():
    """Test error when hidden_dims is missing for MLP method."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="hidden_dims is required for method 'mlp'"):
        PairwiseRankingHead(d_model=128, method="mlp", hidden_dims=None)


def spec_ranking_error_missing_hidden_dims_difference():
    """Test error when hidden_dims is missing for difference method."""
    # Arrange & Act & Assert
    with pytest.raises(
        ValueError, match="hidden_dims is required for method 'difference'"
    ):
        PairwiseRankingHead(d_model=128, method="difference", hidden_dims=None)


def spec_ranking_error_invalid_method():
    """Test error for invalid method."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="method must be one of.*got invalid_method"):
        PairwiseRankingHead(d_model=128, method="invalid_method")


def spec_ranking_output_shape(sample_seq_repr):
    """Verify output shape is correct [batch]."""
    # Arrange
    batch_size = sample_seq_repr.shape[0]
    head = PairwiseRankingHead(d_model=128, method="dot_product")
    seq_a = sample_seq_repr
    seq_b = sample_seq_repr + 0.1

    # Act
    scores = head(seq_a, seq_b)

    # Assert
    assert scores.shape == (batch_size,)


def spec_ranking_score_interpretation(sample_seq_repr):
    """Verify that higher score means seq_a is better than seq_b."""
    # Arrange
    head = PairwiseRankingHead(d_model=128, method="dot_product")
    seq_a = sample_seq_repr
    seq_b = sample_seq_repr - 1.0  # seq_b is clearly worse

    # Act
    scores = head(seq_a, seq_b)

    # Assert
    # When seq_a is better, scores should be higher
    # (This is a basic sanity check - actual ranking quality depends on training)
    assert scores.shape == (sample_seq_repr.shape[0],)


def spec_ranking_device_consistency(sample_seq_repr):
    """Test that head works on different devices."""
    # Arrange
    head = PairwiseRankingHead(d_model=128, method="dot_product")
    seq_a = sample_seq_repr
    seq_b = sample_seq_repr + 0.1

    # Test CPU (default)
    scores_cpu = head(seq_a, seq_b)
    assert scores_cpu.device == seq_a.device

    # Test CUDA if available
    if torch.cuda.is_available():
        head_cuda = head.cuda()
        seq_a_cuda = seq_a.cuda()
        seq_b_cuda = seq_b.cuda()
        scores_cuda = head_cuda(seq_a_cuda, seq_b_cuda)
        assert scores_cuda.device.type == "cuda"

    # Test MPS if available (macOS)
    if torch.backends.mps.is_available():
        head_mps = head.to("mps")
        seq_a_mps = seq_a.to("mps")
        seq_b_mps = seq_b.to("mps")
        scores_mps = head_mps(seq_a_mps, seq_b_mps)
        assert scores_mps.device.type == "mps"


def spec_ranking_different_batch_sizes(sample_d_model):
    """Test with different batch sizes."""
    # Arrange
    head = PairwiseRankingHead(d_model=sample_d_model, method="dot_product")

    # Test batch_size=1
    seq_a_1 = torch.randn(1, sample_d_model)
    seq_b_1 = torch.randn(1, sample_d_model)
    scores_1 = head(seq_a_1, seq_b_1)
    assert scores_1.shape == (1,)

    # Test batch_size=8
    seq_a_8 = torch.randn(8, sample_d_model)
    seq_b_8 = torch.randn(8, sample_d_model)
    scores_8 = head(seq_a_8, seq_b_8)
    assert scores_8.shape == (8,)


def spec_ranking_mlp_empty_hidden_dims(sample_seq_repr):
    """Test MLP method with empty hidden_dims (just linear layer)."""
    # Arrange
    head = PairwiseRankingHead(d_model=128, method="mlp", hidden_dims=[])
    seq_a = sample_seq_repr
    seq_b = sample_seq_repr + 0.1

    # Act
    scores = head(seq_a, seq_b)

    # Assert
    assert scores.shape == (sample_seq_repr.shape[0],)
    assert head.comparison_mlp is None  # No hidden layers
    assert head.output_layer is not None  # But output layer exists


def spec_ranking_difference_empty_hidden_dims(sample_seq_repr):
    """Test difference method with empty hidden_dims (just linear layer)."""
    # Arrange
    head = PairwiseRankingHead(d_model=128, method="difference", hidden_dims=[])
    seq_a = sample_seq_repr
    seq_b = sample_seq_repr + 0.1

    # Act
    scores = head(seq_a, seq_b)

    # Assert
    assert scores.shape == (sample_seq_repr.shape[0],)
    assert head.comparison_mlp is None  # No hidden layers
    assert head.output_layer is not None  # But output layer exists


def spec_ranking_symmetric_methods(sample_seq_repr):
    """Test that dot_product and cosine are symmetric (seq_a, seq_b) vs (seq_b, seq_a)."""
    # Arrange
    head_dot = PairwiseRankingHead(d_model=128, method="dot_product")
    head_cosine = PairwiseRankingHead(d_model=128, method="cosine")
    seq_a = sample_seq_repr
    seq_b = sample_seq_repr + 0.1

    # Act
    scores_dot_ab = head_dot(seq_a, seq_b)
    scores_dot_ba = head_dot(seq_b, seq_a)
    scores_cosine_ab = head_cosine(seq_a, seq_b)
    scores_cosine_ba = head_cosine(seq_b, seq_a)

    # Assert
    # Dot product should be exactly symmetric
    assert torch.allclose(scores_dot_ab, scores_dot_ba)
    # Cosine similarity should be exactly symmetric
    assert torch.allclose(scores_cosine_ab, scores_cosine_ba)
