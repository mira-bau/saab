"""Tests for loss function factory and loss modules."""

import pytest
import torch
import torch.nn as nn

from saab_v3.training.loss import (
    create_loss_fn,
    create_classification_loss,
    create_ranking_loss,
    create_regression_loss,
    create_token_classification_loss,
    PairwiseHingeRankingLoss,
    PairwiseLogisticRankingLoss,
)


# ============================================================================
# Classification Loss Tests
# ============================================================================


def spec_classification_binary():
    """Test binary classification loss (CrossEntropyLoss)."""
    # Arrange
    loss_fn = create_classification_loss(num_classes=2, multi_label=False)
    batch_size = 4
    logits = torch.randn(batch_size, 2)
    labels = torch.randint(0, 2, (batch_size,))

    # Act
    loss = loss_fn(logits, labels)

    # Assert
    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    assert loss.shape == ()  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0


def spec_classification_multiclass():
    """Test multi-class classification loss (CrossEntropyLoss)."""
    # Arrange
    loss_fn = create_classification_loss(num_classes=10, multi_label=False)
    batch_size = 4
    logits = torch.randn(batch_size, 10)
    labels = torch.randint(0, 10, (batch_size,))

    # Act
    loss = loss_fn(logits, labels)

    # Assert
    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    assert loss.shape == ()  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def spec_classification_multilabel():
    """Test multi-label classification loss (BCEWithLogitsLoss)."""
    # Arrange
    loss_fn = create_classification_loss(num_classes=10, multi_label=True)
    batch_size = 4
    logits = torch.randn(batch_size, 10)
    labels = torch.randint(0, 2, (batch_size, 10)).float()

    # Act
    loss = loss_fn(logits, labels)

    # Assert
    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)
    assert loss.shape == ()  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def spec_classification_class_weights():
    """Test classification loss with class weights."""
    # Arrange
    weight = torch.tensor([0.5, 1.0, 2.0])
    loss_fn = create_classification_loss(
        num_classes=3, weight=weight, multi_label=False
    )
    batch_size = 4
    logits = torch.randn(batch_size, 3)
    labels = torch.randint(0, 3, (batch_size,))

    # Act
    loss = loss_fn(logits, labels)

    # Assert
    assert loss.shape == ()
    assert not torch.isnan(loss)


def spec_classification_label_smoothing():
    """Test classification loss with label smoothing."""
    # Arrange
    loss_fn = create_classification_loss(
        num_classes=5, label_smoothing=0.1, multi_label=False
    )
    batch_size = 4
    logits = torch.randn(batch_size, 5)
    labels = torch.randint(0, 5, (batch_size,))

    # Act
    loss = loss_fn(logits, labels)

    # Assert
    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def spec_classification_pos_weight():
    """Test multi-label classification loss with positive class weights."""
    # Arrange
    pos_weight = torch.tensor([1.0, 2.0, 0.5, 1.5, 1.0])
    loss_fn = create_classification_loss(
        num_classes=5, multi_label=True, pos_weight=pos_weight
    )
    batch_size = 4
    logits = torch.randn(batch_size, 5)
    labels = torch.randint(0, 2, (batch_size, 5)).float()

    # Act
    loss = loss_fn(logits, labels)

    # Assert
    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def spec_classification_reduction_modes():
    """Test classification loss with different reduction modes."""
    batch_size = 4
    logits = torch.randn(batch_size, 3)
    labels = torch.randint(0, 3, (batch_size,))

    # Mean reduction (default)
    loss_mean = create_classification_loss(reduction="mean", multi_label=False)
    loss = loss_mean(logits, labels)
    assert loss.shape == ()

    # Sum reduction
    loss_sum = create_classification_loss(reduction="sum", multi_label=False)
    loss = loss_sum(logits, labels)
    assert loss.shape == ()

    # None reduction
    loss_none = create_classification_loss(reduction="none", multi_label=False)
    loss = loss_none(logits, labels)
    assert loss.shape == (batch_size,)


def spec_classification_multilabel_reduction_modes():
    """Test multi-label classification loss with different reduction modes."""
    batch_size = 4
    logits = torch.randn(batch_size, 5)
    labels = torch.randint(0, 2, (batch_size, 5)).float()

    # Mean reduction (default)
    loss_mean = create_classification_loss(reduction="mean", multi_label=True)
    loss = loss_mean(logits, labels)
    assert loss.shape == ()

    # Sum reduction
    loss_sum = create_classification_loss(reduction="sum", multi_label=True)
    loss = loss_sum(logits, labels)
    assert loss.shape == ()

    # None reduction
    loss_none = create_classification_loss(reduction="none", multi_label=True)
    loss = loss_none(logits, labels)
    assert loss.shape == (batch_size, 5)


# ============================================================================
# Ranking Loss Tests
# ============================================================================


def spec_ranking_hinge():
    """Test pairwise hinge ranking loss."""
    # Arrange
    loss_fn = PairwiseHingeRankingLoss(margin=1.0, reduction="mean")
    batch_size = 4
    score_positive = torch.randn(batch_size) + 2.0  # Higher scores
    score_negative = torch.randn(batch_size) - 2.0  # Lower scores

    # Act
    loss = loss_fn(score_positive, score_negative)

    # Assert
    assert loss.shape == ()  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0  # Loss should be non-negative


def spec_ranking_hinge_different_margins():
    """Test hinge ranking loss with different margins."""
    batch_size = 4
    score_positive = torch.randn(batch_size)
    score_negative = torch.randn(batch_size) - 1.0

    # Small margin
    loss_small = PairwiseHingeRankingLoss(margin=0.5, reduction="mean")
    loss1 = loss_small(score_positive, score_negative)

    # Large margin
    loss_large = PairwiseHingeRankingLoss(margin=2.0, reduction="mean")
    loss2 = loss_large(score_positive, score_negative)

    # Assert
    assert loss1.shape == ()
    assert loss2.shape == ()
    # Larger margin should generally produce larger loss when violated
    assert not torch.isnan(loss1)
    assert not torch.isnan(loss2)


def spec_ranking_hinge_zero_loss():
    """Test hinge ranking loss when margin is satisfied (zero loss)."""
    # Arrange
    loss_fn = PairwiseHingeRankingLoss(margin=1.0, reduction="mean")
    batch_size = 4
    score_positive = torch.ones(batch_size) * 5.0  # Much higher
    score_negative = torch.ones(batch_size) * 1.0  # Much lower

    # Act
    loss = loss_fn(score_positive, score_negative)

    # Assert
    # When score_pos - score_neg > margin, loss should be 0
    assert loss.shape == ()
    assert loss.item() == 0.0


def spec_ranking_logistic():
    """Test pairwise logistic ranking loss."""
    # Arrange
    loss_fn = PairwiseLogisticRankingLoss(reduction="mean")
    batch_size = 4
    score_positive = torch.randn(batch_size) + 2.0
    score_negative = torch.randn(batch_size) - 2.0

    # Act
    loss = loss_fn(score_positive, score_negative)

    # Assert
    assert loss.shape == ()  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0  # Should always be positive


def spec_ranking_logistic_equal_scores():
    """Test logistic ranking loss when scores are equal."""
    # Arrange
    loss_fn = PairwiseLogisticRankingLoss(reduction="mean")
    batch_size = 4
    scores = torch.randn(batch_size)

    # Act
    loss = loss_fn(scores, scores)  # Equal scores

    # Assert
    # When diff = 0, loss = log(1 + exp(0)) = log(2) ≈ 0.693
    expected = torch.log(torch.tensor(2.0))
    assert loss.shape == ()
    assert torch.allclose(loss, expected, atol=1e-5)


def spec_ranking_margin():
    """Test PyTorch's built-in margin ranking loss."""
    # Arrange
    loss_fn = create_ranking_loss(method="margin", margin=1.0, reduction="mean")
    batch_size = 4
    score_positive = torch.randn(batch_size)
    score_negative = torch.randn(batch_size) - 1.0
    targets = torch.ones(batch_size)  # 1 means score_pos should be higher

    # Act
    loss = loss_fn(score_positive, score_negative, targets)

    # Assert
    assert isinstance(loss_fn, nn.MarginRankingLoss)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def spec_ranking_invalid_method():
    """Test error handling for invalid ranking loss method."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="Unknown ranking loss method"):
        create_ranking_loss(method="invalid_method")


def spec_ranking_reduction_modes():
    """Test ranking losses with different reduction modes."""
    batch_size = 4
    score_positive = torch.randn(batch_size)
    score_negative = torch.randn(batch_size) - 1.0

    # Hinge - mean
    loss_hinge_mean = PairwiseHingeRankingLoss(reduction="mean")
    loss = loss_hinge_mean(score_positive, score_negative)
    assert loss.shape == ()

    # Hinge - sum
    loss_hinge_sum = PairwiseHingeRankingLoss(reduction="sum")
    loss = loss_hinge_sum(score_positive, score_negative)
    assert loss.shape == ()

    # Hinge - none
    loss_hinge_none = PairwiseHingeRankingLoss(reduction="none")
    loss = loss_hinge_none(score_positive, score_negative)
    assert loss.shape == (batch_size,)

    # Logistic - mean
    loss_logistic_mean = PairwiseLogisticRankingLoss(reduction="mean")
    loss = loss_logistic_mean(score_positive, score_negative)
    assert loss.shape == ()

    # Logistic - sum
    loss_logistic_sum = PairwiseLogisticRankingLoss(reduction="sum")
    loss = loss_logistic_sum(score_positive, score_negative)
    assert loss.shape == ()

    # Logistic - none
    loss_logistic_none = PairwiseLogisticRankingLoss(reduction="none")
    loss = loss_logistic_none(score_positive, score_negative)
    assert loss.shape == (batch_size,)


def spec_ranking_numerical_stability():
    """Test ranking losses for numerical stability."""
    # Arrange
    loss_hinge = PairwiseHingeRankingLoss(reduction="mean")
    loss_logistic = PairwiseLogisticRankingLoss(reduction="mean")

    # Large score differences
    score_positive = torch.ones(4) * 100.0
    score_negative = torch.ones(4) * -100.0

    # Act
    loss_h = loss_hinge(score_positive, score_negative)
    loss_l = loss_logistic(score_positive, score_negative)

    # Assert
    assert not torch.isnan(loss_h)
    assert not torch.isinf(loss_h)
    assert not torch.isnan(loss_l)
    assert not torch.isinf(loss_l)


# ============================================================================
# Regression Loss Tests
# ============================================================================


def spec_regression_mse():
    """Test MSE regression loss."""
    # Arrange
    loss_fn = create_regression_loss(reduction="mean")
    batch_size = 4
    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)

    # Act
    loss = loss_fn(predictions, targets)

    # Assert
    assert isinstance(loss_fn, nn.MSELoss)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def spec_regression_reduction_modes():
    """Test regression loss with different reduction modes."""
    batch_size = 4
    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)

    # Mean reduction
    loss_mean = create_regression_loss(reduction="mean")
    loss = loss_mean(predictions, targets)
    assert loss.shape == ()

    # Sum reduction
    loss_sum = create_regression_loss(reduction="sum")
    loss = loss_sum(predictions, targets)
    assert loss.shape == ()

    # None reduction
    loss_none = create_regression_loss(reduction="none")
    loss = loss_none(predictions, targets)
    assert loss.shape == (batch_size, 1)


# ============================================================================
# Token Classification Loss Tests
# ============================================================================


def spec_token_classification_basic():
    """Test token classification loss (CrossEntropyLoss)."""
    # Arrange
    loss_fn = create_token_classification_loss(num_labels=5, reduction="mean")
    batch_size = 4
    seq_len = 10
    logits = torch.randn(batch_size, seq_len, 5)
    labels = torch.randint(0, 5, (batch_size, seq_len))

    # Act
    loss = loss_fn(logits.view(-1, 5), labels.view(-1))

    # Assert
    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def spec_token_classification_class_weights():
    """Test token classification loss with class weights."""
    # Arrange
    weight = torch.tensor([0.5, 1.0, 2.0, 1.5, 1.0])
    loss_fn = create_token_classification_loss(num_labels=5, weight=weight)
    batch_size = 4
    seq_len = 10
    logits = torch.randn(batch_size, seq_len, 5)
    labels = torch.randint(0, 5, (batch_size, seq_len))

    # Act
    loss = loss_fn(logits.view(-1, 5), labels.view(-1))

    # Assert
    assert loss.shape == ()
    assert not torch.isnan(loss)


def spec_token_classification_label_smoothing():
    """Test token classification loss with label smoothing."""
    # Arrange
    loss_fn = create_token_classification_loss(num_labels=5, label_smoothing=0.1)
    batch_size = 4
    seq_len = 10
    logits = torch.randn(batch_size, seq_len, 5)
    labels = torch.randint(0, 5, (batch_size, seq_len))

    # Act
    loss = loss_fn(logits.view(-1, 5), labels.view(-1))

    # Assert
    assert loss.shape == ()
    assert not torch.isnan(loss)


def spec_token_classification_reduction_modes():
    """Test token classification loss with different reduction modes."""
    batch_size = 4
    seq_len = 10
    logits = torch.randn(batch_size, seq_len, 5)
    labels = torch.randint(0, 5, (batch_size, seq_len))

    # Mean reduction
    loss_mean = create_token_classification_loss(reduction="mean")
    loss = loss_mean(logits.view(-1, 5), labels.view(-1))
    assert loss.shape == ()

    # Sum reduction
    loss_sum = create_token_classification_loss(reduction="sum")
    loss = loss_sum(logits.view(-1, 5), labels.view(-1))
    assert loss.shape == ()

    # None reduction
    loss_none = create_token_classification_loss(reduction="none")
    loss = loss_none(logits.view(-1, 5), labels.view(-1))
    assert loss.shape == (batch_size * seq_len,)


# ============================================================================
# Factory Function Tests
# ============================================================================


def spec_factory_classification():
    """Test factory function for classification loss."""
    # Multi-class
    loss_fn = create_loss_fn("classification", num_classes=10)
    assert isinstance(loss_fn, nn.CrossEntropyLoss)

    # Multi-label
    loss_fn = create_loss_fn("classification", num_classes=10, multi_label=True)
    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)


def spec_factory_regression():
    """Test factory function for regression loss."""
    loss_fn = create_loss_fn("regression")
    assert isinstance(loss_fn, nn.MSELoss)


def spec_factory_ranking():
    """Test factory function for ranking loss."""
    # Hinge
    loss_fn = create_loss_fn("ranking", method="hinge", margin=1.0)
    assert isinstance(loss_fn, PairwiseHingeRankingLoss)

    # Logistic
    loss_fn = create_loss_fn("ranking", method="logistic")
    assert isinstance(loss_fn, PairwiseLogisticRankingLoss)

    # Margin
    loss_fn = create_loss_fn("ranking", method="margin", margin=1.0)
    assert isinstance(loss_fn, nn.MarginRankingLoss)


def spec_factory_token_classification():
    """Test factory function for token classification loss."""
    loss_fn = create_loss_fn("token_classification", num_labels=5)
    assert isinstance(loss_fn, nn.CrossEntropyLoss)


def spec_factory_invalid_task_type():
    """Test factory function error handling for invalid task type."""
    with pytest.raises(ValueError, match="Unknown task_type"):
        create_loss_fn("invalid_task")


def spec_factory_backward_compatibility():
    """Test that factory function maintains backward compatibility."""
    # Old code should still work
    loss_fn = create_loss_fn("classification", num_classes=10)
    assert isinstance(loss_fn, nn.CrossEntropyLoss)

    loss_fn = create_loss_fn("regression")
    assert isinstance(loss_fn, nn.MSELoss)

    loss_fn = create_loss_fn("token_classification", num_labels=5)
    assert isinstance(loss_fn, nn.CrossEntropyLoss)


def spec_factory_parameter_passing():
    """Test that factory function correctly passes parameters."""
    # Classification with all parameters
    loss_fn = create_loss_fn(
        "classification",
        num_classes=5,
        multi_label=False,
        reduction="sum",
        label_smoothing=0.1,
    )
    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    assert loss_fn.reduction == "sum"
    assert loss_fn.label_smoothing == 0.1

    # Ranking with all parameters
    loss_fn = create_loss_fn("ranking", method="hinge", margin=2.0, reduction="sum")
    assert isinstance(loss_fn, PairwiseHingeRankingLoss)
    assert loss_fn.margin == 2.0
    assert loss_fn.reduction == "sum"
