"""Specs for evaluators - happy path only."""

import pytest
import torch

from saab_v3.training.evaluators.classification import ClassificationEvaluator
from saab_v3.training.evaluators.ranking import RankingEvaluator
from saab_v3.training.evaluators.regression import RegressionEvaluator
from saab_v3.training.evaluators.token_classification import TokenClassificationEvaluator


# ============================================================================
# Classification Evaluator Specs
# ============================================================================


def spec_classification_evaluator_initialization():
    """Verify ClassificationEvaluator initializes correctly."""
    # Act
    evaluator = ClassificationEvaluator(
        num_classes=10, multi_label=False, device="cpu"
    )

    # Assert
    assert evaluator.num_classes == 10
    assert evaluator.multi_label is False
    assert evaluator.device.type == "cpu"


def spec_classification_evaluator_reset():
    """Verify ClassificationEvaluator reset clears accumulators."""
    # Arrange
    evaluator = ClassificationEvaluator(num_classes=3, device="cpu")
    outputs = torch.randn(2, 3)  # [batch, num_classes]
    labels = torch.tensor([0, 1])

    # Act
    evaluator.accumulate_batch(outputs, labels)
    evaluator.reset()

    # Assert
    assert len(evaluator._all_predictions) == 0
    assert len(evaluator._all_labels) == 0
    assert evaluator._total_correct == 0
    assert evaluator._total_samples == 0


def spec_classification_evaluator_accumulate_batch():
    """Verify ClassificationEvaluator accumulates batches correctly."""
    # Arrange
    evaluator = ClassificationEvaluator(num_classes=3, device="cpu")
    outputs = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]])  # [batch, num_classes]
    labels = torch.tensor([0, 1])  # Correct predictions

    # Act
    evaluator.accumulate_batch(outputs, labels)

    # Assert
    assert len(evaluator._all_predictions) == 1
    assert len(evaluator._all_labels) == 1
    assert evaluator._total_samples == 2


def spec_classification_evaluator_compute_batch_metrics():
    """Verify ClassificationEvaluator computes batch metrics."""
    # Arrange
    evaluator = ClassificationEvaluator(num_classes=3, device="cpu")
    outputs = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]])
    labels = torch.tensor([0, 1])  # All correct

    # Act
    metrics = evaluator.compute_batch_metrics(outputs, labels, ["accuracy"])

    # Assert
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0  # All correct


def spec_classification_evaluator_compute_aggregated_metrics():
    """Verify ClassificationEvaluator computes aggregated metrics."""
    # Arrange
    evaluator = ClassificationEvaluator(num_classes=3, device="cpu")
    outputs = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]])
    labels = torch.tensor([0, 1])

    # Act
    evaluator.accumulate_batch(outputs, labels)
    metrics = evaluator.compute_aggregated_metrics(["accuracy"])

    # Assert
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def spec_classification_evaluator_multilabel():
    """Verify ClassificationEvaluator handles multi-label correctly."""
    # Arrange
    evaluator = ClassificationEvaluator(num_classes=3, multi_label=True, device="cpu")
    outputs = torch.randn(2, 3)
    labels = torch.tensor([0, 1])

    # Act
    predictions = evaluator._get_predictions(outputs)

    # Assert
    assert predictions.shape == (2,)
    assert predictions.dtype == torch.long


# ============================================================================
# Ranking Evaluator Specs
# ============================================================================


def spec_ranking_evaluator_initialization():
    """Verify RankingEvaluator initializes correctly."""
    # Act
    evaluator = RankingEvaluator(device="cpu")

    # Assert
    assert evaluator.device.type == "cpu"


def spec_ranking_evaluator_accumulate_batch():
    """Verify RankingEvaluator accumulates batches correctly."""
    # Arrange
    evaluator = RankingEvaluator(device="cpu")
    outputs = torch.tensor([1.5, -0.5, 2.0])  # Scores
    labels = torch.tensor([1, 0, 1])  # 1 = first better, 0 = second better

    # Act
    evaluator.accumulate_batch(outputs, labels)

    # Assert
    assert evaluator._total_pairs == 3


def spec_ranking_evaluator_compute_aggregated_metrics():
    """Verify RankingEvaluator computes aggregated metrics."""
    # Arrange
    evaluator = RankingEvaluator(device="cpu")
    outputs = torch.tensor([1.5, -0.5, 2.0])
    labels = torch.tensor([1, 0, 1])

    # Act
    evaluator.accumulate_batch(outputs, labels)
    metrics = evaluator.compute_aggregated_metrics(["pairwise_accuracy"])

    # Assert
    assert "pairwise_accuracy" in metrics
    assert 0.0 <= metrics["pairwise_accuracy"] <= 1.0


# ============================================================================
# Regression Evaluator Specs
# ============================================================================


def spec_regression_evaluator_initialization():
    """Verify RegressionEvaluator initializes correctly."""
    # Act
    evaluator = RegressionEvaluator(device="cpu")

    # Assert
    assert evaluator.device.type == "cpu"


def spec_regression_evaluator_accumulate_batch():
    """Verify RegressionEvaluator accumulates batches correctly."""
    # Arrange
    evaluator = RegressionEvaluator(device="cpu")
    outputs = torch.tensor([1.5, 2.0, 3.0])
    labels = torch.tensor([1.0, 2.0, 3.0])

    # Act
    evaluator.accumulate_batch(outputs, labels)

    # Assert
    assert evaluator._total_samples == 3
    assert evaluator._sum_squared_error >= 0


def spec_regression_evaluator_compute_batch_metrics():
    """Verify RegressionEvaluator computes batch metrics."""
    # Arrange
    evaluator = RegressionEvaluator(device="cpu")
    outputs = torch.tensor([1.0, 2.0, 3.0])
    labels = torch.tensor([1.0, 2.0, 3.0])  # Perfect predictions

    # Act
    metrics = evaluator.compute_batch_metrics(outputs, labels, ["mse", "mae"])

    # Assert
    assert "mse" in metrics
    assert "mae" in metrics
    assert metrics["mse"] == 0.0  # Perfect predictions
    assert metrics["mae"] == 0.0


def spec_regression_evaluator_compute_aggregated_metrics():
    """Verify RegressionEvaluator computes aggregated metrics."""
    # Arrange
    evaluator = RegressionEvaluator(device="cpu")
    outputs = torch.tensor([1.0, 2.0, 3.0])
    labels = torch.tensor([1.0, 2.0, 3.0])

    # Act
    evaluator.accumulate_batch(outputs, labels)
    metrics = evaluator.compute_aggregated_metrics(["mse", "mae", "rmse"])

    # Assert
    assert "mse" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


# ============================================================================
# Token Classification Evaluator Specs
# ============================================================================


def spec_token_classification_evaluator_initialization():
    """Verify TokenClassificationEvaluator initializes correctly."""
    # Act
    evaluator = TokenClassificationEvaluator(num_labels=5, device="cpu")

    # Assert
    assert evaluator.num_labels == 5
    assert evaluator.device.type == "cpu"


def spec_token_classification_evaluator_accumulate_batch():
    """Verify TokenClassificationEvaluator accumulates batches correctly."""
    # Arrange
    evaluator = TokenClassificationEvaluator(num_labels=3, device="cpu")
    batch_size, seq_len = 2, 5
    outputs = torch.randn(batch_size, seq_len, 3)  # [batch, seq, num_labels]
    labels = torch.randint(0, 3, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Act
    evaluator.accumulate_batch(outputs, labels, attention_mask)

    # Assert
    assert evaluator._total_valid_tokens > 0
    assert evaluator._total_sequences == batch_size


def spec_token_classification_evaluator_compute_batch_metrics():
    """Verify TokenClassificationEvaluator computes batch metrics."""
    # Arrange
    evaluator = TokenClassificationEvaluator(num_labels=3, device="cpu")
    batch_size, seq_len = 2, 5
    outputs = torch.randn(batch_size, seq_len, 3)
    labels = torch.randint(0, 3, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Act
    metrics = evaluator.compute_batch_metrics(
        outputs, labels, ["token_accuracy"], attention_mask
    )

    # Assert
    assert "token_accuracy" in metrics
    assert 0.0 <= metrics["token_accuracy"] <= 1.0


def spec_token_classification_evaluator_compute_aggregated_metrics():
    """Verify TokenClassificationEvaluator computes aggregated metrics."""
    # Arrange
    evaluator = TokenClassificationEvaluator(num_labels=3, device="cpu")
    batch_size, seq_len = 2, 5
    outputs = torch.randn(batch_size, seq_len, 3)
    labels = torch.randint(0, 3, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Act
    evaluator.accumulate_batch(outputs, labels, attention_mask)
    metrics = evaluator.compute_aggregated_metrics(["token_accuracy", "exact_match"])

    # Assert
    assert "token_accuracy" in metrics
    assert "exact_match" in metrics
    assert 0.0 <= metrics["token_accuracy"] <= 1.0
    assert 0.0 <= metrics["exact_match"] <= 1.0


def spec_token_classification_evaluator_with_padding():
    """Verify TokenClassificationEvaluator handles padding correctly."""
    # Arrange
    evaluator = TokenClassificationEvaluator(num_labels=3, device="cpu")
    batch_size, seq_len = 2, 5
    outputs = torch.randn(batch_size, seq_len, 3)
    labels = torch.randint(0, 3, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[:, -2:] = 0  # Last 2 tokens are padding

    # Act
    evaluator.accumulate_batch(outputs, labels, attention_mask)

    # Assert
    # Should only count valid tokens (not padding)
    assert evaluator._total_valid_tokens == batch_size * (seq_len - 2)

