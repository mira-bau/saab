"""Specs for evaluator factory - happy path only."""

import pytest
import torch

from saab_v3.training.evaluator import create_evaluator
from saab_v3.training.evaluators.classification import ClassificationEvaluator
from saab_v3.training.evaluators.ranking import RankingEvaluator
from saab_v3.training.evaluators.regression import RegressionEvaluator
from saab_v3.training.evaluators.token_classification import TokenClassificationEvaluator


def spec_evaluator_factory_classification():
    """Verify factory creates ClassificationEvaluator correctly."""
    # Act
    evaluator = create_evaluator(
        task_type="classification",
        num_classes=10,
        multi_label=False,
        device="cpu",
    )

    # Assert
    assert isinstance(evaluator, ClassificationEvaluator)
    assert evaluator.num_classes == 10
    assert evaluator.multi_label is False
    assert evaluator.device.type == "cpu"


def spec_evaluator_factory_classification_multilabel():
    """Verify factory creates ClassificationEvaluator for multi-label."""
    # Act
    evaluator = create_evaluator(
        task_type="classification",
        num_classes=5,
        multi_label=True,
        device="cpu",
    )

    # Assert
    assert isinstance(evaluator, ClassificationEvaluator)
    assert evaluator.multi_label is True


def spec_evaluator_factory_ranking():
    """Verify factory creates RankingEvaluator correctly."""
    # Act
    evaluator = create_evaluator(task_type="ranking", device="cpu")

    # Assert
    assert isinstance(evaluator, RankingEvaluator)
    assert evaluator.device.type == "cpu"


def spec_evaluator_factory_regression():
    """Verify factory creates RegressionEvaluator correctly."""
    # Act
    evaluator = create_evaluator(task_type="regression", device="cpu")

    # Assert
    assert isinstance(evaluator, RegressionEvaluator)
    assert evaluator.device.type == "cpu"


def spec_evaluator_factory_token_classification():
    """Verify factory creates TokenClassificationEvaluator correctly."""
    # Act
    evaluator = create_evaluator(
        task_type="token_classification",
        num_labels=5,
        device="cpu",
    )

    # Assert
    assert isinstance(evaluator, TokenClassificationEvaluator)
    assert evaluator.num_labels == 5
    assert evaluator.device.type == "cpu"


def spec_evaluator_factory_device_string():
    """Verify factory handles device string correctly."""
    # Act
    evaluator = create_evaluator(
        task_type="classification",
        num_classes=3,
        device="cpu",
    )

    # Assert
    assert evaluator.device.type == "cpu"


def spec_evaluator_factory_device_torch_device():
    """Verify factory handles torch.device correctly."""
    # Act
    evaluator = create_evaluator(
        task_type="classification",
        num_classes=3,
        device=torch.device("cpu"),
    )

    # Assert
    assert evaluator.device.type == "cpu"


def spec_evaluator_factory_invalid_task_type():
    """Verify factory raises error for invalid task type."""
    # Act & Assert
    with pytest.raises(ValueError, match="Unknown task_type"):
        create_evaluator(task_type="invalid_task", device="cpu")

