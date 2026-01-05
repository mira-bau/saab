"""Factory function for creating task-specific evaluators."""

import torch

from saab_v3.training.evaluators.classification import ClassificationEvaluator
from saab_v3.training.evaluators.ranking import RankingEvaluator
from saab_v3.training.evaluators.regression import RegressionEvaluator
from saab_v3.training.evaluators.token_classification import (
    TokenClassificationEvaluator,
)
from saab_v3.utils.device import get_device


def create_evaluator(
    task_type: str,
    device: str | torch.device | None = None,
    **kwargs,
):
    """Factory function to create task-specific evaluator.

    Args:
        task_type: Task type ("classification", "ranking", "regression", "token_classification")
        device: Device string ("cpu", "cuda", "mps", "auto") or torch.device.
                If None, defaults to "cpu".
        **kwargs: Task-specific parameters:
            - classification: num_classes, multi_label
            - ranking: (no additional params)
            - regression: (no additional params)
            - token_classification: num_labels

    Returns:
        BaseEvaluator instance configured for the task.

    Raises:
        ValueError: If task_type is not recognized.
    """
    # Convert device string to torch.device
    device_obj = get_device(device) if device is not None else torch.device("cpu")

    if task_type == "classification":
        return ClassificationEvaluator(device=device_obj, **kwargs)
    elif task_type == "ranking":
        return RankingEvaluator(device=device_obj, **kwargs)
    elif task_type == "regression":
        return RegressionEvaluator(device=device_obj, **kwargs)
    elif task_type == "token_classification":
        return TokenClassificationEvaluator(device=device_obj, **kwargs)
    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Must be one of: 'classification', 'ranking', 'regression', 'token_classification'"
        )
