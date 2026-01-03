"""Loss function factory for different task types."""

import torch
import torch.nn as nn


def create_loss_fn(task_type: str, **kwargs) -> nn.Module:
    """Create loss function for task type.

    Args:
        task_type: Task type ("classification", "regression", "ranking", "token_classification")
        **kwargs: Task-specific parameters

    Returns:
        Loss function module

    Raises:
        ValueError: If task_type is not supported

    Examples:
        >>> # Classification loss
        >>> loss_fn = create_loss_fn("classification", num_classes=10)
        >>> 
        >>> # Regression loss
        >>> loss_fn = create_loss_fn("regression")
        >>> 
        >>> # Token classification loss
        >>> loss_fn = create_loss_fn("token_classification", num_labels=5)
    """
    task_type = task_type.lower()

    if task_type == "classification":
        return create_classification_loss(**kwargs)
    elif task_type == "regression":
        return create_regression_loss(**kwargs)
    elif task_type == "token_classification":
        return create_token_classification_loss(**kwargs)
    elif task_type == "ranking":
        # Ranking loss will be implemented when task heads are added
        raise NotImplementedError("Ranking loss not yet implemented")
    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Supported types: 'classification', 'regression', 'token_classification', 'ranking'"
        )


def create_classification_loss(
    num_classes: int | None = None,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Create classification loss function.

    Args:
        num_classes: Number of classes (for validation)
        weight: Optional class weights tensor
        reduction: Reduction method ("mean", "sum", "none")
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        CrossEntropyLoss instance
    """
    return nn.CrossEntropyLoss(
        weight=weight,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def create_regression_loss(
    reduction: str = "mean",
) -> nn.Module:
    """Create regression loss function.

    Args:
        reduction: Reduction method ("mean", "sum", "none")

    Returns:
        MSELoss instance
    """
    return nn.MSELoss(reduction=reduction)


def create_token_classification_loss(
    num_labels: int | None = None,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Create token classification loss function.

    Args:
        num_labels: Number of labels (for validation)
        weight: Optional class weights tensor
        reduction: Reduction method ("mean", "sum", "none")
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        CrossEntropyLoss instance (same as classification, but for per-token predictions)
    """
    return nn.CrossEntropyLoss(
        weight=weight,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def create_ranking_loss(
    margin: float = 1.0,
    reduction: str = "mean",
) -> nn.Module:
    """Create ranking loss function (for future use).

    Args:
        margin: Margin for margin ranking loss
        reduction: Reduction method ("mean", "sum", "none")

    Returns:
        MarginRankingLoss instance
    """
    return nn.MarginRankingLoss(margin=margin, reduction=reduction)

