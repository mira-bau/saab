"""Loss function factory for different task types."""

import torch
import torch.nn as nn


def create_loss_fn(task_type: str, **kwargs) -> nn.Module:
    """Create loss function for task type.

    Args:
        task_type: Task type ("classification", "regression", "ranking", "token_classification")
        **kwargs: Task-specific parameters
            - classification: num_classes, multi_label, weight, reduction, label_smoothing, pos_weight
            - regression: reduction
            - ranking: method ("hinge", "logistic", "margin"), margin, reduction
            - token_classification: num_labels, weight, reduction, label_smoothing

    Returns:
        Loss function module

    Raises:
        ValueError: If task_type is not supported

    Examples:
        >>> # Classification loss (multi-class)
        >>> loss_fn = create_loss_fn("classification", num_classes=10)
        >>>
        >>> # Classification loss (multi-label)
        >>> loss_fn = create_loss_fn("classification", num_classes=10, multi_label=True)
        >>>
        >>> # Regression loss
        >>> loss_fn = create_loss_fn("regression")
        >>>
        >>> # Ranking loss (hinge)
        >>> loss_fn = create_loss_fn("ranking", method="hinge", margin=1.0)
        >>>
        >>> # Ranking loss (logistic)
        >>> loss_fn = create_loss_fn("ranking", method="logistic")
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
        return create_ranking_loss(**kwargs)
    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Supported types: 'classification', 'regression', 'token_classification', 'ranking'"
        )


def create_classification_loss(
    num_classes: int | None = None,
    multi_label: bool = False,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    pos_weight: torch.Tensor | None = None,
) -> nn.Module:
    """Create classification loss function.

    Args:
        num_classes: Number of classes (for validation)
        multi_label: If True, uses BCEWithLogitsLoss (multi-label).
                     If False, uses CrossEntropyLoss (binary/multi-class).
        weight: Optional class weights tensor
        reduction: Reduction method ("mean", "sum", "none")
        label_smoothing: Label smoothing factor (0.0 = no smoothing, only for CrossEntropyLoss)
        pos_weight: Optional positive class weights for multi-label (BCEWithLogitsLoss)

    Returns:
        CrossEntropyLoss or BCEWithLogitsLoss instance
    """
    if multi_label:
        # Multi-label classification: use BCEWithLogitsLoss
        return nn.BCEWithLogitsLoss(
            weight=weight,
            pos_weight=pos_weight,
            reduction=reduction,
        )
    else:
        # Binary/multi-class classification: use CrossEntropyLoss
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


class PairwiseHingeRankingLoss(nn.Module):
    """Pairwise ranking loss with hinge (margin) formulation.

    Formula: L = max(0, margin - (score_positive - score_negative))

    Use Case: Pairwise ranking where positive should score higher than negative.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        """Initialize pairwise hinge ranking loss.

        Args:
            margin: Margin for the hinge loss (default: 1.0)
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        score_positive: torch.Tensor,  # [batch]
        score_negative: torch.Tensor,  # [batch]
    ) -> torch.Tensor:
        """Compute pairwise hinge ranking loss.

        Args:
            score_positive: Scores for positive examples [batch]
            score_negative: Scores for negative examples [batch]

        Returns:
            Loss tensor (scalar if reduction="mean"/"sum", [batch] if "none")
        """
        # Compute margin violation: margin - (score_pos - score_neg)
        violation = self.margin - (score_positive - score_negative)
        # Hinge: max(0, violation)
        loss = torch.clamp(violation, min=0.0)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


class PairwiseLogisticRankingLoss(nn.Module):
    """Pairwise ranking loss with logistic (smooth) formulation.

    Formula: L = log(1 + exp(-(score_positive - score_negative)))

    Use Case: Pairwise ranking (smooth alternative to hinge loss).
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize pairwise logistic ranking loss.

        Args:
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        score_positive: torch.Tensor,  # [batch]
        score_negative: torch.Tensor,  # [batch]
    ) -> torch.Tensor:
        """Compute pairwise logistic ranking loss.

        Args:
            score_positive: Scores for positive examples [batch]
            score_negative: Scores for negative examples [batch]

        Returns:
            Loss tensor (scalar if reduction="mean"/"sum", [batch] if "none")
        """
        # Compute score difference
        diff = score_positive - score_negative
        # Logistic loss: log(1 + exp(-diff))
        # Use log1p(exp(-diff)) for numerical stability
        loss = torch.log1p(torch.exp(-diff))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


def create_ranking_loss(
    method: str = "hinge",
    margin: float = 1.0,
    reduction: str = "mean",
) -> nn.Module:
    """Create ranking loss function.

    Args:
        method: Ranking loss method ("hinge", "logistic", "margin")
        margin: Margin for hinge/margin losses (default: 1.0)
        reduction: Reduction method ("mean", "sum", "none")

    Returns:
        Ranking loss module

    Raises:
        ValueError: If method is not supported
    """
    method = method.lower()

    if method == "hinge":
        return PairwiseHingeRankingLoss(margin=margin, reduction=reduction)
    elif method == "logistic":
        return PairwiseLogisticRankingLoss(reduction=reduction)
    elif method == "margin":
        # PyTorch's built-in MarginRankingLoss (for backward compatibility)
        return nn.MarginRankingLoss(margin=margin, reduction=reduction)
    else:
        raise ValueError(
            f"Unknown ranking loss method: {method}. "
            f"Supported methods: 'hinge', 'logistic', 'margin'"
        )
