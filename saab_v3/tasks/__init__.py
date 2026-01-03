"""Task heads for downstream tasks on top of encoder models."""

from saab_v3.tasks.base import BaseTaskHead
from saab_v3.tasks.classification import ClassificationHead
from saab_v3.tasks.pooling import CLSPooling, MaxPooling, MeanPooling
from saab_v3.tasks.ranking import PairwiseRankingHead
from saab_v3.tasks.regression import RegressionHead
from saab_v3.tasks.token_classification import TokenClassificationHead

__all__ = [
    "BaseTaskHead",
    "CLSPooling",
    "MeanPooling",
    "MaxPooling",
    "ClassificationHead",
    "RegressionHead",
    "TokenClassificationHead",
    "PairwiseRankingHead",
]

