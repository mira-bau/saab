"""Task heads for downstream tasks on top of encoder models."""

from saab_v3.tasks.base import BaseTaskHead
from saab_v3.tasks.classification import ClassificationHead
from saab_v3.tasks.config import (
    ClassificationTaskConfig,
    RankingTaskConfig,
    RegressionTaskConfig,
    TaskConfig,
    TokenClassificationTaskConfig,
)
from saab_v3.tasks.config_schema import RankingMethod, TaskName, validate_task_config
from saab_v3.tasks.factory import create_task_head_from_config
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
    "create_task_head_from_config",
    "validate_task_config",
    "TaskName",
    "RankingMethod",
    "TaskConfig",
    "ClassificationTaskConfig",
    "RankingTaskConfig",
    "RegressionTaskConfig",
    "TokenClassificationTaskConfig",
]
