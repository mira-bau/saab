"""Factory function for creating task heads from configuration."""

from typing import Any

import torch.nn as nn

from saab_v3.tasks.base import BaseTaskHead
from saab_v3.tasks.classification import ClassificationHead
from saab_v3.tasks.config import (
    ClassificationTaskConfig,
    RankingTaskConfig,
    RegressionTaskConfig,
    TaskConfig,
    TokenClassificationTaskConfig,
)
from saab_v3.tasks.config_schema import (
    TaskName,
    validate_task_config,
)
from saab_v3.tasks.pooling import CLSPooling, MaxPooling, MeanPooling
from saab_v3.tasks.ranking import PairwiseRankingHead
from saab_v3.tasks.regression import RegressionHead
from saab_v3.tasks.token_classification import TokenClassificationHead


def create_task_head_from_config(
    config: TaskConfig | dict[str, Any],
    d_model: int,
) -> BaseTaskHead:
    """Create task head from Pydantic config or dict.

    Single entry point for creating task heads.
    Config is the only source of truth.

    Args:
        config: Task configuration (Pydantic model or dict with 'task.name' and 'task.params')
        d_model: Model dimension (from encoder)

    Returns:
        TaskHead instance (ClassificationHead, RegressionHead, etc.)

    Raises:
        KeyError: If required fields are missing
        ValueError: If config is invalid

    Example:
        >>> # Using Pydantic config
        >>> from saab_v3.tasks.config import ClassificationTaskConfig
        >>> config = ClassificationTaskConfig(num_classes=10, multi_label=False)
        >>> head = create_task_head_from_config(config, d_model=768)
        >>>
        >>> # Using dict (backward compatible)
        >>> config = {
        ...     "task": {
        ...         "name": "classification",
        ...         "params": {
        ...             "num_classes": 10,
        ...             "multi_label": False
        ...         }
        ...     }
        ... }
        >>> head = create_task_head_from_config(config, d_model=768)
    """
    # Convert dict to Pydantic if needed
    if isinstance(config, dict):
        # Validate dict config
        validate_task_config(config)
        task_config = _dict_to_task_config(config)
    else:
        # Already a Pydantic model
        task_config = config

    # Extract task name and params from Pydantic model
    if isinstance(task_config, ClassificationTaskConfig):
        task_name = TaskName.CLASSIFICATION
        task_params = task_config.model_dump()
    elif isinstance(task_config, RankingTaskConfig):
        task_name = TaskName.RANKING
        task_params = task_config.model_dump()
    elif isinstance(task_config, RegressionTaskConfig):
        task_name = TaskName.REGRESSION
        task_params = task_config.model_dump()
    elif isinstance(task_config, TokenClassificationTaskConfig):
        task_name = TaskName.TOKEN_CLASSIFICATION
        task_params = task_config.model_dump()
    else:
        valid_tasks = [t.value for t in TaskName]
        raise ValueError(
            f"Unknown task config type: {type(task_config)}. Valid tasks: {valid_tasks}"
        )

    # Task registry
    task_registry = {
        TaskName.CLASSIFICATION: _create_classification_head,
        TaskName.RANKING: _create_ranking_head,
        TaskName.REGRESSION: _create_regression_head,
        TaskName.TOKEN_CLASSIFICATION: _create_token_classification_head,
    }

    # Create task head
    return task_registry[task_name](d_model=d_model, **task_params)


def _dict_to_task_config(config: dict[str, Any]) -> TaskConfig:
    """Convert dict config to Pydantic task config.

    Args:
        config: Dict with 'task.name' and 'task.params'

    Returns:
        Pydantic TaskConfig instance

    Raises:
        ValueError: If task name is invalid
    """
    task_name = config["task"]["name"]
    task_params = config["task"]["params"]

    if task_name == TaskName.CLASSIFICATION:
        return ClassificationTaskConfig(**task_params)
    elif task_name == TaskName.RANKING:
        return RankingTaskConfig(**task_params)
    elif task_name == TaskName.REGRESSION:
        return RegressionTaskConfig(**task_params)
    elif task_name == TaskName.TOKEN_CLASSIFICATION:
        return TokenClassificationTaskConfig(**task_params)
    else:
        valid_tasks = [t.value for t in TaskName]
        raise ValueError(f"Unknown task: {task_name!r}. Valid tasks: {valid_tasks}")


def _get_pooling_from_config(pooling_str: str | None) -> nn.Module | None:
    """Convert pooling string to pooling module.

    Args:
        pooling_str: Pooling strategy string ("cls", "mean", "max", or None)

    Returns:
        Pooling module or None (uses default CLSPooling)

    Raises:
        ValueError: If pooling_str is invalid
    """
    if pooling_str is None:
        return None  # Use default (CLSPooling)

    pooling_map = {
        "cls": CLSPooling(),
        "mean": MeanPooling(),
        "max": MaxPooling(),
    }

    if pooling_str not in pooling_map:
        raise ValueError(
            f"Invalid pooling strategy: {pooling_str!r}. "
            f"Valid options: {list(pooling_map.keys())}"
        )

    return pooling_map[pooling_str]


def _create_classification_head(d_model: int, **params) -> ClassificationHead:
    """Create classification head from parameters.

    Args:
        d_model: Model dimension
        **params: Classification parameters

    Returns:
        ClassificationHead instance
    """
    # Extract pooling if present
    pooling = _get_pooling_from_config(params.pop("pooling", None))
    
    # Remove loss-related parameters that aren't part of the head
    params.pop("label_smoothing", None)

    return ClassificationHead(d_model=d_model, pooling=pooling, **params)


def _create_ranking_head(d_model: int, **params) -> PairwiseRankingHead:
    """Create ranking head from parameters.

    Args:
        d_model: Model dimension
        **params: Ranking parameters

    Returns:
        PairwiseRankingHead instance
    """
    return PairwiseRankingHead(d_model=d_model, **params)


def _create_regression_head(d_model: int, **params) -> RegressionHead:
    """Create regression head from parameters.

    Args:
        d_model: Model dimension
        **params: Regression parameters

    Returns:
        RegressionHead instance
    """
    # Extract pooling if present
    pooling = _get_pooling_from_config(params.pop("pooling", None))

    return RegressionHead(d_model=d_model, pooling=pooling, **params)


def _create_token_classification_head(
    d_model: int, **params
) -> TokenClassificationHead:
    """Create token classification head from parameters.

    Args:
        d_model: Model dimension
        **params: Token classification parameters

    Returns:
        TokenClassificationHead instance
    """
    return TokenClassificationHead(d_model=d_model, **params)
