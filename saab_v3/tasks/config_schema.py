"""Task configuration schema and validation."""

from enum import Enum
from typing import Any


class TaskName(str, Enum):
    """Valid task names."""

    CLASSIFICATION = "classification"
    RANKING = "ranking"
    REGRESSION = "regression"
    TOKEN_CLASSIFICATION = "token_classification"


class RankingMethod(str, Enum):
    """Valid ranking methods."""

    DOT_PRODUCT = "dot_product"
    COSINE = "cosine"
    MLP = "mlp"
    DIFFERENCE = "difference"


def validate_task_config(config: dict[str, Any]) -> None:
    """Validate task configuration at startup.

    Args:
        config: Task configuration dict with 'task.name' and 'task.params'

    Raises:
        KeyError: If required fields are missing
        ValueError: If config values are invalid
    """
    # Validate structure
    if "task" not in config:
        raise KeyError("Config must contain 'task' key")

    task_config = config["task"]

    if "name" not in task_config:
        raise KeyError("Config must contain 'task.name'")

    if "params" not in task_config:
        raise KeyError("Config must contain 'task.params'")

    task_name = task_config["name"]
    task_params = task_config["params"]

    # Validate task name
    valid_tasks = [t.value for t in TaskName]
    if task_name not in valid_tasks:
        raise ValueError(
            f"Invalid task name: {task_name!r}. Valid tasks: {valid_tasks}"
        )

    # Validate task-specific parameters
    if task_name == TaskName.CLASSIFICATION:
        _validate_classification_params(task_params)
    elif task_name == TaskName.RANKING:
        _validate_ranking_params(task_params)
    elif task_name == TaskName.REGRESSION:
        _validate_regression_params(task_params)
    elif task_name == TaskName.TOKEN_CLASSIFICATION:
        _validate_token_classification_params(task_params)


def _validate_classification_params(params: dict[str, Any]) -> None:
    """Validate classification task parameters.

    Args:
        params: Classification parameters dict

    Raises:
        KeyError: If required parameters are missing
        ValueError: If parameter values are invalid
    """
    if "num_classes" not in params:
        raise KeyError("Classification task requires 'num_classes' parameter")

    num_classes = params["num_classes"]
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}")

    if "multi_label" in params:
        if not isinstance(params["multi_label"], bool):
            raise ValueError(
                f"multi_label must be a boolean, got {params['multi_label']!r}"
            )

    if "hidden_dims" in params and params["hidden_dims"] is not None:
        if not isinstance(params["hidden_dims"], list):
            raise ValueError(
                f"hidden_dims must be None or a list of integers, "
                f"got {type(params['hidden_dims']).__name__}"
            )
        if len(params["hidden_dims"]) == 0:
            raise ValueError(
                "hidden_dims must be None or a non-empty list of positive integers"
            )
        if not all(isinstance(d, int) and d > 0 for d in params["hidden_dims"]):
            raise ValueError("hidden_dims must contain only positive integers")

    if "dropout" in params:
        dropout = params["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")

    if "pooling" in params:
        pooling = params["pooling"]
        valid_pooling = ["cls", "mean", "max"]
        if pooling not in valid_pooling:
            raise ValueError(f"pooling must be one of {valid_pooling}, got {pooling!r}")


def _validate_ranking_params(params: dict[str, Any]) -> None:
    """Validate ranking task parameters.

    Args:
        params: Ranking parameters dict

    Raises:
        KeyError: If required parameters are missing
        ValueError: If parameter values are invalid
    """
    if "method" not in params:
        raise KeyError("Ranking task requires 'method' parameter")

    method = params["method"]
    valid_methods = [m.value for m in RankingMethod]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid ranking method: {method!r}. Valid methods: {valid_methods}"
        )

    # Validate hidden_dims for MLP and difference methods
    if method in [RankingMethod.MLP, RankingMethod.DIFFERENCE]:
        if "hidden_dims" not in params or params["hidden_dims"] is None:
            raise ValueError(
                f"Ranking method '{method}' requires 'hidden_dims' parameter"
            )

    if "hidden_dims" in params and params["hidden_dims"] is not None:
        if not isinstance(params["hidden_dims"], list):
            raise ValueError(
                f"hidden_dims must be None or a list of integers, "
                f"got {type(params['hidden_dims']).__name__}"
            )
        if len(params["hidden_dims"]) == 0:
            raise ValueError(
                "hidden_dims must be None or a non-empty list of positive integers"
            )
        if not all(isinstance(d, int) and d > 0 for d in params["hidden_dims"]):
            raise ValueError("hidden_dims must contain only positive integers")

    if "dropout" in params:
        dropout = params["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")


def _validate_regression_params(params: dict[str, Any]) -> None:
    """Validate regression task parameters.

    Args:
        params: Regression parameters dict

    Raises:
        ValueError: If parameter values are invalid
    """
    if "num_targets" in params:
        num_targets = params["num_targets"]
        if not isinstance(num_targets, int) or num_targets <= 0:
            raise ValueError(
                f"num_targets must be a positive integer, got {num_targets!r}"
            )

    if "hidden_dims" in params and params["hidden_dims"] is not None:
        if not isinstance(params["hidden_dims"], list):
            raise ValueError(
                f"hidden_dims must be None or a list of integers, "
                f"got {type(params['hidden_dims']).__name__}"
            )
        if len(params["hidden_dims"]) == 0:
            raise ValueError(
                "hidden_dims must be None or a non-empty list of positive integers"
            )
        if not all(isinstance(d, int) and d > 0 for d in params["hidden_dims"]):
            raise ValueError("hidden_dims must contain only positive integers")

    if "dropout" in params:
        dropout = params["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")

    if "pooling" in params:
        pooling = params["pooling"]
        valid_pooling = ["cls", "mean", "max"]
        if pooling not in valid_pooling:
            raise ValueError(f"pooling must be one of {valid_pooling}, got {pooling!r}")


def _validate_token_classification_params(params: dict[str, Any]) -> None:
    """Validate token classification task parameters.

    Args:
        params: Token classification parameters dict

    Raises:
        KeyError: If required parameters are missing
        ValueError: If parameter values are invalid
    """
    if "num_labels" not in params:
        raise KeyError("Token classification task requires 'num_labels' parameter")

    num_labels = params["num_labels"]
    if not isinstance(num_labels, int) or num_labels <= 0:
        raise ValueError(f"num_labels must be a positive integer, got {num_labels!r}")

    if "hidden_dims" in params and params["hidden_dims"] is not None:
        if not isinstance(params["hidden_dims"], list):
            raise ValueError(
                f"hidden_dims must be None or a list of integers, "
                f"got {type(params['hidden_dims']).__name__}"
            )
        if len(params["hidden_dims"]) == 0:
            raise ValueError(
                "hidden_dims must be None or a non-empty list of positive integers"
            )
        if not all(isinstance(d, int) and d > 0 for d in params["hidden_dims"]):
            raise ValueError("hidden_dims must contain only positive integers")

    if "dropout" in params:
        dropout = params["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
