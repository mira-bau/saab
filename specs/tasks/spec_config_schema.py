"""Tests for task configuration schema and validation."""

import pytest

from saab_v3.tasks.config_schema import (
    RankingMethod,
    TaskName,
    validate_task_config,
)


# ============================================================================
# Structure Validation Tests
# ============================================================================


def spec_config_missing_task_key():
    """Test error when 'task' key is missing."""
    config = {"wrong_key": {"name": "classification"}}

    with pytest.raises(KeyError, match="Config must contain 'task' key"):
        validate_task_config(config)


def spec_config_missing_task_name():
    """Test error when 'task.name' is missing."""
    config = {"task": {"params": {"num_classes": 10}}}

    with pytest.raises(KeyError, match="Config must contain 'task.name'"):
        validate_task_config(config)


def spec_config_missing_task_params():
    """Test error when 'task.params' is missing."""
    config = {"task": {"name": "classification"}}

    with pytest.raises(KeyError, match="Config must contain 'task.params'"):
        validate_task_config(config)


# ============================================================================
# Task Name Validation Tests
# ============================================================================


def spec_config_invalid_task_name():
    """Test error when task name is invalid."""
    config = {
        "task": {
            "name": "invalid_task",
            "params": {"num_classes": 10},
        }
    }

    with pytest.raises(ValueError, match="Invalid task name"):
        validate_task_config(config)


def spec_config_valid_task_names():
    """Test that all valid task names are accepted."""
    valid_tasks = [
        ("classification", {"num_classes": 10}),
        ("ranking", {"method": "dot_product"}),
        ("regression", {}),
        ("token_classification", {"num_labels": 10}),
    ]

    for task_name, params in valid_tasks:
        config = {"task": {"name": task_name, "params": params}}
        # Should not raise
        validate_task_config(config)


# ============================================================================
# Classification Parameter Validation Tests
# ============================================================================


def spec_classification_missing_num_classes():
    """Test error when num_classes is missing."""
    config = {
        "task": {
            "name": "classification",
            "params": {"multi_label": False},
        }
    }

    with pytest.raises(KeyError, match="Classification task requires 'num_classes'"):
        validate_task_config(config)


def spec_classification_invalid_num_classes_type():
    """Test error when num_classes is not an integer."""
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": "10"},
        }
    }

    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        validate_task_config(config)


def spec_classification_invalid_num_classes_value():
    """Test error when num_classes is not positive."""
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": -5},
        }
    }

    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        validate_task_config(config)


def spec_classification_invalid_multi_label_type():
    """Test error when multi_label is not boolean."""
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": 10, "multi_label": "false"},
        }
    }

    with pytest.raises(ValueError, match="multi_label must be a boolean"):
        validate_task_config(config)


def spec_classification_invalid_hidden_dims_type():
    """Test error when hidden_dims is not a list."""
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": 10, "hidden_dims": "256"},
        }
    }

    with pytest.raises(ValueError, match="hidden_dims must be None or a list"):
        validate_task_config(config)


def spec_classification_invalid_hidden_dims_values():
    """Test error when hidden_dims contains non-positive integers."""
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": 10, "hidden_dims": [256, -128]},
        }
    }

    with pytest.raises(ValueError, match="hidden_dims must contain only positive integers"):
        validate_task_config(config)


def spec_classification_invalid_dropout():
    """Test error when dropout is out of range."""
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": 10, "dropout": 1.5},
        }
    }

    with pytest.raises(ValueError, match="dropout must be in \\[0, 1\\)"):
        validate_task_config(config)


def spec_classification_invalid_pooling():
    """Test error when pooling is invalid."""
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": 10, "pooling": "invalid"},
        }
    }

    with pytest.raises(ValueError, match="pooling must be one of"):
        validate_task_config(config)


def spec_classification_valid_config():
    """Test valid classification config."""
    config = {
        "task": {
            "name": "classification",
            "params": {
                "num_classes": 10,
                "multi_label": False,
                "hidden_dims": [256, 128],
                "dropout": 0.1,
                "pooling": "cls",
            },
        }
    }

    # Should not raise
    validate_task_config(config)


# ============================================================================
# Ranking Parameter Validation Tests
# ============================================================================


def spec_ranking_missing_method():
    """Test error when method is missing."""
    config = {
        "task": {
            "name": "ranking",
            "params": {},
        }
    }

    with pytest.raises(KeyError, match="Ranking task requires 'method'"):
        validate_task_config(config)


def spec_ranking_invalid_method():
    """Test error when method is invalid."""
    config = {
        "task": {
            "name": "ranking",
            "params": {"method": "invalid_method"},
        }
    }

    with pytest.raises(ValueError, match="Invalid ranking method"):
        validate_task_config(config)


def spec_ranking_mlp_missing_hidden_dims():
    """Test error when MLP method is missing hidden_dims."""
    config = {
        "task": {
            "name": "ranking",
            "params": {"method": "mlp"},
        }
    }

    with pytest.raises(ValueError, match="Ranking method 'mlp' requires 'hidden_dims'"):
        validate_task_config(config)


def spec_ranking_difference_missing_hidden_dims():
    """Test error when difference method is missing hidden_dims."""
    config = {
        "task": {
            "name": "ranking",
            "params": {"method": "difference"},
        }
    }

    with pytest.raises(ValueError, match="Ranking method 'difference' requires 'hidden_dims'"):
        validate_task_config(config)


def spec_ranking_valid_configs():
    """Test valid ranking configs."""
    valid_configs = [
        {"method": "dot_product"},
        {"method": "cosine"},
        {"method": "mlp", "hidden_dims": [256, 128]},
        {"method": "difference", "hidden_dims": [256, 128]},
    ]

    for params in valid_configs:
        config = {
            "task": {
                "name": "ranking",
                "params": params,
            }
        }
        # Should not raise
        validate_task_config(config)


# ============================================================================
# Regression Parameter Validation Tests
# ============================================================================


def spec_regression_invalid_num_targets():
    """Test error when num_targets is invalid."""
    config = {
        "task": {
            "name": "regression",
            "params": {"num_targets": -1},
        }
    }

    with pytest.raises(ValueError, match="num_targets must be a positive integer"):
        validate_task_config(config)


def spec_regression_valid_config():
    """Test valid regression config."""
    config = {
        "task": {
            "name": "regression",
            "params": {
                "num_targets": 1,
                "hidden_dims": [256, 128],
                "dropout": 0.1,
                "pooling": "mean",
            },
        }
    }

    # Should not raise
    validate_task_config(config)


# ============================================================================
# Token Classification Parameter Validation Tests
# ============================================================================


def spec_token_classification_missing_num_labels():
    """Test error when num_labels is missing."""
    config = {
        "task": {
            "name": "token_classification",
            "params": {},
        }
    }

    with pytest.raises(KeyError, match="Token classification task requires 'num_labels'"):
        validate_task_config(config)


def spec_token_classification_invalid_num_labels():
    """Test error when num_labels is invalid."""
    config = {
        "task": {
            "name": "token_classification",
            "params": {"num_labels": 0},
        }
    }

    with pytest.raises(ValueError, match="num_labels must be a positive integer"):
        validate_task_config(config)


def spec_token_classification_valid_config():
    """Test valid token classification config."""
    config = {
        "task": {
            "name": "token_classification",
            "params": {
                "num_labels": 10,
                "hidden_dims": [256, 128],
                "dropout": 0.1,
            },
        }
    }

    # Should not raise
    validate_task_config(config)


# ============================================================================
# Edge Cases
# ============================================================================


def spec_config_null_hidden_dims():
    """Test that null/None hidden_dims is valid."""
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": 10, "hidden_dims": None},
        }
    }

    # Should not raise
    validate_task_config(config)


def spec_config_empty_hidden_dims():
    """Test that empty list hidden_dims is invalid."""
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": 10, "hidden_dims": []},
        }
    }

    # Empty list should be invalid (no positive integers)
    with pytest.raises(ValueError, match="hidden_dims must be None or a non-empty list"):
        validate_task_config(config)


def spec_config_boundary_dropout():
    """Test boundary values for dropout."""
    # Valid: 0.0
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": 10, "dropout": 0.0},
        }
    }
    validate_task_config(config)

    # Invalid: 1.0 (should be < 1)
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": 10, "dropout": 1.0},
        }
    }
    with pytest.raises(ValueError, match="dropout must be in \\[0, 1\\)"):
        validate_task_config(config)

    # Invalid: negative
    config = {
        "task": {
            "name": "classification",
            "params": {"num_classes": 10, "dropout": -0.1},
        }
    }
    with pytest.raises(ValueError, match="dropout must be in \\[0, 1\\)"):
        validate_task_config(config)

