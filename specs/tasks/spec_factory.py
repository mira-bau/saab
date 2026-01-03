"""Tests for task head factory function."""

import pytest
import torch

from saab_v3.tasks import create_task_head_from_config
from saab_v3.tasks.classification import ClassificationHead
from saab_v3.tasks.ranking import PairwiseRankingHead
from saab_v3.tasks.regression import RegressionHead
from saab_v3.tasks.token_classification import TokenClassificationHead


# ============================================================================
# Factory Function Tests
# ============================================================================


def spec_factory_creates_classification_head():
    """Test factory creates ClassificationHead correctly."""
    config = {
        "task": {
            "name": "classification",
            "params": {
                "num_classes": 10,
                "multi_label": False,
            },
        }
    }

    head = create_task_head_from_config(config, d_model=128)

    assert isinstance(head, ClassificationHead)
    assert head.num_classes == 10
    assert head.multi_label is False


def spec_factory_creates_ranking_head():
    """Test factory creates PairwiseRankingHead correctly."""
    config = {
        "task": {
            "name": "ranking",
            "params": {
                "method": "dot_product",
            },
        }
    }

    head = create_task_head_from_config(config, d_model=128)

    assert isinstance(head, PairwiseRankingHead)
    assert head.method == "dot_product"


def spec_factory_creates_regression_head():
    """Test factory creates RegressionHead correctly."""
    config = {
        "task": {
            "name": "regression",
            "params": {
                "num_targets": 1,
            },
        }
    }

    head = create_task_head_from_config(config, d_model=128)

    assert isinstance(head, RegressionHead)
    assert head.num_targets == 1


def spec_factory_creates_token_classification_head():
    """Test factory creates TokenClassificationHead correctly."""
    config = {
        "task": {
            "name": "token_classification",
            "params": {
                "num_labels": 10,
            },
        }
    }

    head = create_task_head_from_config(config, d_model=128)

    assert isinstance(head, TokenClassificationHead)
    assert head.num_labels == 10


def spec_factory_passes_parameters_correctly():
    """Test factory passes all parameters correctly."""
    config = {
        "task": {
            "name": "classification",
            "params": {
                "num_classes": 5,
                "multi_label": True,
                "hidden_dims": [256, 128],
                "dropout": 0.2,
            },
        }
    }

    head = create_task_head_from_config(config, d_model=128)

    assert head.num_classes == 5
    assert head.multi_label is True
    assert head.hidden_dims == [256, 128]
    assert head.dropout == 0.2
    assert head.mlp is not None  # MLP should be created


def spec_factory_handles_pooling():
    """Test factory handles pooling parameter correctly."""
    config = {
        "task": {
            "name": "classification",
            "params": {
                "num_classes": 10,
                "pooling": "mean",
            },
        }
    }

    head = create_task_head_from_config(config, d_model=128)

    assert isinstance(head.pooling, type(head.pooling))  # Should be MeanPooling
    # Check that it's not CLSPooling (default)
    from saab_v3.tasks.pooling import CLSPooling, MeanPooling
    assert not isinstance(head.pooling, CLSPooling)
    assert isinstance(head.pooling, MeanPooling)


def spec_factory_validates_before_creation():
    """Test factory validates config before creating task head."""
    config = {
        "task": {
            "name": "classification",
            "params": {
                # Missing num_classes
            },
        }
    }

    with pytest.raises(KeyError, match="Classification task requires 'num_classes'"):
        create_task_head_from_config(config, d_model=128)


def spec_factory_handles_invalid_task_name():
    """Test factory handles invalid task name."""
    config = {
        "task": {
            "name": "invalid_task",
            "params": {},
        }
    }

    with pytest.raises(ValueError, match="Invalid task name"):
        create_task_head_from_config(config, d_model=128)


def spec_factory_ranking_with_mlp():
    """Test factory creates ranking head with MLP method."""
    config = {
        "task": {
            "name": "ranking",
            "params": {
                "method": "mlp",
                "hidden_dims": [256, 128],
                "dropout": 0.1,
            },
        }
    }

    head = create_task_head_from_config(config, d_model=128)

    assert isinstance(head, PairwiseRankingHead)
    assert head.method == "mlp"
    assert head.comparison_mlp is not None


def spec_factory_creates_working_heads():
    """Test that factory-created heads actually work."""
    configs = [
        {
            "task": {
                "name": "classification",
                "params": {"num_classes": 5},
            }
        },
        {
            "task": {
                "name": "regression",
                "params": {"num_targets": 1},
            }
        },
        {
            "task": {
                "name": "token_classification",
                "params": {"num_labels": 5},
            }
        },
    ]

    for config in configs:
        head = create_task_head_from_config(config, d_model=128)

        # Create dummy encoder output
        batch_size = 2
        seq_len = 10
        encoder_output = torch.randn(batch_size, seq_len, 128)

        # Test forward pass
        if config["task"]["name"] == "token_classification":
            output = head(encoder_output)
            assert output.shape == (batch_size, seq_len, 5)
        else:
            output = head(encoder_output)
            if config["task"]["name"] == "classification":
                assert output.shape == (batch_size, 5)
            else:  # regression
                assert output.shape == (batch_size, 1)


def spec_factory_ranking_head_works():
    """Test that factory-created ranking head works."""
    config = {
        "task": {
            "name": "ranking",
            "params": {
                "method": "dot_product",
            },
        }
    }

    head = create_task_head_from_config(config, d_model=128)

    # Create dummy sequence representations
    batch_size = 2
    seq_a_repr = torch.randn(batch_size, 128)
    seq_b_repr = torch.randn(batch_size, 128)

    # Test forward pass
    scores = head(seq_a_repr, seq_b_repr)
    assert scores.shape == (batch_size,)


def spec_factory_all_task_types():
    """Test factory handles all task types."""
    task_configs = [
        ("classification", {"num_classes": 10}, ClassificationHead),
        ("ranking", {"method": "dot_product"}, PairwiseRankingHead),
        ("regression", {"num_targets": 1}, RegressionHead),
        ("token_classification", {"num_labels": 10}, TokenClassificationHead),
    ]

    for task_name, params, expected_class in task_configs:
        config = {
            "task": {
                "name": task_name,
                "params": params,
            }
        }

        head = create_task_head_from_config(config, d_model=128)
        assert isinstance(head, expected_class)


def spec_factory_error_handling():
    """Test factory error handling for various invalid configs."""
    # Missing task key
    with pytest.raises(KeyError):
        create_task_head_from_config({}, d_model=128)

    # Missing task.name
    with pytest.raises(KeyError):
        create_task_head_from_config({"task": {"params": {}}}, d_model=128)

    # Missing task.params
    with pytest.raises(KeyError):
        create_task_head_from_config({"task": {"name": "classification"}}, d_model=128)

    # Invalid task name
    with pytest.raises(ValueError):
        create_task_head_from_config(
            {"task": {"name": "invalid", "params": {}}}, d_model=128
        )

