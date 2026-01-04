"""Configuration loader for experiment configs from YAML files."""

import yaml
from pathlib import Path
from typing import TYPE_CHECKING

from saab_v3.models.config import ModelConfig
from saab_v3.training.config import PreprocessingConfig, TrainingConfig

if TYPE_CHECKING:
    pass


def load_experiment_config(
    config_path: Path | str | None = None,
) -> tuple[PreprocessingConfig, ModelConfig, TrainingConfig, dict | None]:
    """Load experiment configuration from YAML file or use defaults.

    Args:
        config_path: Optional path to YAML config file. If None, returns defaults.

    Returns:
        Tuple of (PreprocessingConfig, ModelConfig, TrainingConfig, task_config_dict).
        task_config_dict is None if not provided in YAML.

    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist
        ValueError: If config validation fails
    """
    if config_path is None:
        # Return defaults (with required fields set for stable training)
        return (
            PreprocessingConfig(),
            ModelConfig(),
            TrainingConfig(
                num_epochs=1,
                lr_schedule="constant",  # Constant schedule for stability
                warmup_steps=None,  # Required for constant schedule
            ),
            None,
        )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        raise ValueError(f"Config file must contain a YAML dictionary, got {type(config_dict)}")

    # Extract sections (handle missing sections gracefully)
    preprocessing_dict = config_dict.get("preprocessing", {})
    model_dict = config_dict.get("model", {})
    training_dict = config_dict.get("training", {})
    task_dict = config_dict.get("task", None)

    # Create config objects (Pydantic will use defaults for missing fields)
    preprocessing_config = PreprocessingConfig(**preprocessing_dict)
    model_config = ModelConfig(**model_dict)
    training_config = TrainingConfig(**training_dict)

    return (preprocessing_config, model_config, training_config, task_dict)

