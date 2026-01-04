"""Configuration module for shared and phase-specific configs."""

from saab_v3.config.base import BaseConfig

__all__ = ["BaseConfig"]

# Lazy import to avoid circular dependencies
# Import load_experiment_config directly when needed:
# from saab_v3.config.loader import load_experiment_config

