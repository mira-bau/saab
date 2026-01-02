"""Artifact management for preprocessing vocabularies and encoders."""

from pathlib import Path

from saab_v3.training.preprocessor import Preprocessor


def save_preprocessing_artifacts(
    preprocessor: Preprocessor,
    dataset_name: str,
    base_path: Path | None = None,
) -> None:
    """Save all preprocessing artifacts.

    This is a convenience wrapper around Preprocessor.save_artifacts().

    Args:
        preprocessor: Preprocessor instance to save
        dataset_name: Name of dataset (creates data/artifacts/dataset_name/)
        base_path: Optional override for data directory
    """
    preprocessor.save_artifacts(dataset_name, base_path=base_path)


def load_preprocessing_artifacts(
    dataset_name: str,
    base_path: Path | None = None,
) -> Preprocessor:
    """Load preprocessing artifacts and reconstruct Preprocessor.

    This is a convenience wrapper around Preprocessor.load_artifacts().

    Args:
        dataset_name: Name of dataset
        base_path: Optional override for data directory

    Returns:
        Preprocessor instance with loaded vocabularies
    """
    return Preprocessor.load_artifacts(dataset_name, base_path=base_path)
