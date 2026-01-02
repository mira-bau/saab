"""Specs for artifact management - happy path only."""

import pytest

from saab_v3.training.preprocessor import Preprocessor
from saab_v3.training.artifacts import (
    save_preprocessing_artifacts,
    load_preprocessing_artifacts,
)


def spec_save_artifacts(fitted_preprocessor, temp_data_dir):
    """Verify artifacts can be saved."""
    # Arrange
    dataset_name = "test_dataset"

    # Act
    fitted_preprocessor.save_artifacts(dataset_name, base_path=temp_data_dir)

    # Assert
    artifacts_dir = temp_data_dir / "artifacts" / dataset_name
    assert artifacts_dir.exists()
    assert (artifacts_dir / "config.json").exists()
    assert (artifacts_dir / "vocabularies" / "value_vocab.json").exists()
    assert (artifacts_dir / "vocabularies" / "tag_vocabs" / "field.json").exists()


def spec_load_artifacts(fitted_preprocessor, temp_data_dir):
    """Verify artifacts can be loaded."""
    # Arrange
    dataset_name = "test_dataset"
    fitted_preprocessor.save_artifacts(dataset_name, base_path=temp_data_dir)

    # Act
    loaded_preprocessor = Preprocessor.load_artifacts(
        dataset_name, base_path=temp_data_dir
    )

    # Assert
    assert loaded_preprocessor._is_fitted is True
    assert loaded_preprocessor.tokenizer._is_built is True
    assert loaded_preprocessor.tag_encoder._is_built is True


def spec_artifacts_roundtrip(fitted_preprocessor, sample_dataframe, temp_data_dir):
    """Verify save → load → use roundtrip works."""
    # Arrange
    dataset_name = "test_dataset"
    fitted_preprocessor.save_artifacts(dataset_name, base_path=temp_data_dir)

    # Act
    loaded_preprocessor = Preprocessor.load_artifacts(
        dataset_name, base_path=temp_data_dir
    )
    sequences = loaded_preprocessor.transform(sample_dataframe)
    encoded = loaded_preprocessor.encode(sequences)

    # Assert
    assert len(encoded) > 0
    assert all(isinstance(item, tuple) and len(item) == 3 for item in encoded)


def spec_save_artifacts_wrapper(fitted_preprocessor, temp_data_dir):
    """Verify save_preprocessing_artifacts wrapper function works."""
    # Arrange
    dataset_name = "test_dataset"

    # Act
    save_preprocessing_artifacts(
        fitted_preprocessor, dataset_name, base_path=temp_data_dir
    )

    # Assert
    artifacts_dir = temp_data_dir / "artifacts" / dataset_name
    assert artifacts_dir.exists()
    assert (artifacts_dir / "config.json").exists()


def spec_load_artifacts_wrapper(fitted_preprocessor, temp_data_dir):
    """Verify load_preprocessing_artifacts wrapper function works."""
    # Arrange
    dataset_name = "test_dataset"
    fitted_preprocessor.save_artifacts(dataset_name, base_path=temp_data_dir)

    # Act
    loaded_preprocessor = load_preprocessing_artifacts(
        dataset_name, base_path=temp_data_dir
    )

    # Assert
    assert loaded_preprocessor._is_fitted is True
    assert loaded_preprocessor.tokenizer._is_built is True


def spec_load_artifacts_raises_if_not_found(temp_data_dir):
    """Verify load_artifacts raises FileNotFoundError if artifacts don't exist."""
    # Arrange
    dataset_name = "nonexistent_dataset"

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        Preprocessor.load_artifacts(dataset_name, base_path=temp_data_dir)
