"""Fixtures for training component specs."""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from saab_v3.training.config import PreprocessingConfig
from saab_v3.training.preprocessor import Preprocessor


@pytest.fixture
def sample_config():
    """Sample PreprocessingConfig."""
    return PreprocessingConfig(
        vocab_size=1000,
        max_seq_len=128,
        preserve_original_tags=False,
    )


@pytest.fixture
def sample_dataframe():
    """Simple DataFrame for testing."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [85.5, 90.0, 88.5],
        }
    )


@pytest.fixture
def sample_preprocessor(sample_config):
    """Preprocessor instance with config."""
    return Preprocessor(sample_config)


@pytest.fixture
def fitted_preprocessor(sample_preprocessor, sample_dataframe):
    """Preprocessor that has been fitted on sample data."""
    preprocessor = Preprocessor(sample_preprocessor.config)
    preprocessor.fit(sample_dataframe)
    return preprocessor


@pytest.fixture
def temp_data_dir():
    """Temporary directory for testing artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)
