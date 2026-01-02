"""Specs for StructuredDataset class - happy path only."""

import pandas as pd
from pathlib import Path
import tempfile
import json

from saab_v3.training.dataset import StructuredDataset
from saab_v3.data.structures import TokenizedSequence, EncodedTag


def spec_dataset_initialization(fitted_preprocessor, sample_dataframe):
    """Verify StructuredDataset can be initialized."""
    # Arrange: fitted_preprocessor and sample_dataframe

    # Act
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor, split="train")

    # Assert
    assert dataset.preprocessor == fitted_preprocessor
    assert dataset.split == "train"
    assert len(dataset) > 0


def spec_dataset_len(fitted_preprocessor, sample_dataframe):
    """Verify __len__ returns correct number of sequences."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)

    # Act
    length = len(dataset)

    # Assert
    assert length > 0
    assert isinstance(length, int)


def spec_dataset_getitem(fitted_preprocessor, sample_dataframe):
    """Verify __getitem__ returns correct tuple structure."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)

    # Act
    item = dataset[0]

    # Assert
    assert isinstance(item, tuple)
    assert len(item) == 3
    seq, token_ids, encoded_tags = item
    assert isinstance(seq, TokenizedSequence)
    assert isinstance(token_ids, list)
    assert isinstance(encoded_tags, list)
    assert len(token_ids) == len(encoded_tags)
    assert all(isinstance(tid, int) for tid in token_ids)
    assert all(isinstance(etag, EncodedTag) for etag in encoded_tags)


def spec_dataset_load_from_dataframe(fitted_preprocessor, sample_dataframe):
    """Verify dataset can load from pandas DataFrame."""
    # Arrange: fitted_preprocessor and sample_dataframe

    # Act
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)

    # Assert
    assert len(dataset) > 0


def spec_dataset_load_from_csv(fitted_preprocessor, temp_data_dir):
    """Verify dataset can load from CSV file."""
    # Arrange
    csv_path = temp_data_dir / "test.csv"
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    df.to_csv(csv_path, index=False)

    # Act
    dataset = StructuredDataset(str(csv_path), fitted_preprocessor)

    # Assert
    assert len(dataset) > 0


def spec_dataset_load_from_json(fitted_preprocessor, temp_data_dir):
    """Verify dataset can load from JSON file."""
    # Arrange
    json_path = temp_data_dir / "test.json"
    data = {"name": "Alice", "age": 25, "scores": [85, 90]}
    with open(json_path, "w") as f:
        json.dump(data, f)

    # Act
    dataset = StructuredDataset(str(json_path), fitted_preprocessor)

    # Assert
    assert len(dataset) > 0


def spec_dataset_load_from_list(fitted_preprocessor, sample_dataframe):
    """Verify dataset can load from list of files."""
    # Arrange
    # Create temporary CSV files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        csv1 = temp_path / "part1.csv"
        csv2 = temp_path / "part2.csv"

        df1 = pd.DataFrame({"name": ["Alice"], "age": [25]})
        df2 = pd.DataFrame({"name": ["Bob"], "age": [30]})
        df1.to_csv(csv1, index=False)
        df2.to_csv(csv2, index=False)

        # Act
        dataset = StructuredDataset([str(csv1), str(csv2)], fitted_preprocessor)

        # Assert
        assert len(dataset) > 0
