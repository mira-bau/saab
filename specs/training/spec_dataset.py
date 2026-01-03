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
    assert len(item) == 4  # (seq, token_ids, encoded_tags, label)
    seq, token_ids, encoded_tags, label = item
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


# ============================================================================
# Ranking Dataset Specs
# ============================================================================


def spec_dataset_ranking_load_from_csv(fitted_preprocessor, temp_data_dir):
    """Verify dataset can load ranking pairs from CSV with sequence_a, sequence_b, label columns."""
    # Arrange
    csv_path = temp_data_dir / "ranking.csv"
    # Create ranking data with sequence_a and sequence_b columns
    df = pd.DataFrame({
        "sequence_a": [pd.DataFrame({"name": ["Alice"], "age": [25]}), pd.DataFrame({"name": ["Bob"], "age": [30]})],
        "sequence_b": [pd.DataFrame({"name": ["Charlie"], "age": [35]}), pd.DataFrame({"name": ["David"], "age": [40]})],
        "label": [1, -1]
    })
    # Convert DataFrames to dict format for CSV
    df["sequence_a"] = df["sequence_a"].apply(lambda x: x.to_dict("records")[0] if isinstance(x, pd.DataFrame) else x)
    df["sequence_b"] = df["sequence_b"].apply(lambda x: x.to_dict("records")[0] if isinstance(x, pd.DataFrame) else x)
    df.to_csv(csv_path, index=False)

    # Act
    dataset = StructuredDataset(str(csv_path), fitted_preprocessor, task_type="ranking")

    # Assert
    assert len(dataset) == 2
    assert dataset.task_type == "ranking"


def spec_dataset_ranking_getitem_returns_pair(fitted_preprocessor, sample_dataframe):
    """Verify __getitem__ returns (seq_a_data, seq_b_data, label) tuple for ranking."""
    # Arrange
    # Create ranking data as DataFrame with sequence_a and sequence_b
    ranking_data = pd.DataFrame({
        "sequence_a": [
            pd.DataFrame({"name": ["Alice"], "age": [25]}),
            pd.DataFrame({"name": ["Bob"], "age": [30]})
        ],
        "sequence_b": [
            pd.DataFrame({"name": ["Charlie"], "age": [35]}),
            pd.DataFrame({"name": ["David"], "age": [40]})
        ],
        "label": [1, -1]
    })
    # Convert to dict format
    ranking_data["sequence_a"] = ranking_data["sequence_a"].apply(
        lambda x: x.to_dict("records")[0] if isinstance(x, pd.DataFrame) else x
    )
    ranking_data["sequence_b"] = ranking_data["sequence_b"].apply(
        lambda x: x.to_dict("records")[0] if isinstance(x, pd.DataFrame) else x
    )
    dataset = StructuredDataset(ranking_data, fitted_preprocessor, task_type="ranking")

    # Act
    item = dataset[0]

    # Assert
    assert isinstance(item, tuple)
    assert len(item) == 3
    seq_a_data, seq_b_data, label = item
    assert isinstance(seq_a_data, tuple)
    assert isinstance(seq_b_data, tuple)
    assert len(seq_a_data) == 3  # (TokenizedSequence, token_ids, encoded_tags)
    assert len(seq_b_data) == 3
    # Label can be int, numpy int64, or None (pandas returns numpy types)
    assert label is None or isinstance(label, (int, type(None))) or (hasattr(label, 'item') and isinstance(label.item(), int))


def spec_dataset_ranking_encodes_both_sequences(fitted_preprocessor, sample_dataframe):
    """Verify both sequences are encoded separately for ranking."""
    # Arrange
    ranking_data = pd.DataFrame({
        "sequence_a": [pd.DataFrame({"name": ["Alice"], "age": [25]})],
        "sequence_b": [pd.DataFrame({"name": ["Bob"], "age": [30]})],
        "label": [1]
    })
    ranking_data["sequence_a"] = ranking_data["sequence_a"].apply(
        lambda x: x.to_dict("records")[0] if isinstance(x, pd.DataFrame) else x
    )
    ranking_data["sequence_b"] = ranking_data["sequence_b"].apply(
        lambda x: x.to_dict("records")[0] if isinstance(x, pd.DataFrame) else x
    )
    dataset = StructuredDataset(ranking_data, fitted_preprocessor, task_type="ranking")

    # Act
    seq_a_data, seq_b_data, label = dataset[0]
    seq_a, token_ids_a, encoded_tags_a = seq_a_data
    seq_b, token_ids_b, encoded_tags_b = seq_b_data

    # Assert
    assert isinstance(seq_a, TokenizedSequence)
    assert isinstance(seq_b, TokenizedSequence)
    assert isinstance(token_ids_a, list)
    assert isinstance(token_ids_b, list)
    assert isinstance(encoded_tags_a, list)
    assert isinstance(encoded_tags_b, list)
    assert len(token_ids_a) == len(encoded_tags_a)
    assert len(token_ids_b) == len(encoded_tags_b)
