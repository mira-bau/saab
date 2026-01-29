"""PyTorch Dataset for structured data preprocessing."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset
import networkx as nx

from saab_v3.data.structures import EncodedTag, TokenizedSequence

from saab_v3.training.preprocessor import Preprocessor


class StructuredDataset(Dataset):
    """PyTorch Dataset that loads and provides preprocessed sequences."""

    def __init__(
        self,
        data: Any | list[Any] | str | Path,
        preprocessor: Preprocessor,
        split: str = "train",
        task_type: str | None = None,
        lazy: bool = False,
    ):
        """Initialize dataset.

        Args:
            data: Input data (file path, DataFrame, dict, Graph, or list of these)
            preprocessor: Fitted Preprocessor instance
            split: Dataset split name ("train", "val", "test")
            task_type: Task type ("ranking" for ranking pairs, None for single sequences)
            lazy: If True, load data on-demand (not implemented in Phase 1)
        """
        self.preprocessor = preprocessor
        self.split = split
        self.task_type = task_type
        self.lazy = lazy
        
        # Store original data path for fingerprinting (if it's a file path)
        if isinstance(data, (str, Path)):
            self._data_path = Path(data)
        else:
            self._data_path = None

        # Load data
        self.raw_data = self._load_data(data)

        # For ranking tasks, we handle pairs differently
        if task_type == "ranking":
            # Store raw data for ranking pairs (will encode on-demand in __getitem__)
            self.encoded_sequences = None
        else:
            # Exclude label columns from preprocessing to prevent data leakage
            # Label columns: "label", "target", "y"
            feature_data = self._exclude_label_columns(self.raw_data)
            
            # Transform: extract and tokenize (only from feature columns)
            sequences = self.preprocessor.transform(feature_data)

            # Encode: convert to token IDs and encoded tags
            self.encoded_sequences = self.preprocessor.encode(sequences)

    def _load_data(self, data: Any | list[Any] | str | Path) -> Any:
        """Load data from various sources.

        Args:
            data: Input data (file path, DataFrame, dict, Graph, or list)

        Returns:
            Loaded data in appropriate format
        """
        # Handle list of files
        if isinstance(data, list):
            loaded_items = []
            for item in data:
                loaded_items.append(self._load_single_item(item))
            # For tables, concatenate DataFrames; for others, combine lists
            if all(isinstance(item, pd.DataFrame) for item in loaded_items):
                return pd.concat(loaded_items, ignore_index=True)
            else:
                # For JSON/Graph, return list
                result = []
                for item in loaded_items:
                    if isinstance(item, list):
                        result.extend(item)
                    else:
                        result.append(item)
                return result

        return self._load_single_item(data)

    def _load_single_item(self, data: Any) -> Any:
        """Load a single data item (file or in-memory object).

        Args:
            data: Single data item

        Returns:
            Loaded data
        """
        # If it's already a supported type, return as-is
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, (dict, list)):
            return data
        if nx is not None and isinstance(data, nx.Graph):
            return data

        # If it's a file path, load based on extension
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")

            suffix = path.suffix.lower()

            # CSV or Excel
            if suffix in [".csv", ".xlsx", ".xls"]:
                if suffix == ".csv":
                    return pd.read_csv(path)
                else:
                    return pd.read_excel(path)

            # JSON
            if suffix == ".json":
                with open(path, "r") as f:
                    return json.load(f)

            # Try to infer format
            # If it looks like CSV, try CSV
            try:
                return pd.read_csv(path)
            except Exception:
                pass

            # If it looks like JSON, try JSON
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass

            raise ValueError(
                f"Cannot determine file format for {path}. "
                "Supported: .csv, .xlsx, .xls, .json"
            )

        # Unknown type
        return data

    def _exclude_label_columns(self, data: Any) -> Any:
        """Exclude label columns from data to prevent data leakage.
        
        Args:
            data: Input data (DataFrame, dict, list, etc.)
            
        Returns:
            Data with label columns removed (for DataFrame) or original data (for other types)
        """
        # Label column names to exclude
        label_columns = ["label", "target", "y"]
        
        # For DataFrames (CSV/Excel), drop label columns
        if isinstance(data, pd.DataFrame):
            columns_to_drop = [col for col in label_columns if col in data.columns]
            if columns_to_drop:
                # Create a copy without label columns
                return data.drop(columns=columns_to_drop)
        
        # For dict/list data, we can't easily exclude without knowing structure
        # In this case, the user should structure their data properly
        # For now, return as-is (this is mainly for JSON/Graph data where labels
        # are typically not in the same structure as features)
        return data

    def __len__(self) -> int:
        """Return number of sequences."""
        if self.task_type == "ranking":
            # For ranking, count pairs from raw data
            if isinstance(self.raw_data, pd.DataFrame):
                return len(self.raw_data)
            elif isinstance(self.raw_data, list):
                return len(self.raw_data)
            elif isinstance(self.raw_data, dict):
                # Check for sequence_a or pairs list
                if "sequence_a" in self.raw_data and "sequence_b" in self.raw_data:
                    seq_a = self.raw_data["sequence_a"]
                    return len(seq_a) if isinstance(seq_a, list) else 1
                return 0
            return 0
        return len(self.encoded_sequences)

    def __getitem__(
        self, idx: int
    ) -> tuple[TokenizedSequence, list[int], list[EncodedTag], Any | None] | tuple[
        tuple[TokenizedSequence, list[int], list[EncodedTag]],
        tuple[TokenizedSequence, list[int], list[EncodedTag]],
        Any | None,
    ]:
        """Return encoded sequence(s) ready for batching.

        Args:
            idx: Sequence index

        Returns:
            For single-sequence tasks: (TokenizedSequence, token_ids, encoded_tags, label) tuple
            For ranking tasks: ((seq_a_data), (seq_b_data), label) tuple where each data is (TokenizedSequence, token_ids, encoded_tags)
            label is None if not present in data
        """
        if self.task_type == "ranking":
            return self._get_ranking_item(idx)
        else:
            return self._get_single_item(idx)

    def _get_single_item(
        self, idx: int
    ) -> tuple[TokenizedSequence, list[int], list[EncodedTag], Any | None]:
        """Return single sequence (existing logic).

        Args:
            idx: Sequence index

        Returns:
            (TokenizedSequence, token_ids, encoded_tags, label) tuple
        """
        tokenized_seq, token_ids, encoded_tags = self.encoded_sequences[idx]
        label = self._extract_label(idx)
        return tokenized_seq, token_ids, encoded_tags, label

    def _get_ranking_item(
        self, idx: int
    ) -> tuple[
        tuple[TokenizedSequence, list[int], list[EncodedTag]],
        tuple[TokenizedSequence, list[int], list[EncodedTag]],
        Any | None,
    ]:
        """Return ranking pair.

        Args:
            idx: Pair index

        Returns:
            ((seq_a_data), (seq_b_data), label) tuple where each data is (TokenizedSequence, token_ids, encoded_tags)
        """
        # Extract sequence_a and sequence_b from raw data
        if isinstance(self.raw_data, pd.DataFrame):
            # Check for sequence_a, sequence_b columns
            if "sequence_a" not in self.raw_data.columns or "sequence_b" not in self.raw_data.columns:
                raise ValueError(
                    "Ranking task requires 'sequence_a' and 'sequence_b' columns in CSV"
                )

            seq_a_raw = self.raw_data.iloc[idx]["sequence_a"]
            seq_b_raw = self.raw_data.iloc[idx]["sequence_b"]
            label = self._extract_ranking_label(idx)

        elif isinstance(self.raw_data, list):
            # List of dicts with sequence_a, sequence_b
            if idx >= len(self.raw_data):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(self.raw_data)}")
            item = self.raw_data[idx]
            if not isinstance(item, dict):
                raise ValueError("Ranking task requires list of dicts with 'sequence_a' and 'sequence_b'")
            if "sequence_a" not in item or "sequence_b" not in item:
                raise ValueError("Ranking task requires 'sequence_a' and 'sequence_b' keys in dict")
            seq_a_raw = item["sequence_a"]
            seq_b_raw = item["sequence_b"]
            label = item.get("label") or item.get("target") or item.get("y")

        elif isinstance(self.raw_data, dict):
            # Dict with sequence_a and sequence_b lists
            if "sequence_a" not in self.raw_data or "sequence_b" not in self.raw_data:
                raise ValueError("Ranking task requires 'sequence_a' and 'sequence_b' keys in dict")
            seq_a_list = self.raw_data["sequence_a"]
            seq_b_list = self.raw_data["sequence_b"]
            if idx >= len(seq_a_list) or idx >= len(seq_b_list):
                raise IndexError(f"Index {idx} out of range")
            seq_a_raw = seq_a_list[idx]
            seq_b_raw = seq_b_list[idx]
            labels = self.raw_data.get("labels") or self.raw_data.get("targets")
            label = labels[idx] if labels and idx < len(labels) else None

        else:
            raise ValueError(f"Unsupported raw_data type for ranking: {type(self.raw_data)}")

        # Transform and encode both sequences separately
        # Handle dict/DataFrame: convert dict to DataFrame to ensure single sequence
        if isinstance(seq_a_raw, dict):
            # Convert dict to DataFrame with one row so TableExtractor produces one sequence
            seq_a_input = pd.DataFrame([seq_a_raw])
        elif isinstance(seq_a_raw, pd.DataFrame):
            # DataFrame should produce one sequence per row
            seq_a_input = seq_a_raw
        else:
            seq_a_input = seq_a_raw

        if isinstance(seq_b_raw, dict):
            # Convert dict to DataFrame with one row so TableExtractor produces one sequence
            seq_b_input = pd.DataFrame([seq_b_raw])
        elif isinstance(seq_b_raw, pd.DataFrame):
            seq_b_input = seq_b_raw
        else:
            seq_b_input = seq_b_raw

        # Transform sequence_a
        sequences_a = self.preprocessor.transform(seq_a_input)
        if len(sequences_a) != 1:
            raise ValueError(f"Expected single sequence for sequence_a at index {idx}, got {len(sequences_a)}")
        encoded_a = self.preprocessor.encode(sequences_a)
        seq_a_data = encoded_a[0]  # (TokenizedSequence, token_ids, encoded_tags)

        # Transform sequence_b
        sequences_b = self.preprocessor.transform(seq_b_input)
        if len(sequences_b) != 1:
            raise ValueError(f"Expected single sequence for sequence_b at index {idx}, got {len(sequences_b)}")
        encoded_b = self.preprocessor.encode(sequences_b)
        seq_b_data = encoded_b[0]  # (TokenizedSequence, token_ids, encoded_tags)

        return (seq_a_data, seq_b_data, label)

    def _extract_ranking_label(self, idx: int) -> Any | None:
        """Extract ranking label from raw data.

        Args:
            idx: Pair index

        Returns:
            Label value (int: 1 = a better, -1 = b better, or binary 0/1) or None
        """
        if isinstance(self.raw_data, pd.DataFrame):
            # Try common column names
            for col_name in ["label", "target", "y"]:
                if col_name in self.raw_data.columns:
                    label_value = self.raw_data.iloc[idx][col_name]
                    if pd.isna(label_value):
                        return None
                    return label_value
        return None

    def _extract_label(self, idx: int) -> Any | None:
        """Extract label from raw data for given index.

        Args:
            idx: Sequence index

        Returns:
            Label value (int, float, list, etc.) or None if not present
        """
        # Try to extract label from raw data
        # Default field names: "label", "target", "y"
        if isinstance(self.raw_data, pd.DataFrame):
            # CSV/Excel: try common column names
            for col_name in ["label", "target", "y"]:
                if col_name in self.raw_data.columns:
                    label_value = self.raw_data.iloc[idx][col_name]
                    # Handle NaN
                    if pd.isna(label_value):
                        return None
                    return label_value
        elif isinstance(self.raw_data, list):
            # List of dicts or objects
            if idx < len(self.raw_data):
                item = self.raw_data[idx]
                if isinstance(item, dict):
                    # Try common field names
                    for field_name in ["label", "target", "y"]:
                        if field_name in item:
                            return item[field_name]
        elif isinstance(self.raw_data, dict):
            # Single dict: check if it has a labels list
            if "labels" in self.raw_data and idx < len(self.raw_data["labels"]):
                return self.raw_data["labels"][idx]

        # No label found
        return None
