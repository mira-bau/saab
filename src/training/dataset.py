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
        lazy: bool = False,
    ):
        """Initialize dataset.

        Args:
            data: Input data (file path, DataFrame, dict, Graph, or list of these)
            preprocessor: Fitted Preprocessor instance
            split: Dataset split name ("train", "val", "test")
            lazy: If True, load data on-demand (not implemented in Phase 1)
        """
        self.preprocessor = preprocessor
        self.split = split
        self.lazy = lazy

        # Load data
        loaded_data = self._load_data(data)

        # Transform: extract and tokenize
        sequences = self.preprocessor.transform(loaded_data)

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

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.encoded_sequences)

    def __getitem__(
        self, idx: int
    ) -> tuple[TokenizedSequence, list[int], list[EncodedTag]]:
        """Return encoded sequence ready for batching.

        Args:
            idx: Sequence index

        Returns:
            (TokenizedSequence, token_ids, encoded_tags) tuple
        """
        return self.encoded_sequences[idx]
