"""Table extractor for pandas DataFrames, CSV, and Excel files."""

from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from saab_v3.data.extractors.base import StructuralExtractor
from saab_v3.data.structures import StructureTag, Token


def _infer_token_type(dtype: str) -> str:
    """Infer token type from pandas dtype."""
    if pd.api.types.is_integer_dtype(dtype):
        return "number"
    elif pd.api.types.is_float_dtype(dtype):
        return "number"
    elif pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "date"
    else:
        return "text"


def _convert_value_to_string(value: Any) -> str:
    """Convert value to string, handling special types."""
    if pd.isna(value):
        return ""  # Empty string for missing values
    # Handle NumPy datetime64
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    # Handle NumPy scalars
    if isinstance(value, (np.integer, np.floating)):
        return str(value.item())  # Convert to Python native type first
    return str(value)


class TableExtractor(StructuralExtractor):
    """Extract tokens from tabular data (pandas DataFrame, CSV, Excel)."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a pandas DataFrame, CSV file, or Excel file."""
        if isinstance(data, pd.DataFrame):
            return True
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix.lower() in [".csv", ".xlsx", ".xls"]:
                return path.exists()
        return False

    def extract(self, data: Any, schema: dict | None = None) -> list[Token]:
        """Extract tokens from table data.

        Args:
            data: pandas DataFrame, CSV file path, or Excel file path
            schema: Optional schema with primary_keys, foreign_keys, relationships

        Returns:
            List of Token objects in row-major order
        """
        # Load data if file path provided
        df = self._load_data(data)

        if df.empty:
            raise ValueError("DataFrame is empty")

        # Pre-compute all column-level metadata
        col_token_types = {col: _infer_token_type(df[col].dtype) for col in df.columns}
        primary_keys = schema.get("primary_keys", []) if schema else []
        foreign_keys = schema.get("foreign_keys", []) if schema else []
        foreign_key_map = (
            {fk["column"]: fk.get("references", "") for fk in foreign_keys}
            if foreign_keys
            else {}
        )

        # Convert to NumPy array for fast access
        values = df.values  # Shape: (n_rows, n_cols)
        index_values = df.index.values
        col_names = df.columns.tolist()
        n_rows, n_cols = values.shape

        # Pre-compute entity IDs for all rows
        entity_ids = [self._get_entity_id(idx, df) for idx in index_values]

        tokens: list[Token] = []
        position = 0

        # Use tqdm for large DataFrames
        row_iterator = (
            tqdm(range(n_rows), desc="Extracting tokens")
            if n_rows > 1000
            else range(n_rows)
        )

        # Fast NumPy-based iteration
        for row_pos in row_iterator:
            entity_id = entity_ids[row_pos]

            for col_pos in range(n_cols):
                value = values[row_pos, col_pos]

                # Fast NaN check using pandas
                if pd.isna(value):
                    continue

                value_str = _convert_value_to_string(value)
                if not value_str:
                    continue

                col_name = col_names[col_pos]

                # Build structure tag (token type pre-computed)
                structure_tag = StructureTag(
                    field=col_name,
                    entity=entity_id,
                    token_type=col_token_types[col_name],
                )

                # Add role/edge if applicable
                if col_name in primary_keys:
                    structure_tag.role = "primary_key"
                if col_name in foreign_key_map:
                    structure_tag.edge = foreign_key_map[col_name]
                    structure_tag.role = "foreign_key"

                # Create token
                token = Token(
                    value=value_str,
                    structure_tag=structure_tag,
                    position=position,
                )
                tokens.append(token)
                position += 1

        return tokens

    def _load_data(self, data: Any) -> pd.DataFrame:
        """Load data into pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, (str, Path)):
            path = Path(data)
            suffix = path.suffix.lower()

            if suffix == ".csv":
                return pd.read_csv(path)
            elif suffix in [".xlsx", ".xls"]:
                return pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        raise ValueError(f"Cannot load data from type: {type(data)}")

    def _get_entity_id(self, row_idx: Any, df: pd.DataFrame) -> str:
        """Get entity identifier for a row."""
        # Use index name if it's a named index
        if df.index.name:
            return f"{df.index.name}_{row_idx}"

        # Use index value if it's meaningful (not just position)
        if isinstance(df.index, pd.RangeIndex):
            return f"row_{row_idx}"
        else:
            return str(row_idx)
