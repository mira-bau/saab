"""JSON extractor for Python dict/list and JSON strings."""

import json
from typing import Any

from tqdm import tqdm

from saab_v3.data.extractors.base import StructuralExtractor
from saab_v3.data.structures import StructureTag, Token


def _infer_token_type(value: Any) -> str:
    """Infer token type from Python type."""
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "number"
    elif isinstance(value, float):
        return "number"
    elif isinstance(value, str):
        return "text"
    else:
        return "text"


def _convert_value_to_string(value: Any) -> str:
    """Convert value to string."""
    if value is None:
        return ""
    return str(value)


class JSONExtractor(StructuralExtractor):
    """Extract tokens from JSON data (Python dict/list or JSON strings)."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a dict, list, or JSON string."""
        if isinstance(data, (dict, list)):
            return True
        if isinstance(data, str):
            try:
                json.loads(data)
                return True
            except (json.JSONDecodeError, ValueError):
                return False
        return False

    def extract(self, data: Any, schema: dict | None = None) -> list[Token]:
        """Extract tokens from JSON data using depth-first traversal.

        Args:
            data: Python dict/list or JSON string
            schema: Optional schema (currently unused, reserved for future use)

        Returns:
            List of Token objects in depth-first order
        """
        # Parse JSON string if needed
        if isinstance(data, str):
            data = json.loads(data)

        if not isinstance(data, (dict, list)):
            raise ValueError(f"Expected dict or list, got {type(data)}")

        # Quick count of leaf values for progress bar
        total_estimate = self._estimate_token_count(data)

        tokens: list[Token] = []
        position = 0

        # Create progress bar if large enough
        pbar = (
            tqdm(total=total_estimate, desc="Extracting tokens")
            if total_estimate > 100
            else None
        )

        try:
            # Depth-first traversal with progress
            if isinstance(data, dict):
                position = self._extract_dict(
                    data, tokens, position, path="", pbar=pbar
                )
            elif isinstance(data, list):
                position = self._extract_list(
                    data, tokens, position, path="", pbar=pbar
                )
        finally:
            if pbar:
                pbar.close()

        return tokens

    def _estimate_token_count(self, data: Any) -> int:
        """Quick estimate of total leaf values for progress bar."""
        if isinstance(data, dict):
            return sum(self._estimate_token_count(v) for v in data.values())
        elif isinstance(data, list):
            return sum(self._estimate_token_count(item) for item in data)
        else:
            return 1  # Leaf value

    def _extract_dict(
        self,
        obj: dict,
        tokens: list[Token],
        position: int,
        path: str,
        pbar: tqdm | None = None,
    ) -> int:
        """Extract tokens from dictionary (depth-first)."""
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            entity_id = f"entity_{current_path}"

            if isinstance(value, dict):
                # Nested object - recurse
                position = self._extract_dict(
                    value, tokens, position, current_path, pbar
                )
            elif isinstance(value, list):
                # Array - recurse
                position = self._extract_list(
                    value, tokens, position, current_path, pbar
                )
            else:
                # Leaf value - create token
                value_str = _convert_value_to_string(value)
                if not value_str:
                    continue

                structure_tag = StructureTag(
                    field=current_path,
                    entity=entity_id,
                    token_type=_infer_token_type(value),
                )

                token = Token(
                    value=value_str,
                    structure_tag=structure_tag,
                    position=position,
                )
                tokens.append(token)
                position += 1

                # Update progress bar
                if pbar:
                    pbar.update(1)

        return position

    def _extract_list(
        self,
        arr: list,
        tokens: list[Token],
        position: int,
        path: str,
        pbar: tqdm | None = None,
    ) -> int:
        """Extract tokens from list (depth-first)."""
        for idx, item in enumerate(arr):
            current_path = f"{path}[{idx}]" if path else f"[{idx}]"
            entity_id = f"entity_{current_path}"

            if isinstance(item, dict):
                # Nested object - recurse
                position = self._extract_dict(
                    item, tokens, position, current_path, pbar
                )
            elif isinstance(item, list):
                # Nested array - recurse
                position = self._extract_list(
                    item, tokens, position, current_path, pbar
                )
            else:
                # Leaf value - create token
                value_str = _convert_value_to_string(item)
                if not value_str:
                    continue

                structure_tag = StructureTag(
                    field=current_path,
                    entity=entity_id,
                    token_type=_infer_token_type(item),
                )

                token = Token(
                    value=value_str,
                    structure_tag=structure_tag,
                    position=position,
                )
                tokens.append(token)
                position += 1

                # Update progress bar
                if pbar:
                    pbar.update(1)

        return position
