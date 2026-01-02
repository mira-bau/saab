"""Base abstract class for structural extractors."""

from abc import ABC, abstractmethod
from typing import Any

from saab_v3.data.structures import Token


class StructuralExtractor(ABC):
    """Abstract base class for structural extractors.

    Extractors convert different input formats into the canonical tokenized format
    (list[Token]) with structure tags.
    """

    @abstractmethod
    def extract(self, data: Any, schema: dict | None = None) -> list[Token]:
        """Extract tokens with structure tags from input data.

        Args:
            data: Input data in format specific to this extractor
            schema: Optional schema providing explicit relationships/roles

        Returns:
            List of Token objects with sequential positions (0, 1, 2, ...)

        Raises:
            ValueError: If data format is invalid or cannot be processed
        """
        pass

    @abstractmethod
    def can_handle(self, data: Any) -> bool:
        """Check if this extractor can handle the input format.

        Args:
            data: Input data to check

        Returns:
            True if extractor can process the data, False otherwise
        """
        pass
