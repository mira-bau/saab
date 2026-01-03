"""Preprocessing configuration model."""

from pathlib import Path
from pydantic import ConfigDict, field_validator

from saab_v3.config.base import BaseConfig


class PreprocessingConfig(BaseConfig):
    """Configuration for preprocessing pipeline.

    Inherits device field from BaseConfig.
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    vocab_size: int = 30000
    max_seq_len: int = 512
    extractor_type: str | None = None  # "table", "json", "graph", or None (auto-detect)
    extractor_schema: dict | None = None  # Optional schema for extractors
    # device inherited from BaseConfig
    data_dir: Path | None = None  # Override default data/ directory (optional)

    @field_validator("vocab_size")
    @classmethod
    def validate_vocab_size(cls, v: int) -> int:
        """Validate that vocab_size is positive."""
        if v <= 0:
            raise ValueError(f"vocab_size must be > 0, got {v}")
        return v

    @field_validator("max_seq_len")
    @classmethod
    def validate_max_seq_len(cls, v: int) -> int:
        """Validate that max_seq_len is positive."""
        if v <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {v}")
        return v

    @field_validator("extractor_type")
    @classmethod
    def validate_extractor_type(cls, v: str | None) -> str | None:
        """Validate that extractor_type is valid."""
        if v is not None and v not in ["table", "json", "graph"]:
            raise ValueError(
                f"extractor_type must be one of ['table', 'json', 'graph', None], got {v}"
            )
        return v

    @field_validator("data_dir")
    @classmethod
    def validate_data_dir(cls, v: Path | str | None) -> Path | None:
        """Convert string to Path if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v
