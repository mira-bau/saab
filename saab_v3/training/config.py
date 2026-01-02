"""Preprocessing configuration model."""

from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_validator


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing pipeline."""

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    vocab_size: int = 30000
    max_seq_len: int = 512
    extractor_type: str | None = None  # "table", "json", "graph", or None (auto-detect)
    preserve_original_tags: bool = (
        False  # True for SAAB (preserve StructureTag objects)
    )
    extractor_schema: dict | None = None  # Optional schema for extractors
    device: str = "cpu"  # "cpu", "cuda", "mps"
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

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate that device is valid."""
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"device must be one of ['cpu', 'cuda', 'mps'], got {v}")
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
