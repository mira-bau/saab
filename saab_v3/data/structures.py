"""Core data structures for preprocessing pipeline using Pydantic models."""

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class StructureTag(BaseModel):
    """Symbolic structure tag representing structural primitives for a token."""

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    field: str | None = None  # e.g., "name", "price"
    entity: str | None = None  # e.g., "user_123", "order_456"
    time: str | None = None  # e.g., "2023-Q1", "2023-01-15"
    edge: str | None = None  # e.g., "parent_of", "contains"
    role: str | None = None  # e.g., "primary_key", "foreign_key"
    token_type: str | None = None  # e.g., "text", "number", "date"

    @field_validator("field", "entity", "time", "edge", "role", "token_type")
    @classmethod
    def validate_string_fields(cls, v: str | None) -> str | None:
        """Validate that string fields are non-empty if provided."""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("String fields must be non-empty if provided")
        return v

    @model_validator(mode="after")
    def validate_at_least_one_field(self) -> "StructureTag":
        """Validate that at least one field is present."""
        if self.is_empty():
            raise ValueError("At least one field must be present in StructureTag")
        return self

    def is_empty(self) -> bool:
        """Check if all fields are None."""
        return all(
            getattr(self, field) is None
            for field in ["field", "entity", "time", "edge", "role", "token_type"]
        )


class Token(BaseModel):
    """Single token with value and structure tags."""

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    value: str  # Raw value: "John", "42.5", "2023-01-15"
    structure_tag: StructureTag  # Structural metadata
    position: int  # Position in sequence (0-indexed)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: str) -> str:
        """Validate that value is non-empty."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Token value must be non-empty")
        return v

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: int) -> int:
        """Validate that position is non-negative."""
        if v < 0:
            raise ValueError(f"Token position must be >= 0, got {v}")
        return v


class TokenizedSequence(BaseModel):
    """Canonical format: tokenized sequence with structure tags."""

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    tokens: list[Token]  # List of tokens in sequence
    sequence_id: str | None = None  # Identifier for tracking/debugging

    @field_validator("tokens")
    @classmethod
    def validate_tokens(cls, v: list[Token]) -> list[Token]:
        """Validate that tokens list is non-empty."""
        if not v or len(v) == 0:
            raise ValueError("TokenizedSequence must contain at least one token")
        return v

    @field_validator("sequence_id")
    @classmethod
    def validate_sequence_id(cls, v: str | None) -> str | None:
        """Validate that sequence_id is non-empty if provided."""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("sequence_id must be non-empty if provided")
        return v

    @model_validator(mode="after")
    def validate_sequential_positions(self) -> "TokenizedSequence":
        """Validate that token positions are sequential (0, 1, 2, ...)."""
        positions = [token.position for token in self.tokens]
        expected_positions = list(range(len(self.tokens)))
        if positions != expected_positions:
            raise ValueError(
                f"Token positions must be sequential starting from 0. "
                f"Got {positions}, expected {expected_positions}"
            )
        return self

    def __len__(self) -> int:
        """Return number of tokens."""
        return len(self.tokens)

    def get_tokens_by_field(self, field: str) -> list[Token]:
        """Filter tokens by field tag."""
        return [token for token in self.tokens if token.structure_tag.field == field]


class EncodedTag(BaseModel):
    """Encoded tag with indices for embeddings."""

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    field_idx: int | None = None
    entity_idx: int | None = None
    time_idx: int | None = None
    edge_idx: int | None = None
    role_idx: int | None = None
    token_type_idx: int | None = None

    @field_validator(
        "field_idx",
        "entity_idx",
        "time_idx",
        "edge_idx",
        "role_idx",
        "token_type_idx",
    )
    @classmethod
    def validate_indices(cls, v: int | None) -> int | None:
        """Validate that indices are non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError(f"Index must be >= 0, got {v}")
        return v

    def get_indices(self) -> dict[str, int | None]:
        """Get all indices as dictionary."""
        return {
            "field_idx": self.field_idx,
            "entity_idx": self.entity_idx,
            "time_idx": self.time_idx,
            "edge_idx": self.edge_idx,
            "role_idx": self.role_idx,
            "token_type_idx": self.token_type_idx,
        }


class Batch(BaseModel):
    """Batched sequences ready for model input."""

    model_config = ConfigDict(
        validate_assignment=True, frozen=False, arbitrary_types_allowed=True
    )

    token_ids: torch.Tensor  # [batch_size, seq_len]
    attention_mask: torch.Tensor  # [batch_size, seq_len] (1 = valid, 0 = pad)
    field_ids: torch.Tensor  # [batch_size, seq_len]
    entity_ids: torch.Tensor  # [batch_size, seq_len]
    time_ids: torch.Tensor  # [batch_size, seq_len]
    edge_ids: torch.Tensor | None = None  # [batch_size, seq_len] or None
    role_ids: torch.Tensor | None = None  # [batch_size, seq_len] or None
    token_type_ids: torch.Tensor  # [batch_size, seq_len]
    sequence_lengths: list[int]  # Actual lengths (for unpacking)
    sequence_ids: list[str] | None = None  # For tracking/debugging
    labels: torch.Tensor | None = None  # Task-specific labels (optional)
    # Format depends on task type:
    # - Classification (multi-class): [batch] (class indices)
    # - Classification (multi-label): [batch, num_classes] (binary vectors)
    # - Regression: [batch, num_targets] (continuous values)
    # - Token Classification: [batch, seq_len] (label indices per token)
    # - Ranking: Not stored in Batch (handled separately with pairs)

    @field_validator("attention_mask")
    @classmethod
    def validate_attention_mask(cls, v: torch.Tensor) -> torch.Tensor:
        """Validate that attention_mask contains only 0s and 1s."""
        if not torch.all((v == 0) | (v == 1)):
            raise ValueError("attention_mask must contain only 0s and 1s")
        return v

    @model_validator(mode="after")
    def validate_tensor_shapes(self) -> "Batch":
        """Validate that all tensors have consistent shapes."""
        batch_size = self.token_ids.shape[0]
        seq_len = self.token_ids.shape[1]

        # Check all required tensors
        required_tensors = {
            "token_ids": self.token_ids,
            "attention_mask": self.attention_mask,
            "field_ids": self.field_ids,
            "entity_ids": self.entity_ids,
            "time_ids": self.time_ids,
            "token_type_ids": self.token_type_ids,
        }

        for name, tensor in required_tensors.items():
            if tensor.shape != (batch_size, seq_len):
                raise ValueError(
                    f"{name} must have shape [batch_size, seq_len] = [{batch_size}, {seq_len}], "
                    f"got {tensor.shape}"
                )

        # Check optional tensors if provided
        optional_tensors = {
            "edge_ids": self.edge_ids,
            "role_ids": self.role_ids,
        }

        for name, tensor in optional_tensors.items():
            if tensor is not None and tensor.shape != (batch_size, seq_len):
                raise ValueError(
                    f"{name} must have shape [batch_size, seq_len] = [{batch_size}, {seq_len}], "
                    f"got {tensor.shape}"
                )

        # Validate sequence_lengths
        if len(self.sequence_lengths) != batch_size:
            raise ValueError(
                f"sequence_lengths length ({len(self.sequence_lengths)}) must match batch_size ({batch_size})"
            )

        for i, length in enumerate(self.sequence_lengths):
            if length < 0:
                raise ValueError(f"sequence_lengths[{i}] must be >= 0, got {length}")
            if length > seq_len:
                raise ValueError(
                    f"sequence_lengths[{i}] ({length}) must be <= seq_len ({seq_len})"
                )

        # Validate sequence_ids if provided
        if self.sequence_ids is not None:
            if len(self.sequence_ids) != batch_size:
                raise ValueError(
                    f"sequence_ids length ({len(self.sequence_ids)}) must match batch_size ({batch_size})"
                )

        # Validate labels if provided
        # Note: Shape validation is task-specific and can be deferred to Trainer
        # Here we just check that batch_size matches if labels are present
        if self.labels is not None:
            if self.labels.shape[0] != batch_size:
                raise ValueError(
                    f"labels batch dimension ({self.labels.shape[0]}) must match batch_size ({batch_size})"
                )

        return self

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self.token_ids.shape[0]

    @property
    def seq_len(self) -> int:
        """Return sequence length."""
        return self.token_ids.shape[1]

    def to(self, device: torch.device) -> "Batch":
        """Move all tensors to specified device (MPS/GPU/CPU)."""
        return Batch(
            token_ids=self.token_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            field_ids=self.field_ids.to(device),
            entity_ids=self.entity_ids.to(device),
            time_ids=self.time_ids.to(device),
            edge_ids=self.edge_ids.to(device) if self.edge_ids is not None else None,
            role_ids=self.role_ids.to(device) if self.role_ids is not None else None,
            token_type_ids=self.token_type_ids.to(device),
            sequence_lengths=self.sequence_lengths,
            sequence_ids=self.sequence_ids,
            labels=self.labels.to(device) if self.labels is not None else None,
        )
