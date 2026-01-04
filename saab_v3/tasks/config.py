"""Task configuration Pydantic models."""

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing import Optional

from saab_v3.tasks.config_schema import RankingMethod


class ClassificationTaskConfig(BaseModel):
    """Configuration for classification tasks."""

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    num_classes: int
    multi_label: bool = False
    hidden_dims: Optional[list[int]] = None
    dropout: float = 0.1
    pooling: str = "cls"
    label_smoothing: float = 0.0

    @field_validator("num_classes")
    @classmethod
    def validate_num_classes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"num_classes must be > 0, got {v}")
        return v

    @field_validator("multi_label")
    @classmethod
    def validate_multi_label(cls, v: bool) -> bool:
        if not isinstance(v, bool):
            raise ValueError(f"multi_label must be a boolean, got {v!r}")
        return v

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None:
            if not isinstance(v, list):
                raise ValueError(
                    f"hidden_dims must be None or a list of integers, "
                    f"got {type(v).__name__}"
                )
            if len(v) == 0:
                raise ValueError(
                    "hidden_dims must be None or a non-empty list of positive integers"
                )
            if not all(isinstance(d, int) and d > 0 for d in v):
                raise ValueError("hidden_dims must contain only positive integers")
        return v

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or not (0 <= v < 1):
            raise ValueError(f"dropout must be in [0, 1), got {v!r}")
        return float(v)

    @field_validator("pooling")
    @classmethod
    def validate_pooling(cls, v: str) -> str:
        valid_pooling = ["cls", "mean", "max"]
        if v not in valid_pooling:
            raise ValueError(f"pooling must be one of {valid_pooling}, got {v!r}")
        return v

    @field_validator("label_smoothing")
    @classmethod
    def validate_label_smoothing(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or not (0 <= v < 1):
            raise ValueError(f"label_smoothing must be in [0, 1), got {v!r}")
        return float(v)


class RankingTaskConfig(BaseModel):
    """Configuration for ranking tasks."""

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    method: str  # "dot_product", "cosine", "mlp", "difference"
    hidden_dims: Optional[list[int]] = None
    dropout: float = 0.1

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid_methods = [m.value for m in RankingMethod]
        if v not in valid_methods:
            raise ValueError(
                f"Invalid ranking method: {v!r}. Valid methods: {valid_methods}"
            )
        return v

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None:
            if not isinstance(v, list):
                raise ValueError(
                    f"hidden_dims must be None or a list of integers, "
                    f"got {type(v).__name__}"
                )
            if len(v) == 0:
                raise ValueError(
                    "hidden_dims must be None or a non-empty list of positive integers"
                )
            if not all(isinstance(d, int) and d > 0 for d in v):
                raise ValueError("hidden_dims must contain only positive integers")
        return v

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or not (0 <= v < 1):
            raise ValueError(f"dropout must be in [0, 1), got {v!r}")
        return float(v)

    @model_validator(mode="after")
    def validate_method_requires_hidden_dims(self) -> "RankingTaskConfig":
        """Validate that MLP and difference methods require hidden_dims."""
        if self.method in [RankingMethod.MLP, RankingMethod.DIFFERENCE]:
            if self.hidden_dims is None:
                raise ValueError(
                    f"Ranking method '{self.method}' requires 'hidden_dims' parameter"
                )
        return self


class RegressionTaskConfig(BaseModel):
    """Configuration for regression tasks."""

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    num_targets: int = 1
    hidden_dims: Optional[list[int]] = None
    dropout: float = 0.1
    pooling: str = "cls"

    @field_validator("num_targets")
    @classmethod
    def validate_num_targets(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"num_targets must be > 0, got {v}")
        return v

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None:
            if not isinstance(v, list):
                raise ValueError(
                    f"hidden_dims must be None or a list of integers, "
                    f"got {type(v).__name__}"
                )
            if len(v) == 0:
                raise ValueError(
                    "hidden_dims must be None or a non-empty list of positive integers"
                )
            if not all(isinstance(d, int) and d > 0 for d in v):
                raise ValueError("hidden_dims must contain only positive integers")
        return v

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or not (0 <= v < 1):
            raise ValueError(f"dropout must be in [0, 1), got {v!r}")
        return float(v)

    @field_validator("pooling")
    @classmethod
    def validate_pooling(cls, v: str) -> str:
        valid_pooling = ["cls", "mean", "max"]
        if v not in valid_pooling:
            raise ValueError(f"pooling must be one of {valid_pooling}, got {v!r}")
        return v


class TokenClassificationTaskConfig(BaseModel):
    """Configuration for token classification tasks."""

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    num_labels: int
    hidden_dims: Optional[list[int]] = None
    dropout: float = 0.1

    @field_validator("num_labels")
    @classmethod
    def validate_num_labels(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"num_labels must be > 0, got {v}")
        return v

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None:
            if not isinstance(v, list):
                raise ValueError(
                    f"hidden_dims must be None or a list of integers, "
                    f"got {type(v).__name__}"
                )
            if len(v) == 0:
                raise ValueError(
                    "hidden_dims must be None or a non-empty list of positive integers"
                )
            if not all(isinstance(d, int) and d > 0 for d in v):
                raise ValueError("hidden_dims must contain only positive integers")
        return v

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or not (0 <= v < 1):
            raise ValueError(f"dropout must be in [0, 1), got {v!r}")
        return float(v)


# Union type for task configs
TaskConfig = (
    ClassificationTaskConfig
    | RankingTaskConfig
    | RegressionTaskConfig
    | TokenClassificationTaskConfig
)
