"""Base configuration with shared settings across all phases."""

import warnings
from pydantic import BaseModel, ConfigDict, field_validator

from saab_v3.utils.device import get_device, validate_device


class BaseConfig(BaseModel):
    """Base configuration with shared settings across all phases.

    All phase-specific configs (PreprocessingConfig, ModelConfig, TrainingConfig, etc.)
    should inherit from this class to get shared settings like device management.
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    device: str = "cpu"  # "cpu", "cuda", "mps", "auto"

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device string and check availability.

        Args:
            v: Device string ("cpu", "cuda", "mps", "auto")

        Returns:
            Validated device string

        Raises:
            ValueError: If device string is invalid
        """
        valid_devices = ["cpu", "cuda", "mps", "auto"]
        if v not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got {v}"
            )

        # If device is not "auto" or "cpu", check availability
        if v not in ["auto", "cpu"]:
            if not validate_device(v):
                warnings.warn(
                    f"Device '{v}' is not available. Falling back to CPU. "
                    f"Use 'auto' to automatically detect the best available device.",
                    UserWarning,
                    stacklevel=2,
                )
                return "cpu"

        return v

