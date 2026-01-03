"""Device utility functions for consistent device detection, validation, and conversion."""

import warnings
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Get torch.device object from string, torch.device, or auto-detect.

    This is the single function for all device conversion needs. It handles:
    - torch.device objects (returns as-is)
    - Device strings ("cpu", "cuda", "mps", "auto")
    - None (auto-detects)
    - Auto-detection when device is "auto" or None

    Args:
        device: Device string ("cpu", "cuda", "mps", "auto"), torch.device object, or None

    Returns:
        torch.device object

    Examples:
        >>> get_device("cpu")
        device(type='cpu')
        >>> get_device("auto")  # Auto-detects best available
        device(type='mps')  # or 'cuda' or 'cpu'
        >>> get_device(torch.device("cpu"))
        device(type='cpu')
        >>> get_device(None)  # Same as "auto"
        device(type='mps')
    """
    # If already a torch.device, return as-is
    if isinstance(device, torch.device):
        return device

    # Handle None or "auto" - auto-detect best available device
    if device is None or device == "auto":
        # Priority: MPS (Mac) → CUDA (GPU) → CPU
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    # Handle string device
    if isinstance(device, str):
        # Validate device string
        valid_devices = ["cpu", "cuda", "mps"]
        if device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices} or 'auto', got {device}"
            )

        # Check availability and warn if unavailable
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "CUDA is not available. Falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        elif device == "mps" and not torch.backends.mps.is_available():
            warnings.warn(
                "MPS is not available. Falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")

        return torch.device(device)

    raise TypeError(
        f"device must be str, torch.device, or None, got {type(device).__name__}"
    )


def validate_device(device: str) -> bool:
    """Check if a device string is available.

    Args:
        device: Device string ("cpu", "cuda", "mps")

    Returns:
        True if device is available, False otherwise

    Examples:
        >>> validate_device("cpu")
        True
        >>> validate_device("cuda")  # True if CUDA available, False otherwise
        False
        >>> validate_device("mps")  # True if MPS available, False otherwise
        True
    """
    if device == "cpu":
        return True
    elif device == "cuda":
        return torch.cuda.is_available()
    elif device == "mps":
        return torch.backends.mps.is_available()
    else:
        return False


def get_available_devices() -> list[str]:
    """Get list of available device strings.

    Returns:
        List of available device strings (e.g., ["cpu", "mps"] or ["cpu", "cuda"])

    Examples:
        >>> get_available_devices()
        ['cpu', 'mps']  # On Mac with MPS
        >>> get_available_devices()
        ['cpu', 'cuda']  # On system with CUDA
        >>> get_available_devices()
        ['cpu']  # CPU-only system
    """
    devices = ["cpu"]  # CPU is always available

    if torch.cuda.is_available():
        devices.append("cuda")

    if torch.backends.mps.is_available():
        devices.append("mps")

    return devices

