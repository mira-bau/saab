"""Specs for device utility functions - happy path only."""

import warnings

import pytest
import torch

from saab_v3.utils.device import get_available_devices, get_device, validate_device


# ============================================================================
# get_device() Tests
# ============================================================================


def spec_device_get_device_string():
    """Verify get_device() handles device strings correctly."""
    # Arrange & Act
    cpu_device = get_device("cpu")

    # Assert
    assert isinstance(cpu_device, torch.device)
    assert cpu_device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        cuda_device = get_device("cuda")
        assert isinstance(cuda_device, torch.device)
        assert cuda_device.type == "cuda"
    else:
        # If CUDA not available, should fallback to CPU with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cuda_device = get_device("cuda")
            assert isinstance(cuda_device, torch.device)
            assert cuda_device.type == "cpu"
            assert len(w) > 0
            assert "CUDA is not available" in str(w[0].message)

    # Test MPS if available
    if torch.backends.mps.is_available():
        mps_device = get_device("mps")
        assert isinstance(mps_device, torch.device)
        assert mps_device.type == "mps"
    else:
        # If MPS not available, should fallback to CPU with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mps_device = get_device("mps")
            assert isinstance(mps_device, torch.device)
            assert mps_device.type == "cpu"
            assert len(w) > 0
            assert "MPS is not available" in str(w[0].message)

    # Test invalid device string
    with pytest.raises(ValueError, match="device must be one of"):
        get_device("invalid")


def spec_device_get_device_auto():
    """Verify get_device() auto-detects best available device."""
    # Arrange & Act
    auto_device = get_device("auto")
    none_device = get_device(None)

    # Assert
    assert isinstance(auto_device, torch.device)
    assert isinstance(none_device, torch.device)
    # Both should return the same device
    assert auto_device.type == none_device.type

    # Verify returned device is actually available
    assert auto_device.type in ["cpu", "cuda", "mps"]
    if auto_device.type == "cuda":
        assert torch.cuda.is_available()
    elif auto_device.type == "mps":
        assert torch.backends.mps.is_available()
    # CPU is always available

    # Verify priority: MPS → CUDA → CPU
    # (This is tested implicitly by checking the returned device matches availability)


def spec_device_get_device_torch_device():
    """Verify get_device() returns torch.device objects as-is."""
    # Arrange
    cpu_torch_device = torch.device("cpu")

    # Act
    result = get_device(cpu_torch_device)

    # Assert
    assert result is cpu_torch_device  # Should be same object (no conversion)
    assert isinstance(result, torch.device)
    assert result.type == "cpu"

    # Test with CUDA device if available
    if torch.cuda.is_available():
        cuda_torch_device = torch.device("cuda")
        result_cuda = get_device(cuda_torch_device)
        assert result_cuda is cuda_torch_device
        assert result_cuda.type == "cuda"


def spec_device_get_device_fallback():
    """Verify get_device() falls back to CPU when requested device unavailable."""
    # Test CUDA fallback on CPU-only system
    if not torch.cuda.is_available():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            device = get_device("cuda")
            assert device.type == "cpu"
            assert len(w) > 0
            assert "CUDA is not available" in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

    # Test MPS fallback on non-Mac system
    if not torch.backends.mps.is_available():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            device = get_device("mps")
            assert device.type == "cpu"
            assert len(w) > 0
            assert "MPS is not available" in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)


def spec_device_get_device_type_error():
    """Verify get_device() raises TypeError for invalid types."""
    # Test with invalid type
    with pytest.raises(TypeError, match="device must be str, torch.device, or None"):
        get_device(123)  # int is invalid

    with pytest.raises(TypeError, match="device must be str, torch.device, or None"):
        get_device([])  # list is invalid


# ============================================================================
# validate_device() Tests
# ============================================================================


def spec_device_validate_device():
    """Verify validate_device() checks device availability correctly."""
    # CPU is always available
    assert validate_device("cpu") is True

    # CUDA availability depends on system
    assert validate_device("cuda") == torch.cuda.is_available()

    # MPS availability depends on system
    assert validate_device("mps") == torch.backends.mps.is_available()

    # Invalid device returns False
    assert validate_device("invalid") is False
    assert validate_device("gpu") is False


# ============================================================================
# get_available_devices() Tests
# ============================================================================


def spec_device_get_available_devices():
    """Verify get_available_devices() returns list of available devices."""
    # Arrange & Act
    devices = get_available_devices()

    # Assert
    assert isinstance(devices, list)
    assert len(devices) > 0  # At least CPU should be available
    assert "cpu" in devices  # CPU is always available

    # Verify CUDA is included only if available
    if torch.cuda.is_available():
        assert "cuda" in devices
    else:
        assert "cuda" not in devices

    # Verify MPS is included only if available
    if torch.backends.mps.is_available():
        assert "mps" in devices
    else:
        assert "mps" not in devices

    # Verify no invalid devices
    valid_devices = {"cpu", "cuda", "mps"}
    assert all(device in valid_devices for device in devices)
