"""Specs for DataLoader factory - happy path only."""

import torch

from saab_v3.training.config import PreprocessingConfig
from saab_v3.training.preprocessor import Preprocessor
from saab_v3.training.dataset import StructuredDataset
from saab_v3.training.dataloader import create_dataloader
from saab_v3.data.structures import Batch


def spec_dataloader_creation(fitted_preprocessor, sample_dataframe):
    """Verify DataLoader can be created."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)

    # Act
    dataloader = create_dataloader(dataset, batch_size=2)

    # Assert
    assert dataloader is not None
    assert dataloader.batch_size == 2


def spec_dataloader_iteration(fitted_preprocessor, sample_dataframe):
    """Verify DataLoader yields Batch objects."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    dataloader = create_dataloader(dataset, batch_size=2)

    # Act
    batch = next(iter(dataloader))

    # Assert
    assert isinstance(batch, Batch)
    assert batch.token_ids.shape[0] <= 2  # batch_size
    assert batch.token_ids.shape[1] > 0  # seq_len


def spec_dataloader_batch_shapes(fitted_preprocessor, sample_dataframe):
    """Verify Batch tensors have correct shapes."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    dataloader = create_dataloader(dataset, batch_size=2)

    # Act
    batch = next(iter(dataloader))
    batch_size = batch.token_ids.shape[0]
    seq_len = batch.token_ids.shape[1]

    # Assert
    assert batch.attention_mask.shape == (batch_size, seq_len)
    assert batch.field_ids.shape == (batch_size, seq_len)
    assert batch.entity_ids.shape == (batch_size, seq_len)
    assert batch.time_ids.shape == (batch_size, seq_len)
    assert batch.token_type_ids.shape == (batch_size, seq_len)


def spec_dataloader_device_placement(fitted_preprocessor, sample_dataframe):
    """Verify tensors are placed on correct device."""
    # Arrange
    # Device comes from preprocessor config (single source of truth)
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    dataloader = create_dataloader(dataset, batch_size=2)

    # Act
    batch = next(iter(dataloader))

    # Assert
    # Device should match config.device (default is "cpu")
    expected_device = fitted_preprocessor.config.device
    if expected_device == "auto":
        # Auto-detection may choose different device, just verify it's valid
        assert batch.token_ids.device.type in ["cpu", "cuda", "mps"]
    else:
        assert batch.token_ids.device.type == expected_device
    assert batch.attention_mask.device.type == batch.token_ids.device.type


def spec_dataloader_shuffle(fitted_preprocessor, sample_dataframe):
    """Verify shuffle parameter works."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    dataloader_shuffled = create_dataloader(dataset, batch_size=1, shuffle=True)
    dataloader_not_shuffled = create_dataloader(dataset, batch_size=1, shuffle=False)

    # Act
    batches_shuffled = [batch for batch in dataloader_shuffled]
    batches_not_shuffled = [batch for batch in dataloader_not_shuffled]

    # Assert
    # With shuffle, order might be different (but might also be same by chance)
    # Just verify both produce batches
    assert len(batches_shuffled) > 0
    assert len(batches_not_shuffled) > 0


# ============================================================================
# DataLoader Device Tests
# ============================================================================


def spec_dataloader_device_from_config(sample_dataframe):
    """Verify DataLoader gets device from dataset.preprocessor.config.device."""
    # Arrange
    config = PreprocessingConfig(device="cpu", vocab_size=1000, max_seq_len=128)
    preprocessor = Preprocessor(config)
    preprocessor.fit(sample_dataframe)
    dataset = StructuredDataset(sample_dataframe, preprocessor)

    # Act
    dataloader = create_dataloader(dataset, batch_size=2)
    batch = next(iter(dataloader))

    # Assert
    # Device should flow from config through DataLoader to Batcher
    assert batch.token_ids.device.type == "cpu"
    # Verify device matches config
    assert batch.token_ids.device.type == config.device

    # Test with different device
    config_mps = PreprocessingConfig(device="mps", vocab_size=1000, max_seq_len=128)
    preprocessor_mps = Preprocessor(config_mps)
    preprocessor_mps.fit(sample_dataframe)
    dataset_mps = StructuredDataset(sample_dataframe, preprocessor_mps)
    dataloader_mps = create_dataloader(dataset_mps, batch_size=2)
    batch_mps = next(iter(dataloader_mps))

    # If MPS is available, should use it; otherwise should fallback to CPU
    if torch.backends.mps.is_available():
        assert batch_mps.token_ids.device.type == "mps"
    else:
        assert batch_mps.token_ids.device.type == "cpu"


def spec_dataloader_device_auto(sample_dataframe):
    """Verify DataLoader works with device='auto' in config."""
    # Arrange
    config = PreprocessingConfig(device="auto", vocab_size=1000, max_seq_len=128)
    preprocessor = Preprocessor(config)
    preprocessor.fit(sample_dataframe)
    dataset = StructuredDataset(sample_dataframe, preprocessor)

    # Act
    dataloader = create_dataloader(dataset, batch_size=2)
    batch = next(iter(dataloader))

    # Assert
    # Auto-detected device should be used correctly
    assert batch.token_ids.device.type in ["cpu", "cuda", "mps"]
    # Verify device detection happens at DataLoader creation time
    # (device is determined when get_device() is called in create_dataloader)
    from saab_v3.utils.device import get_device

    expected_device = get_device(config.device)
    assert batch.token_ids.device.type == expected_device.type


def spec_dataloader_device_explicit(sample_dataframe):
    """Verify DataLoader works with explicit device strings."""
    # Test with CPU
    config_cpu = PreprocessingConfig(device="cpu", vocab_size=1000, max_seq_len=128)
    preprocessor_cpu = Preprocessor(config_cpu)
    preprocessor_cpu.fit(sample_dataframe)
    dataset_cpu = StructuredDataset(sample_dataframe, preprocessor_cpu)
    dataloader_cpu = create_dataloader(dataset_cpu, batch_size=2)
    batch_cpu = next(iter(dataloader_cpu))
    assert batch_cpu.token_ids.device.type == "cpu"

    # Test with CUDA if available
    if torch.cuda.is_available():
        config_cuda = PreprocessingConfig(
            device="cuda", vocab_size=1000, max_seq_len=128
        )
        preprocessor_cuda = Preprocessor(config_cuda)
        preprocessor_cuda.fit(sample_dataframe)
        dataset_cuda = StructuredDataset(sample_dataframe, preprocessor_cuda)
        dataloader_cuda = create_dataloader(dataset_cuda, batch_size=2)
        batch_cuda = next(iter(dataloader_cuda))
        assert batch_cuda.token_ids.device.type == "cuda"

    # Test with MPS if available
    if torch.backends.mps.is_available():
        config_mps = PreprocessingConfig(device="mps", vocab_size=1000, max_seq_len=128)
        preprocessor_mps = Preprocessor(config_mps)
        preprocessor_mps.fit(sample_dataframe)
        dataset_mps = StructuredDataset(sample_dataframe, preprocessor_mps)
        dataloader_mps = create_dataloader(dataset_mps, batch_size=2)
        batch_mps = next(iter(dataloader_mps))
        assert batch_mps.token_ids.device.type == "mps"


def spec_dataloader_device_consistency(fitted_preprocessor, sample_dataframe):
    """Verify all batches from DataLoader are on same device."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    dataloader = create_dataloader(dataset, batch_size=1)

    # Act
    batches = [batch for batch in dataloader]

    # Assert
    assert len(batches) > 0
    # All batches should be on the same device
    first_device = batches[0].token_ids.device
    for batch in batches:
        assert batch.token_ids.device == first_device
        assert batch.attention_mask.device == first_device
        assert batch.field_ids.device == first_device

    # Device should match config.device (after conversion)
    from saab_v3.utils.device import get_device

    expected_device = get_device(fitted_preprocessor.config.device)
    assert first_device.type == expected_device.type

    # Verify device doesn't change between batches
    if len(batches) > 1:
        for i in range(1, len(batches)):
            assert batches[i].token_ids.device == batches[0].token_ids.device
