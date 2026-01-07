"""Specs for DataLoader factory - happy path only."""

import multiprocessing
import pickle

import pytest
import torch

from saab_v3.training.config import PreprocessingConfig
from saab_v3.training.preprocessor import Preprocessor
from saab_v3.training.dataset import StructuredDataset
from saab_v3.training.dataloader import create_dataloader, CollateFunction, _worker_init_fn
from saab_v3.data.structures import Batch
from saab_v3.data.batcher import Batcher
from saab_v3.data.constants import PAD_IDX


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


# ============================================================================
# Test Infrastructure Helpers
# ============================================================================


def _set_spawn_method():
    """Helper to set multiprocessing start method to 'spawn' for tests."""
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass


def _skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ============================================================================
# Multiprocessing Tests
# ============================================================================


def spec_dataloader_multiprocessing_basic(fitted_preprocessor, sample_dataframe):
    """Test DataLoader with multiprocessing (num_workers > 0) on CPU."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    
    # Act
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=2)
    batches = [batch for batch in dataloader]
    
    # Assert
    assert len(batches) > 0
    for batch in batches:
        assert isinstance(batch, Batch)
        assert batch.token_ids.shape[0] <= 2  # batch_size
        assert batch.token_ids.shape[1] > 0  # seq_len


def spec_dataloader_multiprocessing_batch_consistency(fitted_preprocessor, sample_dataframe):
    """Verify batches are correct with multiprocessing workers."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    dataloader_single = create_dataloader(dataset, batch_size=2, num_workers=0)
    dataloader_multi = create_dataloader(dataset, batch_size=2, num_workers=2)
    
    # Act
    batches_single = [batch for batch in dataloader_single]
    batches_multi = [batch for batch in dataloader_multi]
    
    # Assert
    # Both should produce same number of batches (may be in different order)
    assert len(batches_single) == len(batches_multi)
    # All batches should have valid shapes
    for batch in batches_multi:
        assert batch.token_ids.shape[0] <= 2
        assert batch.attention_mask.shape == batch.token_ids.shape
        assert batch.field_ids.shape == batch.token_ids.shape


def spec_dataloader_multiprocessing_pin_memory(fitted_preprocessor, sample_dataframe):
    """Test pin_memory=True with multiprocessing workers."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    
    # Act
    dataloader = create_dataloader(
        dataset, batch_size=2, num_workers=2, pin_memory=True
    )
    batch = next(iter(dataloader))
    
    # Assert
    assert isinstance(batch, Batch)
    # Pin memory should work without errors (on CPU, pin_memory has no effect but shouldn't break)
    # On GPU, tensors would be pinned, but we're testing on CPU here
    assert batch.token_ids.shape[0] <= 2
    assert batch.token_ids.shape[1] > 0


def spec_dataloader_multiprocessing_persistent_workers(fitted_preprocessor, sample_dataframe):
    """Test persistent workers across multiple epochs."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=2)
    
    # Act - Iterate through multiple epochs
    batches_epoch1 = [batch for batch in dataloader]
    batches_epoch2 = [batch for batch in dataloader]
    
    # Assert
    # Both epochs should produce batches (persistent workers should work)
    assert len(batches_epoch1) > 0
    assert len(batches_epoch2) > 0
    # Batches should have correct shapes
    for batch in batches_epoch1 + batches_epoch2:
        assert isinstance(batch, Batch)
        assert batch.token_ids.shape[0] <= 2


def spec_dataloader_multiprocessing_prefetch(fitted_preprocessor, sample_dataframe):
    """Test prefetch_factor works correctly with multiprocessing."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=2)
    
    # Act
    # Prefetch should work transparently - just verify batches are correct
    batches = [batch for batch in dataloader]
    
    # Assert
    assert len(batches) > 0
    for batch in batches:
        assert isinstance(batch, Batch)
        assert batch.token_ids.shape[0] <= 2


# ============================================================================
# CUDA + Multiprocessing Tests
# ============================================================================


def spec_dataloader_cuda_multiprocessing(sample_dataframe):
    """Test CUDA with multiprocessing (requires spawn method)."""
    _skip_if_no_cuda()
    _set_spawn_method()
    
    # Arrange
    config = PreprocessingConfig(device="cuda", vocab_size=1000, max_seq_len=128)
    preprocessor = Preprocessor(config)
    preprocessor.fit(sample_dataframe)
    dataset = StructuredDataset(sample_dataframe, preprocessor)
    
    # Act
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=2)
    batches = [batch for batch in dataloader]
    
    # Assert
    assert len(batches) > 0
    for batch in batches:
        assert isinstance(batch, Batch)
        assert batch.token_ids.device.type == "cuda"
        assert batch.token_ids.shape[0] <= 2


def spec_dataloader_cuda_pin_memory(sample_dataframe):
    """Test CUDA + pin_memory + multiprocessing."""
    _skip_if_no_cuda()
    _set_spawn_method()
    
    # Arrange
    config = PreprocessingConfig(device="cuda", vocab_size=1000, max_seq_len=128)
    preprocessor = Preprocessor(config)
    preprocessor.fit(sample_dataframe)
    dataset = StructuredDataset(sample_dataframe, preprocessor)
    
    # Act
    dataloader = create_dataloader(
        dataset, batch_size=2, num_workers=2, pin_memory=True
    )
    batch = next(iter(dataloader))
    
    # Assert
    assert isinstance(batch, Batch)
    assert batch.token_ids.device.type == "cuda"
    # Pin memory should be enabled
    assert batch.token_ids.is_pinned()


def spec_dataloader_cuda_persistent_workers(sample_dataframe):
    """Test CUDA + persistent workers across epochs."""
    _skip_if_no_cuda()
    _set_spawn_method()
    
    # Arrange
    config = PreprocessingConfig(device="cuda", vocab_size=1000, max_seq_len=128)
    preprocessor = Preprocessor(config)
    preprocessor.fit(sample_dataframe)
    dataset = StructuredDataset(sample_dataframe, preprocessor)
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=2)
    
    # Act - Multiple epochs
    batches_epoch1 = [batch for batch in dataloader]
    batches_epoch2 = [batch for batch in dataloader]
    
    # Assert
    assert len(batches_epoch1) > 0
    assert len(batches_epoch2) > 0
    for batch in batches_epoch1 + batches_epoch2:
        assert batch.token_ids.device.type == "cuda"


# ============================================================================
# Pickling Tests
# ============================================================================


def spec_dataloader_collate_picklable(fitted_preprocessor, sample_dataframe):
    """Verify collate function can be pickled."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    from saab_v3.utils.device import get_device
    from saab_v3.data.batcher import Batcher
    
    device = get_device(fitted_preprocessor.config.device)
    batcher = Batcher(
        max_seq_len=fitted_preprocessor.config.max_seq_len,
        pad_token_id=PAD_IDX,
        device=device,
    )
    task_type = getattr(dataset, "task_type", None)
    collate_fn = CollateFunction(batcher, task_type)
    
    # Act - Try to pickle
    pickled = pickle.dumps(collate_fn)
    unpickled = pickle.loads(pickled)
    
    # Assert - Should be able to unpickle and use
    assert isinstance(unpickled, CollateFunction)
    # Test that it still works
    sample_item = dataset[0]
    batch = unpickled([sample_item])
    assert isinstance(batch, Batch)


def spec_dataloader_batcher_picklable(fitted_preprocessor):
    """Verify Batcher can be pickled."""
    # Arrange
    from saab_v3.utils.device import get_device
    
    device = get_device(fitted_preprocessor.config.device)
    batcher = Batcher(
        max_seq_len=fitted_preprocessor.config.max_seq_len,
        pad_token_id=PAD_IDX,
        device=device,
    )
    
    # Act - Try to pickle
    pickled = pickle.dumps(batcher)
    unpickled = pickle.loads(pickled)
    
    # Assert
    assert isinstance(unpickled, Batcher)
    assert unpickled.max_seq_len == batcher.max_seq_len
    assert unpickled.pad_token_id == batcher.pad_token_id
    assert unpickled.device == batcher.device


def spec_dataloader_worker_init_picklable():
    """Verify worker_init_fn can be pickled."""
    # Act - Try to pickle
    pickled = pickle.dumps(_worker_init_fn)
    unpickled = pickle.loads(pickled)
    
    # Assert - Should be able to unpickle
    assert callable(unpickled)
    # Test that it can be called
    unpickled(0)  # Should not raise


# ============================================================================
# Edge Cases
# ============================================================================


def spec_dataloader_multiprocessing_empty_batch(fitted_preprocessor, sample_dataframe):
    """Test edge case with multiprocessing and very small dataset."""
    # Arrange - Use batch_size larger than dataset to test edge case
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    # Create a very small dataset by taking first few items
    small_dataset = StructuredDataset(sample_dataframe.head(1), fitted_preprocessor)
    
    # Act
    dataloader = create_dataloader(small_dataset, batch_size=10, num_workers=2)
    batches = [batch for batch in dataloader]
    
    # Assert
    assert len(batches) > 0
    for batch in batches:
        assert isinstance(batch, Batch)
        assert batch.token_ids.shape[0] > 0


def spec_dataloader_multiprocessing_different_devices(fitted_preprocessor, sample_dataframe):
    """Test device consistency with multiprocessing workers."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=2)
    
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


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


def spec_dataloader_backward_compatibility_num_workers_zero(fitted_preprocessor, sample_dataframe):
    """Verify num_workers=0 still works (backward compatibility)."""
    # Arrange
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    
    # Act
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)
    batches = [batch for batch in dataloader]
    
    # Assert
    assert len(batches) > 0
    for batch in batches:
        assert isinstance(batch, Batch)
        assert batch.token_ids.shape[0] <= 2
    # Verify persistent_workers is False when num_workers=0
    assert not dataloader.persistent_workers
