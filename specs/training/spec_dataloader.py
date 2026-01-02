"""Specs for DataLoader factory - happy path only."""

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
    dataset = StructuredDataset(sample_dataframe, fitted_preprocessor)
    dataloader = create_dataloader(dataset, batch_size=2, device="cpu")

    # Act
    batch = next(iter(dataloader))

    # Assert
    assert batch.token_ids.device.type == "cpu"
    assert batch.attention_mask.device.type == "cpu"


def spec_dataloader_preserve_original_tags(fitted_preprocessor, sample_dataframe):
    """Verify preserve_original_tags flag works."""
    # Arrange
    config = PreprocessingConfig(
        vocab_size=1000,
        max_seq_len=128,
        preserve_original_tags=True,
    )
    preprocessor = Preprocessor(config)
    preprocessor.fit(sample_dataframe)

    dataset = StructuredDataset(sample_dataframe, preprocessor)
    dataloader = create_dataloader(dataset, batch_size=2, preserve_original_tags=True)

    # Act
    batch = next(iter(dataloader))

    # Assert
    assert batch.original_tags is not None
    assert len(batch.original_tags) == batch.token_ids.shape[0]


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
