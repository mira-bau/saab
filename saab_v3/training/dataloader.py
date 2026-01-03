"""DataLoader factory for creating DataLoaders with proper batching."""

import torch
from torch.utils.data import DataLoader

from saab_v3.data.batcher import Batcher
from saab_v3.data.constants import PAD_IDX
from saab_v3.utils.device import get_device

from saab_v3.training.dataset import StructuredDataset


def create_dataloader(
    dataset: StructuredDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create DataLoader with proper batching.

    Args:
        dataset: StructuredDataset instance
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker processes (0 = main process)
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader that yields Batch objects

    Note:
        Device is automatically obtained from dataset.preprocessor.config.device
        (single source of truth). No need to pass device parameter.
    """
    # Get max_seq_len from dataset's preprocessor config
    max_seq_len = dataset.preprocessor.config.max_seq_len

    # Get device from config (single source of truth)
    device = get_device(dataset.preprocessor.config.device)

    # Create Batcher with device
    batcher = Batcher(max_seq_len=max_seq_len, pad_token_id=PAD_IDX, device=device)

    # Get task_type from dataset if available
    task_type = getattr(dataset, "task_type", None)

    # Create collate function
    def collate_fn(batch_items):
        """Collate function that collects items and creates Batch."""
        # batch_items is a list of (TokenizedSequence, token_ids, encoded_tags, label) tuples
        # or for ranking: ((seq_a_data), (seq_b_data), label) tuples
        # label can be None if not present in data
        # Pass directly to batcher (device is already set in batcher)
        return batcher.batch(batch_items, task_type=task_type)

    # Create DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
