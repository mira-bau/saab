"""DataLoader factory for creating DataLoaders with proper batching."""

import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np

from saab_v3.data.batcher import Batcher
from saab_v3.data.constants import PAD_IDX
from saab_v3.utils.device import get_device

from saab_v3.training.dataset import StructuredDataset


class CollateFunction:
    """Picklable collate function for DataLoader multiprocessing.

    Performance: Minimal overhead - class instantiation is one-time,
    method calls are as fast as function calls.
    """

    def __init__(self, batcher: Batcher, task_type: str | None = None):
        """Initialize collate function.

        Args:
            batcher: Batcher instance to use for batching
            task_type: Task type for batching ("ranking" or None)
        """
        self.batcher = batcher
        self.task_type = task_type

    def __call__(self, batch_items):
        """Collate function that collects items and creates Batch.

        Args:
            batch_items: List of (TokenizedSequence, token_ids, encoded_tags, label) tuples
                or for ranking: ((seq_a_data), (seq_b_data), label) tuples

        Returns:
            Batch object with all tensors and labels
        """
        return self.batcher.batch(batch_items, task_type=self.task_type)


def _worker_init_fn(worker_id: int) -> None:
    """Initialize worker process for multiprocessing.

    Sets environment variables and ensures proper worker setup.
    Called once per worker process.

    Args:
        worker_id: Worker process ID (0-indexed)
    """
    # Disable tokenizers parallelism in worker processes to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Optional: Set worker-specific random seed if needed for reproducibility
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dataloader(
    dataset: StructuredDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create DataLoader with proper batching and multiprocessing support.

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

    Performance Optimizations (when num_workers > 0):
        - persistent_workers=True: Reuses workers across epochs (high impact,
          avoids expensive worker re-initialization)
        - prefetch_factor=2: Prefetches batches for better pipelining (medium impact,
          reduces GPU idle time)
        - worker_init_fn: Initializes worker processes (disables tokenizers parallelism)

    Multiprocessing Requirements:
        - For CUDA: multiprocessing start method must be 'spawn' (not 'fork')
        - Set before creating DataLoader: multiprocessing.set_start_method('spawn', force=True)
        - The collate function is picklable and works with both 'fork' and 'spawn' methods
    """
    # Get max_seq_len from dataset's preprocessor config
    max_seq_len = dataset.preprocessor.config.max_seq_len

    # Get device from config (single source of truth)
    device = get_device(dataset.preprocessor.config.device)

    # Create Batcher with device
    batcher = Batcher(max_seq_len=max_seq_len, pad_token_id=PAD_IDX, device=device)

    # Get task_type from dataset if available
    task_type = getattr(dataset, "task_type", None)

    # Create picklable collate function
    collate_fn = CollateFunction(batcher, task_type)

    # Performance optimizations for multiprocessing
    persistent_workers = num_workers > 0  # Reuse workers across epochs (high impact)
    prefetch_factor = 2 if num_workers > 0 else None  # Prefetch batches (medium impact)
    worker_init_fn = _worker_init_fn if num_workers > 0 else None  # Worker setup

    # Create DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,  # Avoid worker re-initialization
        prefetch_factor=prefetch_factor,  # Better pipelining
        worker_init_fn=worker_init_fn,  # Worker initialization
    )
