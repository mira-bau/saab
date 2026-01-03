"""Time bucket embedding module."""

import torch
import torch.nn as nn

from saab_v3.data.constants import PAD_IDX


class TimeEmbedding(nn.Module):
    """Time bucket embedding layer.

    Converts time bucket indices to dense embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = PAD_IDX,
    ) -> None:
        """Initialize time embedding.

        Args:
            vocab_size: Size of time bucket vocabulary
            d_model: Embedding dimension
            padding_idx: Index for padding token (default: 0)
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

    def forward(self, time_ids: torch.Tensor) -> torch.Tensor:
        """Apply time embedding.

        Args:
            time_ids: Time bucket indices tensor of shape [batch_size, seq_len]

        Returns:
            Embedded time buckets of shape [batch_size, seq_len, d_model]
        """
        return self.embedding(time_ids)
