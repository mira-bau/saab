"""Entity embedding module."""

import torch
import torch.nn as nn

from saab_v3.data.constants import PAD_IDX


class EntityEmbedding(nn.Module):
    """Entity embedding layer.

    Converts entity indices to dense embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = PAD_IDX,
    ) -> None:
        """Initialize entity embedding.

        Args:
            vocab_size: Size of entity vocabulary
            d_model: Embedding dimension
            padding_idx: Index for padding token (default: 0)
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Apply entity embedding.

        Args:
            entity_ids: Entity indices tensor of shape [batch_size, seq_len]

        Returns:
            Embedded entities of shape [batch_size, seq_len, d_model]
        """
        return self.embedding(entity_ids)
