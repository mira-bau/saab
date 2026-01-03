"""Token type embedding module."""

import torch
import torch.nn as nn

from saab_v3.data.constants import PAD_IDX


class TokenTypeEmbedding(nn.Module):
    """Token type embedding layer.

    Converts token type indices to dense embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = PAD_IDX,
    ) -> None:
        """Initialize token type embedding.

        Args:
            vocab_size: Size of token type vocabulary
            d_model: Embedding dimension
            padding_idx: Index for padding token (default: 0)
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

    def forward(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """Apply token type embedding.

        Args:
            token_type_ids: Token type indices tensor of shape [batch_size, seq_len]

        Returns:
            Embedded token types of shape [batch_size, seq_len, d_model]
        """
        return self.embedding(token_type_ids)
