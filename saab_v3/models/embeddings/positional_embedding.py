"""Positional embedding module."""

import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Positional embedding layer.

    Provides positional information to the model. Supports both learned
    (default, like BERT) and sinusoidal (fixed) embeddings.
    """

    def __init__(
        self,
        max_seq_len: int = 512,
        d_model: int = 128,
        learned: bool = True,
    ) -> None:
        """Initialize positional embedding.

        Args:
            max_seq_len: Maximum sequence length
            d_model: Embedding dimension
            learned: Whether to use learned embeddings (default: True)
                If False, uses fixed sinusoidal embeddings
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.learned = learned

        if learned:
            # Learned positional embeddings (like BERT)
            self.embedding = nn.Embedding(max_seq_len, d_model)
        else:
            # Sinusoidal embeddings (fixed, not learnable)
            self.register_buffer(
                "pe",
                self._create_sinusoidal_embeddings(max_seq_len, d_model),
            )

    def _create_sinusoidal_embeddings(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create fixed sinusoidal positional embeddings.

        Args:
            max_len: Maximum sequence length
            d_model: Embedding dimension

        Returns:
            Sinusoidal embeddings of shape [max_len, d_model]
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
                (used to infer sequence length)

        Returns:
            Positional embeddings of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        if self.learned:
            # Create position indices [0, 1, 2, ..., seq_len-1]
            positions = torch.arange(
                seq_len, device=x.device, dtype=torch.long
            ).unsqueeze(0)  # [1, seq_len]
            # Expand to batch size
            positions = positions.expand(batch_size, -1)  # [batch_size, seq_len]
            # Get embeddings
            pos_emb = self.embedding(positions)  # [batch_size, seq_len, d_model]
        else:
            # Use fixed sinusoidal embeddings
            pos_emb = self.pe[:seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]
            pos_emb = pos_emb.expand(
                batch_size, -1, -1
            )  # [batch_size, seq_len, d_model]

        return pos_emb
