"""Flat Transformer: Minimal baseline with token + positional embeddings only."""

import torch

from saab_v3.data.structures import Batch
from saab_v3.models.base_transformer import BaseTransformer


class FlatTransformer(BaseTransformer):
    """Flat Transformer: Weak baseline with minimal embeddings.

    Uses only token and positional embeddings. No structural information.
    Serves as a lower bound to demonstrate that structure-agnostic modeling
    is insufficient for structured data.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        vocab_sizes: dict[str, int],
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: torch.nn.Module = torch.nn.GELU(),
        layer_norm_eps: float = 1e-5,
        positional_learned: bool = True,
    ) -> None:
        """Initialize Flat Transformer.

        Args:
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            vocab_sizes: Dictionary with vocabulary sizes.
                Must contain: "token_vocab_size"
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function for FFN
            layer_norm_eps: Epsilon for layer normalization
            positional_learned: Whether to use learned positional embeddings
        """
        super().__init__(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            vocab_sizes=vocab_sizes,
            max_seq_len=max_seq_len,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            use_token_type=False,
            use_field=False,
            use_entity=False,
            use_time=False,
            positional_learned=positional_learned,
        )

    def forward(
        self,
        batch: Batch,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through Flat Transformer.

        Args:
            batch: Batch object with token and tag indices
            return_attention_weights: Whether to return attention weights from all layers

        Returns:
            If return_attention_weights=False:
                Output tensor of shape [batch_size, seq_len, d_model]
            If return_attention_weights=True:
                Tuple of (output tensor, list of attention weights)
        """
        return super().forward(batch, return_attention_weights=return_attention_weights)
