"""Base Transformer class with shared functionality."""

import torch
import torch.nn as nn

from saab_v3.data.structures import Batch
from saab_v3.models.components.encoder_layer import TransformerEncoderLayer
from saab_v3.models.embeddings.combined_embedding import CombinedEmbedding


class BaseTransformer(nn.Module):
    """Base Transformer encoder with shared architecture.

    This class provides common functionality for Flat, Scratch, and SAAB transformers.
    Subclasses should override embedding configuration and attention mechanism.
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
        activation: nn.Module = nn.GELU(),
        layer_norm_eps: float = 1e-5,
        use_token_type: bool = False,
        use_field: bool = False,
        use_entity: bool = False,
        use_time: bool = False,
        positional_learned: bool = True,
    ) -> None:
        """Initialize base transformer.

        Args:
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            vocab_sizes: Dictionary with vocabulary sizes
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function for FFN
            layer_norm_eps: Epsilon for layer normalization
            use_token_type: Whether to use token type embeddings
            use_field: Whether to use field embeddings
            use_entity: Whether to use entity embeddings
            use_time: Whether to use time embeddings
            positional_learned: Whether to use learned positional embeddings
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embeddings = CombinedEmbedding(
            d_model=d_model,
            vocab_sizes=vocab_sizes,
            max_seq_len=max_seq_len,
            use_token_type=use_token_type,
            use_field=use_field,
            use_entity=use_entity,
            use_time=use_time,
            positional_learned=positional_learned,
        )

        # Encoder layers (standard attention)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        batch: Batch,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through transformer.

        Args:
            batch: Batch object with token and tag indices
            return_attention_weights: Whether to return attention weights from all layers

        Returns:
            If return_attention_weights=False:
                Output tensor of shape [batch_size, seq_len, d_model]
            If return_attention_weights=True:
                Tuple of (output tensor, list of attention weights)
                Each attention weight tensor has shape [batch_size, num_heads, seq_len, seq_len]
        """
        # Apply embeddings
        x = self.embeddings(batch)  # [batch_size, seq_len, d_model]

        # Apply encoder layers
        attention_weights_list = []
        for layer in self.layers:
            if return_attention_weights:
                x, attn_weights = layer(
                    x,
                    attention_mask=batch.attention_mask,
                    return_attention_weights=True,
                )
                attention_weights_list.append(attn_weights)
            else:
                x = layer(
                    x,
                    attention_mask=batch.attention_mask,
                    return_attention_weights=False,
                )

        if return_attention_weights:
            return x, attention_weights_list
        return x
