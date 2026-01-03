"""Transformer encoder layer component."""

import torch
import torch.nn as nn

from saab_v3.models.components.attention import MultiHeadAttention
from saab_v3.models.components.ffn import FeedForward
from saab_v3.models.components.normalization import LayerNorm


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with pre-norm architecture.

    Implements a complete encoder layer with:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization (pre-norm)
    - Residual connections
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        layer_norm_eps: float = 1e-5,
    ) -> None:
        """Initialize Transformer encoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            dropout: Dropout probability
            activation: Activation function for FFN (default: GELU)
            layer_norm_eps: Epsilon for layer normalization
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = FeedForward(
            d_model=d_model,
            ffn_dim=ffn_dim,
            activation=activation,
            dropout=dropout,
        )
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply Transformer encoder layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
                where 1 = valid token, 0 = padding (masked)
            return_attention_weights: Whether to return attention weights

        Returns:
            If return_attention_weights=False:
                Output tensor of shape [batch_size, seq_len, d_model]
            If return_attention_weights=True:
                Tuple of (output tensor, attention weights)
                Attention weights shape: [batch_size, num_heads, seq_len, seq_len]
        """
        # Pre-norm architecture: LayerNorm before attention
        norm_x = self.norm1(x)
        attn_output = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights,
        )

        if return_attention_weights:
            attn_output, attn_weights = attn_output
            x = x + attn_output  # Residual connection
        else:
            x = x + attn_output  # Residual connection

        # Pre-norm architecture: LayerNorm before FFN
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + ffn_output  # Residual connection

        if return_attention_weights:
            return x, attn_weights
        return x
