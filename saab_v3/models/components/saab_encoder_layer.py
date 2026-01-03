"""SAAB Transformer encoder layer with structure-aware attention."""

import torch
import torch.nn as nn

from saab_v3.data.structures import StructureTag
from saab_v3.models.components.saab_attention import SAABAttention
from saab_v3.models.components.ffn import FeedForward
from saab_v3.models.components.normalization import LayerNorm


class SAABEncoderLayer(nn.Module):
    """Single SAAB Transformer encoder layer with pre-norm architecture.

    Similar to TransformerEncoderLayer but uses SAABAttention instead of
    standard MultiHeadAttention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        layer_norm_eps: float = 1e-5,
        lambda_bias: float = 1.0,
        learnable_lambda: bool = False,
        bias_normalization: float = 1.0,
    ) -> None:
        """Initialize SAAB encoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            dropout: Dropout probability
            activation: Activation function for FFN (default: GELU)
            layer_norm_eps: Epsilon for layer normalization
            lambda_bias: Bias strength parameter (λ)
            learnable_lambda: If True, lambda is learnable
            bias_normalization: Normalization factor for structural bias
        """
        super().__init__()
        self.self_attn = SAABAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            lambda_bias=lambda_bias,
            learnable_lambda=learnable_lambda,
            bias_normalization=bias_normalization,
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
        original_tags: list[list[StructureTag]] | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply SAAB Transformer encoder layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
                where 1 = valid token, 0 = padding (masked)
            original_tags: List of lists of StructureTag objects for SAAB bias.
                Required if lambda_bias != 0. Shape: [batch_size, seq_len]
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
            original_tags=original_tags,
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
