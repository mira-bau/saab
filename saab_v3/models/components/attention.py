"""Multi-head attention component."""

import torch
import torch.nn as nn

from saab_v3.models.components.dropout import Dropout


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.

    Implements standard Transformer attention mechanism with support for
    attention masks. Can optionally return attention weights for analysis.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        """Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** (-0.5)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout
        self.dropout = Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention.

        Args:
            query: Query tensor of shape [batch_size, seq_len, d_model]
            key: Key tensor of shape [batch_size, seq_len, d_model]
            value: Value tensor of shape [batch_size, seq_len, d_model]
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
        batch_size, seq_len, _ = query.shape

        # Project Q, K, V
        Q = self.q_proj(query)  # [batch_size, seq_len, d_model]
        K = self.k_proj(key)  # [batch_size, seq_len, d_model]
        V = self.v_proj(value)  # [batch_size, seq_len, d_model]

        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores shape: [batch_size, num_heads, seq_len, seq_len]

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            # 1 = valid, 0 = masked
            mask = attention_mask.unsqueeze(1).unsqueeze(
                2
            )  # [batch_size, 1, 1, seq_len]
            # Set masked positions to -inf
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # attn_output shape: [batch_size, num_heads, seq_len, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        # [batch_size, seq_len, d_model]

        # Output projection
        output = self.out_proj(attn_output)

        if return_attention_weights:
            return output, attn_weights
        return output
