"""SAAB (Structure-Aware Attention Bias) attention mechanism."""

import torch
import torch.nn as nn

from saab_v3.data.structures import StructureTag
from saab_v3.data.saab_utils import compute_structural_relationship, is_pad_tag
from saab_v3.models.components.dropout import Dropout


class SAABAttention(nn.Module):
    """Multi-head attention with structural bias for SAAB Transformer.

    Implements: Attention = softmax((QK^T / √d) + λ · B_struct) V

    When λ=0, this is bitwise-equivalent to standard MultiHeadAttention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        lambda_bias: float = 1.0,
        learnable_lambda: bool = False,
        bias_normalization: float = 1.0,
    ) -> None:
        """Initialize SAAB attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
            lambda_bias: Initial value for bias strength (λ)
            learnable_lambda: If True, lambda is a learnable parameter
            bias_normalization: Normalization factor for B_struct to avoid
                dominating QK^T scores. Default 1.0 means no normalization.
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
        self.bias_normalization = bias_normalization

        # Lambda parameter (bias strength)
        if learnable_lambda:
            self.lambda_bias = nn.Parameter(torch.tensor(float(lambda_bias)))
        else:
            self.register_buffer("lambda_bias", torch.tensor(float(lambda_bias)))

        # Standard attention components
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout
        self.dropout = Dropout(dropout)

    def compute_bias_matrix(
        self,
        original_tags: list[list[StructureTag]],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute structural bias matrix B_struct from original tags.

        Args:
            original_tags: List of lists of StructureTag objects,
                shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
                where 1 = valid token, 0 = padding (masked)

        Returns:
            Bias matrix of shape [batch_size, seq_len, seq_len]
        """
        batch_size = len(original_tags)
        if batch_size == 0:
            raise ValueError("original_tags cannot be empty")

        seq_len = len(original_tags[0])
        device = self.lambda_bias.device

        # Initialize bias matrix
        bias_matrix = torch.zeros(
            (batch_size, seq_len, seq_len), dtype=torch.float32, device=device
        )

        # Compute bias for each sequence
        for batch_idx, tags in enumerate(original_tags):
            for i in range(seq_len):
                for j in range(seq_len):
                    tag_i = tags[i]
                    tag_j = tags[j]

                    # Skip padding tags
                    if is_pad_tag(tag_i) or is_pad_tag(tag_j):
                        bias_matrix[batch_idx, i, j] = float("-inf")
                        continue

                    # Compute structural relationship
                    relationship = compute_structural_relationship(tag_i, tag_j)

                    # Convert relationship to scalar bias
                    # This is an implementation choice - can be adjusted
                    bias = 0.0
                    if relationship["same_field"]:
                        bias += 1.0
                    if relationship["same_entity"]:
                        bias += 1.0
                    if relationship["same_time"]:
                        bias += 0.5
                    if relationship["has_edge"]:
                        bias += 1.5
                    if relationship["same_role"]:
                        bias += 0.5
                    if relationship["same_token_type"]:
                        bias += 0.3

                    bias_matrix[batch_idx, i, j] = bias

        # Normalize bias to avoid dominating QK^T scores
        # Scale by normalization factor (typically 1/sqrt(seq_len) or similar)
        if self.bias_normalization != 1.0:
            bias_matrix = bias_matrix * self.bias_normalization

        # Apply attention mask: set padding positions to -inf
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len], 1 = valid, 0 = padding
            # Expand to [batch_size, seq_len, seq_len]
            mask_i = attention_mask.unsqueeze(2)  # [batch_size, seq_len, 1]
            mask_j = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
            mask_2d = mask_i * mask_j  # [batch_size, seq_len, seq_len]
            # Set masked positions to -inf
            bias_matrix = bias_matrix.masked_fill(mask_2d == 0, float("-inf"))

        return bias_matrix

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        original_tags: list[list[StructureTag]] | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply SAAB attention with structural bias.

        Args:
            query: Query tensor of shape [batch_size, seq_len, d_model]
            key: Key tensor of shape [batch_size, seq_len, d_model]
            value: Value tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
                where 1 = valid token, 0 = padding (masked)
            original_tags: List of lists of StructureTag objects for bias computation.
                Required if lambda_bias != 0. Shape: [batch_size, seq_len]
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

        # Apply structural bias if lambda != 0 and original_tags provided
        # When lambda=0, skip bias computation for efficiency (bitwise equivalent to standard attention)
        if self.lambda_bias.item() != 0.0:
            if original_tags is None:
                raise ValueError(
                    "original_tags is required when lambda_bias != 0 for SAAB attention"
                )

            # Compute bias matrix: [batch_size, seq_len, seq_len]
            B_struct = self.compute_bias_matrix(original_tags, attention_mask)

            # Expand bias to match scores shape: [batch_size, 1, seq_len, seq_len]
            B_struct = B_struct.unsqueeze(1)

            # Apply bias: scores = scores + λ · B_struct
            scores = scores + self.lambda_bias * B_struct
        # When lambda=0, no bias is applied - this is bitwise equivalent to standard MultiHeadAttention

        # Apply attention mask if provided (after bias addition)
        if attention_mask is not None:
            # Convert mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
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
