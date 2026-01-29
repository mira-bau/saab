"""SAAB (Structure-Aware Attention Bias) attention mechanism."""

import torch
import torch.nn as nn

from saab_v3.data.constants import PAD_IDX
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
        field_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        time_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        edge_ids: torch.Tensor | None = None,
        role_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute structural bias matrix B_struct from tag index tensors.

        Uses vectorized PyTorch operations for efficient computation on GPU.

        Args:
            field_ids: Field indices tensor of shape [batch_size, seq_len]
            entity_ids: Entity indices tensor of shape [batch_size, seq_len]
            time_ids: Time indices tensor of shape [batch_size, seq_len]
            token_type_ids: Token type indices tensor of shape [batch_size, seq_len]
            edge_ids: Optional edge indices tensor of shape [batch_size, seq_len]
            role_ids: Optional role indices tensor of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
                where 1 = valid token, 0 = padding (masked)

        Returns:
            Bias matrix of shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len = field_ids.shape
        device = field_ids.device

        # Detect padding positions using attention_mask (ground truth)
        # NOTE: Cannot use field_ids == PAD_IDX because field_idx=0 could be a valid field index
        # attention_mask: 1 = valid token, 0 = padding
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len], 1 = valid, 0 = padding
            is_pad = attention_mask == 0  # [batch_size, seq_len]
            pad_mask = is_pad.unsqueeze(2) | is_pad.unsqueeze(1)  # [batch_size, seq_len, seq_len]
        else:
            # Fallback: if no attention_mask, assume all tokens are valid
            # This should not happen in practice, but handle gracefully
            pad_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)

        # Vectorized comparisons using broadcasting
        # unsqueeze(2): [batch_size, seq_len, 1]
        # unsqueeze(1): [batch_size, 1, seq_len]
        # Result: [batch_size, seq_len, seq_len] via broadcasting

        # Same field: field_ids[i] == field_ids[j]
        same_field = field_ids.unsqueeze(2) == field_ids.unsqueeze(1)  # [batch_size, seq_len, seq_len]

        # Same entity: entity_ids[i] == entity_ids[j]
        same_entity = entity_ids.unsqueeze(2) == entity_ids.unsqueeze(1)  # [batch_size, seq_len, seq_len]

        # Same time: time_ids[i] == time_ids[j]
        same_time = time_ids.unsqueeze(2) == time_ids.unsqueeze(1)  # [batch_size, seq_len, seq_len]

        # Same token_type: token_type_ids[i] == token_type_ids[j]
        same_token_type = token_type_ids.unsqueeze(2) == token_type_ids.unsqueeze(1)  # [batch_size, seq_len, seq_len]

        # Initialize bias matrix
        bias_matrix = torch.zeros(
            (batch_size, seq_len, seq_len), dtype=torch.float32, device=device
        )

        # Combine relationships into bias matrix
        # Convert boolean tensors to float and apply weights
        bias_matrix = bias_matrix + (same_field.float() * 1.0)
        bias_matrix = bias_matrix + (same_entity.float() * 1.0)
        bias_matrix = bias_matrix + (same_time.float() * 0.5)
        bias_matrix = bias_matrix + (same_token_type.float() * 0.3)

        # Optional: has_edge (either token has non-PAD edge)
        if edge_ids is not None:
            has_edge = (edge_ids.unsqueeze(2) != PAD_IDX) | (edge_ids.unsqueeze(1) != PAD_IDX)
            bias_matrix = bias_matrix + (has_edge.float() * 1.5)

        # Optional: same_role
        if role_ids is not None:
            same_role = role_ids.unsqueeze(2) == role_ids.unsqueeze(1)
            bias_matrix = bias_matrix + (same_role.float() * 0.5)

        # Set padding positions to -inf (using attention_mask-based pad_mask)
        bias_matrix = bias_matrix.masked_fill(pad_mask, float("-inf"))

        # Normalize bias to avoid dominating QK^T scores
        if self.bias_normalization != 1.0:
            bias_matrix = bias_matrix * self.bias_normalization

        return bias_matrix

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        field_ids: torch.Tensor | None = None,
        entity_ids: torch.Tensor | None = None,
        time_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        edge_ids: torch.Tensor | None = None,
        role_ids: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply SAAB attention with structural bias.

        Args:
            query: Query tensor of shape [batch_size, seq_len, d_model]
            key: Key tensor of shape [batch_size, seq_len, d_model]
            value: Value tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
                where 1 = valid token, 0 = padding (masked)
            field_ids: Field indices tensor of shape [batch_size, seq_len]
            entity_ids: Entity indices tensor of shape [batch_size, seq_len]
            time_ids: Time indices tensor of shape [batch_size, seq_len]
            token_type_ids: Token type indices tensor of shape [batch_size, seq_len]
            edge_ids: Optional edge indices tensor of shape [batch_size, seq_len]
            role_ids: Optional role indices tensor of shape [batch_size, seq_len]
            return_attention_weights: Whether to return attention weights

        Returns:
            If return_attention_weights=False:
                Output tensor of shape [batch_size, seq_len, d_model]
            If return_attention_weights=True:
                Tuple of (output tensor, attention weights)
                Attention weights shape: [batch_size, num_heads, seq_len, seq_len]

        Raises:
            ValueError: If required tag indices are None when lambda_bias != 0
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

        # Apply structural bias if lambda != 0 and tag indices provided
        # When lambda=0, skip bias computation for efficiency (bitwise equivalent to standard attention)
        if self.lambda_bias.item() != 0.0:
            # Validate required tag indices
            if (
                field_ids is None
                or entity_ids is None
                or time_ids is None
                or token_type_ids is None
            ):
                raise ValueError(
                    "Tag indices (field_ids, entity_ids, time_ids, token_type_ids) "
                    "are required when lambda_bias != 0 for SAAB attention"
                )

            # Ensure all tensors are on the same device
            device = query.device
            field_ids = field_ids.to(device)
            entity_ids = entity_ids.to(device)
            time_ids = time_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            if edge_ids is not None:
                edge_ids = edge_ids.to(device)
            if role_ids is not None:
                role_ids = role_ids.to(device)

            # Compute bias matrix: [batch_size, seq_len, seq_len]
            B_struct = self.compute_bias_matrix(
                field_ids=field_ids,
                entity_ids=entity_ids,
                time_ids=time_ids,
                token_type_ids=token_type_ids,
                edge_ids=edge_ids,
                role_ids=role_ids,
                attention_mask=attention_mask,
            )

            # Expand bias to match scores shape: [batch_size, 1, seq_len, seq_len]
            B_struct = B_struct.unsqueeze(1)

            # Apply bias: scores = scores + λ · B_struct
            scores = scores + self.lambda_bias * B_struct
        # When lambda=0, no bias is applied - this is bitwise equivalent to standard MultiHeadAttention

        # Apply attention mask if provided (after bias addition)
        # NOTE: The bias matrix already sets -inf for padding positions (both query and key).
        # Here we only need to mask keys for the attention mask (standard practice).
        # However, we must handle the case where all scores are -inf to prevent NaN in softmax.
        if attention_mask is not None:
            # Convert mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            # This masks keys: valid queries won't attend to padding keys
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            # Set masked positions to -inf
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
            # CRITICAL: Prevent NaN in softmax when all scores are -inf for a query
            # This can happen for padding queries (all keys masked) or edge cases
            # Solution: If all scores are -inf, set them to 0 (softmax will produce uniform distribution)
            all_keys_masked = (scores == float("-inf")).all(dim=-1, keepdim=True)  # [batch_size, num_heads, seq_len, 1]
            if all_keys_masked.any():
                # Set all -inf scores to 0 for queries with all keys masked
                # This prevents NaN: softmax([0, 0, ..., 0]) = uniform distribution
                # Expand all_keys_masked to match scores shape: [batch_size, num_heads, seq_len, 1] -> [batch_size, num_heads, seq_len, seq_len]
                all_keys_masked_expanded = all_keys_masked.expand_as(scores)  # [batch_size, num_heads, seq_len, seq_len]
                scores = scores.masked_fill(all_keys_masked_expanded, 0.0)

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
