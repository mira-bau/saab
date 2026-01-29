"""MSM-Field head for masked field prediction (self-supervised task)."""

import torch
import torch.nn as nn

from saab_v3.tasks.base import BaseTaskHead


class MSMFieldHead(BaseTaskHead):
    """MSM-Field head for per-token field prediction.

    Predicts field_idx for each token position.
    Uses ALL token representations (no pooling).
    Outputs logits per token.

    Returns logits per token (no activation applied). User should apply softmax per token as needed.
    """

    def __init__(
        self,
        d_model: int,
        num_fields: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        """Initialize MSM-Field head.

        Args:
            d_model: Model dimension
            num_fields: Number of field classes
            hidden_dims: Hidden dimensions for MLP (None = simple linear)
            dropout: Dropout probability (only used if hidden_dims is not None)

        Note: Pooling is not used for MSM-Field (uses all tokens).
        """
        # Don't use pooling for MSM-Field
        super().__init__(d_model, hidden_dims, dropout, pooling=None)
        self.num_fields = num_fields

        # Build output layer
        if hidden_dims is not None:
            input_dim = hidden_dims[-1]
        else:
            input_dim = d_model

        self.output_layer = nn.Linear(input_dim, num_fields)

    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch, seq_len, d_model]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [batch, seq_len, num_fields]
        """Forward pass.

        Args:
            encoder_output: Encoder output tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask (not used in computation, but kept for interface consistency)

        Returns:
            Logits tensor of shape [batch_size, seq_len, num_fields] (logits per token, no activation applied).
            User should apply softmax per token as needed.
        """
        # Apply MLP per-token if present
        if self.mlp is not None:
            batch_size, seq_len, d_model = encoder_output.shape
            # Reshape to [batch * seq_len, d_model]
            encoder_flat = encoder_output.view(-1, d_model)
            # Apply MLP
            encoder_flat = self.mlp(encoder_flat)
            # Reshape back to [batch, seq_len, hidden_dim]
            new_d_model = encoder_flat.shape[-1]
            encoder_output = encoder_flat.view(batch_size, seq_len, new_d_model)

        # Output layer (applied per-token)
        logits = self.output_layer(encoder_output)  # [batch, seq_len, num_fields]

        return logits


def make_msm_field_mask(
    attention_mask: torch.Tensor,
    mask_prob: float,
    seed: int,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    """Create deterministic boolean mask for MSM-Field task.
    
    Randomly selects mask_prob fraction of non-padding tokens to mask.
    Masking is deterministic given seed and step for reproducibility.
    
    Args:
        attention_mask: Attention mask tensor [batch_size, seq_len] (1 = valid, 0 = pad)
        mask_prob: Probability of masking each non-padding token (0 < mask_prob <= 1)
        seed: Random seed for determinism
        step: Training step for determinism (seed + step used as actual seed)
        device: Device to create mask on
    
    Returns:
        Boolean mask tensor [batch_size, seq_len] (True = masked, False = not masked)
        Only non-padding positions (attention_mask == 1) can be True.
    """
    batch_size, seq_len = attention_mask.shape
    
    # Create deterministic generator
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + step)
    
    # Initialize mask (all False)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Only mask non-padding positions
    non_padding = attention_mask == 1
    
    if non_padding.sum() == 0:
        # No valid tokens to mask
        return mask
    
    # Generate random values for all non-padding positions
    # Use uniform distribution [0, 1)
    random_values = torch.rand(
        batch_size, seq_len, generator=generator, device=device
    )
    
    # Set random values to 1.0 for padding positions (so they're never selected)
    random_values = torch.where(non_padding, random_values, torch.ones_like(random_values))
    
    # Select positions where random value < mask_prob
    # This gives us approximately mask_prob fraction of non-padding tokens
    candidate_mask = random_values < mask_prob
    
    # Only mask non-padding positions
    mask = candidate_mask & non_padding
    
    return mask


def make_msm_field_mask_balanced(
    field_ids: torch.Tensor,
    *,
    mask_prob: float,
    mask_field_id: int,
    pad_field_id: int,
    seed: int,
    step: int,
) -> torch.Tensor:
    """Create deterministic field-balanced boolean mask for MSM-Field task.
    
    Masks approximately mask_prob fraction of non-padding tokens, with masking
    budget balanced across field IDs (excluding PAD and MASK_FIELD_ID).
    Masking is deterministic given seed and step for reproducibility.
    
    Args:
        field_ids: Field ID tensor [batch_size, seq_len] (pre-mask field IDs)
        mask_prob: Target fraction of non-padding tokens to mask (0 < mask_prob <= 1)
        mask_field_id: MASK_FIELD_ID value (positions with this ID are never masked)
        pad_field_id: PAD_FIELD_ID value (positions with this ID are never masked)
        seed: Random seed for determinism
        step: Training step for determinism (seed + step used as actual seed)
    
    Returns:
        Boolean mask tensor [batch_size, seq_len] (True = masked, False = not masked)
        Only non-padding, non-mask positions can be True.
    """
    batch_size, seq_len = field_ids.shape
    device = field_ids.device
    
    # Create deterministic generator
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + step)
    
    # Initialize mask (all False)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Find valid positions (non-padding, non-mask)
    valid = (field_ids != pad_field_id) & (field_ids != mask_field_id)
    
    if valid.sum() == 0:
        # No valid tokens to mask
        return mask
    
    # Get unique field IDs present in valid positions (excluding pad and mask)
    valid_field_ids = field_ids[valid]
    unique_fields = torch.unique(valid_field_ids)
    num_fields_present = len(unique_fields)
    
    if num_fields_present == 0:
        return mask
    
    # Calculate total tokens to mask
    num_valid = valid.sum().item()
    total_to_mask = round(mask_prob * num_valid)
    
    if total_to_mask == 0:
        return mask
    
    # Allocate masking budget per field: base quota + remainder distribution
    k_base = total_to_mask // num_fields_present
    remainder = total_to_mask % num_fields_present
    
    # Count tokens per field (only valid positions)
    field_counts = {}
    for field_id in unique_fields:
        field_id_item = field_id.item()
        # Count only valid positions (non-padding, non-mask) with this field_id
        count = ((field_ids == field_id_item) & valid).sum().item()
        field_counts[field_id_item] = count
    
    # Allocate per-field quotas (base + remainder distributed cyclically)
    field_quotas = {}
    for idx, field_id in enumerate(unique_fields):
        field_id_item = field_id.item()
        quota = k_base + (1 if idx < remainder else 0)
        # Clip to available count
        quota = min(quota, field_counts[field_id_item])
        field_quotas[field_id_item] = quota
    
    # Reallocate leftover budget to fields with remaining capacity
    allocated = sum(field_quotas.values())
    leftover = total_to_mask - allocated
    
    if leftover > 0:
        # Find fields with remaining capacity
        fields_with_capacity = [
            (fid, field_counts[fid] - field_quotas[fid])
            for fid in field_quotas.keys()
            if field_counts[fid] > field_quotas[fid]
        ]
        fields_with_capacity.sort(key=lambda x: x[1], reverse=True)  # Sort by capacity
        
        # Distribute leftover
        for idx in range(min(leftover, len(fields_with_capacity))):
            field_id = fields_with_capacity[idx][0]
            field_quotas[field_id] += 1
    
    # Mask tokens within each field deterministically
    for field_id_item, quota in field_quotas.items():
        if quota == 0:
            continue
        
        # Get all positions with this field ID
        field_positions = (field_ids == field_id_item).nonzero(as_tuple=False)  # [n, 2] where n = count
        n = len(field_positions)
        
        if n == 0:
            continue
        
        # Select exactly quota positions deterministically
        if quota >= n:
            # Mask all positions for this field
            selected_indices = torch.arange(n, device=device)
        else:
            # Permute and take first quota
            perm = torch.randperm(n, generator=generator, device=device)
            selected_indices = perm[:quota]
        
        # Set mask at selected positions
        selected_positions = field_positions[selected_indices]
        for pos in selected_positions:
            b, l = pos[0].item(), pos[1].item()
            mask[b, l] = True
    
    return mask
