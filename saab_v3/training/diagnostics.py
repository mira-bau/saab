"""Diagnostic functions for MSM training comparisons."""

import torch


def compute_attention_entropy(
    attn_weights: torch.Tensor,
    attention_mask: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute average attention entropy per layer.

    For each query token, computes entropy = -sum(p_i * log(p_i + eps)) over keys.
    Averages over heads, tokens (non-padding), and batch.

    Args:
        attn_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        attention_mask: Attention mask [batch_size, seq_len] where 1 = valid, 0 = padding
        eps: Small epsilon to avoid log(0)

    Returns:
        Average entropy scalar (float)
    """
    batch_size, num_heads, seq_len, _ = attn_weights.shape

    # Compute entropy for each query token: H = -sum(p * log(p + eps))
    # attn_weights: [B, H, L, L] where last dim is attention distribution over keys
    entropy = -torch.sum(
        attn_weights * torch.log(attn_weights + eps), dim=-1
    )  # [B, H, L]

    # Mask out padding positions (attention_mask == 0)
    # attention_mask: [B, L] -> [B, 1, L] for broadcasting
    mask = attention_mask.unsqueeze(1)  # [B, 1, L]
    entropy = entropy * mask  # [B, H, L]

    # Average over heads, tokens (non-padding), and batch
    # Count non-padding tokens
    num_valid_tokens = mask.sum().item()  # Total non-padding tokens across batch
    if num_valid_tokens == 0:
        return 0.0

    # Sum over all dimensions and divide by number of valid tokens
    total_entropy = entropy.sum().item()
    avg_entropy = total_entropy / (num_valid_tokens * num_heads)

    return avg_entropy


def compute_same_field_mass(
    attn_weights: torch.Tensor,
    field_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """Compute average same-field attention mass per layer.

    For each query token i, sums attention weights to keys j where field_ids[j] == field_ids[i].
    Averages over heads, tokens (non-padding), and batch.

    Args:
        attn_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        field_ids: Field IDs [batch_size, seq_len] (field index per token, should be PRE-MASK for MSM-Field)
        attention_mask: Attention mask [batch_size, seq_len] where 1 = valid, 0 = padding

    Returns:
        Average same-field mass scalar (float)
    """
    """Compute average same-field attention mass per layer.

    For each query token i, sums attention weights to keys j where field_ids[j] == field_ids[i].
    Averages over heads, tokens (non-padding), and batch.

    Args:
        attn_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        field_ids: Field IDs [batch_size, seq_len] (field index per token)
        attention_mask: Attention mask [batch_size, seq_len] where 1 = valid, 0 = padding

    Returns:
        Average same-field mass scalar (float)
    """
    batch_size, num_heads, seq_len, _ = attn_weights.shape

    # Create field comparison matrix: [B, L, L] where [b, i, j] = (field_ids[b, i] == field_ids[b, j])
    # field_ids: [B, L] -> [B, L, 1] and [B, 1, L] for broadcasting
    field_ids_i = field_ids.unsqueeze(-1)  # [B, L, 1]
    field_ids_j = field_ids.unsqueeze(1)  # [B, 1, L]
    same_field_mask = (field_ids_i == field_ids_j).float()  # [B, L, L]

    # Expand to include heads: [B, L, L] -> [B, 1, L, L]
    same_field_mask = same_field_mask.unsqueeze(1)  # [B, 1, L, L]

    # For each query token i, sum attention weights where field_ids[j] == field_ids[i]
    # attn_weights: [B, H, L, L], same_field_mask: [B, 1, L, L]
    same_field_attn = attn_weights * same_field_mask  # [B, H, L, L]
    same_field_mass = same_field_attn.sum(dim=-1)  # [B, H, L] - sum over keys

    # Mask out padding positions (attention_mask == 0)
    # attention_mask: [B, L] -> [B, 1, L] for broadcasting
    mask = attention_mask.unsqueeze(1)  # [B, 1, L]
    same_field_mass = same_field_mass * mask  # [B, H, L]

    # Average over heads, tokens (non-padding), and batch
    # Count non-padding tokens
    num_valid_tokens = mask.sum().item()  # Total non-padding tokens across batch
    if num_valid_tokens == 0:
        return 0.0

    # Sum over all dimensions and divide by number of valid tokens
    total_mass = same_field_mass.sum().item()
    avg_mass = total_mass / (num_valid_tokens * num_heads)

    return avg_mass
