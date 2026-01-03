"""Pooling strategies for extracting sequence-level representations from encoder outputs."""

import torch
import torch.nn as nn


class CLSPooling(nn.Module):
    """Extract [CLS] token representation (position 0).

    This is the default pooling strategy, matching BERT-style approach.
    The [CLS] token is expected to be at position 0 in the sequence.
    """

    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch_size, seq_len, d_model]
        attention_mask: torch.Tensor | None = None,  # [batch_size, seq_len]
    ) -> torch.Tensor:  # [batch_size, d_model]
        """Extract [CLS] token at position 0.

        Args:
            encoder_output: Encoder output tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask (not used for CLS pooling, but kept for interface consistency)

        Returns:
            Sequence representation of shape [batch_size, d_model]
        """
        return encoder_output[:, 0, :]


class MeanPooling(nn.Module):
    """Mean pooling over all tokens (excluding padding).

    Computes the mean of all valid tokens (where attention_mask == 1),
    excluding padding tokens (where attention_mask == 0).
    """

    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch_size, seq_len, d_model]
        attention_mask: torch.Tensor,  # [batch_size, seq_len] (required)
    ) -> torch.Tensor:  # [batch_size, d_model]
        """Mean pool over valid tokens only.

        Args:
            encoder_output: Encoder output tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Attention mask of shape [batch_size, seq_len] (1 = valid, 0 = padding)

        Returns:
            Sequence representation of shape [batch_size, d_model]

        Raises:
            ValueError: If attention_mask is not provided
        """
        if attention_mask is None:
            raise ValueError("attention_mask is required for MeanPooling")

        # Expand attention mask to match encoder_output dimensions
        # [batch_size, seq_len] -> [batch_size, seq_len, 1]
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # Mask out padding tokens (set to 0)
        masked_output = encoder_output * mask_expanded

        # Sum over sequence length dimension
        # [batch_size, seq_len, d_model] -> [batch_size, d_model]
        sum_output = masked_output.sum(dim=1)

        # Count valid tokens per sequence
        # [batch_size, seq_len] -> [batch_size]
        valid_token_counts = attention_mask.sum(dim=1).float()

        # Avoid division by zero (if all tokens are padding, return zeros)
        # [batch_size] -> [batch_size, 1]
        valid_token_counts = valid_token_counts.unsqueeze(-1)
        valid_token_counts = torch.clamp(valid_token_counts, min=1.0)

        # Compute mean
        mean_output = sum_output / valid_token_counts

        return mean_output


class MaxPooling(nn.Module):
    """Max pooling over all tokens (excluding padding).

    Computes the maximum over all valid tokens (where attention_mask == 1),
    excluding padding tokens (where attention_mask == 0).
    """

    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch_size, seq_len, d_model]
        attention_mask: torch.Tensor,  # [batch_size, seq_len] (required)
    ) -> torch.Tensor:  # [batch_size, d_model]
        """Max pool over valid tokens only.

        Args:
            encoder_output: Encoder output tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Attention mask of shape [batch_size, seq_len] (1 = valid, 0 = padding)

        Returns:
            Sequence representation of shape [batch_size, d_model]

        Raises:
            ValueError: If attention_mask is not provided
        """
        if attention_mask is None:
            raise ValueError("attention_mask is required for MaxPooling")

        # Expand attention mask to match encoder_output dimensions
        # [batch_size, seq_len] -> [batch_size, seq_len, 1]
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # Mask out padding tokens by setting them to a very large negative value
        # Use a large negative value instead of -inf to avoid NaN issues
        large_negative = torch.finfo(encoder_output.dtype).min
        mask_inverted = (1.0 - mask_expanded) * large_negative

        # Apply mask: valid tokens remain unchanged, padding becomes large negative
        masked_output = encoder_output + mask_inverted

        # Max over sequence length dimension
        # [batch_size, seq_len, d_model] -> [batch_size, d_model]
        max_output, _ = masked_output.max(dim=1)

        return max_output
