"""Layer normalization component."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization wrapper for consistency.

    Wraps PyTorch's nn.LayerNorm to provide a consistent interface
    across the codebase.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        """Initialize layer normalization.

        Args:
            d_model: Normalized shape (last dimension)
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model] or any shape

        Returns:
            Normalized tensor of same shape as input
        """
        return self.norm(x)
