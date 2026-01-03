"""Dropout component."""

import torch
import torch.nn as nn


class Dropout(nn.Module):
    """Dropout wrapper for consistency.

    Wraps PyTorch's nn.Dropout to provide a consistent interface
    across the codebase.
    """

    def __init__(self, p: float = 0.1) -> None:
        """Initialize dropout.

        Args:
            p: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout.

        Args:
            x: Input tensor

        Returns:
            Tensor with dropout applied
        """
        return self.dropout(x)
