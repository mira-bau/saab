"""Feed-forward network component."""

import torch
import torch.nn as nn

from saab_v3.models.components.dropout import Dropout


class FeedForward(nn.Module):
    """Feed-forward network (FFN) for Transformer encoder.

    Standard two-layer MLP with configurable activation and dropout.
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        """Initialize feed-forward network.

        Args:
            d_model: Model dimension
            ffn_dim: Feed-forward network dimension (typically 4 * d_model)
            activation: Activation function (default: GELU)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim, bias=bias)
        self.activation = activation
        self.dropout1 = Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, d_model, bias=bias)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x
