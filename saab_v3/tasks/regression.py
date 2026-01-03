"""Regression task head for continuous value prediction."""

import torch
import torch.nn as nn

from saab_v3.tasks.base import BaseTaskHead


class RegressionHead(BaseTaskHead):
    """Regression task head for continuous value prediction.

    Supports:
    - Single-target regression (num_targets=1)
    - Multi-target regression (num_targets > 1)

    Returns continuous values (no activation applied).
    """

    def __init__(
        self,
        d_model: int,
        num_targets: int = 1,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        pooling: nn.Module | None = None,
    ):
        """Initialize regression head.

        Args:
            d_model: Model dimension
            num_targets: Number of target values (default: 1)
            hidden_dims: Hidden dimensions for MLP (None = simple linear)
            dropout: Dropout probability (only used if hidden_dims is not None)
            pooling: Pooling strategy (default: CLSPooling)
        """
        super().__init__(d_model, hidden_dims, dropout, pooling)
        self.num_targets = num_targets

        # Build output layer
        if hidden_dims is not None:
            input_dim = hidden_dims[-1]
        else:
            input_dim = d_model

        self.output_layer = nn.Linear(input_dim, num_targets)

    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch, seq_len, d_model]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [batch, num_targets]
        """Forward pass.

        Args:
            encoder_output: Encoder output tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]

        Returns:
            Predictions tensor of shape [batch_size, num_targets] (continuous values, no activation).
        """
        # Pool sequence
        seq_repr = self._pool_sequence(encoder_output, attention_mask)

        # Apply MLP if present
        if self.mlp is not None:
            seq_repr = self.mlp(seq_repr)

        # Output layer (linear, no activation)
        predictions = self.output_layer(seq_repr)

        return predictions
