"""Classification task head for binary/multi-class/multi-label tasks."""

import torch
import torch.nn as nn

from saab_v3.tasks.base import BaseTaskHead


class ClassificationHead(BaseTaskHead):
    """Classification task head for binary/multi-class/multi-label tasks.

    Supports:
    - Binary classification (num_classes=2)
    - Multi-class classification (num_classes > 2, multi_label=False)
    - Multi-label classification (num_classes > 2, multi_label=True)

    Returns logits (no activation applied). User should apply:
    - Softmax for binary/multi-class (multi_label=False)
    - Sigmoid for multi-label (multi_label=True)
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        multi_label: bool = False,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        pooling: nn.Module | None = None,
    ):
        """Initialize classification head.

        Args:
            d_model: Model dimension
            num_classes: Number of output classes
            multi_label: If True, uses sigmoid (multi-label). If False, uses softmax (multi-class)
            hidden_dims: Hidden dimensions for MLP (None = simple linear)
            dropout: Dropout probability (only used if hidden_dims is not None)
            pooling: Pooling strategy (default: CLSPooling)
        """
        super().__init__(d_model, hidden_dims, dropout, pooling)
        self.num_classes = num_classes
        self.multi_label = multi_label

        # Build output layer
        if hidden_dims is not None:
            input_dim = hidden_dims[-1]  # Last hidden layer dimension
        else:
            input_dim = d_model  # Simple linear mode

        self.output_layer = nn.Linear(input_dim, num_classes)

    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch, seq_len, d_model]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [batch, num_classes]
        """Forward pass.

        Args:
            encoder_output: Encoder output tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]

        Returns:
            Logits tensor of shape [batch_size, num_classes] (no activation applied).
            User should apply softmax (multi-class) or sigmoid (multi-label) as needed.
        """
        # Pool sequence
        seq_repr = self._pool_sequence(encoder_output, attention_mask)

        # Apply MLP if present
        if self.mlp is not None:
            seq_repr = self.mlp(seq_repr)

        # Output layer (logits)
        logits = self.output_layer(seq_repr)

        return logits
