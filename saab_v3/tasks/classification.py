"""Classification task head for binary/multi-class/multi-label tasks."""

import torch
import torch.nn as nn

from saab_v3.tasks.base import BaseTaskHead
from saab_v3.tasks.pooling import MeanPooling


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

        # Add LayerNorm before output layer to stabilize pooled representation
        # This prevents logits explosion and reduces saturation
        self.layer_norm = nn.LayerNorm(input_dim)
        
        self.output_layer = nn.Linear(input_dim, num_classes)
        
        # Initialize output layer with smaller weights to prevent extreme logits
        # This helps with training stability and reduces logit saturation
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

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
        # Assert: MeanPooling requires attention_mask
        if isinstance(self.pooling, MeanPooling):
            assert attention_mask is not None, (
                "attention_mask is required when pooling='mean'. "
                "MeanPooling needs attention_mask to exclude padding tokens."
            )
            assert attention_mask.ndim == 2, (
                f"attention_mask must be 2D [B, L], got shape {attention_mask.shape}"
            )
            assert attention_mask.shape[0] == encoder_output.shape[0], (
                f"attention_mask batch size {attention_mask.shape[0]} must match "
                f"encoder_output batch size {encoder_output.shape[0]}"
            )
            assert attention_mask.shape[1] == encoder_output.shape[1], (
                f"attention_mask sequence length {attention_mask.shape[1]} must match "
                f"encoder_output sequence length {encoder_output.shape[1]}"
            )
        
        # Pool sequence
        seq_repr = self._pool_sequence(encoder_output, attention_mask)

        # Apply MLP if present
        if self.mlp is not None:
            seq_repr = self.mlp(seq_repr)

        # Apply LayerNorm to stabilize representation before classifier
        seq_repr = self.layer_norm(seq_repr)

        # Output layer (logits)
        logits = self.output_layer(seq_repr)

        return logits
