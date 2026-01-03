"""Token classification head for per-token labeling (NER, POS tagging)."""

import torch
import torch.nn as nn

from saab_v3.tasks.base import BaseTaskHead


class TokenClassificationHead(BaseTaskHead):
    """Token classification head for per-token labeling (NER, POS tagging).

    Uses ALL token representations (no pooling).
    Outputs logits per token.

    Returns logits per token (no activation applied). User should apply softmax per token as needed.
    """

    def __init__(
        self,
        d_model: int,
        num_labels: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        """Initialize token classification head.

        Args:
            d_model: Model dimension
            num_labels: Number of label classes
            hidden_dims: Hidden dimensions for MLP (None = simple linear)
            dropout: Dropout probability (only used if hidden_dims is not None)

        Note: Pooling is not used for token classification (uses all tokens).
        """
        # Don't use pooling for token classification
        super().__init__(d_model, hidden_dims, dropout, pooling=None)
        self.num_labels = num_labels

        # Build output layer
        if hidden_dims is not None:
            input_dim = hidden_dims[-1]
        else:
            input_dim = d_model

        self.output_layer = nn.Linear(input_dim, num_labels)

    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch, seq_len, d_model]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [batch, seq_len, num_labels]
        """Forward pass.

        Args:
            encoder_output: Encoder output tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask (not used in computation, but kept for interface consistency)

        Returns:
            Logits tensor of shape [batch_size, seq_len, num_labels] (logits per token, no activation applied).
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
        logits = self.output_layer(encoder_output)  # [batch, seq_len, num_labels]

        return logits
