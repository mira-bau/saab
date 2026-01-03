"""Pairwise ranking head for comparing two sequences."""

import torch
import torch.nn as nn

from saab_v3.tasks.base import BaseTaskHead


class PairwiseRankingHead(BaseTaskHead):
    """Pairwise ranking head for comparing two sequences.

    Takes TWO sequence representations (not encoder output directly).
    User must pool sequences first before passing to this head.

    Supports multiple comparison methods:
    - dot_product: Dot product of representations
    - cosine: Cosine similarity
    - mlp: MLP on concatenated representations
    - difference: MLP on difference vector

    Returns scores where higher score means seq_a is better than seq_b.
    """

    def __init__(
        self,
        d_model: int,
        method: str = "dot_product",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        """Initialize pairwise ranking head.

        Args:
            d_model: Model dimension (for each sequence representation)
            method: Comparison method ("dot_product", "cosine", "mlp", "difference")
            hidden_dims: Hidden dimensions for MLP methods (required for "mlp" and "difference")
            dropout: Dropout probability (only used if hidden_dims is not None)

        Note: Pooling is not used (user provides already-pooled representations).

        Raises:
            ValueError: If method is invalid or hidden_dims is missing for MLP methods
        """
        # Don't use pooling (user provides pooled representations)
        super().__init__(d_model, hidden_dims, dropout, pooling=None)
        self.method = method

        # Validate method
        valid_methods = ["dot_product", "cosine", "mlp", "difference"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")

        # Build comparison layers based on method
        if method in ["mlp", "difference"]:
            if hidden_dims is None:
                raise ValueError(f"hidden_dims is required for method '{method}'")
            # MLP for comparison
            if method == "mlp":
                # Concatenate representations: [batch, 2 * d_model]
                input_dim = 2 * d_model
            else:  # difference
                # Difference vector: [batch, d_model]
                input_dim = d_model

            # Build MLP ending with single output
            if len(hidden_dims) > 0:
                # MLP with hidden layers + final output layer
                self.comparison_mlp = self._build_mlp(input_dim, hidden_dims, dropout)
                self.output_layer = nn.Linear(hidden_dims[-1], 1)
            else:
                # No hidden layers, just linear
                self.comparison_mlp = None
                self.output_layer = nn.Linear(input_dim, 1)
        else:
            # dot_product and cosine don't need layers
            self.comparison_mlp = None
            self.output_layer = None

    def forward(
        self,
        seq_a_repr: torch.Tensor,  # [batch, d_model]
        seq_b_repr: torch.Tensor,  # [batch, d_model]
    ) -> torch.Tensor:  # [batch]
        """Forward pass.

        Args:
            seq_a_repr: First sequence representation [batch_size, d_model]
            seq_b_repr: Second sequence representation [batch_size, d_model]

        Returns:
            Scores tensor of shape [batch_size] where higher score means seq_a is better than seq_b.
        """
        if self.method == "dot_product":
            # Dot product: score = seq_a @ seq_b
            scores = (seq_a_repr * seq_b_repr).sum(dim=1)  # [batch]

        elif self.method == "cosine":
            # Cosine similarity: score = cosine(seq_a, seq_b)
            # Normalize both vectors
            seq_a_norm = seq_a_repr / (seq_a_repr.norm(dim=1, keepdim=True) + 1e-8)
            seq_b_norm = seq_b_repr / (seq_b_repr.norm(dim=1, keepdim=True) + 1e-8)
            scores = (seq_a_norm * seq_b_norm).sum(dim=1)  # [batch]

        elif self.method == "mlp":
            # Concatenate representations
            combined = torch.cat(
                [seq_a_repr, seq_b_repr], dim=1
            )  # [batch, 2 * d_model]
            # Apply MLP
            if self.comparison_mlp is not None:
                combined = self.comparison_mlp(combined)
            # Output layer
            scores = self.output_layer(combined).squeeze(-1)  # [batch]

        elif self.method == "difference":
            # Difference vector
            diff = seq_a_repr - seq_b_repr  # [batch, d_model]
            # Apply MLP
            if self.comparison_mlp is not None:
                diff = self.comparison_mlp(diff)
            # Output layer
            scores = self.output_layer(diff).squeeze(-1)  # [batch]

        return scores
