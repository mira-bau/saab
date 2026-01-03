"""Base task head class with shared functionality for all task heads."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from saab_v3.tasks.pooling import CLSPooling


class BaseTaskHead(nn.Module, ABC):
    """Abstract base class for all task heads.

    Provides shared functionality:
    - Pooling strategy for extracting sequence representations
    - MLP builder for configurable hidden layers
    - Consistent interface across all task heads
    """

    def __init__(
        self,
        d_model: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        pooling: nn.Module | None = None,
    ):
        """Initialize base task head.

        Args:
            d_model: Model dimension (from encoder)
            hidden_dims: Hidden dimensions for MLP (None = simple linear, [256, 128] = MLP)
            dropout: Dropout probability (only used if hidden_dims is not None)
            pooling: Pooling strategy (default: CLSPooling)
        """
        super().__init__()
        self.d_model = d_model
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Default to CLS pooling if not specified
        if pooling is None:
            pooling = CLSPooling()
        self.pooling = pooling

        # Build shared MLP if hidden_dims provided
        if hidden_dims is not None:
            self.mlp = self._build_mlp(d_model, hidden_dims, dropout)
        else:
            self.mlp = None

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> nn.Module:
        """Build MLP with specified hidden dimensions.

        Architecture:
        Input → Linear → Dropout → ReLU → ... → Linear → Output

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability

        Returns:
            Sequential MLP module

        Example:
            >>> mlp = self._build_mlp(768, [256, 128], 0.1)
            >>> # Creates: Linear(768→256) → Dropout → ReLU → Linear(256→128) → Dropout → ReLU
        """
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        return nn.Sequential(*layers)

    @abstractmethod
    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch_size, seq_len, d_model]
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through task head.

        Must be implemented by subclasses.

        Args:
            encoder_output: Encoder output tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            **kwargs: Task-specific arguments

        Returns:
            Task-specific output tensor
        """
        pass

    def _pool_sequence(
        self,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract sequence representation using pooling strategy.

        Args:
            encoder_output: Encoder output of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]

        Returns:
            Sequence representation of shape [batch_size, d_model]
        """
        return self.pooling(encoder_output, attention_mask)
