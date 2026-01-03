"""SAAB Transformer: Proposed method with structure-aware attention bias."""

import torch
import torch.nn as nn

from saab_v3.data.structures import Batch, StructureTag
from saab_v3.models.components.saab_encoder_layer import SAABEncoderLayer
from saab_v3.models.embeddings.combined_embedding import CombinedEmbedding


class SAABTransformer(nn.Module):
    """SAAB Transformer: Proposed method with structure-aware attention bias.

    Uses full structural embeddings (identical to Scratch) plus SAAB attention
    with structural bias. Structure affects attention explicitly through B_struct.
    When lambda=0, this is bitwise-equivalent to ScratchTransformer.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        vocab_sizes: dict[str, int],
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        layer_norm_eps: float = 1e-5,
        positional_learned: bool = True,
        lambda_bias: float = 1.0,
        learnable_lambda: bool = False,
        bias_normalization: float = 1.0,
    ) -> None:
        """Initialize SAAB Transformer.

        Args:
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            vocab_sizes: Dictionary with vocabulary sizes.
                Must contain: "token_vocab_size", "token_type_vocab_size",
                "field_vocab_size", "entity_vocab_size", "time_vocab_size"
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function for FFN
            layer_norm_eps: Epsilon for layer normalization
            positional_learned: Whether to use learned positional embeddings
            lambda_bias: Bias strength parameter (λ). When 0, equivalent to Scratch.
            learnable_lambda: If True, lambda is a learnable parameter
            bias_normalization: Normalization factor for structural bias
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        self.lambda_bias = lambda_bias

        # Embeddings (identical to Scratch)
        self.embeddings = CombinedEmbedding(
            d_model=d_model,
            vocab_sizes=vocab_sizes,
            max_seq_len=max_seq_len,
            use_token_type=True,
            use_field=True,
            use_entity=True,
            use_time=True,
            positional_learned=positional_learned,
        )

        # SAAB encoder layers (with structure-aware attention)
        self.layers = nn.ModuleList(
            [
                SAABEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                    lambda_bias=lambda_bias,
                    learnable_lambda=learnable_lambda,
                    bias_normalization=bias_normalization,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        batch: Batch,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through SAAB Transformer.

        Args:
            batch: Batch object with token and tag indices.
                Must have batch.original_tags if lambda_bias != 0
            return_attention_weights: Whether to return attention weights from all layers

        Returns:
            If return_attention_weights=False:
                Output tensor of shape [batch_size, seq_len, d_model]
            If return_attention_weights=True:
                Tuple of (output tensor, list of attention weights)
                Each attention weight tensor has shape [batch_size, num_heads, seq_len, seq_len]

        Raises:
            ValueError: If original_tags is None when lambda_bias != 0
        """
        # Validate original_tags if needed
        if self.lambda_bias != 0.0 and batch.original_tags is None:
            raise ValueError(
                "batch.original_tags is required for SAAB Transformer when lambda_bias != 0. "
                "Set preserve_original_tags=True in preprocessing."
            )

        # Apply embeddings
        x = self.embeddings(batch)  # [batch_size, seq_len, d_model]

        # Apply SAAB encoder layers
        attention_weights_list = []
        for layer in self.layers:
            if return_attention_weights:
                x, attn_weights = layer(
                    x,
                    attention_mask=batch.attention_mask,
                    original_tags=batch.original_tags,
                    return_attention_weights=True,
                )
                attention_weights_list.append(attn_weights)
            else:
                x = layer(
                    x,
                    attention_mask=batch.attention_mask,
                    original_tags=batch.original_tags,
                    return_attention_weights=False,
                )

        if return_attention_weights:
            return x, attention_weights_list
        return x
