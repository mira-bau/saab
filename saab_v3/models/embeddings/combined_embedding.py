"""Combined embedding module that sums all embedding types."""

import torch
import torch.nn as nn

from saab_v3.data.structures import Batch

from saab_v3.models.embeddings.entity_embedding import EntityEmbedding
from saab_v3.models.embeddings.field_embedding import FieldEmbedding
from saab_v3.models.embeddings.positional_embedding import PositionalEmbedding
from saab_v3.models.embeddings.time_embedding import TimeEmbedding
from saab_v3.models.embeddings.token_embedding import TokenEmbedding
from saab_v3.models.embeddings.token_type_embedding import TokenTypeEmbedding


class CombinedEmbedding(nn.Module):
    """Combined embedding module that sums all enabled embedding types.

    Supports different configurations for Flat (token + positional only)
    and Scratch/SAAB (all embeddings) models.
    """

    def __init__(
        self,
        d_model: int,
        vocab_sizes: dict[str, int],
        max_seq_len: int = 512,
        use_token_type: bool = False,
        use_field: bool = False,
        use_entity: bool = False,
        use_time: bool = False,
        positional_learned: bool = True,
    ) -> None:
        """Initialize combined embedding.

        Args:
            d_model: Embedding dimension
            vocab_sizes: Dictionary with vocabulary sizes:
                - `token_vocab_size`: Required
                - `field_vocab_size`: Required if use_field=True
                - `entity_vocab_size`: Required if use_entity=True
                - `time_vocab_size`: Required if use_time=True
                - `token_type_vocab_size`: Required if use_token_type=True
            max_seq_len: Maximum sequence length for positional embeddings
            use_token_type: Whether to include token type embeddings
            use_field: Whether to include field embeddings
            use_entity: Whether to include entity embeddings
            use_time: Whether to include time embeddings
            positional_learned: Whether to use learned positional embeddings
        """
        super().__init__()
        self.d_model = d_model
        self.use_token_type = use_token_type
        self.use_field = use_field
        self.use_entity = use_entity
        self.use_time = use_time

        # Token embedding (always required)
        if "token_vocab_size" not in vocab_sizes:
            raise ValueError("token_vocab_size is required in vocab_sizes")
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_sizes["token_vocab_size"],
            d_model=d_model,
        )

        # Positional embedding (always required)
        self.positional_embedding = PositionalEmbedding(
            max_seq_len=max_seq_len,
            d_model=d_model,
            learned=positional_learned,
        )

        # Optional embeddings
        if use_token_type:
            if "token_type_vocab_size" not in vocab_sizes:
                raise ValueError(
                    "token_type_vocab_size is required when use_token_type=True"
                )
            self.token_type_embedding = TokenTypeEmbedding(
                vocab_size=vocab_sizes["token_type_vocab_size"],
                d_model=d_model,
            )
        else:
            self.token_type_embedding = None

        if use_field:
            if "field_vocab_size" not in vocab_sizes:
                raise ValueError("field_vocab_size is required when use_field=True")
            self.field_embedding = FieldEmbedding(
                vocab_size=vocab_sizes["field_vocab_size"],
                d_model=d_model,
            )
        else:
            self.field_embedding = None

        if use_entity:
            if "entity_vocab_size" not in vocab_sizes:
                raise ValueError("entity_vocab_size is required when use_entity=True")
            self.entity_embedding = EntityEmbedding(
                vocab_size=vocab_sizes["entity_vocab_size"],
                d_model=d_model,
            )
        else:
            self.entity_embedding = None

        if use_time:
            if "time_vocab_size" not in vocab_sizes:
                raise ValueError("time_vocab_size is required when use_time=True")
            self.time_embedding = TimeEmbedding(
                vocab_size=vocab_sizes["time_vocab_size"],
                d_model=d_model,
            )
        else:
            self.time_embedding = None

    def forward(self, batch: Batch) -> torch.Tensor:
        """Apply combined embedding.

        Args:
            batch: Batch object containing all token and tag indices

        Returns:
            Combined embeddings of shape [batch_size, seq_len, d_model]
        """
        # Token embedding (always)
        token_emb = self.token_embedding(batch.token_ids)

        # Positional embedding (always)
        # Use token_emb to infer sequence length
        pos_emb = self.positional_embedding(token_emb)

        # Start with token + positional
        combined = token_emb + pos_emb

        # Add optional embeddings
        if self.use_token_type and self.token_type_embedding is not None:
            combined = combined + self.token_type_embedding(batch.token_type_ids)

        if self.use_field and self.field_embedding is not None:
            combined = combined + self.field_embedding(batch.field_ids)

        if self.use_entity and self.entity_embedding is not None:
            combined = combined + self.entity_embedding(batch.entity_ids)

        if self.use_time and self.time_embedding is not None:
            combined = combined + self.time_embedding(batch.time_ids)

        return combined
