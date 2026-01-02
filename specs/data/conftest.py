"""Fixtures for data structure specs."""

import torch
import pytest

from saab_v3.data.structures import StructureTag, Token


@pytest.fixture
def sample_structure_tag():
    """Minimal valid StructureTag."""
    return StructureTag(field="name")


@pytest.fixture
def sample_structure_tag_multiple():
    """StructureTag with multiple fields."""
    return StructureTag(field="name", entity="user_123", token_type="text")


@pytest.fixture
def sample_structure_tag_all_fields():
    """StructureTag with all fields."""
    return StructureTag(
        field="name",
        entity="user_123",
        time="2023-Q1",
        edge="parent_of",
        role="primary_key",
        token_type="text",
    )


@pytest.fixture
def sample_token(sample_structure_tag):
    """Minimal valid Token."""
    return Token(value="John", structure_tag=sample_structure_tag, position=0)


@pytest.fixture
def sample_tokens(sample_structure_tag):
    """List of sample tokens for TokenizedSequence."""
    return [
        Token(value="John", structure_tag=StructureTag(field="name"), position=0),
        Token(value="Doe", structure_tag=StructureTag(field="name"), position=1),
        Token(value="42", structure_tag=StructureTag(field="age"), position=2),
    ]


@pytest.fixture
def sample_batch_tensors():
    """Sample tensors for Batch creation."""
    batch_size, seq_len = 2, 5
    return {
        "token_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "field_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "entity_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "time_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "token_type_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "sequence_lengths": [seq_len, seq_len],
    }
