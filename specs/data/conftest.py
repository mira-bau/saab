"""Fixtures for data structure specs."""

import torch
import pytest
import pandas as pd
import networkx as nx

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


# ============================================================================
# Extractor Fixtures
# ============================================================================


@pytest.fixture
def sample_dataframe():
    """Simple DataFrame for testing TableExtractor."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [85.5, 90.0, 88.5],
        }
    )


@pytest.fixture
def sample_dataframe_with_nan():
    """DataFrame with NaN values for testing."""
    return pd.DataFrame(
        {
            "name": ["Alice", None, "Charlie"],
            "age": [25, 30, None],
            "score": [85.5, None, 88.5],
        }
    )


@pytest.fixture
def sample_json_dict():
    """Simple dict for testing JSONExtractor."""
    return {"name": "Alice", "age": 25, "city": "NYC"}


@pytest.fixture
def sample_json_nested():
    """Nested dict structure for testing JSONExtractor."""
    return {
        "user": {
            "name": "Alice",
            "address": {"city": "NYC", "zip": "10001"},
        }
    }


@pytest.fixture
def sample_graph():
    """Simple NetworkX Graph for testing GraphExtractor."""
    graph = nx.Graph()
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    return graph


@pytest.fixture
def sample_graph_with_attributes():
    """Graph with node and edge attributes for testing."""
    graph = nx.Graph()
    graph.add_node(1, name="node1", value=10)
    graph.add_node(2, name="node2", value=20)
    graph.add_node(3, name="node3", value=30)
    graph.add_edge(1, 2, weight=1.5, label="edge1")
    graph.add_edge(2, 3, weight=2.0, label="edge2")
    return graph
