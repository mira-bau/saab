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


# ============================================================================
# Tokenization Fixtures
# ============================================================================


@pytest.fixture
def sample_vocabulary():
    """Pre-built vocabulary for testing."""
    from saab_v3.data.vocabulary import Vocabulary
    from saab_v3.data.constants import VALUE_SPECIAL_TOKENS

    vocab = Vocabulary(special_tokens=VALUE_SPECIAL_TOKENS)
    tokens = ["hello", "world", "hello", "test", "world", "python"]
    vocab.build_from_tokens(tokens)
    return vocab


@pytest.fixture
def sample_tokenized_sequences():
    """Sample TokenizedSequence objects for tokenizer/encoder testing."""
    from saab_v3.data.structures import TokenizedSequence, Token, StructureTag

    seq1 = TokenizedSequence(
        tokens=[
            Token(
                value="Alice",
                structure_tag=StructureTag(field="name", entity="user_1"),
                position=0,
            ),
            Token(
                value="25",
                structure_tag=StructureTag(field="age", entity="user_1"),
                position=1,
            ),
        ],
        sequence_id="seq1",
    )

    seq2 = TokenizedSequence(
        tokens=[
            Token(
                value="Bob",
                structure_tag=StructureTag(field="name", entity="user_2"),
                position=0,
            ),
            Token(
                value="30",
                structure_tag=StructureTag(field="age", entity="user_2"),
                position=1,
            ),
        ],
        sequence_id="seq2",
    )

    return [seq1, seq2]


@pytest.fixture
def sample_structure_tags():
    """Various structure tags for testing."""
    from saab_v3.data.structures import StructureTag

    return {
        "minimal": StructureTag(field="name"),
        "multiple": StructureTag(field="name", entity="user_1", token_type="text"),
        "all_fields": StructureTag(
            field="name",
            entity="user_1",
            time="2023-Q1",
            edge="parent_of",
            role="primary_key",
            token_type="text",
        ),
        "missing_fields": StructureTag(field="name", entity="user_1"),
    }


# ============================================================================
# Batcher Fixtures
# ============================================================================


@pytest.fixture
def sample_encoded_sequences(sample_tokenized_sequences):
    """Pre-encoded sequences (TokenizedSequence, token_ids, encoded_tags) for batcher testing."""
    from saab_v3.data import ValueTokenizer, TagEncoder

    # Build tokenizer and encoder
    tokenizer = ValueTokenizer(vocab_size=100)
    tokenizer.build_vocab(sample_tokenized_sequences)

    encoder = TagEncoder()
    encoder.build_vocabs(sample_tokenized_sequences)

    # Encode sequences
    encoded = []
    for seq in sample_tokenized_sequences:
        seq_enc, token_ids = tokenizer.encode_sequence(seq)
        seq_final, encoded_tags = encoder.encode_sequence(seq_enc)
        encoded.append((seq_final, token_ids, encoded_tags, None))  # Add None label

    return encoded


@pytest.fixture
def sample_encoded_sequences_different_lengths():
    """Encoded sequences with different lengths for padding testing."""
    from saab_v3.data.structures import TokenizedSequence, Token, StructureTag
    from saab_v3.data import ValueTokenizer, TagEncoder

    # Create sequences with different lengths
    seq1 = TokenizedSequence(
        tokens=[
            Token(value="A", structure_tag=StructureTag(field="col1"), position=0),
            Token(value="B", structure_tag=StructureTag(field="col2"), position=1),
        ],
        sequence_id="short",
    )

    seq2 = TokenizedSequence(
        tokens=[
            Token(value="X", structure_tag=StructureTag(field="col1"), position=0),
            Token(value="Y", structure_tag=StructureTag(field="col2"), position=1),
            Token(value="Z", structure_tag=StructureTag(field="col3"), position=2),
            Token(value="W", structure_tag=StructureTag(field="col4"), position=3),
        ],
        sequence_id="long",
    )

    sequences = [seq1, seq2]

    # Build tokenizer and encoder
    tokenizer = ValueTokenizer(vocab_size=100)
    tokenizer.build_vocab(sequences)

    encoder = TagEncoder()
    encoder.build_vocabs(sequences)

    # Encode sequences
    encoded = []
    for seq in sequences:
        seq_enc, token_ids = tokenizer.encode_sequence(seq)
        seq_final, encoded_tags = encoder.encode_sequence(seq_enc)
        encoded.append((seq_final, token_ids, encoded_tags, None))  # Add None label

    return encoded


@pytest.fixture
def sample_encoded_sequences_with_edges_roles():
    """Encoded sequences with edge and role tags for optional tag testing."""
    from saab_v3.data.structures import TokenizedSequence, Token, StructureTag
    from saab_v3.data import ValueTokenizer, TagEncoder

    # Create sequences with edges and roles
    seq1 = TokenizedSequence(
        tokens=[
            Token(
                value="node1",
                structure_tag=StructureTag(
                    field="id", entity="graph_1", edge="connects", role="source"
                ),
                position=0,
            ),
            Token(
                value="node2",
                structure_tag=StructureTag(
                    field="id", entity="graph_1", edge="connects", role="target"
                ),
                position=1,
            ),
        ],
        sequence_id="with_edges",
    )

    sequences = [seq1]

    # Build tokenizer and encoder
    tokenizer = ValueTokenizer(vocab_size=100)
    tokenizer.build_vocab(sequences)

    encoder = TagEncoder()
    encoder.build_vocabs(sequences)

    # Encode sequences
    encoded = []
    for seq in sequences:
        seq_enc, token_ids = tokenizer.encode_sequence(seq)
        seq_final, encoded_tags = encoder.encode_sequence(seq_enc)
        encoded.append((seq_final, token_ids, encoded_tags, None))  # Add None label

    return encoded
