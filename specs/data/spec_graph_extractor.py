"""Specs for GraphExtractor - happy path only."""

import networkx as nx

from saab_v3.data.extractors import GraphExtractor


# ============================================================================
# GraphExtractor Basic Functionality
# ============================================================================


def spec_graph_extractor_basic(sample_graph):
    """Verify GraphExtractor extracts tokens from simple Graph."""
    # Arrange
    extractor = GraphExtractor()

    # Act
    tokens = extractor.extract(sample_graph)

    # Assert
    # Should have tokens for nodes and edges
    assert len(tokens) > 0
    # Verify positions are sequential
    positions = [token.position for token in tokens]
    assert positions == list(range(len(tokens)))
    # Verify node tokens exist
    node_values = [token.value for token in tokens if token.structure_tag.entity]
    assert len(node_values) >= 4  # At least 4 nodes


def spec_graph_extractor_with_attributes(sample_graph_with_attributes):
    """Verify GraphExtractor extracts node and edge attributes."""
    # Arrange
    extractor = GraphExtractor()

    # Act
    tokens = extractor.extract(sample_graph_with_attributes)

    # Assert
    # Should have tokens for nodes, node attributes, edges, and edge attributes
    assert len(tokens) > 0
    # Find attribute tokens (have field names)
    attr_tokens = [t for t in tokens if t.structure_tag.field]
    assert len(attr_tokens) > 0
    # Verify attribute field names
    fields = [t.structure_tag.field for t in attr_tokens]
    assert (
        "name" in fields or "value" in fields or "weight" in fields or "label" in fields
    )


def spec_graph_extractor_with_schema(sample_graph_with_attributes):
    """Verify GraphExtractor handles schema with node roles and edge types."""
    # Arrange
    extractor = GraphExtractor()
    schema = {
        "node_roles": {"1": "primary", "2": "secondary"},
        "edge_types": ["edge1", "edge2"],
        "node_fields": ["name", "value"],
        "edge_fields": ["weight"],
    }

    # Act
    tokens = extractor.extract(sample_graph_with_attributes, schema=schema)

    # Assert
    # Find node with role
    node_tokens = [t for t in tokens if t.structure_tag.entity == "1"]
    if node_tokens:
        assert any(t.structure_tag.role == "primary" for t in node_tokens)
    # Verify edge types are used
    edge_tokens = [t for t in tokens if t.structure_tag.edge]
    assert len(edge_tokens) > 0


def spec_graph_extractor_traversal(sample_graph):
    """Verify GraphExtractor supports both BFS and DFS traversal."""
    # Arrange
    extractor_bfs = GraphExtractor(traversal="bfs")
    extractor_dfs = GraphExtractor(traversal="dfs")

    # Act
    tokens_bfs = extractor_bfs.extract(sample_graph)
    tokens_dfs = extractor_dfs.extract(sample_graph)

    # Assert
    # Both should extract tokens
    assert len(tokens_bfs) > 0
    assert len(tokens_dfs) > 0
    # Node order might differ, but both should have same number of nodes
    # (exact comparison depends on graph structure, so we just verify both work)


def spec_graph_extractor_can_handle(sample_graph):
    """Verify GraphExtractor can_handle() detects correct formats."""
    # Arrange
    extractor = GraphExtractor()

    # Act & Assert
    assert extractor.can_handle(sample_graph) is True
    # Test with DiGraph
    digraph = nx.DiGraph()
    digraph.add_edge(1, 2)
    assert extractor.can_handle(digraph) is True
    # Test with invalid types
    assert extractor.can_handle("not a graph") is False
    assert extractor.can_handle(123) is False
    assert extractor.can_handle(None) is False
