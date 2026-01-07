"""Graph extractor for NetworkX graphs. DGL support deferred."""

from typing import Any

import networkx as nx
from tqdm import tqdm

from saab_v3.data.extractors.base import StructuralExtractor
from saab_v3.data.structures import StructureTag, Token


def _is_dgl_graph(data: Any) -> bool:
    """Check if data is a DGL graph."""
    # Check for DGL graph without importing (to avoid dependency)
    if hasattr(data, "__class__"):
        class_name = data.__class__.__name__
        module_name = getattr(data.__class__, "__module__", "")
        if "dgl" in module_name.lower() or "DGLGraph" in class_name:
            return True
    return False


class GraphExtractor(StructuralExtractor):
    """Extract tokens from graph structures (NetworkX supported, DGL deferred)."""

    def __init__(self, traversal: str = "bfs"):
        """Initialize graph extractor.

        Args:
            traversal: Traversal method, "bfs" (breadth-first) or "dfs" (depth-first)
        """
        if traversal not in ["bfs", "dfs"]:
            raise ValueError(f"traversal must be 'bfs' or 'dfs', got {traversal}")
        self.traversal = traversal

    def can_handle(self, data: Any) -> bool:
        """Check if data is a NetworkX graph.

        DGL graphs return False (not yet supported).
        """
        if _is_dgl_graph(data):
            return False  # DGL not supported

        if nx is None:
            return False

        # Check if it's a NetworkX graph type
        try:
            return isinstance(
                data, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
            )
        except (TypeError, AttributeError):
            return False

    def extract(self, data: Any, schema: dict | None = None) -> list[Token]:
        """Extract tokens from graph data.

        Args:
            data: NetworkX graph object
            schema: Optional schema with node_roles, edge_types, node_fields, edge_fields

        Returns:
            List of Token objects (nodes first, then edges)

        Raises:
            NotImplementedError: If DGL graph is provided
            ValueError: If data is not a NetworkX graph
        """
        # Check for DGL graph
        if _is_dgl_graph(data):
            raise NotImplementedError(
                "DGL graph support is not yet implemented. "
                "Please use NetworkX graphs or convert DGL graph to NetworkX first."
            )

        if nx is None:
            raise ImportError("networkx is required for GraphExtractor")

        # Verify it's a NetworkX graph
        try:
            if not isinstance(
                data, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
            ):
                raise ValueError(f"Expected NetworkX graph, got {type(data)}")
        except (TypeError, AttributeError):
            raise ValueError(f"Expected NetworkX graph, got {type(data)}")

        tokens: list[Token] = []
        position = 0

        # Extract schema information if provided
        node_roles = schema.get("node_roles", {}) if schema else {}
        edge_types = schema.get("edge_types", []) if schema else []
        node_fields = schema.get("node_fields", []) if schema else None
        edge_fields = schema.get("edge_fields", []) if schema else None

        # Get node order based on traversal
        nodes = self._get_node_order(data)
        n_nodes = len(nodes)

        # Get edges upfront
        edges = list(data.edges(data=True))
        n_edges = len(edges)

        # Create progress bars for large graphs
        node_pbar = (
            tqdm(total=n_nodes, desc="Extracting nodes") if n_nodes > 1000 else None
        )
        edge_pbar = (
            tqdm(total=n_edges, desc="Extracting edges") if n_edges > 1000 else None
        )

        try:
            # Extract node tokens
            for node_id in nodes:
                node_data = data.nodes[node_id]

                # Extract node attributes as fields
                # Exclude label attributes to prevent data leakage
                label_keys = ["label", "target", "y"]
                if node_fields is None:
                    # Extract all attributes except label keys
                    attrs = {
                        k: v
                        for k, v in node_data.items()
                        if k not in label_keys
                    }
                else:
                    # Extract only specified fields (already excludes label keys if not in schema)
                    attrs = {k: v for k, v in node_data.items() if k in node_fields}

                # Create token for node ID
                structure_tag = StructureTag(
                    entity=str(node_id),
                    role=node_roles.get(str(node_id)),
                )

                token = Token(
                    value=str(node_id),
                    structure_tag=structure_tag,
                    position=position,
                )
                tokens.append(token)
                position += 1

                # Create tokens for node attributes
                for attr_name, attr_value in attrs.items():
                    value_str = str(attr_value) if attr_value is not None else ""
                    if not value_str:
                        continue

                    structure_tag = StructureTag(
                        field=attr_name,
                        entity=str(node_id),
                    )

                    token = Token(
                        value=value_str,
                        structure_tag=structure_tag,
                        position=position,
                    )
                    tokens.append(token)
                    position += 1

                # Update progress bar
                if node_pbar:
                    node_pbar.update(1)

            # Extract edge tokens
            for source, target, edge_data in edges:
                # Determine edge type
                edge_type = edge_data.get("type") or edge_data.get("label") or "edge"
                if edge_types and edge_type not in edge_types:
                    edge_type = "edge"  # Default if not in schema

                # Extract edge attributes as fields
                if edge_fields is None:
                    # Extract all attributes except type/label
                    attrs = {
                        k: v for k, v in edge_data.items() if k not in ["type", "label"]
                    }
                else:
                    # Extract only specified fields
                    attrs = {k: v for k, v in edge_data.items() if k in edge_fields}

                # Create token for edge relationship
                structure_tag = StructureTag(
                    entity=str(source),
                    edge=edge_type,
                )

                token = Token(
                    value=str(target),
                    structure_tag=structure_tag,
                    position=position,
                )
                tokens.append(token)
                position += 1

                # Create tokens for edge attributes
                for attr_name, attr_value in attrs.items():
                    value_str = str(attr_value) if attr_value is not None else ""
                    if not value_str:
                        continue

                    structure_tag = StructureTag(
                        field=attr_name,
                        entity=str(source),
                        edge=edge_type,
                    )

                    token = Token(
                        value=value_str,
                        structure_tag=structure_tag,
                        position=position,
                    )
                    tokens.append(token)
                    position += 1

                # Update progress bar
                if edge_pbar:
                    edge_pbar.update(1)
        finally:
            if node_pbar:
                node_pbar.close()
            if edge_pbar:
                edge_pbar.close()

        return tokens

    def _get_node_order(self, graph: Any) -> list:
        """Get node order based on traversal method."""
        if self.traversal == "bfs":
            # Breadth-first: start from first node
            if len(graph.nodes) == 0:
                return []
            start_node = list(graph.nodes)[0]
            return list(nx.bfs_tree(graph, start_node).nodes())
        else:  # dfs
            # Depth-first: start from first node
            if len(graph.nodes) == 0:
                return []
            start_node = list(graph.nodes)[0]
            return list(nx.dfs_tree(graph, start_node).nodes())
