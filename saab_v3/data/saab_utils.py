"""Utilities for SAAB (Structure-Aware Attention Bias) computation."""

from saab_v3.data.structures import StructureTag, EncodedTag, TokenizedSequence
from saab_v3.data.constants import PAD_TAG_FIELD


def extract_original_tags(
    sequences: list[tuple[TokenizedSequence, list[int], list[EncodedTag]]],
) -> list[list[StructureTag]]:
    """Extract original tags from encoded sequences for SAAB bias computation.

    Args:
        sequences: List of (TokenizedSequence, token_ids, encoded_tags) tuples

    Returns:
        List of lists of StructureTag objects, one list per sequence
    """
    return [
        [tag.original_tag for tag in encoded_tags] for _, _, encoded_tags in sequences
    ]


def is_pad_tag(tag: StructureTag) -> bool:
    """Check if a tag is a padding tag.

    Args:
        tag: StructureTag to check

    Returns:
        True if tag is a padding tag (field == PAD_TAG_FIELD)
    """
    return tag.field == PAD_TAG_FIELD


def compute_structural_relationship(
    tag_i: StructureTag, tag_j: StructureTag
) -> dict[str, bool | str | None]:
    """Compute structural relationships between two tags for SAAB bias computation.

    Args:
        tag_i: First StructureTag
        tag_j: Second StructureTag

    Returns:
        Dictionary with relationship flags:
        - same_field: bool - True if both tags have the same field
        - same_entity: bool - True if both tags have the same entity
        - same_time: bool - True if both tags have the same time
        - has_edge: bool - True if either tag has an edge
        - edge_type: str | None - Edge type if present (from tag_i or tag_j)
        - same_role: bool - True if both tags have the same role
        - same_token_type: bool - True if both tags have the same token_type
    """
    # Skip padding tags
    if is_pad_tag(tag_i) or is_pad_tag(tag_j):
        return {
            "same_field": False,
            "same_entity": False,
            "same_time": False,
            "has_edge": False,
            "edge_type": None,
            "same_role": False,
            "same_token_type": False,
        }

    same_field = (
        tag_i.field is not None
        and tag_j.field is not None
        and tag_i.field == tag_j.field
    )

    same_entity = (
        tag_i.entity is not None
        and tag_j.entity is not None
        and tag_i.entity == tag_j.entity
    )

    same_time = (
        tag_i.time is not None and tag_j.time is not None and tag_i.time == tag_j.time
    )

    has_edge = tag_i.edge is not None or tag_j.edge is not None
    edge_type = tag_i.edge if tag_i.edge is not None else tag_j.edge

    same_role = (
        tag_i.role is not None and tag_j.role is not None and tag_i.role == tag_j.role
    )

    same_token_type = (
        tag_i.token_type is not None
        and tag_j.token_type is not None
        and tag_i.token_type == tag_j.token_type
    )

    return {
        "same_field": same_field,
        "same_entity": same_entity,
        "same_time": same_time,
        "has_edge": has_edge,
        "edge_type": edge_type,
        "same_role": same_role,
        "same_token_type": same_token_type,
    }


def same_field(tag_i: StructureTag, tag_j: StructureTag) -> bool:
    """Check if two tags have the same field.

    Args:
        tag_i: First StructureTag
        tag_j: Second StructureTag

    Returns:
        True if both tags have the same non-None field
    """
    return (
        tag_i.field is not None
        and tag_j.field is not None
        and tag_i.field == tag_j.field
        and not is_pad_tag(tag_i)
        and not is_pad_tag(tag_j)
    )


def same_entity(tag_i: StructureTag, tag_j: StructureTag) -> bool:
    """Check if two tags have the same entity.

    Args:
        tag_i: First StructureTag
        tag_j: Second StructureTag

    Returns:
        True if both tags have the same non-None entity
    """
    return (
        tag_i.entity is not None
        and tag_j.entity is not None
        and tag_i.entity == tag_j.entity
        and not is_pad_tag(tag_i)
        and not is_pad_tag(tag_j)
    )


def has_edge(tag_i: StructureTag, tag_j: StructureTag) -> bool:
    """Check if either tag has an edge.

    Args:
        tag_i: First StructureTag
        tag_j: Second StructureTag

    Returns:
        True if either tag has a non-None edge
    """
    if is_pad_tag(tag_i) or is_pad_tag(tag_j):
        return False
    return tag_i.edge is not None or tag_j.edge is not None
