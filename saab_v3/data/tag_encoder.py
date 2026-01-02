"""High-performance tag encoder with preserved symbols for SAAB."""

from saab_v3.data.constants import TAG_SPECIAL_TOKENS, NONE_TOKEN
from saab_v3.data.structures import TokenizedSequence, StructureTag, EncodedTag
from saab_v3.data.vocabulary import Vocabulary


class TagEncoder:
    """High-performance tag encoder with preserved symbols for SAAB."""

    def __init__(self):
        """Initialize tag encoder with separate vocabularies per tag type."""
        self.tag_vocabs = {
            "field": Vocabulary(special_tokens=TAG_SPECIAL_TOKENS),
            "entity": Vocabulary(special_tokens=TAG_SPECIAL_TOKENS),
            "time": Vocabulary(special_tokens=TAG_SPECIAL_TOKENS),
            "edge": Vocabulary(special_tokens=TAG_SPECIAL_TOKENS),
            "role": Vocabulary(special_tokens=TAG_SPECIAL_TOKENS),
            "token_type": Vocabulary(special_tokens=TAG_SPECIAL_TOKENS),
        }
        self._is_built = False

    def build_vocabs(self, sequences: list[TokenizedSequence]):
        """Build all vocabularies efficiently in one pass.

        Args:
            sequences: List of TokenizedSequence objects to build vocabularies from

        Raises:
            ValueError: If vocabularies already built
        """
        if self._is_built:
            raise ValueError("Vocabularies already built")

        # Extract tags by type in single pass
        tag_collections = {
            "field": [],
            "entity": [],
            "time": [],
            "edge": [],
            "role": [],
            "token_type": [],
        }

        for seq in sequences:
            for token in seq.tokens:
                tag = token.structure_tag
                if tag.field:
                    tag_collections["field"].append(tag.field)
                if tag.entity:
                    tag_collections["entity"].append(tag.entity)
                if tag.time:
                    tag_collections["time"].append(tag.time)
                if tag.edge:
                    tag_collections["edge"].append(tag.edge)
                if tag.role:
                    tag_collections["role"].append(tag.role)
                if tag.token_type:
                    tag_collections["token_type"].append(tag.token_type)

        # Build vocabularies (can be parallelized if needed)
        # Build all vocabularies, even if empty (will only have special tokens)
        for tag_type, tokens in tag_collections.items():
            self.tag_vocabs[tag_type].build_from_tokens(tokens if tokens else [])

        self._is_built = True

    def encode(self, tag: StructureTag) -> EncodedTag:
        """Encode tag efficiently, preserving original for SAAB.

        Args:
            tag: StructureTag to encode

        Returns:
            EncodedTag with indices and original tag

        Raises:
            ValueError: If vocabularies not built
        """
        if not self._is_built:
            raise ValueError("Vocabularies not built. Call build_vocabs() first.")

        return EncodedTag(
            field_idx=(
                self.tag_vocabs["field"].encode(tag.field)
                if tag.field
                else self.tag_vocabs["field"].encode(NONE_TOKEN)
            ),
            entity_idx=(
                self.tag_vocabs["entity"].encode(tag.entity)
                if tag.entity
                else self.tag_vocabs["entity"].encode(NONE_TOKEN)
            ),
            time_idx=(
                self.tag_vocabs["time"].encode(tag.time)
                if tag.time
                else self.tag_vocabs["time"].encode(NONE_TOKEN)
            ),
            edge_idx=(
                self.tag_vocabs["edge"].encode(tag.edge)
                if tag.edge
                else self.tag_vocabs["edge"].encode(NONE_TOKEN)
            ),
            role_idx=(
                self.tag_vocabs["role"].encode(tag.role)
                if tag.role
                else self.tag_vocabs["role"].encode(NONE_TOKEN)
            ),
            token_type_idx=(
                self.tag_vocabs["token_type"].encode(tag.token_type)
                if tag.token_type
                else self.tag_vocabs["token_type"].encode(NONE_TOKEN)
            ),
            original_tag=tag,  # Preserve for SAAB bias computation
        )

    def encode_sequence(
        self, sequence: TokenizedSequence
    ) -> tuple[TokenizedSequence, list[EncodedTag]]:
        """Encode all tags in sequence efficiently.

        Args:
            sequence: TokenizedSequence to encode

        Returns:
            Tuple of (TokenizedSequence with preserved structure, list of EncodedTag objects)

        Raises:
            ValueError: If vocabularies not built
        """
        if not self._is_built:
            raise ValueError("Vocabularies not built. Call build_vocabs() first.")

        # Batch encode all tags
        encoded_tags = [self.encode(token.structure_tag) for token in sequence.tokens]

        # Return original sequence (structure preserved) and encoded tags separately
        # Encoded tags can be used later for creating Batch tensors
        return sequence, encoded_tags
