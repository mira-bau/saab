"""Preprocessor orchestrator for the preprocessing pipeline."""

from pathlib import Path
from typing import Any

from saab_v3.data.extractors import (
    GraphExtractor,
    JSONExtractor,
    StructuralExtractor,
    TableExtractor,
)
from saab_v3.data.structures import EncodedTag, Token, TokenizedSequence
from saab_v3.data.tag_encoder import TagEncoder
from saab_v3.data.tokenizer import ValueTokenizer
from saab_v3.data.vocabulary import Vocabulary

from saab_v3.training.config import PreprocessingConfig


class Preprocessor:
    """High-level orchestrator that coordinates the preprocessing pipeline."""

    def __init__(self, config: PreprocessingConfig):
        """Initialize with configuration.

        Args:
            config: PreprocessingConfig instance
        """
        self.config = config
        self.tokenizer = ValueTokenizer(vocab_size=config.vocab_size)
        self.tag_encoder = TagEncoder()

        # Initialize extractors
        self.extractors: list[StructuralExtractor] = [
            TableExtractor(),
            JSONExtractor(),
            GraphExtractor(),
        ]

        # Selected extractor (set during fit or transform)
        self._selected_extractor: StructuralExtractor | None = None
        self._is_fitted = False

    def _group_tokens_into_sequences(
        self, tokens: list[Token]
    ) -> list[TokenizedSequence]:
        """Group tokens into TokenizedSequence objects.

        For tables: Group by entity (each row becomes a sequence)
        For other formats: Group all tokens into a single sequence (or by entity if available)

        Args:
            tokens: List of Token objects

        Returns:
            List of TokenizedSequence objects
        """
        if not tokens:
            return []

        # Check if tokens have entity tags (tables typically do)
        has_entities = any(token.structure_tag.entity for token in tokens)

        if has_entities:
            # Group by entity
            entity_groups: dict[str, list[Token]] = {}
            for token in tokens:
                entity = token.structure_tag.entity or "default"
                if entity not in entity_groups:
                    entity_groups[entity] = []
                entity_groups[entity].append(token)

            sequences = []
            for entity, entity_tokens in entity_groups.items():
                # Reset positions to be sequential within each sequence
                for i, token in enumerate(entity_tokens):
                    token.position = i
                sequences.append(
                    TokenizedSequence(tokens=entity_tokens, sequence_id=entity)
                )
        else:
            # No entities: create single sequence with all tokens
            # Reset positions to be sequential
            for i, token in enumerate(tokens):
                token.position = i
            sequences = [TokenizedSequence(tokens=tokens)]

        return sequences

    def _auto_detect_extractor(self, data: Any) -> StructuralExtractor:
        """Auto-detect format and select appropriate extractor.

        Args:
            data: Input data to detect format for

        Returns:
            Selected StructuralExtractor instance

        Raises:
            ValueError: If no extractor can handle the data
        """
        # If extractor_type is specified, use that
        if self.config.extractor_type:
            extractor_map = {
                "table": TableExtractor(),
                "json": JSONExtractor(),
                "graph": GraphExtractor(),
            }
            extractor = extractor_map.get(self.config.extractor_type)
            if extractor is None:
                raise ValueError(
                    f"Invalid extractor_type: {self.config.extractor_type}"
                )
            if not extractor.can_handle(data):
                raise ValueError(
                    f"Extractor '{self.config.extractor_type}' cannot handle the provided data"
                )
            return extractor

        # Try each extractor in order
        for extractor in self.extractors:
            if extractor.can_handle(data):
                return extractor

        raise ValueError(
            "No extractor can handle the provided data. "
            "Supported formats: pandas DataFrame, CSV, Excel, JSON (dict/string), NetworkX Graph"
        )

    def fit(self, train_data: Any, schema: dict | None = None):
        """Build vocabularies from training data only.

        Args:
            train_data: Training data (DataFrame, dict, Graph, file path, etc.)
            schema: Optional schema dict for extractors

        Raises:
            ValueError: If fit() called multiple times
        """
        if self._is_fitted:
            raise ValueError(
                "Preprocessor already fitted. Create a new instance to fit again."
            )

        # Use provided schema or config schema
        schema = schema or self.config.extractor_schema

        # Auto-detect format and select extractor
        self._selected_extractor = self._auto_detect_extractor(train_data)

        # Extract tokens
        tokens = self._selected_extractor.extract(train_data, schema=schema)

        if not tokens:
            raise ValueError("No tokens extracted from training data")

        # Group tokens into sequences (by entity for tables, or all tokens for other formats)
        sequences = self._group_tokens_into_sequences(tokens)

        if not sequences:
            raise ValueError("No sequences created from extracted tokens")

        # Build value vocabulary
        self.tokenizer.build_vocab(sequences)

        # Build tag vocabularies
        self.tag_encoder.build_vocabs(sequences)

        self._is_fitted = True

    def transform(
        self, data: Any, schema: dict | None = None
    ) -> list[TokenizedSequence]:
        """Extract and tokenize data (no encoding yet).

        Args:
            data: Input data (any format)
            schema: Optional schema dict

        Returns:
            List of TokenizedSequence objects
        """
        # Use provided schema or config schema
        schema = schema or self.config.extractor_schema

        # Select extractor (use fitted one if it can handle the data, otherwise auto-detect)
        if self._selected_extractor is not None and self._selected_extractor.can_handle(
            data
        ):
            extractor = self._selected_extractor
        else:
            extractor = self._auto_detect_extractor(data)
            # Store for future use
            if self._is_fitted:
                self._selected_extractor = extractor

        # Extract tokens
        tokens = extractor.extract(data, schema=schema)

        # Group tokens into sequences
        sequences = self._group_tokens_into_sequences(tokens)

        return sequences

    def encode(
        self, sequences: list[TokenizedSequence]
    ) -> list[tuple[TokenizedSequence, list[int], list[EncodedTag]]]:
        """Encode sequences using fitted vocabularies.

        Args:
            sequences: List of TokenizedSequence objects

        Returns:
            List of (TokenizedSequence, token_ids, encoded_tags) tuples

        Raises:
            ValueError: If not fitted yet
        """
        if not self._is_fitted:
            raise ValueError(
                "Preprocessor not fitted. Call fit() first to build vocabularies."
            )

        encoded = []
        for seq in sequences:
            # Encode values
            _, token_ids = self.tokenizer.encode_sequence(seq)

            # Encode tags
            _, encoded_tags = self.tag_encoder.encode_sequence(seq)

            # Return tuple matching Batcher expectations
            encoded.append((seq, token_ids, encoded_tags))

        return encoded

    def save_artifacts(self, dataset_name: str, base_path: Path | None = None):
        """Save vocabularies and config to artifacts directory.

        Args:
            dataset_name: Name of dataset (creates dataset/artifacts/dataset_name/)
            base_path: Optional override for base directory (default: saab_v3/dataset/)
        """
        if not self._is_fitted:
            raise ValueError("Cannot save artifacts: Preprocessor not fitted yet.")

        # Determine artifacts path
        if base_path is None:
            if self.config.data_dir is not None:
                base_path = self.config.data_dir
            else:
                # Default: dataset/artifacts relative to saab_v3 package
                base_path = Path(__file__).parent.parent.parent / "dataset"

        artifacts_dir = base_path / "artifacts" / dataset_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save vocabularies
        vocab_dir = artifacts_dir / "vocabularies"
        vocab_dir.mkdir(parents=True, exist_ok=True)

        # Save value vocabulary (will implement save method in Vocabulary)
        value_vocab_path = vocab_dir / "value_vocab.json"
        self.tokenizer.vocab.save(str(value_vocab_path))

        # Save tag vocabularies
        tag_vocabs_dir = vocab_dir / "tag_vocabs"
        tag_vocabs_dir.mkdir(parents=True, exist_ok=True)

        for tag_type, vocab in self.tag_encoder.tag_vocabs.items():
            tag_vocab_path = tag_vocabs_dir / f"{tag_type}.json"
            vocab.save(str(tag_vocab_path))

        # Save config
        config_path = artifacts_dir / "config.json"
        with open(config_path, "w") as f:
            f.write(self.config.model_dump_json())

    @classmethod
    def load_artifacts(
        cls, dataset_name: str, base_path: Path | None = None
    ) -> "Preprocessor":
        """Load vocabularies and config from artifacts directory.

        Args:
            dataset_name: Name of dataset
            base_path: Optional override for base directory (default: saab_v3/dataset/)

        Returns:
            Preprocessor instance with loaded vocabularies

        Raises:
            FileNotFoundError: If artifacts don't exist
        """
        # Determine artifacts path
        if base_path is None:
            # Default: dataset/artifacts relative to saab_v3 package
            default_path = Path(__file__).parent.parent.parent / "dataset"
            artifacts_dir = default_path / "artifacts" / dataset_name
        else:
            artifacts_dir = base_path / "artifacts" / dataset_name

        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

        # Load config
        config_path = artifacts_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_json = f.read()

        config = PreprocessingConfig.model_validate_json(config_json)

        # Create Preprocessor instance
        preprocessor = cls(config)

        # Load value vocabulary
        value_vocab_path = artifacts_dir / "vocabularies" / "value_vocab.json"
        if not value_vocab_path.exists():
            raise FileNotFoundError(f"Value vocabulary not found: {value_vocab_path}")

        preprocessor.tokenizer.vocab = Vocabulary.load(str(value_vocab_path))
        preprocessor.tokenizer._is_built = True

        # Load tag vocabularies
        tag_vocabs_dir = artifacts_dir / "vocabularies" / "tag_vocabs"
        if not tag_vocabs_dir.exists():
            raise FileNotFoundError(
                f"Tag vocabularies directory not found: {tag_vocabs_dir}"
            )

        for tag_type in ["field", "entity", "time", "edge", "role", "token_type"]:
            tag_vocab_path = tag_vocabs_dir / f"{tag_type}.json"
            if not tag_vocab_path.exists():
                raise FileNotFoundError(f"Tag vocabulary not found: {tag_vocab_path}")

            preprocessor.tag_encoder.tag_vocabs[tag_type] = Vocabulary.load(
                str(tag_vocab_path)
            )

        preprocessor.tag_encoder._is_built = True
        preprocessor._is_fitted = True

        return preprocessor
