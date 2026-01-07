"""High-performance value tokenizer for converting values to token IDs."""

from typing import Any, Protocol

from saab_v3.data.constants import VALUE_SPECIAL_TOKENS
from saab_v3.data.structures import TokenizedSequence
from saab_v3.data.vocabulary import Vocabulary


class TextIdTokenizer(Protocol):
    """Protocol for text tokenizers that convert text to token IDs."""

    def encode(self, text: str) -> list[int]:
        """Encode text string to list of token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        ...

    @property
    def unk_id(self) -> int:
        """Return the UNK token ID."""
        ...


try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False
    tokenizers = None  # type: ignore


class TokenizersBPETokenizer:
    """BPE tokenizer wrapper using HuggingFace tokenizers library."""

    def __init__(self, tokenizer: Any):
        """Initialize BPE tokenizer wrapper.

        Args:
            tokenizer: Trained tokenizers.Tokenizer instance
        """
        if not HAS_TOKENIZERS:
            raise ImportError(
                "tokenizers library is required. Install with: poetry install --extras tokenizers"
            )
        self.tok = tokenizer
        # Extract UNK ID from tokenizer
        try:
            self._unk_id = tokenizer.token_to_id("[UNK]")
        except (AttributeError, KeyError):
            # Fallback: try to find UNK token
            try:
                self._unk_id = tokenizer.token_to_id("<unk>")
            except (AttributeError, KeyError):
                # Default to 0 if not found
                self._unk_id = 0

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        enc = self.tok.encode(text)
        return enc.ids

    @property
    def unk_id(self) -> int:
        """Return UNK token ID."""
        return self._unk_id


class ValueTokenizer:
    """High-performance value tokenizer."""

    def __init__(
        self,
        vocab_size: int = 30000,
        text_tokenizer: TextIdTokenizer | None = None,
    ):
        """Initialize value tokenizer.

        Args:
            vocab_size: Maximum vocabulary size (including special tokens)
            text_tokenizer: Optional text tokenizer for subword tokenization
        """
        self.vocab_size = vocab_size
        self.vocab = Vocabulary(special_tokens=VALUE_SPECIAL_TOKENS)
        self.text_tokenizer = text_tokenizer
        self._is_built = False

    def build_vocab(self, sequences: list[TokenizedSequence]):
        """Build vocabulary efficiently from all sequences.

        If text_tokenizer is enabled, skip text tokens and only build vocab
        from non-text tokens (numbers, enums, field markers, etc.).

        Args:
            sequences: List of TokenizedSequence objects to build vocabulary from

        Raises:
            ValueError: If vocabulary is already built
        """
        if self._is_built:
            raise ValueError("Vocabulary already built")

        # Extract all values in one pass (memory efficient)
        # Skip text tokens if text_tokenizer is enabled
        all_values = []
        for seq in sequences:
            for token in seq.tokens:
                # If text_tokenizer is enabled, skip tokens with token_type="text"
                if self.text_tokenizer is not None:
                    if token.structure_tag.token_type == "text":
                        continue  # Skip text tokens - they use subword tokenizer
                all_values.append(token.value)

        # Build vocabulary (uses Counter + heapq internally)
        self.vocab.build_from_tokens(all_values, vocab_size=self.vocab_size)
        self._is_built = True

    def encode(self, value: str) -> int:
        """Encode single value to token ID.

        Args:
            value: Value string to encode

        Returns:
            Token ID

        Raises:
            ValueError: If vocabulary not built
        """
        if not self._is_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        return self.vocab.encode(value)

    def encode_sequence(
        self, sequence: TokenizedSequence
    ) -> tuple[TokenizedSequence, list[int]]:
        """Encode sequence efficiently using batch encoding.

        For text tokens (token_type="text"), uses text_tokenizer if available,
        which may return multiple IDs per token. For non-text tokens, uses
        categorical vocab (single ID per token).

        Args:
            sequence: TokenizedSequence to encode

        Returns:
            Tuple of (TokenizedSequence with preserved structure, list of token IDs)
            Note: token_ids may have more elements than sequence.tokens if text
            tokens are expanded to multiple IDs.

        Raises:
            ValueError: If vocabulary not built
        """
        if not self._is_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        token_ids = []
        
        for token in sequence.tokens:
            # Check if this is a text token and text_tokenizer is available
            if (
                self.text_tokenizer is not None
                and token.structure_tag.token_type == "text"
            ):
                # Use subword tokenizer - may return multiple IDs
                ids = self.text_tokenizer.encode(token.value)
                token_ids.extend(ids)
            else:
                # Use categorical vocab - single ID per token
                token_id = self.vocab.encode(token.value)
                token_ids.append(token_id)

        # Return original sequence (structure preserved) and token IDs separately
        # Token IDs can be used later for creating Batch tensors
        return sequence, token_ids

    def decode(self, token_id: int) -> str:
        """Decode token ID back to value string.

        Args:
            token_id: Token ID to decode

        Returns:
            Original value string (or UNK if ID not found)
        """
        return self.vocab.decode(token_id)
