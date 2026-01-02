"""High-performance value tokenizer for converting values to token IDs."""

from saab_v3.data.constants import VALUE_SPECIAL_TOKENS
from saab_v3.data.structures import TokenizedSequence
from saab_v3.data.vocabulary import Vocabulary


class ValueTokenizer:
    """High-performance value tokenizer."""

    def __init__(self, vocab_size: int = 30000):
        """Initialize value tokenizer.

        Args:
            vocab_size: Maximum vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size
        self.vocab = Vocabulary(special_tokens=VALUE_SPECIAL_TOKENS)
        self._is_built = False

    def build_vocab(self, sequences: list[TokenizedSequence]):
        """Build vocabulary efficiently from all sequences.

        Args:
            sequences: List of TokenizedSequence objects to build vocabulary from

        Raises:
            ValueError: If vocabulary is already built
        """
        if self._is_built:
            raise ValueError("Vocabulary already built")

        # Extract all values in one pass (memory efficient)
        all_values = []
        for seq in sequences:
            all_values.extend(token.value for token in seq.tokens)

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

        Args:
            sequence: TokenizedSequence to encode

        Returns:
            Tuple of (TokenizedSequence with preserved structure, list of token IDs)

        Raises:
            ValueError: If vocabulary not built
        """
        if not self._is_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        # Batch encode all values
        values = [token.value for token in sequence.tokens]
        token_ids = self.vocab.encode_batch(values)

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
