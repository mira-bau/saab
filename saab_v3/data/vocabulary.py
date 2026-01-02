"""High-performance vocabulary for mapping tokens to indices."""

import json
from collections import Counter
from pathlib import Path
import heapq

from saab_v3.data.constants import UNK_TOKEN


class Vocabulary:
    """High-performance vocabulary for mapping tokens to indices."""

    def __init__(self, special_tokens: list[str] | None = None):
        """Initialize vocabulary with special tokens.

        Args:
            special_tokens: List of special tokens to add first (e.g., [PAD], [UNK])
        """
        self.special_tokens = special_tokens or []
        self.token_to_idx: dict[str, int] = {}
        self.idx_to_token: list[str] = []  # List for O(1) index access
        self._is_built = False
        self._build_special_tokens()

    def _build_special_tokens(self):
        """Initialize special tokens with fixed indices."""
        for token in self.special_tokens:
            if token not in self.token_to_idx:
                idx = len(self.idx_to_token)
                self.token_to_idx[token] = idx
                self.idx_to_token.append(token)

    def build_from_tokens(self, tokens: list[str], vocab_size: int | None = None):
        """Build vocabulary efficiently using Counter and heapq.

        Args:
            tokens: List of tokens to build vocabulary from
            vocab_size: Maximum vocabulary size (including special tokens).
                       If None, include all tokens.

        Raises:
            ValueError: If vocabulary is already built
        """
        if self._is_built:
            raise ValueError("Vocabulary already built")

        # Count frequencies (O(n))
        counter = Counter(tokens)

        # Get tokens excluding special tokens
        special_set = set(self.special_tokens)
        filtered = [
            (freq, token) for token, freq in counter.items() if token not in special_set
        ]

        if vocab_size:
            # Calculate how many regular tokens we can add
            num_special = len(self.special_tokens)
            max_regular = max(0, vocab_size - num_special)
            if max_regular > 0:
                # Use heapq.nlargest for efficient top-k (O(n log k))
                top_tokens = heapq.nlargest(max_regular, filtered)
            else:
                top_tokens = []
        else:
            # Sort by frequency (descending)
            top_tokens = sorted(filtered, reverse=True)

        # Add tokens to vocabulary
        for freq, token in top_tokens:
            if token not in self.token_to_idx:
                idx = len(self.idx_to_token)
                self.token_to_idx[token] = idx
                self.idx_to_token.append(token)

        self._is_built = True

    def encode(self, token: str) -> int:
        """Fast O(1) lookup for token to index.

        Args:
            token: Token string to encode

        Returns:
            Token index, or UNK index if token not found
        """
        if not self._is_built:
            raise ValueError("Vocabulary not built. Call build_from_tokens() first.")
        unk_idx = self.token_to_idx.get(UNK_TOKEN, 0)
        return self.token_to_idx.get(token, unk_idx)

    def encode_batch(self, tokens: list[str]) -> list[int]:
        """Batch encoding for better performance.

        Args:
            tokens: List of token strings to encode

        Returns:
            List of token indices
        """
        if not self._is_built:
            raise ValueError("Vocabulary not built. Call build_from_tokens() first.")
        unk_idx = self.token_to_idx.get(UNK_TOKEN, 0)
        return [self.token_to_idx.get(token, unk_idx) for token in tokens]

    def decode(self, idx: int) -> str:
        """Fast O(1) list access for index to token.

        Args:
            idx: Token index to decode

        Returns:
            Token string, or UNK token if index out of range
        """
        if 0 <= idx < len(self.idx_to_token):
            return self.idx_to_token[idx]
        return UNK_TOKEN

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.idx_to_token)

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.token_to_idx

    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON file.

        Args:
            path: Path to save vocabulary JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "special_tokens": self.special_tokens,
            "token_to_idx": self.token_to_idx,
            "idx_to_token": self.idx_to_token,
            "_is_built": self._is_built,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load vocabulary from JSON file.

        Args:
            path: Path to vocabulary JSON file

        Returns:
            Vocabulary instance with loaded data
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        vocab = cls(special_tokens=data["special_tokens"])
        vocab.token_to_idx = data["token_to_idx"]
        vocab.idx_to_token = data["idx_to_token"]
        vocab._is_built = data["_is_built"]

        return vocab
