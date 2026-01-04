"""Batching logic for converting encoded sequences to Batch objects."""

import torch
from typing import Any

import numpy as np

from saab_v3.data.constants import PAD_IDX
from saab_v3.data.structures import Batch, EncodedTag, TokenizedSequence


class Batcher:
    """Batches encoded sequences with dynamic padding."""

    def __init__(
        self,
        max_seq_len: int = 512,
        pad_token_id: int = PAD_IDX,
        device: torch.device | None = None,
    ):
        """Initialize batcher.

        Args:
            max_seq_len: Maximum sequence length (truncate if longer)
            pad_token_id: Token ID for padding (default 0 = [PAD])
            device: Device to place tensors on (defaults to CPU if None)
        """
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.device = device if device is not None else torch.device("cpu")

    def batch(
        self,
        sequences: list[
            tuple[TokenizedSequence, list[int], list[EncodedTag], Any | None]
            | tuple[
                tuple[TokenizedSequence, list[int], list[EncodedTag]],
                tuple[TokenizedSequence, list[int], list[EncodedTag]],
                Any | None,
            ]
        ],
        task_type: str | None = None,
    ) -> Batch:
        """Create batch from encoded sequences.

        Args:
            sequences: List of sequence tuples. For single-sequence tasks:
                (TokenizedSequence, token_ids, encoded_tags, label) tuples.
                For ranking tasks:
                ((seq_a_data), (seq_b_data), label) tuples where each data is
                (TokenizedSequence, token_ids, encoded_tags).
            task_type: Task type ("ranking" for ranking pairs, None for single sequences)

        Returns:
            Batch object with all tensors and labels (if provided)

        Raises:
            ValueError: If sequences list is empty
        """
        if not sequences:
            raise ValueError("Cannot create batch from empty sequences list")

        if task_type == "ranking":
            return self._batch_ranking_pairs(sequences)
        else:
            return self._batch_single_sequences(sequences)

    def _batch_single_sequences(
        self,
        sequences: list[
            tuple[TokenizedSequence, list[int], list[EncodedTag], Any | None]
        ],
    ) -> Batch:
        """Create batch from single sequences (existing logic).

        Args:
            sequences: List of (TokenizedSequence, token_ids, encoded_tags, label) tuples
                label can be None if not present

        Returns:
            Batch object with all tensors and labels (if provided)
        """
        # 1. Extract labels (if present)
        labels = [label for _, _, _, label in sequences]
        has_labels = any(label is not None for label in labels)

        # 2. Truncate sequences if needed
        truncated = [
            self._truncate_sequence(seq, token_ids, encoded_tags)
            for seq, token_ids, encoded_tags, _ in sequences
        ]

        # 3. Find max length in batch
        max_len = max(len(token_ids) for _, token_ids, _ in truncated)
        max_len = min(max_len, self.max_seq_len)  # Don't exceed max_seq_len

        # 4. Pad all sequences to max length
        padded_token_ids = []
        padded_encoded_tags = []
        sequence_lengths = []
        sequence_ids = []

        for seq, token_ids, encoded_tags in truncated:
            seq_len = len(token_ids)
            sequence_lengths.append(seq_len)
            sequence_ids.append(seq.sequence_id)

            # Pad token IDs
            padded_token_ids.append(
                self._pad_list(token_ids, max_len, self.pad_token_id)
            )

            # Pad encoded tags
            padded_encoded_tags.append(self._pad_encoded_tags(encoded_tags, max_len))

        # 5. Extract tag indices
        tag_indices = self._extract_tag_indices(padded_encoded_tags)

        # 6. Create attention masks
        attention_masks = self._create_attention_masks(sequence_lengths, max_len)

        # 7. Convert to tensors
        tensors = self._to_tensors(
            padded_token_ids,
            attention_masks,
            tag_indices,
            self.device,
        )

        # 8. Batch labels if present
        labels_tensor = None
        if has_labels:
            labels_tensor = self._batch_labels(labels, max_len, self.device)

        # 9. Create Batch object
        return Batch(
            token_ids=tensors["token_ids"],
            attention_mask=tensors["attention_mask"],
            field_ids=tensors["field_ids"],
            entity_ids=tensors["entity_ids"],
            time_ids=tensors["time_ids"],
            edge_ids=tensors["edge_ids"],
            role_ids=tensors["role_ids"],
            token_type_ids=tensors["token_type_ids"],
            sequence_lengths=sequence_lengths,
            sequence_ids=sequence_ids if any(sequence_ids) else None,
            labels=labels_tensor,
        )

    def _batch_ranking_pairs(
        self,
        sequences: list[
            tuple[
                tuple[TokenizedSequence, list[int], list[EncodedTag]],
                tuple[TokenizedSequence, list[int], list[EncodedTag]],
                Any | None,
            ]
        ],
    ) -> Batch:
        """Create batch from ranking pairs.

        Args:
            sequences: List of ((seq_a_data), (seq_b_data), label) tuples
                where each data is (TokenizedSequence, token_ids, encoded_tags)

        Returns:
            Batch object with all tensors including _b fields for batch B
        """
        # Extract pairs
        seqs_a = [seq_a_data for seq_a_data, _, _ in sequences]
        seqs_b = [seq_b_data for _, seq_b_data, _ in sequences]
        labels = [label for _, _, label in sequences]

        # Batch sequence A (using single sequence batching)
        batch_a = self._batch_single_sequences(
            [
                (seq, token_ids, encoded_tags, None)
                for seq, token_ids, encoded_tags in seqs_a
            ]
        )

        # Batch sequence B (using single sequence batching)
        batch_b = self._batch_single_sequences(
            [
                (seq, token_ids, encoded_tags, None)
                for seq, token_ids, encoded_tags in seqs_b
            ]
        )

        # Batch labels (ranking labels are simple: [batch])
        labels_tensor = None
        if any(label is not None for label in labels):
            # Convert labels to tensor (1 = a better, -1 = b better, or binary 0/1)
            label_values = []
            for label in labels:
                if label is None:
                    label_values.append(0)  # Default to 0 if missing
                else:
                    label_values.append(int(label))
            labels_tensor = torch.tensor(
                label_values, dtype=torch.long, device=self.device
            )

        # Combine into ranking batch
        return Batch(
            # Batch A fields
            token_ids=batch_a.token_ids,
            attention_mask=batch_a.attention_mask,
            field_ids=batch_a.field_ids,
            entity_ids=batch_a.entity_ids,
            time_ids=batch_a.time_ids,
            edge_ids=batch_a.edge_ids,
            role_ids=batch_a.role_ids,
            token_type_ids=batch_a.token_type_ids,
            sequence_lengths=batch_a.sequence_lengths,
            sequence_ids=batch_a.sequence_ids,
            labels=labels_tensor,
            # Batch B fields
            token_ids_b=batch_b.token_ids,
            attention_mask_b=batch_b.attention_mask,
            field_ids_b=batch_b.field_ids,
            entity_ids_b=batch_b.entity_ids,
            time_ids_b=batch_b.time_ids,
            edge_ids_b=batch_b.edge_ids,
            role_ids_b=batch_b.role_ids,
            token_type_ids_b=batch_b.token_type_ids,
            sequence_lengths_b=batch_b.sequence_lengths,
        )

    def _truncate_sequence(
        self,
        sequence: TokenizedSequence,
        token_ids: list[int],
        encoded_tags: list[EncodedTag],
    ) -> tuple[TokenizedSequence, list[int], list[EncodedTag]]:
        """Truncate sequence if longer than max_seq_len.

        Args:
            sequence: Original TokenizedSequence
            token_ids: List of token IDs
            encoded_tags: List of EncodedTag objects

        Returns:
            Truncated (sequence, token_ids, encoded_tags) tuple
        """
        if len(token_ids) <= self.max_seq_len:
            return (sequence, token_ids, encoded_tags)

        # Truncate to first max_seq_len tokens
        truncated_token_ids = token_ids[: self.max_seq_len]
        truncated_encoded_tags = encoded_tags[: self.max_seq_len]
        truncated_tokens = sequence.tokens[: self.max_seq_len]

        truncated_sequence = TokenizedSequence(
            tokens=truncated_tokens, sequence_id=sequence.sequence_id
        )

        return (truncated_sequence, truncated_token_ids, truncated_encoded_tags)

    def _pad_list(self, items: list[Any], target_len: int, pad_value: Any) -> list[Any]:
        """Pad list to target length.

        Args:
            items: List to pad
            target_len: Target length
            pad_value: Value to use for padding

        Returns:
            Padded list
        """
        if len(items) >= target_len:
            return items[:target_len]
        return items + [pad_value] * (target_len - len(items))

    def _pad_encoded_tags(
        self, encoded_tags: list[EncodedTag], target_len: int
    ) -> list[EncodedTag]:
        """Pad encoded tags to target length using PAD indices.

        Args:
            encoded_tags: List of EncodedTag objects
            target_len: Target length

        Returns:
            Padded list of EncodedTag objects
        """
        if len(encoded_tags) >= target_len:
            return encoded_tags[:target_len]

        # Create padding EncodedTag with PAD indices for all tag types
        pad_encoded_tag = EncodedTag(
            field_idx=PAD_IDX,
            entity_idx=PAD_IDX,
            time_idx=PAD_IDX,
            edge_idx=PAD_IDX,
            role_idx=PAD_IDX,
            token_type_idx=PAD_IDX,
        )

        return encoded_tags + [pad_encoded_tag] * (target_len - len(encoded_tags))

    def _extract_tag_indices(
        self, padded_encoded_tags: list[list[EncodedTag]]
    ) -> dict[str, list[list[int]]]:
        """Extract tag indices from EncodedTag objects.

        Args:
            padded_encoded_tags: List of lists of EncodedTag objects (one per sequence)

        Returns:
            Dictionary mapping tag type to list of index lists
        """
        field_indices = []
        entity_indices = []
        time_indices = []
        edge_indices = []
        role_indices = []
        token_type_indices = []

        for seq_tags in padded_encoded_tags:
            field_indices.append([tag.field_idx for tag in seq_tags])
            entity_indices.append([tag.entity_idx for tag in seq_tags])
            time_indices.append([tag.time_idx for tag in seq_tags])
            edge_indices.append([tag.edge_idx for tag in seq_tags])
            role_indices.append([tag.role_idx for tag in seq_tags])
            token_type_indices.append([tag.token_type_idx for tag in seq_tags])

        return {
            "field": field_indices,
            "entity": entity_indices,
            "time": time_indices,
            "edge": edge_indices,
            "role": role_indices,
            "token_type": token_type_indices,
        }

    def _create_attention_masks(
        self, sequence_lengths: list[int], max_len: int
    ) -> list[list[int]]:
        """Create attention masks (1 for valid, 0 for pad).

        Args:
            sequence_lengths: List of actual sequence lengths
            max_len: Maximum sequence length in batch

        Returns:
            List of attention mask lists
        """
        masks = []
        for seq_len in sequence_lengths:
            mask = [1] * seq_len + [0] * (max_len - seq_len)
            masks.append(mask)
        return masks

    def _to_tensors(
        self,
        token_ids: list[list[int]],
        attention_masks: list[list[int]],
        tag_indices: dict[str, list[list[int]]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Convert lists to PyTorch tensors.

        Args:
            token_ids: List of token ID lists
            attention_masks: List of attention mask lists
            tag_indices: Dictionary of tag index lists
            device: Device to place tensors on

        Returns:
            Dictionary of tensors
        """
        # Convert to tensors
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        attention_mask_tensor = torch.tensor(
            attention_masks, dtype=torch.long, device=device
        )
        field_ids_tensor = torch.tensor(
            tag_indices["field"], dtype=torch.long, device=device
        )
        entity_ids_tensor = torch.tensor(
            tag_indices["entity"], dtype=torch.long, device=device
        )
        time_ids_tensor = torch.tensor(
            tag_indices["time"], dtype=torch.long, device=device
        )
        token_type_ids_tensor = torch.tensor(
            tag_indices["token_type"], dtype=torch.long, device=device
        )

        # Optional tensors (edge_ids, role_ids)
        # Check if any sequence has non-PAD edge/role indices (meaning actual edges/roles exist)
        edge_ids_tensor = None
        role_ids_tensor = None

        # Check if edge_ids have any non-PAD values (actual edges, not just padding)
        edge_has_data = any(
            any(idx != PAD_IDX for idx in seq) for seq in tag_indices["edge"]
        )
        if edge_has_data:
            edge_ids_tensor = torch.tensor(
                tag_indices["edge"], dtype=torch.long, device=device
            )

        # Check if role_ids have any non-PAD values (actual roles, not just padding)
        role_has_data = any(
            any(idx != PAD_IDX for idx in seq) for seq in tag_indices["role"]
        )
        if role_has_data:
            role_ids_tensor = torch.tensor(
                tag_indices["role"], dtype=torch.long, device=device
            )

        return {
            "token_ids": token_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "field_ids": field_ids_tensor,
            "entity_ids": entity_ids_tensor,
            "time_ids": time_ids_tensor,
            "edge_ids": edge_ids_tensor,
            "role_ids": role_ids_tensor,
            "token_type_ids": token_type_ids_tensor,
        }

    def _normalize_to_python_type(self, value: Any) -> Any:
        """Convert numpy types to Python native types.

        Args:
            value: Value that might be a numpy type

        Returns:
            Python native type (int, float, or original value)
        """
        # Handle numpy integer types (np.int_ and np.float_ removed in NumPy 2.0)
        if isinstance(
            value,
            (
                np.integer,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(value)

        # Handle numpy float types
        if isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
            return float(value)

        # Already a Python type or other type
        return value

    def _is_scalar_numeric(self, label: Any) -> bool:
        """Check if label is a scalar numeric value.

        Args:
            label: Label value to check

        Returns:
            True if label is int, float, or numpy numeric type
        """
        # Python native types
        if isinstance(label, (int, float)):
            return True

        # Numpy numeric types
        if isinstance(label, (np.integer, np.floating)):
            return True

        return False

    def _is_sequence(self, label: Any) -> bool:
        """Check if label is a sequence (list or tuple).

        Args:
            label: Label value to check

        Returns:
            True if label is list or tuple
        """
        return isinstance(label, (list, tuple))

    def _can_convert_to_int(self, value: Any) -> bool:
        """Check if value can be safely converted to integer.

        Args:
            value: Value to check

        Returns:
            True if value can be converted to int without loss
        """
        try:
            # Normalize to Python type first
            normalized = self._normalize_to_python_type(value)

            # Check if it's already an int
            if isinstance(normalized, int):
                return True

            # Check if it's a float that represents an integer
            if isinstance(normalized, float):
                return normalized.is_integer() and -(2**31) <= normalized <= 2**31 - 1

            return False
        except (ValueError, TypeError, OverflowError):
            return False

    def _batch_multi_label(
        self, labels: list[Any | None], num_classes: int, device: torch.device
    ) -> torch.Tensor:
        """Batch multi-label binary vectors.

        Args:
            labels: List of label values (binary vectors)
            num_classes: Number of classes
            device: Device to place tensor on

        Returns:
            Tensor of shape [batch, num_classes] with dtype float
        """
        label_tensors = []
        for label in labels:
            if label is None:
                label_tensors.append([0.0] * num_classes)
            else:
                label_list = list(label) if isinstance(label, tuple) else label
                label_tensors.append([float(x) for x in label_list])

        return torch.tensor(label_tensors, dtype=torch.float, device=device)

    def _batch_token_classification(
        self, labels: list[Any | None], max_seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Batch token classification labels with padding.

        Args:
            labels: List of label sequences (variable length)
            max_seq_len: Maximum sequence length for padding
            device: Device to place tensor on

        Returns:
            Tensor of shape [batch, max_seq_len] with dtype long
        """
        padded_labels = []
        for label in labels:
            if label is None:
                padded_labels.append([-100] * max_seq_len)
            else:
                label_list = list(label) if isinstance(label, tuple) else label
                # Truncate if too long
                if len(label_list) > max_seq_len:
                    label_list = label_list[:max_seq_len]
                # Pad if too short
                padded = label_list + [-100] * (max_seq_len - len(label_list))
                padded_labels.append(padded)

        return torch.tensor(padded_labels, dtype=torch.long, device=device)

    def _batch_scalar_labels(
        self, labels: list[Any | None], device: torch.device
    ) -> torch.Tensor:
        """Batch scalar labels (classification or regression).

        Args:
            labels: List of scalar label values
            device: Device to place tensor on

        Returns:
            Tensor of shape [batch] (classification) or [batch, 1] (regression)
        """
        # Normalize all labels to Python types
        normalized_labels = [
            self._normalize_to_python_type(label) if label is not None else None
            for label in labels
        ]

        # Check if all can be converted to int (classification)
        valid_labels = [label for label in normalized_labels if label is not None]
        all_int = all(self._can_convert_to_int(label) for label in valid_labels)

        if all_int:
            # Classification: [batch] (class indices)
            label_values = []
            for label in normalized_labels:
                if label is None:
                    label_values.append(0)
                else:
                    # Convert to int (handles float values like 0.0)
                    label_values.append(int(float(label)))

            return torch.tensor(label_values, dtype=torch.long, device=device)
        else:
            # Regression: [batch, 1] (continuous values)
            label_values = []
            for label in normalized_labels:
                if label is None:
                    label_values.append(0.0)
                else:
                    label_values.append(float(label))

            return torch.tensor(
                label_values, dtype=torch.float, device=device
            ).unsqueeze(-1)

    def _batch_sequence_labels(
        self, labels: list[Any | None], max_seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Batch sequence labels (multi-label or token classification).

        Args:
            labels: List of sequence label values
            max_seq_len: Maximum sequence length for padding
            device: Device to place tensor on

        Returns:
            Tensor of appropriate shape and dtype
        """
        # Filter out None labels to find valid ones
        valid_labels = [label for label in labels if label is not None]
        if not valid_labels:
            return None

        first_label = valid_labels[0]

        # Check if first element is numeric
        if len(first_label) > 0:
            first_element = (
                first_label[0]
                if isinstance(first_label, list)
                else list(first_label)[0]
            )
            is_numeric = self._is_scalar_numeric(first_element)
        else:
            # Empty sequence - treat as token classification
            return self._batch_token_classification(labels, max_seq_len, device)

        if not is_numeric:
            # Non-numeric sequences - treat as token classification
            return self._batch_token_classification(labels, max_seq_len, device)

        # Check if all labels have same length (multi-label) or variable length (token classification)
        label_lengths = [
            len(list(label) if isinstance(label, tuple) else label)
            for label in valid_labels
        ]

        if len(set(label_lengths)) == 1:
            # All labels have same length: multi-label classification
            num_classes = label_lengths[0]
            return self._batch_multi_label(labels, num_classes, device)
        else:
            # Variable lengths: token classification
            return self._batch_token_classification(labels, max_seq_len, device)

    def _batch_labels(
        self,
        labels: list[Any | None],
        max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Batch labels into tensor.

        Args:
            labels: List of label values (can be None, int, float, list, etc.)
            max_seq_len: Maximum sequence length (for token classification padding)
            device: Device to place tensor on

        Returns:
            Labels tensor (shape depends on label format) or None if no valid labels
        """
        # Early return if no valid labels
        valid_labels = [label for label in labels if label is not None]
        if not valid_labels:
            return None

        # Infer label format from first valid label
        first_label = valid_labels[0]

        # Route to appropriate handler based on label type
        if self._is_sequence(first_label):
            # Sequence labels: multi-label or token classification
            return self._batch_sequence_labels(labels, max_seq_len, device)
        elif self._is_scalar_numeric(first_label):
            # Scalar labels: classification or regression
            return self._batch_scalar_labels(labels, device)
        else:
            # Unsupported type
            raise ValueError(
                f"Cannot batch labels of type {type(first_label)}. "
                f"Supported types: int, float, list, tuple. "
                f"Got value: {first_label}"
            )
