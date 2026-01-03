"""Batching logic for converting encoded sequences to Batch objects."""

import torch
from typing import Any

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
        sequences: list[tuple[TokenizedSequence, list[int], list[EncodedTag], Any | None]],
    ) -> Batch:
        """Create batch from encoded sequences.

        Args:
            sequences: List of (TokenizedSequence, token_ids, encoded_tags, label) tuples
                label can be None if not present

        Returns:
            Batch object with all tensors and labels (if provided)

        Raises:
            ValueError: If sequences list is empty
        """
        if not sequences:
            raise ValueError("Cannot create batch from empty sequences list")

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

    def _batch_labels(
        self,
        labels: list[Any | None],
        max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Batch labels into tensor.

        Args:
            labels: List of label values (can be None, int, float, list, etc.)
            max_seq_len: Maximum sequence length (for token classification padding)
            device: Device to place tensor on

        Returns:
            Labels tensor (shape depends on label format)
        """
        # Filter out None labels
        valid_labels = [label for label in labels if label is not None]
        if not valid_labels:
            return None

        # Infer label format from first valid label
        first_label = valid_labels[0]

        # Check if labels are sequences (list/tuple)
        if isinstance(first_label, (list, tuple)):
            # Check if first element is numeric (multi-label binary vector or token classification)
            if len(first_label) > 0 and isinstance(first_label[0], (int, float, bool)):
                # Check if all labels have same length (multi-label) or variable length (token classification)
                label_lengths = [
                    len(list(label) if isinstance(label, tuple) else label)
                    for label in valid_labels
                    if label is not None
                ]
                
                if len(set(label_lengths)) == 1:
                    # All labels have same length: multi-label classification [batch, num_classes]
                    num_classes = label_lengths[0]
                    label_tensors = []
                    for label in labels:
                        if label is None:
                            # Use zeros for missing labels
                            label_tensors.append([0.0] * num_classes)
                        else:
                            label_list = list(label) if isinstance(label, tuple) else label
                            label_tensors.append([float(x) for x in label_list])

                    return torch.tensor(label_tensors, dtype=torch.float, device=device)
                else:
                    # Variable lengths: token classification [batch, seq_len]
                    # Pad sequences to max_seq_len
                    padded_labels = []
                    for label in labels:
                        if label is None:
                            # Pad with -100 (ignore_index for CrossEntropyLoss)
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
            else:
                # Empty list or non-numeric: treat as token classification with empty sequence
                # This shouldn't happen in practice, but handle gracefully
                padded_labels = []
                for label in labels:
                    if label is None:
                        padded_labels.append([-100] * max_seq_len)
                    else:
                        label_list = list(label) if isinstance(label, tuple) else label
                        if len(label_list) > max_seq_len:
                            label_list = label_list[:max_seq_len]
                        padded = label_list + [-100] * (max_seq_len - len(label_list))
                        padded_labels.append(padded)

                return torch.tensor(padded_labels, dtype=torch.long, device=device)

        # Check if labels are numeric (classification or regression)
        if isinstance(first_label, (int, float)):
            # Check if all labels are integers (classification) or floats (regression)
            all_int = all(
                isinstance(label_val, int) or (isinstance(label_val, float) and label_val.is_integer())
                for label_val in valid_labels
            )

            label_values = []
            for label in labels:
                if label is None:
                    # Use 0 as placeholder (will be handled by loss function if needed)
                    label_values.append(0)
                else:
                    label_values.append(label)

            if all_int:
                # Classification: [batch] (class indices)
                return torch.tensor(label_values, dtype=torch.long, device=device)
            else:
                # Regression: [batch] or [batch, num_targets] if labels are lists
                # Check if labels are lists of floats (multi-target regression)
                if isinstance(first_label, (list, tuple)) and isinstance(
                    first_label[0], float
                ):
                    # Multi-target regression: [batch, num_targets]
                    num_targets = len(first_label)
                    label_tensors = []
                    for label in labels:
                        if label is None:
                            label_tensors.append([0.0] * num_targets)
                        else:
                            label_list = (
                                list(label) if isinstance(label, tuple) else label
                            )
                            label_tensors.append([float(x) for x in label_list])
                    return torch.tensor(label_tensors, dtype=torch.float, device=device)
                else:
                    # Single-target regression: [batch, 1]
                    return torch.tensor(
                        label_values, dtype=torch.float, device=device
                    ).unsqueeze(-1)

        # Fallback: try to convert to tensor
        try:
            return torch.tensor(labels, dtype=torch.float, device=device)
        except (ValueError, TypeError):
            raise ValueError(
                f"Cannot batch labels of type {type(first_label)}. "
                f"Supported types: int, float, list, tuple"
            )
