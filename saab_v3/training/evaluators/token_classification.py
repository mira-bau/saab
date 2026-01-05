"""Token classification evaluator for computing per-token metrics."""

import numpy as np
import torch

from saab_v3.training.evaluators.base import BaseEvaluator


class TokenClassificationEvaluator(BaseEvaluator):
    """Evaluator for token classification tasks (NER, POS tagging, etc.).

    Metrics:
    - token_accuracy: Per-token accuracy (ignoring padding)
    - f1_per_label: F1 score per label (future)
    - exact_match: Percentage of sequences with all tokens correct
    """

    def __init__(
        self,
        num_labels: int,
        device: str | torch.device | None = None,
    ):
        """Initialize token classification evaluator.

        Args:
            num_labels: Number of label classes
            device: Device string or torch.device
        """
        super().__init__(device=device)
        self.num_labels = num_labels
        self._reset_accumulator()

    def _reset_accumulator(self) -> None:
        """Reset accumulator for new evaluation run."""
        self._all_predictions: list[np.ndarray] = []
        self._all_labels: list[np.ndarray] = []
        self._all_attention_masks: list[np.ndarray] = []
        self._total_correct_tokens = 0
        self._total_valid_tokens = 0
        self._exact_matches = 0
        self._total_sequences = 0

    def reset(self) -> None:
        """Reset accumulators for a new evaluation run."""
        self._reset_accumulator()

    def accumulate_batch(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate batch predictions and labels.

        Args:
            outputs: Model outputs (logits) of shape [batch_size, seq_len, num_labels]
            labels: Ground truth labels of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len] (1 = valid, 0 = padding)
        """
        outputs = outputs.to(self.device)
        labels = labels.to(self.device)

        # Get predictions
        predictions = torch.argmax(outputs, dim=-1)  # [batch_size, seq_len]

        # Handle attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            # Mask out padding tokens
            valid_mask = attention_mask.bool()
        else:
            # Assume all tokens are valid
            valid_mask = torch.ones_like(labels, dtype=torch.bool)

        # Count correct tokens (only for valid positions)
        correct = ((predictions == labels) & valid_mask).sum().item()
        valid_tokens = valid_mask.sum().item()

        self._total_correct_tokens += correct
        self._total_valid_tokens += valid_tokens

        # Check exact matches (all tokens in sequence correct)
        batch_size = labels.shape[0]
        for i in range(batch_size):
            seq_valid_mask = valid_mask[i]
            seq_predictions = predictions[i][seq_valid_mask]
            seq_labels = labels[i][seq_valid_mask]
            if torch.equal(seq_predictions, seq_labels):
                self._exact_matches += 1
            self._total_sequences += 1

        # Store for future metrics (f1_per_label)
        self._all_predictions.append(predictions.detach().cpu().numpy())
        self._all_labels.append(labels.detach().cpu().numpy())
        if attention_mask is not None:
            self._all_attention_masks.append(attention_mask.detach().cpu().numpy())
        else:
            # Create all-ones mask
            self._all_attention_masks.append(
                np.ones_like(labels.detach().cpu().numpy())
            )

    def compute_batch_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        metrics: list[str] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute metrics that can be computed per-batch.

        Args:
            outputs: Model outputs (logits)
            labels: Ground truth labels
            metrics: List of metric names. If None, computes all available.
            attention_mask: Optional attention mask

        Returns:
            Dictionary of metric values.
        """
        if metrics is None:
            metrics = ["token_accuracy"]

        results = {}
        outputs = outputs.to(self.device)
        labels = labels.to(self.device)
        predictions = torch.argmax(outputs, dim=-1)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            valid_mask = attention_mask.bool()
        else:
            valid_mask = torch.ones_like(labels, dtype=torch.bool)

        if "token_accuracy" in metrics:
            correct = ((predictions == labels) & valid_mask).sum().item()
            valid_tokens = valid_mask.sum().item()
            results["token_accuracy"] = (
                correct / valid_tokens if valid_tokens > 0 else 0.0
            )

        return results

    def compute_aggregated_metrics(
        self, metrics: list[str] | None = None
    ) -> dict[str, float]:
        """Compute metrics on accumulated data.

        Args:
            metrics: List of metric names to compute. If None, computes all available.

        Returns:
            Dictionary mapping metric names to values.
        """
        if metrics is None:
            metrics = ["token_accuracy", "exact_match"]

        results = {}

        if "token_accuracy" in metrics:
            results["token_accuracy"] = (
                self._total_correct_tokens / self._total_valid_tokens
                if self._total_valid_tokens > 0
                else 0.0
            )

        if "exact_match" in metrics:
            results["exact_match"] = (
                self._exact_matches / self._total_sequences
                if self._total_sequences > 0
                else 0.0
            )

        # Future: f1_per_label would require per-label F1 computation
        # This would use sklearn.metrics.classification_report or similar

        return results
