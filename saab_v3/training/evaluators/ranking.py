"""Ranking evaluator for computing ranking metrics."""

import torch

from saab_v3.training.evaluators.base import BaseEvaluator


class RankingEvaluator(BaseEvaluator):
    """Evaluator for ranking tasks.

    Metrics:
    - pairwise_accuracy: Percentage of correctly ranked pairs
    - ndcg: Normalized Discounted Cumulative Gain (future)
    - map: Mean Average Precision (future)
    """

    def __init__(self, device: str | torch.device | None = None):
        """Initialize ranking evaluator.

        Args:
            device: Device string or torch.device
        """
        super().__init__(device=device)
        self._reset_accumulator()

    def _reset_accumulator(self) -> None:
        """Reset accumulator for new evaluation run."""
        self._all_scores_a: list[torch.Tensor] = []
        self._all_scores_b: list[torch.Tensor] = []
        self._all_labels: list[torch.Tensor] = []
        self._total_correct = 0
        self._total_pairs = 0

    def reset(self) -> None:
        """Reset accumulators for a new evaluation run."""
        self._reset_accumulator()

    def accumulate_batch(
        self, outputs: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Accumulate batch scores and labels.

        For ranking, outputs should be scores for pairs.
        Labels indicate which item is better (1 = first better, 0 = second better).

        Args:
            outputs: Model scores of shape [batch_size] (difference scores)
            labels: Ground truth labels of shape [batch_size] (1 or 0)
        """
        outputs = outputs.to(self.device)
        labels = labels.to(self.device)

        # For pairwise ranking, predictions are based on score sign
        predictions = (outputs > 0).long()

        # Accumulate for pairwise accuracy
        correct = (predictions == labels).sum().item()
        self._total_correct += correct
        self._total_pairs += labels.numel()

        # Store for future metrics
        self._all_scores_a.append(outputs.detach())
        self._all_labels.append(labels.detach())

    def compute_batch_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute metrics that can be computed per-batch.

        Args:
            outputs: Model scores
            labels: Ground truth labels
            metrics: List of metric names. If None, computes all available.

        Returns:
            Dictionary of metric values.
        """
        if metrics is None:
            metrics = ["pairwise_accuracy"]

        results = {}
        outputs = outputs.to(self.device)
        labels = labels.to(self.device)
        predictions = (outputs > 0).long()

        if "pairwise_accuracy" in metrics:
            correct = (predictions == labels).sum().item()
            total = labels.numel()
            results["pairwise_accuracy"] = correct / total if total > 0 else 0.0

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
            metrics = ["pairwise_accuracy"]

        results = {}

        if "pairwise_accuracy" in metrics:
            results["pairwise_accuracy"] = (
                self._total_correct / self._total_pairs
                if self._total_pairs > 0
                else 0.0
            )

        # Future metrics: NDCG, MAP
        # These would require more complex computation

        return results

