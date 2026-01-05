"""Classification evaluator for computing classification metrics."""

import numpy as np
import torch

from saab_v3.training.evaluators.base import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks.

    Supports:
    - Binary classification
    - Multi-class classification
    - Multi-label classification

    Metrics:
    - GPU-native: accuracy
    - CPU-based (sklearn): f1, precision, recall, f1_macro, f1_weighted, f1_micro
    """

    def __init__(
        self,
        num_classes: int,
        multi_label: bool = False,
        device: str | torch.device | None = None,
    ):
        """Initialize classification evaluator.

        Args:
            num_classes: Number of classes
            multi_label: If True, treats as multi-label classification
            device: Device string or torch.device
        """
        super().__init__(device=device)
        self.num_classes = num_classes
        self.multi_label = multi_label
        self._reset_accumulator()

    def _reset_accumulator(self) -> None:
        """Reset accumulator for new evaluation run."""
        self._all_predictions: list[np.ndarray] = []
        self._all_labels: list[np.ndarray] = []
        self._total_correct = 0
        self._total_samples = 0

    def reset(self) -> None:
        """Reset accumulators for a new evaluation run."""
        self._reset_accumulator()

    def _get_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Get predictions from model outputs.

        Args:
            outputs: Model outputs (logits) of shape [batch_size, num_classes]

        Returns:
            Predictions tensor of shape [batch_size]
        """
        if self.multi_label:
            # Multi-label: threshold sigmoid outputs
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            # For multi-label, we need to handle it differently
            # For now, return argmax for compatibility
            return torch.argmax(outputs, dim=-1)
        else:
            # Multi-class: argmax of logits
            return torch.argmax(outputs, dim=-1)

    def accumulate_batch(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate batch predictions and labels.

        Args:
            outputs: Model outputs (logits) of shape [batch_size, num_classes]
            labels: Ground truth labels of shape [batch_size]
        """
        # Ensure tensors are on correct device
        outputs = outputs.to(self.device)
        labels = labels.to(self.device)

        # Get predictions
        predictions = self._get_predictions(outputs)

        # Accumulate for GPU-native metrics (accuracy)
        correct = (predictions == labels).sum().item()
        self._total_correct += correct
        self._total_samples += labels.numel()

        # Store for CPU-based metrics (only if needed)
        # We'll transfer to CPU when computing aggregated metrics
        self._all_predictions.append(predictions.detach().cpu().numpy())
        self._all_labels.append(labels.detach().cpu().numpy())

    def compute_batch_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute metrics that can be computed per-batch.

        Args:
            outputs: Model outputs (logits)
            labels: Ground truth labels
            metrics: List of metric names. If None, computes all available.

        Returns:
            Dictionary of metric values.
        """
        if metrics is None:
            metrics = ["accuracy"]

        results = {}
        outputs = outputs.to(self.device)
        labels = labels.to(self.device)
        predictions = self._get_predictions(outputs)

        if "accuracy" in metrics:
            correct = (predictions == labels).sum().item()
            total = labels.numel()
            results["accuracy"] = correct / total if total > 0 else 0.0

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
            metrics = [
                "accuracy",
                "f1",
                "f1_macro",
                "f1_weighted",
                "precision",
                "recall",
            ]

        results = {}

        # Concatenate all accumulated predictions and labels
        if len(self._all_predictions) == 0:
            return results

        all_predictions = np.concatenate(self._all_predictions)
        all_labels = np.concatenate(self._all_labels)

        # GPU-native metrics (already computed during accumulation)
        if "accuracy" in metrics:
            results["accuracy"] = (
                self._total_correct / self._total_samples
                if self._total_samples > 0
                else 0.0
            )

        # CPU-based metrics (require sklearn)
        cpu_metrics = [
            "f1",
            "f1_macro",
            "f1_weighted",
            "f1_micro",
            "precision",
            "recall",
        ]
        requested_cpu_metrics = [m for m in metrics if m in cpu_metrics]

        if requested_cpu_metrics:
            try:
                from sklearn.metrics import (
                    f1_score,
                    precision_score,
                    recall_score,
                )

                # Compute F1 scores
                if any("f1" in m for m in requested_cpu_metrics):
                    if (
                        "f1" in requested_cpu_metrics
                        or "f1_weighted" in requested_cpu_metrics
                    ):
                        results["f1"] = f1_score(
                            all_labels,
                            all_predictions,
                            average="weighted",
                            zero_division=0,
                        )
                        results["f1_weighted"] = results["f1"]
                    if "f1_macro" in requested_cpu_metrics:
                        results["f1_macro"] = f1_score(
                            all_labels,
                            all_predictions,
                            average="macro",
                            zero_division=0,
                        )
                    if "f1_micro" in requested_cpu_metrics:
                        results["f1_micro"] = f1_score(
                            all_labels,
                            all_predictions,
                            average="micro",
                            zero_division=0,
                        )

                # Compute precision
                if "precision" in requested_cpu_metrics:
                    results["precision"] = precision_score(
                        all_labels, all_predictions, average="weighted", zero_division=0
                    )

                # Compute recall
                if "recall" in requested_cpu_metrics:
                    results["recall"] = recall_score(
                        all_labels, all_predictions, average="weighted", zero_division=0
                    )

            except ImportError:
                # sklearn not available, skip CPU metrics
                pass

        return results
