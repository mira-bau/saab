"""Regression evaluator for computing regression metrics."""

import numpy as np
import torch

from saab_v3.training.evaluators.base import BaseEvaluator


class RegressionEvaluator(BaseEvaluator):
    """Evaluator for regression tasks.

    Metrics:
    - GPU-native: mse, mae, rmse
    - CPU-based: r2_score
    """

    def __init__(self, device: str | torch.device | None = None):
        """Initialize regression evaluator.

        Args:
            device: Device string or torch.device
        """
        super().__init__(device=device)
        self._reset_accumulator()

    def _reset_accumulator(self) -> None:
        """Reset accumulator for new evaluation run."""
        self._all_predictions: list[np.ndarray] = []
        self._all_labels: list[np.ndarray] = []
        self._sum_squared_error = 0.0
        self._sum_absolute_error = 0.0
        self._total_samples = 0

    def reset(self) -> None:
        """Reset accumulators for a new evaluation run."""
        self._reset_accumulator()

    def accumulate_batch(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate batch predictions and labels.

        Args:
            outputs: Model predictions of shape [batch_size, num_targets] or [batch_size]
            labels: Ground truth values of shape [batch_size, num_targets] or [batch_size]
        """
        outputs = outputs.to(self.device)
        labels = labels.to(self.device)

        # Flatten if needed
        if outputs.dim() > 1:
            outputs = outputs.squeeze()
        if labels.dim() > 1:
            labels = labels.squeeze()

        # Accumulate for GPU-native metrics
        squared_error = ((outputs - labels) ** 2).sum().item()
        absolute_error = (torch.abs(outputs - labels)).sum().item()
        self._sum_squared_error += squared_error
        self._sum_absolute_error += absolute_error
        self._total_samples += labels.numel()

        # Store for CPU-based metrics (r2_score)
        self._all_predictions.append(outputs.detach().cpu().numpy())
        self._all_labels.append(labels.detach().cpu().numpy())

    def compute_batch_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute metrics that can be computed per-batch.

        Args:
            outputs: Model predictions
            labels: Ground truth values
            metrics: List of metric names. If None, computes all available.

        Returns:
            Dictionary of metric values.
        """
        if metrics is None:
            metrics = ["mse", "mae", "rmse"]

        results = {}
        outputs = outputs.to(self.device)
        labels = labels.to(self.device)

        # Flatten if needed
        if outputs.dim() > 1:
            outputs = outputs.squeeze()
        if labels.dim() > 1:
            labels = labels.squeeze()

        if "mse" in metrics:
            mse = ((outputs - labels) ** 2).mean().item()
            results["mse"] = mse

        if "mae" in metrics:
            mae = torch.abs(outputs - labels).mean().item()
            results["mae"] = mae

        if "rmse" in metrics:
            mse = ((outputs - labels) ** 2).mean().item()
            results["rmse"] = np.sqrt(mse)

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
            metrics = ["mse", "mae", "rmse", "r2_score"]

        results = {}

        # GPU-native metrics (already computed during accumulation)
        if self._total_samples > 0:
            if "mse" in metrics:
                results["mse"] = self._sum_squared_error / self._total_samples

            if "mae" in metrics:
                results["mae"] = self._sum_absolute_error / self._total_samples

            if "rmse" in metrics:
                mse = self._sum_squared_error / self._total_samples
                results["rmse"] = np.sqrt(mse)

        # CPU-based metrics (r2_score requires sklearn)
        if "r2_score" in metrics and len(self._all_predictions) > 0:
            try:
                from sklearn.metrics import r2_score

                all_predictions = np.concatenate(self._all_predictions)
                all_labels = np.concatenate(self._all_labels)
                results["r2_score"] = r2_score(all_labels, all_predictions)
            except ImportError:
                # sklearn not available, skip r2_score
                pass

        return results
