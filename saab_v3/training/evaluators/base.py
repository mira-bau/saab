"""Base evaluator abstract class for task-specific metrics computation."""

from abc import ABC, abstractmethod

import torch

from saab_v3.utils.device import get_device


class BaseEvaluator(ABC):
    """Abstract base class for task-specific evaluators.

    Provides common interface for computing metrics on model outputs.
    Supports both GPU-native (fast) and CPU-based (sklearn) metrics.
    """

    def __init__(self, device: str | torch.device | None = None):
        """Initialize evaluator with explicit device configuration.

        Args:
            device: Device string ("cpu", "cuda", "mps", "auto") or torch.device.
                   If None, defaults to "cpu".
        """
        self._device = get_device(device) if device is not None else torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Get the device this evaluator is configured for."""
        return self._device

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulators for a new evaluation run."""
        pass

    @abstractmethod
    def accumulate_batch(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate batch predictions and labels for aggregated metrics.

        Args:
            outputs: Model outputs tensor
            labels: Ground truth labels tensor
        """
        pass

    @abstractmethod
    def compute_aggregated_metrics(
        self, metrics: list[str] | None = None
    ) -> dict[str, float]:
        """Compute metrics on accumulated data.

        Args:
            metrics: List of metric names to compute. If None, computes all available.

        Returns:
            Dictionary mapping metric names to values.
        """
        pass

    def compute_batch_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute metrics that can be computed per-batch (optional).

        Default implementation returns empty dict. Override in subclasses
        for metrics that can be computed immediately without accumulation.

        Args:
            outputs: Model outputs tensor
            labels: Ground truth labels tensor
            metrics: List of metric names to compute. If None, computes all available.

        Returns:
            Dictionary mapping metric names to values.
        """
        return {}
