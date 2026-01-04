"""Metrics logging for training and validation."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saab_v3.training.config import TrainingConfig


class MetricsLogger:
    """Logs training metrics to console, files, and optionally TensorBoard/W&B."""

    def __init__(self, config: "TrainingConfig", experiment_name: str):
        """Initialize metrics logger.

        Args:
            config: TrainingConfig instance
            experiment_name: Name of experiment (for log directory naming)
        """
        self.config = config
        self.experiment_name = experiment_name

        # Determine log directory
        if config.log_dir is not None:
            self.log_dir = Path(config.log_dir)
        else:
            self.log_dir = Path("logs") / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # File logger (JSON lines format)
        self.log_file = self.log_dir / "metrics.jsonl"
        self._log_file_handle = open(self.log_file, "a")

        # TensorBoard logger (optional)
        self.tb_writer = None
        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / "tensorboard"
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            except ImportError:
                print(
                    "Warning: TensorBoard not available. Install with: pip install tensorboard"
                )

        # Weights & Biases logger (optional)
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project or experiment_name,
                    name=config.wandb_run_name or experiment_name,
                    config=config.model_dump() if hasattr(config, "model_dump") else {},
                )
                self.wandb_run = wandb
            except ImportError:
                print(
                    "Warning: Weights & Biases not available. Install with: pip install wandb"
                )

        # Metric accumulation for epoch-level logging
        self._step_metrics: dict[str, list[float]] = {}

    def log_step(self, step: int, metrics: dict, phase: str = "train"):
        """Log metrics for a training step.

        Args:
            step: Current step number
            metrics: Dictionary of metric name -> value
            phase: Phase name ("train", "val", "test")
        """
        # Format metric names with phase prefix
        formatted_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}

        # Console logging (if log_steps matches)
        if step % self.config.log_steps == 0:
            self._log_to_console(step, formatted_metrics, phase)

        # File logging (JSON lines)
        self._log_to_file(step, formatted_metrics, phase)

        # TensorBoard logging
        if self.tb_writer is not None:
            for name, value in formatted_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(name, value, step)

        # W&B logging
        if self.wandb_run is not None:
            self.wandb_run.log({**formatted_metrics, "step": step})

        # Accumulate metrics for epoch-level aggregation
        for name, value in formatted_metrics.items():
            if isinstance(value, (int, float)):
                if name not in self._step_metrics:
                    self._step_metrics[name] = []
                self._step_metrics[name].append(float(value))

    def log_epoch(self, epoch: int, metrics: dict, phase: str = "train"):
        """Log metrics for an epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric name -> value
            phase: Phase name ("train", "val", "test")
        """
        # Format metric names with phase prefix
        formatted_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}

        # Console logging
        if self.config.log_epochs:
            self._log_to_console_epoch(epoch, formatted_metrics, phase)

        # File logging
        self._log_to_file(epoch, formatted_metrics, phase, is_epoch=True)

        # TensorBoard logging
        if self.tb_writer is not None:
            for name, value in formatted_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(name, value, epoch)

        # W&B logging
        if self.wandb_run is not None:
            self.wandb_run.log({**formatted_metrics, "epoch": epoch})

        # Reset accumulated metrics
        self._step_metrics.clear()

    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar metric.

        Args:
            name: Metric name
            value: Metric value
            step: Step number
        """
        self.log_step(step, {name: value})

    def get_epoch_summary(self) -> dict[str, float]:
        """Get summary statistics for accumulated epoch metrics.

        Returns:
            Dictionary with mean values for each metric
        """
        summary = {}
        for name, values in self._step_metrics.items():
            if len(values) > 0:
                summary[f"{name}_mean"] = sum(values) / len(values)
                summary[f"{name}_std"] = (
                    sum((x - summary[f"{name}_mean"]) ** 2 for x in values) / len(values)
                ) ** 0.5
        return summary

    def _log_to_console(self, step: int, metrics: dict, phase: str):
        """Log metrics to console."""
        print(f"\n[{phase.upper()}] Step {step}:")
        for name, value in sorted(metrics.items()):
            if isinstance(value, float):
                # Use scientific notation for very small values (like learning rates during warmup)
                # or for learning_rate specifically
                if "learning_rate" in name.lower() or abs(value) < 1e-5:
                    print(f"  {name}: {value:.2e}")
                else:
                    print(f"  {name}: {value:.6f}")
            else:
                print(f"  {name}: {value}")

    def _log_to_console_epoch(self, epoch: int, metrics: dict, phase: str):
        """Log epoch metrics to console."""
        print(f"\n{'='*60}")
        print(f"[{phase.upper()}] Epoch {epoch}:")
        for name, value in sorted(metrics.items()):
            if isinstance(value, float):
                # Use scientific notation for very small values (like learning rates during warmup)
                # or for learning_rate specifically
                if "learning_rate" in name.lower() or abs(value) < 1e-5:
                    print(f"  {name}: {value:.2e}")
                else:
                    print(f"  {name}: {value:.6f}")
            else:
                print(f"  {name}: {value}")
        print(f"{'='*60}")

    def _log_to_file(self, step_or_epoch: int, metrics: dict, phase: str, is_epoch: bool = False):
        """Log metrics to file (JSON lines format)."""
        log_entry = {
            "step_or_epoch": step_or_epoch,
            "is_epoch": is_epoch,
            "phase": phase,
            "metrics": metrics,
        }
        self._log_file_handle.write(json.dumps(log_entry) + "\n")
        self._log_file_handle.flush()

    def close(self):
        """Close loggers and flush buffers."""
        if self._log_file_handle:
            self._log_file_handle.close()

        if self.tb_writer is not None:
            self.tb_writer.close()

        if self.wandb_run is not None:
            self.wandb_run.finish()

