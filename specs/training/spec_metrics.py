"""Specs for MetricsLogger - happy path only."""

import json
import tempfile
import shutil
from pathlib import Path

from saab_v3.training.config import TrainingConfig
from saab_v3.training.metrics import MetricsLogger


# ============================================================================
# MetricsLogger Specs
# ============================================================================


def spec_metrics_logger_console_logging():
    """Verify MetricsLogger logs to console."""
    # Arrange
    config = TrainingConfig(num_epochs=1, log_steps=1, lr_schedule="constant")
    logger = MetricsLogger(config, "test_experiment")

    # Act: Log step metrics
    logger.log_step(step=1, metrics={"loss": 0.5, "accuracy": 0.9}, phase="train")

    # Assert: No errors (console output is hard to test, so we just verify it doesn't crash)
    logger.close()


def spec_metrics_logger_file_logging():
    """Verify MetricsLogger logs to file."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        config = TrainingConfig(num_epochs=1, log_dir=str(log_dir), lr_schedule="constant")
        logger = MetricsLogger(config, "test_experiment")

        # Act: Log step and epoch metrics
        logger.log_step(step=1, metrics={"loss": 0.5}, phase="train")
        logger.log_epoch(epoch=1, metrics={"loss": 0.4}, phase="train")
        logger.close()

        # Assert: Log file exists and contains entries
        # The logger creates log_dir/metrics.jsonl (not log_dir/experiment_name/metrics.jsonl)
        # when log_dir is explicitly provided
        log_file = log_dir / "metrics.jsonl"
        assert log_file.exists()

        # Read and verify log entries
        with open(log_file) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            assert len(lines) >= 2

            # Parse first entry (step)
            step_entry = json.loads(lines[0])
            assert step_entry["step_or_epoch"] == 1
            assert step_entry["is_epoch"] is False
            assert step_entry["phase"] == "train"
            assert "train/loss" in step_entry["metrics"]

            # Parse second entry (epoch)
            epoch_entry = json.loads(lines[1])
            assert epoch_entry["step_or_epoch"] == 1
            assert epoch_entry["is_epoch"] is True
            assert epoch_entry["phase"] == "train"


def spec_metrics_logger_epoch_summary():
    """Verify MetricsLogger can compute epoch summary statistics."""
    # Arrange
    config = TrainingConfig(num_epochs=1, lr_schedule="constant")
    logger = MetricsLogger(config, "test_experiment")

    # Act: Log multiple steps
    for step in range(10):
        logger.log_step(step=step, metrics={"loss": 0.5 + step * 0.01}, phase="train")

    # Act: Get summary
    summary = logger.get_epoch_summary()

    # Assert: Summary contains mean and std
    assert "train/loss_mean" in summary
    assert "train/loss_std" in summary
    assert summary["train/loss_mean"] > 0

    logger.close()


def spec_metrics_logger_scalar_logging():
    """Verify MetricsLogger can log scalar values."""
    # Arrange
    config = TrainingConfig(num_epochs=1, lr_schedule="constant")
    logger = MetricsLogger(config, "test_experiment")

    # Act: Log scalar
    logger.log_scalar("test_metric", 0.75, step=1)

    # Assert: No errors
    logger.close()


def spec_metrics_logger_validation_phase():
    """Verify MetricsLogger handles validation phase correctly."""
    # Arrange
    config = TrainingConfig(num_epochs=1, lr_schedule="constant")
    logger = MetricsLogger(config, "test_experiment")

    # Act: Log validation metrics
    logger.log_epoch(epoch=1, metrics={"loss": 0.3, "accuracy": 0.95}, phase="val")

    # Assert: No errors (validation metrics logged)
    logger.close()

