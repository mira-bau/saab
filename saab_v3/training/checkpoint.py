"""Checkpoint management for saving and loading model states."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import torch.optim
    import torch.optim.lr_scheduler


class CheckpointManager:
    """Manages model checkpointing and loading."""

    def __init__(self, save_dir: Path | str, keep_checkpoints: int = 3):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            keep_checkpoints: Number of checkpoints to keep (oldest are deleted)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_checkpoints = keep_checkpoints
        self._checkpoint_files: list[Path] = []

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        epoch: int,
        step: int,
        metrics: dict,
        is_best: bool = False,
        is_latest: bool = False,
        config: dict | None = None,
    ) -> Path:
        """Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Optional LR scheduler state
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
            is_latest: Whether this is the latest checkpoint
            config: Optional config dictionary to save

        Returns:
            Path to saved checkpoint
        """
        # Convert config dict to JSON-serializable format (Path -> str)
        serializable_config = None
        if config is not None:
            serializable_config = {}
            for key, value in config.items():
                if isinstance(value, Path):
                    serializable_config[key] = str(value)
                elif isinstance(value, dict):
                    # Recursively convert nested dicts
                    serializable_config[key] = {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_config[key] = value

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "config": serializable_config,
            "timestamp": datetime.now().isoformat(),
        }

        # Determine checkpoint filename
        if is_best:
            checkpoint_path = self.save_dir / "best_model.pt"
        elif is_latest:
            checkpoint_path = self.save_dir / "latest.pt"
        else:
            # Regular checkpoint
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Track checkpoint files (for cleanup)
        if not is_best and not is_latest:
            self._checkpoint_files.append(checkpoint_path)
            self._checkpoint_files.sort(key=lambda p: p.stat().st_mtime)

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path | str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        map_location: str | torch.device | None = None,
    ) -> dict:
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            map_location: Device to load checkpoint to (for CPU/GPU migration)

        Returns:
            Dictionary with loaded state (epoch, step, metrics, config, etc.)

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load with weights_only=False for backward compatibility
        # (checkpoints may contain Path objects and other non-weight data)
        checkpoint_data = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )

        # Load model state
        model.load_state_dict(checkpoint_data["model_state_dict"])

        # Load optimizer state
        if optimizer is not None and checkpoint_data.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        # Load scheduler state
        if (
            scheduler is not None
            and checkpoint_data.get("scheduler_state_dict") is not None
        ):
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

        return {
            "epoch": checkpoint_data.get("epoch", 0),
            "step": checkpoint_data.get("step", 0),
            "metrics": checkpoint_data.get("metrics", {}),
            "config": checkpoint_data.get("config"),
            "timestamp": checkpoint_data.get("timestamp"),
        }

    def get_best_checkpoint(self) -> Path | None:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint, or None if not found
        """
        best_path = self.save_dir / "best_model.pt"
        if best_path.exists():
            return best_path
        return None

    def get_latest_checkpoint(self) -> Path | None:
        """Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if not found
        """
        latest_path = self.save_dir / "latest.pt"
        if latest_path.exists():
            return latest_path
        return None

    def cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if len(self._checkpoint_files) <= self.keep_checkpoints:
            return

        # Keep the most recent N checkpoints
        files_to_keep = self._checkpoint_files[-self.keep_checkpoints :]
        files_to_delete = [
            f for f in self._checkpoint_files if f not in files_to_keep
        ]

        for file_path in files_to_delete:
            try:
                file_path.unlink()
                self._checkpoint_files.remove(file_path)
            except FileNotFoundError:
                # File already deleted, skip
                pass

    def list_checkpoints(self) -> list[Path]:
        """List all checkpoint files in save directory.

        Returns:
            List of checkpoint file paths
        """
        checkpoints = []
        for file_path in self.save_dir.glob("checkpoint_*.pt"):
            checkpoints.append(file_path)
        if (self.save_dir / "best_model.pt").exists():
            checkpoints.append(self.save_dir / "best_model.pt")
        if (self.save_dir / "latest.pt").exists():
            checkpoints.append(self.save_dir / "latest.pt")
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime)

