"""Specs for CheckpointManager - happy path only."""

import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from saab_v3.training.checkpoint import CheckpointManager


# ============================================================================
# CheckpointManager Specs
# ============================================================================


def spec_checkpoint_manager_save_and_load():
    """Verify CheckpointManager can save and load checkpoints."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir)

        # Create a simple model
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        # Act: Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            step=100,
            metrics={"loss": 0.5, "accuracy": 0.9},
            is_best=False,
            is_latest=True,
        )

        # Assert: Checkpoint file exists
        assert checkpoint_path.exists()
        assert checkpoint_path.name == "latest.pt"

        # Act: Load checkpoint
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.1)

        state = manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
        )

        # Assert: State loaded correctly
        assert state["epoch"] == 5
        assert state["step"] == 100
        assert state["metrics"]["loss"] == 0.5
        assert state["metrics"]["accuracy"] == 0.9

        # Assert: Model weights are the same
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


def spec_checkpoint_manager_best_checkpoint():
    """Verify CheckpointManager handles best checkpoint correctly."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir)

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Act: Save best checkpoint
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=10,
            step=200,
            metrics={"loss": 0.3},
            is_best=True,
        )

        # Assert: Best checkpoint exists
        best_path = manager.get_best_checkpoint()
        assert best_path is not None
        assert best_path.name == "best_model.pt"
        assert best_path.exists()


def spec_checkpoint_manager_cleanup():
    """Verify CheckpointManager cleans up old checkpoints."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, keep_checkpoints=2)

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Act: Save multiple checkpoints
        for i in range(5):
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=i,
                step=i * 10,
                metrics={"loss": 0.5},
                is_best=False,
                is_latest=False,
            )

        # Act: Cleanup
        manager.cleanup_old_checkpoints()

        # Assert: Only last 2 checkpoints remain (plus best/latest if any)
        checkpoints = manager.list_checkpoints()
        regular_checkpoints = [c for c in checkpoints if c.name.startswith("checkpoint_")]
        assert len(regular_checkpoints) <= 2


def spec_checkpoint_manager_list_checkpoints():
    """Verify CheckpointManager can list all checkpoints."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir)

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Act: Save multiple checkpoints
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            step=10,
            metrics={"loss": 0.5},
            is_best=True,
        )
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=2,
            step=20,
            metrics={"loss": 0.4},
            is_latest=True,
        )

        # Assert: List checkpoints
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) >= 2
        checkpoint_names = [c.name for c in checkpoints]
        assert "best_model.pt" in checkpoint_names
        assert "latest.pt" in checkpoint_names

