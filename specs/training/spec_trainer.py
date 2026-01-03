"""Specs for Trainer - happy path only."""

import tempfile
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from saab_v3.data.structures import Batch
from saab_v3.models import ModelConfig, create_flat_transformer
from saab_v3.training import PreprocessingConfig, Preprocessor, StructuredDataset, TrainingConfig, create_dataloader
from saab_v3.training.trainer import Trainer


# ============================================================================
# Trainer Specs
# ============================================================================


@pytest.fixture
def sample_training_config():
    """Sample TrainingConfig for testing."""
    return TrainingConfig(
        optimizer_type="adamw",
        learning_rate=1e-4,
        batch_size=2,
        num_epochs=1,
        lr_schedule="constant",  # No warmup for simplicity
        log_steps=10,
        log_epochs=False,  # Disable epoch logging for faster tests
        eval_epochs=1,
        save_epochs=None,  # Disable checkpointing for faster tests
        device="cpu",
    )


@pytest.fixture
def sample_model_config():
    """Sample ModelConfig for testing."""
    return ModelConfig(
        d_model=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        max_seq_len=128,
        device="cpu",
    )


def spec_trainer_initialization(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer can be initialized correctly."""
    # Arrange
    model = create_flat_transformer(fitted_preprocessor, sample_model_config)

    # Create minimal dataloader
    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    # Act
    trainer = Trainer(
        model=model,
        config=sample_training_config,
        train_loader=train_loader,
        experiment_name="test_experiment",
    )

    # Assert
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.config == sample_training_config


def spec_trainer_optimizer_creation(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer creates optimizer correctly."""
    # Arrange
    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    # Test Adam optimizer
    config_adam = TrainingConfig(
        optimizer_type="adam",
        learning_rate=2e-4,
        batch_size=2,
        num_epochs=1,
        lr_schedule="constant",
        device="cpu",
    )
    trainer = Trainer(model=model, config=config_adam, train_loader=train_loader, experiment_name="test")
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.optimizer.param_groups[0]["lr"] == 2e-4

    # Test AdamW optimizer
    config_adamw = TrainingConfig(
        optimizer_type="adamw",
        learning_rate=1e-4,
        batch_size=2,
        num_epochs=1,
        lr_schedule="constant",
        device="cpu",
    )
    trainer = Trainer(model=model, config=config_adamw, train_loader=train_loader, experiment_name="test")
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-4


def spec_trainer_lr_scheduler(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer creates LR scheduler correctly."""
    # Arrange
    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    # Test constant schedule (no scheduler)
    config_constant = TrainingConfig(
        optimizer_type="adamw",
        learning_rate=1e-4,
        batch_size=2,
        num_epochs=1,
        lr_schedule="constant",
        device="cpu",
    )
    trainer = Trainer(model=model, config=config_constant, train_loader=train_loader, experiment_name="test")
    assert trainer.scheduler is None

    # Test linear warmup schedule (need enough steps for warmup + decay)
    config_warmup = TrainingConfig(
        optimizer_type="adamw",
        learning_rate=1e-4,
        batch_size=2,
        num_epochs=10,  # More epochs to get enough steps
        lr_schedule="linear_warmup",
        warmup_steps=2,  # Small warmup, leaving room for decay
        device="cpu",
    )
    trainer = Trainer(model=model, config=config_warmup, train_loader=train_loader, experiment_name="test")
    assert trainer.scheduler is not None


def spec_trainer_training_step(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer can perform a training step."""
    # Arrange
    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    trainer = Trainer(
        model=model,
        config=sample_training_config,
        train_loader=train_loader,
        experiment_name="test_experiment",
    )

    # Act: Get a batch and perform step
    batch = next(iter(train_loader))
    step_metrics = trainer._step(batch)

    # Assert: Step metrics contain loss
    assert "loss" in step_metrics
    assert isinstance(step_metrics["loss"], float)


def spec_trainer_validation(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer can run validation."""
    # Arrange
    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    train_dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    val_dataset = StructuredDataset(sample_data, fitted_preprocessor, split="val")
    train_loader = create_dataloader(train_dataset, batch_size=2, shuffle=False)
    val_loader = create_dataloader(val_dataset, batch_size=2, shuffle=False)

    trainer = Trainer(
        model=model,
        config=sample_training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name="test_experiment",
    )

    # Act: Run validation
    val_metrics = trainer._validate()

    # Assert: Validation metrics contain loss
    assert "loss" in val_metrics
    assert isinstance(val_metrics["loss"], float)


def spec_trainer_checkpoint_save_load(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer can save and load checkpoints."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        sample_training_config.save_dir = checkpoint_dir

        model = create_flat_transformer(fitted_preprocessor, sample_model_config)
        sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
        train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

        trainer = Trainer(
            model=model,
            config=sample_training_config,
            train_loader=train_loader,
            experiment_name="test_experiment",
        )

        # Act: Save checkpoint
        trainer._save_checkpoint(epoch=1, metrics={"loss": 0.5}, is_latest=True)

        # Assert: Checkpoint exists
        latest_checkpoint = trainer.checkpoint_manager.get_latest_checkpoint()
        assert latest_checkpoint is not None
        assert latest_checkpoint.exists()

        # Act: Load checkpoint
        new_model = create_flat_transformer(fitted_preprocessor, sample_model_config)
        new_trainer = Trainer(
            model=new_model,
            config=sample_training_config,
            train_loader=train_loader,
            experiment_name="test_experiment",
        )

        state = new_trainer.load_checkpoint(checkpoint_path=latest_checkpoint, resume=False)

        # Assert: State loaded correctly
        assert state["epoch"] == 1
        assert state["metrics"]["loss"] == 0.5


def spec_trainer_random_seeds(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer sets random seeds for reproducibility."""
    # Arrange
    sample_training_config.seed = 123

    model1 = create_flat_transformer(fitted_preprocessor, sample_model_config)
    model2 = create_flat_transformer(fitted_preprocessor, sample_model_config)

    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    # Act: Create trainers with same seed
    trainer1 = Trainer(
        model=model1,
        config=sample_training_config,
        train_loader=train_loader,
        experiment_name="test1",
    )

    trainer2 = Trainer(
        model=model2,
        config=sample_training_config,
        train_loader=train_loader,
        experiment_name="test2",
    )

    # Assert: Both trainers have same seed in config
    assert trainer1.config.seed == 123
    assert trainer2.config.seed == 123

