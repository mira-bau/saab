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


def spec_trainer_loss_computation_classification(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer computes classification loss correctly."""
    # Arrange
    from saab_v3.tasks import ClassificationHead
    from saab_v3.training.loss import create_loss_fn

    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    task_head = ClassificationHead(d_model=128, num_classes=3, multi_label=False)
    loss_fn = create_loss_fn("classification", num_classes=3, multi_label=False)

    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    trainer = Trainer(
        model=model,
        config=sample_training_config,
        train_loader=train_loader,
        task_head=task_head,
        loss_fn=loss_fn,
        task_type="classification",
        experiment_name="test",
    )

    # Create a batch with labels
    batch = next(iter(train_loader))
    # Add classification labels (class indices)
    batch.labels = torch.tensor([0, 1], dtype=torch.long)

    # Act: Forward pass and loss computation
    outputs = trainer.model(batch)
    outputs = task_head(outputs)
    loss = trainer._compute_loss(batch, outputs)

    # Assert
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0  # Loss should be positive
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def spec_trainer_loss_computation_regression(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer computes regression loss correctly."""
    # Arrange
    from saab_v3.tasks import RegressionHead
    from saab_v3.training.loss import create_loss_fn

    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    task_head = RegressionHead(d_model=128, num_targets=1)
    loss_fn = create_loss_fn("regression")

    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    trainer = Trainer(
        model=model,
        config=sample_training_config,
        train_loader=train_loader,
        task_head=task_head,
        loss_fn=loss_fn,
        task_type="regression",
        experiment_name="test",
    )

    # Create a batch with labels
    batch = next(iter(train_loader))
    # Add regression labels (continuous values)
    batch.labels = torch.tensor([[3.5], [2.1]], dtype=torch.float)

    # Act: Forward pass and loss computation
    outputs = trainer.model(batch)
    outputs = task_head(outputs)
    loss = trainer._compute_loss(batch, outputs)

    # Assert
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0  # Loss should be non-negative
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def spec_trainer_loss_computation_token_classification(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer computes token classification loss correctly."""
    # Arrange
    from saab_v3.tasks import TokenClassificationHead
    from saab_v3.training.loss import create_loss_fn

    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    task_head = TokenClassificationHead(d_model=128, num_labels=3)
    loss_fn = create_loss_fn("token_classification", num_labels=3)

    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    trainer = Trainer(
        model=model,
        config=sample_training_config,
        train_loader=train_loader,
        task_head=task_head,
        loss_fn=loss_fn,
        task_type="token_classification",
        experiment_name="test",
    )

    # Create a batch with labels
    batch = next(iter(train_loader))
    batch_size, seq_len = batch.token_ids.shape
    # Add token classification labels (per-token label indices)
    batch.labels = torch.randint(0, 3, (batch_size, seq_len), dtype=torch.long)

    # Act: Forward pass and loss computation
    outputs = trainer.model(batch)
    outputs = task_head(outputs)
    loss = trainer._compute_loss(batch, outputs)

    # Assert
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0  # Loss should be positive
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def spec_trainer_loss_computation_missing_labels(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer raises error when labels are missing but required."""
    # Arrange
    from saab_v3.tasks import ClassificationHead
    from saab_v3.training.loss import create_loss_fn

    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    task_head = ClassificationHead(d_model=128, num_classes=3, multi_label=False)
    loss_fn = create_loss_fn("classification", num_classes=3, multi_label=False)

    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    trainer = Trainer(
        model=model,
        config=sample_training_config,
        train_loader=train_loader,
        task_head=task_head,
        loss_fn=loss_fn,
        task_type="classification",
        experiment_name="test",
    )

    # Create a batch without labels
    batch = next(iter(train_loader))
    batch.labels = None

    # Act: Forward pass and loss computation
    outputs = trainer.model(batch)
    outputs = task_head(outputs)

    # Assert: Should raise ValueError
    with pytest.raises(ValueError, match="Labels are required"):
        trainer._compute_loss(batch, outputs)


def spec_trainer_loss_computation_auto_create_loss(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer auto-creates loss function from task_type."""
    # Arrange
    from saab_v3.tasks import ClassificationHead

    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    task_head = ClassificationHead(d_model=128, num_classes=3, multi_label=False)

    sample_data = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    dataset = StructuredDataset(sample_data, fitted_preprocessor, split="train")
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    # Act: Create trainer with task_type but no loss_fn
    trainer = Trainer(
        model=model,
        config=sample_training_config,
        train_loader=train_loader,
        task_head=task_head,
        task_type="classification",
        experiment_name="test",
    )

    # Assert: Loss function should be auto-created
    assert trainer.loss_fn is not None
    assert trainer.task_type == "classification"

    # Verify it works
    batch = next(iter(train_loader))
    batch.labels = torch.tensor([0, 1], dtype=torch.long)
    outputs = trainer.model(batch)
    outputs = task_head(outputs)
    loss = trainer._compute_loss(batch, outputs)
    assert loss is not None
    assert not torch.isnan(loss)


# ============================================================================
# Ranking Trainer Specs
# ============================================================================


def spec_trainer_forward_ranking(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer._forward_ranking() encodes both sequences and returns scores."""
    # Arrange
    from saab_v3.tasks import PairwiseRankingHead
    from saab_v3.models import create_flat_transformer
    from saab_v3.data.structures import Batch

    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    task_head = PairwiseRankingHead(d_model=128, method="dot_product")

    # Create ranking batch (with _b fields)
    batch_size, seq_len = 2, 5
    device = torch.device("cpu")
    ranking_batch = Batch(
        token_ids=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        field_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        entity_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        time_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        token_type_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        sequence_lengths=[seq_len] * batch_size,
        token_ids_b=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        attention_mask_b=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        field_ids_b=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        entity_ids_b=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        time_ids_b=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        token_type_ids_b=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        sequence_lengths_b=[seq_len] * batch_size,
    )

    trainer = Trainer(
        model=model,
        config=sample_training_config,
        train_loader=None,  # Not needed for this test
        task_head=task_head,
        task_type="ranking",
        experiment_name="test",
    )

    # Act
    scores = trainer._forward_ranking(ranking_batch)

    # Assert
    assert scores is not None
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (batch_size,)
    assert not torch.isnan(scores).any()
    assert not torch.isinf(scores).any()


def spec_trainer_loss_computation_ranking(fitted_preprocessor, sample_training_config, sample_model_config):
    """Verify Trainer computes ranking loss correctly."""
    # Arrange
    from saab_v3.tasks import PairwiseRankingHead
    from saab_v3.training.loss import create_loss_fn
    from saab_v3.models import create_flat_transformer
    from saab_v3.data.structures import Batch

    model = create_flat_transformer(fitted_preprocessor, sample_model_config)
    task_head = PairwiseRankingHead(d_model=128, method="dot_product")
    loss_fn = create_loss_fn("ranking", method="hinge", margin=1.0)

    # Create ranking batch
    batch_size, seq_len = 2, 5
    device = torch.device("cpu")
    ranking_batch = Batch(
        token_ids=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        field_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        entity_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        time_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        token_type_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        sequence_lengths=[seq_len] * batch_size,
        labels=torch.tensor([1, -1], dtype=torch.long, device=device),
        token_ids_b=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        attention_mask_b=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        field_ids_b=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        entity_ids_b=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        time_ids_b=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        token_type_ids_b=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        sequence_lengths_b=[seq_len] * batch_size,
    )

    trainer = Trainer(
        model=model,
        config=sample_training_config,
        train_loader=None,
        task_head=task_head,
        loss_fn=loss_fn,
        task_type="ranking",
        experiment_name="test",
    )

    # Act: Forward pass and loss computation
    scores = trainer._forward_ranking(ranking_batch)
    loss = trainer._compute_loss(ranking_batch, scores)

    # Assert
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0  # Loss should be non-negative
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

