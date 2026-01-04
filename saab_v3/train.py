"""
Training entrypoint script for Transformer models.

Usage:
    poetry run python -m saab_v3.train --dataset-name <dataset_name> --model-type <model_type>

Required Arguments:
    --dataset-name, --dataset: Name of the dataset directory in dataset/raw/
    --model-type, --model: Model type ('flat', 'scratch', or 'saab')

Optional Arguments:
    --resume: Path to checkpoint to resume training from
    --experiment-name: Name of experiment (default: {dataset_name}_{model_type})

Examples:
    # Basic training without task head (unsupervised/pretraining)
    poetry run python -m saab_v3.train --dataset-name mydataset --model-type saab

    # Resume training from checkpoint
    poetry run python -m saab_v3.train \\
        --dataset-name mydataset \\
        --model-type saab \\
        --resume checkpoints/mydataset_saab/checkpoint_epoch_5.pt

Configuration:
    All configuration uses Pydantic defaults (single source of truth).
    To customize, modify the Pydantic config objects in the code:
    - PreprocessingConfig: preprocessing settings
    - ModelConfig: model architecture settings
    - TrainingConfig: training hyperparameters
    - TaskConfig: task head settings (optional, for supervised learning)

Dataset Requirements:
    The dataset must be located in dataset/raw/{dataset_name}/ directory
    and should contain train.csv and val.csv files.

    For supervised learning, CSV files should include a 'label' or 'target' column.
    Supported label formats:
    - Classification: integer class indices or binary vectors (multi-label)
    - Regression: continuous float values
    - Token Classification: JSON array of label indices per token

Data Locations:
    - Raw data: dataset/raw/{dataset_name}/
    - Artifacts: dataset/artifacts/{dataset_name}/
    - Checkpoints: checkpoints/{experiment_name}/
    - Logs: logs/{experiment_name}/
"""

import argparse
from pathlib import Path

from saab_v3.models import (
    ModelConfig,
    create_flat_transformer,
    create_scratch_transformer,
    create_saab_transformer,
)
from saab_v3.training import (
    PreprocessingConfig,
    Preprocessor,
    StructuredDataset,
    TrainingConfig,
    create_dataloader,
)
from saab_v3.training.trainer import Trainer
from saab_v3.tasks.config import ClassificationTaskConfig

# ============================================================================
# Command-Line Arguments
# ============================================================================

parser = argparse.ArgumentParser(
    description="Train Transformer models on structured data"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    dest="dataset_name",
    help="Name of the dataset directory in dataset/raw/",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["flat", "scratch", "saab"],
    dest="model_type",
    help="Model type: 'flat', 'scratch', or 'saab'",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume from (optional)",
)
parser.add_argument(
    "--experiment-name",
    type=str,
    default=None,
    help="Experiment name (default: {dataset_name}_{model_type})",
)
args = parser.parse_args()

dataset_name = args.dataset_name
model_type = args.model_type
resume_checkpoint = args.resume
experiment_name = args.experiment_name or f"{dataset_name}_{model_type}"

# ============================================================================
# Configuration - Single Source of Truth: Pydantic Defaults
# ============================================================================

print("\nUsing Pydantic defaults for configuration...")

# Simple constructor calls with values
preprocessing_config = PreprocessingConfig()
model_config = ModelConfig(dropout=0.2)  # Increased for regularization
training_config = TrainingConfig(
    num_epochs=5,
    learning_rate=1e-6,
    batch_size=64,
    lr_schedule="reduce_on_plateau",
    max_grad_norm=0.1,
    early_stop_zero_loss_steps=100,
    early_stopping_patience=3,
    early_stopping_min_delta=0.001,
    early_stopping_metric="loss",
    # ReduceLROnPlateau parameters
    lr_mode="min",
    lr_factor=0.5,
    lr_patience=3,
    lr_threshold=1e-4,
    lr_min=1e-8,
    lr_cooldown=0,
)
task_config = ClassificationTaskConfig(
    num_classes=14,  # Hardcoded for mydataset (14 classes)
    multi_label=False,
    label_smoothing=0.1,  # Regularization
)

# Print configuration summary
print(f"\n{'=' * 60}")
print("=" * 60)
print("Preprocessing:")
print(f"  - vocab_size: {preprocessing_config.vocab_size}")
print(f"  - max_seq_len: {preprocessing_config.max_seq_len}")
print(f"  - device: {preprocessing_config.device}")
print("\nModel:")
print(f"  - d_model: {model_config.d_model}")
print(f"  - num_layers: {model_config.num_layers}")
print(f"  - num_heads: {model_config.num_heads}")
print(f"  - dropout: {model_config.dropout}")
print(f"  - device: {model_config.device}")
print("\nTraining:")
print(f"  - learning_rate: {training_config.learning_rate}")
print(f"  - batch_size: {training_config.batch_size}")
print(f"  - num_epochs: {training_config.num_epochs}")
print(f"  - lr_schedule: {training_config.lr_schedule}")
print(f"  - max_grad_norm: {training_config.max_grad_norm}")
print(f"  - early_stopping_patience: {training_config.early_stopping_patience}")
print(f"  - device: {training_config.device}")
if task_config is not None:
    print("\nTask:")
    print("  - task_type: classification")
    print(f"  - num_classes: {task_config.num_classes}")
    print(f"  - label_smoothing: {task_config.label_smoothing}")
print("=" * 60)
print()

# ============================================================================
# Dataset Paths
# ============================================================================

# Dataset directory (all datasets must be in dataset/raw/)
DATASET_DIR = Path(__file__).parent.parent / "dataset" / "raw" / dataset_name

# File paths
train_path = DATASET_DIR / "train.csv"
val_path = DATASET_DIR / "val.csv"
test_path = DATASET_DIR / "test.csv"  # Optional, for inference

# ============================================================================
# Preprocessing
# ============================================================================

# Check if dataset directory exists
if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}\n"
        f"Please ensure the dataset is located in dataset/raw/{dataset_name}/"
    )

# Check if files exist
if not train_path.exists():
    raise FileNotFoundError(f"Training file not found: {train_path}")
if not val_path.exists():
    raise FileNotFoundError(f"Validation file not found: {val_path}")

# Initialize preprocessor
preprocessor = Preprocessor(preprocessing_config)

# Fit on training data (builds vocabularies)
print(f"Fitting preprocessor on training data from {train_path}...")
preprocessor.fit(str(train_path))

# Save artifacts for later use
print(f"Saving preprocessing artifacts for '{dataset_name}'...")
preprocessor.save_artifacts(dataset_name)

# Extract task type for dataset creation (if task config is available)
task_type = None
if task_config is not None:
    # Get task name from Pydantic model
    from saab_v3.tasks.config import (
        ClassificationTaskConfig,
        RankingTaskConfig,
        RegressionTaskConfig,
        TokenClassificationTaskConfig,
    )

    if isinstance(task_config, ClassificationTaskConfig):
        task_type = "classification"
    elif isinstance(task_config, RankingTaskConfig):
        task_type = "ranking"
    elif isinstance(task_config, RegressionTaskConfig):
        task_type = "regression"
    elif isinstance(task_config, TokenClassificationTaskConfig):
        task_type = "token_classification"

    print(f"✓ Task type: {task_type}")

# Create datasets
print("Creating datasets...")
train_dataset = StructuredDataset(
    str(train_path), preprocessor, split="train", task_type=task_type
)
val_dataset = StructuredDataset(
    str(val_path), preprocessor, split="val", task_type=task_type
)

# Create dataloaders
print("Creating dataloaders...")
train_loader = create_dataloader(
    train_dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
)

val_loader = create_dataloader(
    val_dataset,
    batch_size=training_config.batch_size,
    shuffle=False,
)

# ============================================================================
# Model Creation
# ============================================================================

print(f"\nCreating {model_type.upper()} model...")

if model_type == "flat":
    model = create_flat_transformer(preprocessor, model_config)
elif model_type == "scratch":
    model = create_scratch_transformer(preprocessor, model_config)
elif model_type == "saab":
    model = create_saab_transformer(preprocessor, model_config)
else:
    raise ValueError(f"Unknown model_type: {model_type}")

print(f"Model created: {model.__class__.__name__}")
print(f"  - d_model: {model.d_model}")
print(f"  - num_layers: {model.num_layers}")
print(f"  - num_heads: {model.num_heads}")

# ============================================================================
# Task Head and Loss Function
# ============================================================================

task_head = None
loss_fn = None

if task_config is not None:
    # task_config is a Pydantic model
    from saab_v3.tasks import create_task_head_from_config

    # Create task head from Pydantic config
    task_head = create_task_head_from_config(task_config, d_model=model_config.d_model)
    print(f"✓ Task head created: {task_head.__class__.__name__}")

    # Extract task params from Pydantic model and create loss function
    task_params = task_config.model_dump()

    # Add default label_smoothing for classification tasks if not provided
    if task_type == "classification" and "label_smoothing" not in task_params:
        task_params["label_smoothing"] = 0.1  # Default to 0.1 for regularization

    from saab_v3.training.loss import create_loss_fn

    loss_fn = create_loss_fn(task_type, **task_params)
    print(f"✓ Loss function created for task: {task_type}")  # noqa: F541

# ============================================================================
# Training
# ============================================================================

print(f"\nInitializing trainer for experiment: {experiment_name}...")

# Prepare configs for checkpointing
# Convert model_config to dict if it's a Pydantic model
if model_config is not None:
    model_config_dict = (
        model_config.model_dump()
        if hasattr(model_config, "model_dump")
        else model_config.__dict__
    )
else:
    model_config_dict = None

# Convert task_config to dict for checkpointing if it's a Pydantic model
task_config_dict = None
if task_config is not None:
    task_config_dict = (
        task_config.model_dump() if hasattr(task_config, "model_dump") else task_config
    )

# Convert preprocessing_config to dict for checkpointing if it's a Pydantic model
preprocessing_config_dict = None
if preprocessing_config is not None:
    preprocessing_config_dict = (
        preprocessing_config.model_dump()
        if hasattr(preprocessing_config, "model_dump")
        else preprocessing_config.__dict__
    )

trainer = Trainer(
    model=model,
    config=training_config,
    train_loader=train_loader,
    val_loader=val_loader,
    task_head=task_head,
    loss_fn=loss_fn,
    task_type=task_type,
    experiment_name=experiment_name,
    model_config=model_config_dict,
    task_config=task_config_dict,
    preprocessing_config=preprocessing_config_dict,
    dataset_name=dataset_name,
    model_type=model_type,
)

# Resume from checkpoint if provided
if resume_checkpoint:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.load_checkpoint(checkpoint_path=resume_checkpoint, resume=True)

# Run training
print("\nStarting training...")
print("=" * 60)
history = trainer.train()
print("=" * 60)
print("\nTraining complete!")

# Print summary
print("\nTraining Summary:")
print(f"  - Total epochs: {len(history['train_losses'])}")
print(f"  - Final train loss: {history['train_losses'][-1]:.6f}")
if history["val_losses"]:
    print(f"  - Final val loss: {history['val_losses'][-1]:.6f}")
print(f"  - Checkpoints saved to: {trainer.checkpoint_manager.save_dir}")
print(f"  - Logs saved to: {trainer.metrics_logger.log_dir}")


if __name__ == "__main__":
    # Script is executed when run directly
    pass
