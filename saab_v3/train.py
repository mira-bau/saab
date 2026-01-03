"""
Training entrypoint script for Transformer models.

Usage:
    poetry run python -m saab_v3.train --dataset-name <dataset_name> --model-type <model_type> [--task-config <path>]

Required Arguments:
    --dataset-name, --dataset: Name of the dataset directory in dataset/raw/
    --model-type, --model: Model type ('flat', 'scratch', or 'saab')

Optional Arguments:
    --task-config: Path to task configuration YAML file (for supervised learning)
    --resume: Path to checkpoint to resume training from
    --experiment-name: Name of experiment (default: {dataset_name}_{model_type})

Examples:
    # Basic training without task head (unsupervised/pretraining)
    poetry run python -m saab_v3.train --dataset-name mydataset --model-type saab

    # Training with task head (supervised learning)
    poetry run python -m saab_v3.train \\
        --dataset-name mydataset \\
        --model-type saab \\
        --task-config experiments/configs/examples/binary_classification.yaml

    # Resume training from checkpoint
    poetry run python -m saab_v3.train \\
        --dataset-name mydataset \\
        --model-type saab \\
        --task-config experiments/configs/examples/binary_classification.yaml \\
        --resume checkpoints/mydataset_saab/checkpoint_epoch_5.pt

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

Task Configuration:
    Task configs define the task head and loss function. See:
    - experiments/configs/templates/ for template configs
    - experiments/configs/examples/ for example configs
    - experiments/configs/README.md for detailed documentation
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

# ============================================================================
# Command-Line Arguments
# ============================================================================

parser = argparse.ArgumentParser(
    description="Train Transformer models on structured data"
)
parser.add_argument(
    "--dataset-name",
    "--dataset",
    type=str,
    required=True,
    dest="dataset_name",
    help="Name of the dataset directory in dataset/raw/",
)
parser.add_argument(
    "--model-type",
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
parser.add_argument(
    "--task-config",
    type=str,
    default=None,
    help="Path to task configuration YAML file (optional)",
)
args = parser.parse_args()

dataset_name = args.dataset_name
model_type = args.model_type
resume_checkpoint = args.resume
experiment_name = args.experiment_name or f"{dataset_name}_{model_type}"
task_config_path = args.task_config

# ============================================================================
# Configuration
# ============================================================================

# Preprocessing config
preprocessing_config = PreprocessingConfig(
    vocab_size=30000,
    max_seq_len=512,
    device="auto",  # "auto" to auto-detect, or "cpu", "cuda", "mps"
)

# Model config
model_config = ModelConfig(
    d_model=768,
    num_layers=12,
    num_heads=12,
    ffn_dim=3072,
    max_seq_len=512,
    dropout=0.1,
    device="auto",  # Should match preprocessing_config.device
)

# Training config
training_config = TrainingConfig(
    optimizer_type="adamw",
    learning_rate=1e-4,
    weight_decay=0.01,
    batch_size=32,
    num_epochs=10,
    lr_schedule="linear_warmup_cosine",
    warmup_steps=1000,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    seed=42,
    log_steps=100,
    log_epochs=True,
    eval_epochs=1,
    save_epochs=1,
    save_best=True,
    best_metric="loss",
    best_mode="min",
    device="auto",  # Should match preprocessing_config.device
)

# ============================================================================
# Dataset Paths
# ============================================================================

# Dataset directory (all datasets must be in dataset/raw/)
DATASET_DIR = Path(__file__).parent / "dataset" / "raw" / dataset_name

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

# Load task config early if provided (needed for dataset creation)
task_type = None
task_config = None
if task_config_path:
    import yaml

    print(f"\nLoading task configuration from {task_config_path}...")
    with open(task_config_path, "r") as f:
        task_config = yaml.safe_load(f)

    # Extract task type for dataset creation
    task_type = task_config["task"]["name"]
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

if task_config_path:
    # task_config and task_type already loaded above
    # Validate config
    from saab_v3.tasks import validate_task_config, create_task_head_from_config

    validate_task_config(task_config)
    print("✓ Task configuration validated")

    # Create task head
    task_head = create_task_head_from_config(task_config, d_model=model_config.d_model)
    print(f"✓ Task head created: {task_head.__class__.__name__}")

    # Extract task type and create loss function
    task_type = task_config["task"]["name"]
    task_params = task_config["task"]["params"]

    from saab_v3.training.loss import create_loss_fn

    loss_fn = create_loss_fn(task_type, **task_params)
    print(f"✓ Loss function created for task: {task_type}")  # noqa: F541

# ============================================================================
# Training
# ============================================================================

print(f"\nInitializing trainer for experiment: {experiment_name}...")
trainer = Trainer(
    model=model,
    config=training_config,
    train_loader=train_loader,
    val_loader=val_loader,
    task_head=task_head,
    loss_fn=loss_fn,
    task_type=task_type,
    experiment_name=experiment_name,
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
