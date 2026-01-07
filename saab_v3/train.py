"""
Training entrypoint script for Transformer models.

Usage:
    poetry run python -m saab_v3.train --dataset-name <dataset_name> --model-type <model_type>

Required Arguments:
    --dataset-name, --dataset: Name of the dataset directory in dataset/raw/
    --model-type, --model: Model type ('flat', 'scratch', or 'saab')
    --device: Device to use for training ('cuda', 'cpu', etc.)

Optional Arguments:
    --resume: Path to checkpoint to resume training from
    --experiment-name: Name of experiment (default: {dataset_name}_{model_type})

Examples:
    # Basic training without task head (unsupervised/pretraining)
    poetry run python -m saab_v3.train --dataset-name mydataset --model-type saab --device cuda

    # Resume training from checkpoint
    poetry run python -m saab_v3.train \\
        --dataset-name mydataset \\
        --model-type saab \\
        --device cuda \\
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
import os
import multiprocessing
from pathlib import Path

# Set multiprocessing start method to 'spawn' for CUDA compatibility with DataLoader workers
# This must be done before any CUDA operations or DataLoader creation
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

# Set tokenizers parallelism to avoid warnings with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from saab_v3.training.loss import create_loss_fn
from saab_v3.tasks.config import ClassificationTaskConfig
from saab_v3.tasks import create_task_head_from_config

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
parser.add_argument(
    "--device",
    type=str,
    default="cpu",  # Default to CPU to avoid MPS memory issues
    help="Device to use for training (e.g., 'cuda', 'cpu', 'mps', 'auto')",
)
args = parser.parse_args()

dataset_name = args.dataset_name
model_type = args.model_type
resume_checkpoint = args.resume
experiment_name = args.experiment_name or f"{dataset_name}_{model_type}"
device = args.device or "cpu"  # Fallback to CPU if not provided

# ============================================================================
# Configuration - Single Source of Truth: Pydantic Defaults
# ============================================================================

print("\nUsing Pydantic defaults for configuration...")

# Simple constructor calls with values
# Enable text tokenization for DBpedia dataset
if dataset_name == "dbpedia":
    preprocessing_config = PreprocessingConfig(
        device=device,
        text_fields=["title", "content"],  # Mark these fields as text for subword tokenization
        use_text_tokenizer=True,  # Enable BPE tokenizer for text fields
        text_tokenizer_type="bpe",
        text_tokenizer_vocab_size=30000,  # Vocabulary size for text tokenizer
        field_boundary_token=True,  # Add [FIELD_START] tokens before each field
        max_seq_len=256,  # Reduced for faster training
    )
else:
    preprocessing_config = PreprocessingConfig(
        device=device,
    )
model_config = ModelConfig(
    device=device,
    dropout=0.2,  # Increased for regularization
    num_layers=4,
    num_heads=6,
)
training_config = TrainingConfig(
    device=device,
    num_epochs=5,  # Full training: 5 epochs
    max_steps=None,  # Use epochs instead of steps
    learning_rate=0.0002,  # Reduced from 0.0005 to prevent overfitting
    batch_size=512,  # Increased batch size directly (no gradient accumulation needed)
    gradient_accumulation_steps=1,  # Removed gradient accumulation for faster training
    lr_schedule="linear_warmup_cosine",  # Warmup + cosine decay for better convergence
    warmup_ratio=0.1,  # Warmup for 10% of training steps
    min_lr_ratio=0.1,  # Decay to 10% of initial LR by end of training
    max_grad_norm=1.0,  # FIXED: Increased from 0.1 (was clipping gradients by 19,000x!)
    early_stop_zero_loss_steps=100,
    early_stopping_patience=3,
    early_stopping_min_delta=0.001,
    early_stopping_metric="loss",
)
task_config = ClassificationTaskConfig(
    num_classes=14,  # Hardcoded for mydataset (14 classes)
    multi_label=False,
    label_smoothing=0.1,  # Regularization
)
task_type = "classification"  # ranking, regression, token_classification

# Print configuration summary
print(f"\n{'=' * 60}")
print("=" * 60)
print("Preprocessing:")
print(f"  - vocab_size: {preprocessing_config.vocab_size}")
print(f"  - max_seq_len: {preprocessing_config.max_seq_len}")
print(f"  - device: {preprocessing_config.device}")
if preprocessing_config.use_text_tokenizer:
    print(f"  - text_fields: {preprocessing_config.text_fields}")
    print(f"  - text_tokenizer_type: {preprocessing_config.text_tokenizer_type}")
    print(f"  - text_tokenizer_vocab_size: {preprocessing_config.text_tokenizer_vocab_size}")
    print(f"  - field_boundary_token: {preprocessing_config.field_boundary_token}")
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
# NOTE: test dataset will be used for evaluation

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
# Enable pin_memory and num_workers for faster data loading on GPU
is_cuda = device == "cuda" or (isinstance(device, str) and "cuda" in device.lower())
train_loader = create_dataloader(
    train_dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
    num_workers=4,  # Use 4 worker processes for parallel data loading
    pin_memory=is_cuda,  # Pin memory for faster CPU->GPU transfer (CUDA only)
)

val_loader = create_dataloader(
    val_dataset,
    batch_size=training_config.batch_size,
    shuffle=False,
    num_workers=4,  # Use 4 worker processes for parallel data loading
    pin_memory=is_cuda,  # Pin memory for faster CPU->GPU transfer (CUDA only)
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

# Create task head from Pydantic config
task_head = create_task_head_from_config(task_config, d_model=model_config.d_model)
print(f"✓ Task head created: {task_head.__class__.__name__}")

# Extract loss params from task config (filters out task head params)
loss_params = task_config.get_loss_params()

# Create loss function from task type and loss params
loss_fn = create_loss_fn(task_type, **loss_params)
print(f"✓ Loss function created for task: {task_type}")

# ============================================================================
# Training
# ============================================================================

print(f"\nInitializing trainer for experiment: {experiment_name}...")

# Prepare configs for checkpointing
model_config_dict = (
    model_config.model_dump()
    if hasattr(model_config, "model_dump")
    else model_config.__dict__
)

# Convert task_config to dict for checkpointing if it's a Pydantic model
task_config_dict = (
    task_config.model_dump() if hasattr(task_config, "model_dump") else task_config
)

# Convert preprocessing_config to dict for checkpointing if it's a Pydantic model
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
