"""
Standalone evaluation script for trained models.

Usage:
    poetry run python -m saab_v3.evaluate --checkpoint <path> --dataset-name <name> --split <split> --device <device>

Required Arguments:
    --checkpoint: Path to checkpoint file (required)
    --dataset-name: Name of the dataset directory in dataset/raw/ (required)
    --split: Data split to evaluate on ("val", "test", default: "test")
    --device: Device to use ("cpu", "cuda", "mps", "auto")

Optional Arguments:
    --batch-size: Batch size for evaluation (default: 64)

Examples:
    # Evaluate on test set
    poetry run python -m saab_v3.evaluate \\
        --checkpoint checkpoints/dbpedia_saab/best_model.pt \\
        --dataset-name dbpedia \\
        --split test \\
        --device cuda

    # Evaluate on validation set
    poetry run python -m saab_v3.evaluate \\
        --checkpoint checkpoints/dbpedia_saab/best_model.pt \\
        --dataset-name dbpedia \\
        --split val \\
        --device cuda

Output:
    Results are automatically saved to:
    dataset/{dataset_name}/evaluation_results_{split}_{experiment_name}.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from saab_v3.data.structures import Batch
from saab_v3.models import (
    ModelConfig,
    create_flat_transformer,
    create_scratch_transformer,
    create_saab_transformer,
)
from saab_v3.tasks import create_task_head_from_config
from saab_v3.tasks.config import (
    ClassificationTaskConfig,
    RankingTaskConfig,
    RegressionTaskConfig,
    TokenClassificationTaskConfig,
)
from saab_v3.training import (
    PreprocessingConfig,
    StructuredDataset,
    create_dataloader,
    load_preprocessing_artifacts,
)
from saab_v3.training.checkpoint import CheckpointManager
from saab_v3.training.evaluator import create_evaluator
from saab_v3.training.loss import create_loss_fn
from saab_v3.utils.device import get_device

# ============================================================================
# Command-Line Arguments
# ============================================================================

parser = argparse.ArgumentParser(
    description="Evaluate trained Transformer models on test/validation data"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to checkpoint file",
)
parser.add_argument(
    "--dataset-name",
    type=str,
    required=True,
    help="Name of the dataset directory in dataset/raw/",
)
parser.add_argument(
    "--split",
    type=str,
    default="test",
    choices=["val", "test"],
    help="Data split to evaluate on (default: test)",
)
parser.add_argument(
    "--device",
    type=str,
    required=True,
    help="Device to use ('cpu', 'cuda', 'mps', 'auto')",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size for evaluation (default: 64)",
)
args = parser.parse_args()

checkpoint_path = Path(args.checkpoint)
dataset_name = args.dataset_name
split = args.split
device = args.device
batch_size = args.batch_size

# ============================================================================
# Metrics Configuration (defined in script, not as args)
# ============================================================================

# Task-specific metric lists
CLASSIFICATION_METRICS = [
    "loss",
    "accuracy",
    "f1",
    "f1_macro",
    "f1_weighted",
    "precision",
    "recall",
]

RANKING_METRICS = [
    "loss",
    "pairwise_accuracy",
]

REGRESSION_METRICS = [
    "loss",
    "mse",
    "mae",
    "rmse",
    "r2_score",
]

TOKEN_CLASSIFICATION_METRICS = [
    "loss",
    "token_accuracy",
    "exact_match",
]

# ============================================================================
# Load Checkpoint
# ============================================================================

print(f"\nLoading checkpoint from {checkpoint_path}...")

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# Load checkpoint data
checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Extract configs from checkpoint
checkpoint_config = checkpoint_data.get("config", {})
model_config_dict = checkpoint_config.get("model_config", {})
task_config_dict = checkpoint_config.get("task_config", {})
preprocessing_config_dict = checkpoint_config.get("preprocessing_config", {})
model_type = checkpoint_config.get("model_type", "scratch")
dataset_name_from_checkpoint = checkpoint_config.get("dataset_name", dataset_name)

# Use dataset name from checkpoint if available, otherwise use provided
if dataset_name_from_checkpoint:
    dataset_name = dataset_name_from_checkpoint

# Reconstruct configs
model_config = ModelConfig(**model_config_dict)
preprocessing_config = PreprocessingConfig(**preprocessing_config_dict)
preprocessing_config.device = device  # Override with provided device
model_config.device = device  # Override with provided device

# Reconstruct task config
if "num_classes" in task_config_dict:
    task_config = ClassificationTaskConfig(**task_config_dict)
    task_type = "classification"
elif "num_labels" in task_config_dict:
    task_config = TokenClassificationTaskConfig(**task_config_dict)
    task_type = "token_classification"
elif "num_targets" in task_config_dict:
    task_config = RegressionTaskConfig(**task_config_dict)
    task_type = "regression"
elif "method" in task_config_dict:
    task_config = RankingTaskConfig(**task_config_dict)
    task_type = "ranking"
else:
    raise ValueError("Could not determine task type from checkpoint")

# Select metrics based on task type
if task_type == "classification":
    metrics_to_compute = CLASSIFICATION_METRICS
elif task_type == "ranking":
    metrics_to_compute = RANKING_METRICS
elif task_type == "regression":
    metrics_to_compute = REGRESSION_METRICS
elif task_type == "token_classification":
    metrics_to_compute = TOKEN_CLASSIFICATION_METRICS
else:
    raise ValueError(f"Unknown task type: {task_type}")

print(f"Task type: {task_type}")
print(f"Model type: {model_type}")
print(f"Metrics to compute: {', '.join(metrics_to_compute)}")

# ============================================================================
# Load Preprocessor
# ============================================================================

print(f"\nLoading preprocessor artifacts for '{dataset_name}'...")
preprocessor = load_preprocessing_artifacts(dataset_name)
preprocessor.config.device = device  # Override device

# ============================================================================
# Create Model
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

# Load model state
model.load_state_dict(checkpoint_data["model_state_dict"])
model.eval()

print(f"Model created: {model.__class__.__name__}")

# ============================================================================
# Create Task Head
# ============================================================================

print("Creating task head...")
task_head = create_task_head_from_config(task_config, d_model=model_config.d_model)

# Load task head state if available
if checkpoint_data.get("task_head_state_dict") is not None:
    task_head.load_state_dict(checkpoint_data["task_head_state_dict"])

task_head = task_head.to(get_device(device))
task_head.eval()

print(f"Task head created: {task_head.__class__.__name__}")

# ============================================================================
# Create Loss Function
# ============================================================================

print("Creating loss function...")
loss_params = (
    task_config.get_loss_params() if hasattr(task_config, "get_loss_params") else {}
)
loss_fn = create_loss_fn(task_type, **loss_params)
loss_fn = loss_fn.to(get_device(device))
print(f"Loss function created for task: {task_type}")

# ============================================================================
# Create Dataset and Dataloader
# ============================================================================

# Dataset directory
DATASET_DIR = Path(__file__).parent.parent / "dataset" / "raw" / dataset_name
data_path = DATASET_DIR / f"{split}.csv"

if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

print(f"\nCreating dataset from {data_path}...")
dataset = StructuredDataset(
    str(data_path), preprocessor, split=split, task_type=task_type
)

print("Creating dataloader...")
dataloader = create_dataloader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
)

# ============================================================================
# Create Evaluator
# ============================================================================

print("Creating evaluator...")
evaluator_kwargs = {}
if task_type == "classification":
    evaluator_kwargs = {
        "num_classes": task_config.num_classes,
        "multi_label": task_config.multi_label,
    }
elif task_type == "token_classification":
    evaluator_kwargs = {"num_labels": task_config.num_labels}

evaluator = create_evaluator(
    task_type=task_type,
    device=device,
    **evaluator_kwargs,
)
print(f"Evaluator created: {evaluator.__class__.__name__}")

# ============================================================================
# Run Evaluation
# ============================================================================

print(f"\n{'=' * 60}")
print(f"Running evaluation on {split} set...")
print(f"{'=' * 60}")

evaluator.reset()
val_losses = []
batch_count = 0

device_obj = get_device(device)

with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device_obj)

        # Forward pass
        if task_type == "ranking":
            # Ranking requires special handling
            encoder_output_a = model(batch)
            encoder_output_b = model(
                Batch(
                    token_ids=batch.token_ids_b,
                    attention_mask=batch.attention_mask_b,
                    field_ids=batch.field_ids_b,
                    entity_ids=batch.entity_ids_b,
                    time_ids=batch.time_ids_b,
                    token_type_ids=batch.token_type_ids,
                    sequence_lengths=batch.sequence_lengths,
                    sequence_ids=batch.sequence_ids,
                )
            )
            outputs = task_head(encoder_output_a, encoder_output_b)
        else:
            encoder_output = model(batch)
            outputs = task_head(encoder_output, batch.attention_mask)

        # Compute loss
        loss = loss_fn(outputs, batch.labels)
        val_losses.append(loss.item())

        # Accumulate for metrics
        # Token classification needs attention_mask, others don't
        if task_type == "token_classification":
            # TokenClassificationEvaluator accepts attention_mask as optional parameter
            evaluator.accumulate_batch(
                outputs, batch.labels, attention_mask=batch.attention_mask
            )
        else:
            evaluator.accumulate_batch(outputs, batch.labels)

        batch_count += 1

# Compute final metrics
val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

# Get metrics (excluding loss, which we compute separately)
metrics_to_compute_no_loss = [m for m in metrics_to_compute if m != "loss"]
all_metrics = evaluator.compute_aggregated_metrics(metrics_to_compute_no_loss)
all_metrics["loss"] = val_loss

# ============================================================================
# Print Results
# ============================================================================

print(f"\n{'=' * 60}")
print(f"Evaluation Results ({split} set):")
print(f"{'=' * 60}")
for metric_name, metric_value in sorted(all_metrics.items()):
    if isinstance(metric_value, float):
        print(f"  {metric_name}: {metric_value:.6f}")
    else:
        print(f"  {metric_name}: {metric_value}")
print(f"{'=' * 60}")

# ============================================================================
# Save Results
# ============================================================================

# Extract experiment name from checkpoint path
# e.g., "checkpoints/dbpedia_saab/best_model.pt" -> "dbpedia_saab"
experiment_name = checkpoint_path.parent.name

# Create output directory
output_dir = Path(__file__).parent.parent / "dataset" / dataset_name
output_dir.mkdir(parents=True, exist_ok=True)

# Create output filename
output_filename = f"evaluation_results_{split}_{experiment_name}.json"
output_path = output_dir / output_filename

# Prepare results dictionary
results = {
    "checkpoint_path": str(checkpoint_path),
    "dataset_name": dataset_name,
    "split": split,
    "model_type": model_type,
    "task_type": task_type,
    "device": device,
    "batch_size": batch_size,
    "timestamp": datetime.now().isoformat(),
    "metrics": all_metrics,
    "num_batches": batch_count,
    "dataset_size": len(dataset),
}

# Save results
print(f"\nSaving results to {output_path}...")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_path}")

if __name__ == "__main__":
    pass
