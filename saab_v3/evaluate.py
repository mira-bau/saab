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

import numpy as np
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

# ============================================================================
# Checkpoint Verification Diagnostics
# ============================================================================
print(f"\n{'=' * 60}")
print("CHECKPOINT VERIFICATION")
print(f"{'=' * 60}")
print(f"Absolute checkpoint path: {checkpoint_path.resolve()}")
print(f"Checkpoint file size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
print(f"Checkpoint mtime: {datetime.fromtimestamp(checkpoint_path.stat().st_mtime).isoformat()}")

# Determine if checkpoint is "best" or "last"
checkpoint_filename = checkpoint_path.name.lower()
is_best = "best" in checkpoint_filename
is_latest = "latest" in checkpoint_filename or "last" in checkpoint_filename
checkpoint_type = "best" if is_best else ("latest" if is_latest else "step-based")
print(f"Checkpoint type: {checkpoint_type}")

# Load checkpoint data
checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Extract configs from checkpoint
checkpoint_config = checkpoint_data.get("config", {})
model_config_dict = checkpoint_config.get("model_config", {})
task_config_dict = checkpoint_config.get("task", {})
preprocessing_config_dict = checkpoint_config.get("preprocessing_config", {})
model_type = checkpoint_config.get("model_type", "scratch")
dataset_name_from_checkpoint = checkpoint_config.get("dataset_name", dataset_name)

# Print checkpoint metadata
print(f"\nCheckpoint Metadata:")
print(f"  - Model type: {model_type}")
print(f"  - Dataset name: {dataset_name_from_checkpoint}")
if "epoch" in checkpoint_data:
    print(f"  - Epoch: {checkpoint_data['epoch']}")
if "step" in checkpoint_data:
    print(f"  - Global step: {checkpoint_data['step']}")
if "metrics" in checkpoint_data:
    print(f"  - Metrics in checkpoint: {list(checkpoint_data['metrics'].keys())}")
    if "loss" in checkpoint_data["metrics"]:
        print(f"  - Checkpoint loss: {checkpoint_data['metrics']['loss']:.6f}")

# Verify checkpoint matches expected model type from path
checkpoint_dir = checkpoint_path.parent.name.lower()
if "scratch" in checkpoint_dir and model_type != "scratch":
    print(f"\n⚠️  WARNING: Checkpoint directory suggests 'scratch' but model_type is '{model_type}'")
    print(f"    Checkpoint path: {checkpoint_path}")
    print(f"    This may indicate loading the wrong checkpoint!")
elif "saab" in checkpoint_dir and model_type != "saab":
    print(f"\n⚠️  WARNING: Checkpoint directory suggests 'saab' but model_type is '{model_type}'")
    print(f"    Checkpoint path: {checkpoint_path}")
    print(f"    This may indicate loading the wrong checkpoint!")
elif "flat" in checkpoint_dir and model_type != "flat":
    print(f"\n⚠️  WARNING: Checkpoint directory suggests 'flat' but model_type is '{model_type}'")
    print(f"    Checkpoint path: {checkpoint_path}")
    print(f"    This may indicate loading the wrong checkpoint!")

print(f"{'=' * 60}\n")

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
    try:
        task_head.load_state_dict(checkpoint_data["task_head_state_dict"], strict=True)
    except RuntimeError as e:
        # Handle case where checkpoint was saved before LayerNorm was added
        if "layer_norm" in str(e):
            print("⚠️  Warning: Checkpoint missing layer_norm (likely from before LayerNorm was added)")
            print("  Loading with strict=False (missing layer_norm will use default initialization)")
            task_head.load_state_dict(checkpoint_data["task_head_state_dict"], strict=False)
        else:
            raise

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

# Collect predictions and probabilities for diagnostics (classification only)
all_predictions_list = []
all_labels_list = []
all_probs_list = []
all_outputs_list = []

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

        # Collect diagnostics for classification tasks
        if task_type == "classification":
            # Store outputs (logits) and compute probabilities
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(outputs, dim=-1)
            
            all_outputs_list.append(outputs.detach().cpu())
            all_probs_list.append(probs.detach().cpu())
            all_predictions_list.append(preds.detach().cpu())
            all_labels_list.append(batch.labels.detach().cpu())

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
# Prediction Diagnostics (Classification only)
# ============================================================================
if task_type == "classification" and all_predictions_list:
    print(f"\n{'=' * 60}")
    print("PREDICTION DIAGNOSTICS")
    print(f"{'=' * 60}")
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions_list, dim=0)
    all_labels = torch.cat(all_labels_list, dim=0)
    all_probs = torch.cat(all_probs_list, dim=0)
    all_outputs = torch.cat(all_outputs_list, dim=0)
    
    # Predictions histogram
    unique_preds, pred_counts = torch.unique(all_predictions, return_counts=True)
    print(f"\nPredictions histogram:")
    for pred, count in zip(unique_preds, pred_counts):
        print(f"  Class {pred.item()}: {count.item()} predictions ({100*count.item()/len(all_predictions):.2f}%)")
    
    # Labels histogram
    unique_labels, label_counts = torch.unique(all_labels, return_counts=True)
    print(f"\nLabels histogram:")
    for label, count in zip(unique_labels, label_counts):
        print(f"  Class {label.item()}: {count.item()} samples ({100*count.item()/len(all_labels):.2f}%)")
    
    # Probability statistics
    top1_probs = all_probs.max(dim=-1).values
    top1_prob_mean = top1_probs.mean().item()
    top1_prob_min = top1_probs.min().item()
    top1_prob_max = top1_probs.max().item()
    print(f"\nTop-1 probability statistics:")
    print(f"  Mean: {top1_prob_mean:.6f}")
    print(f"  Min: {top1_prob_min:.6f}")
    print(f"  Max: {top1_prob_max:.6f}")
    print(f"  Std: {top1_probs.std().item():.6f}")
    
    # Entropy statistics
    entropy = (-all_probs * (all_probs + 1e-12).log()).sum(dim=-1)
    entropy_mean = entropy.mean().item()
    max_entropy = np.log(task_config.num_classes)  # Maximum entropy for uniform distribution
    print(f"\nPrediction entropy statistics:")
    print(f"  Mean entropy: {entropy_mean:.6f}")
    print(f"  Max possible entropy (uniform): {max_entropy:.6f}")
    print(f"  Normalized entropy: {entropy_mean/max_entropy:.6f} (1.0 = uniform, 0.0 = deterministic)")
    
    # Number of unique predictions
    num_unique_preds = len(unique_preds)
    print(f"\nNumber of unique predictions: {num_unique_preds} / {task_config.num_classes} classes")
    
    # Check for model collapse
    if num_unique_preds == 1:
        collapsed_class = unique_preds[0].item()
        print(f"\n⚠️  MODEL COLLAPSE DETECTED!")
        print(f"  Model predicts only class {collapsed_class} for all samples.")
        print(f"  Expected accuracy on balanced data: ~{1/task_config.num_classes:.6f} ({100/task_config.num_classes:.2f}%)")
        print(f"  This suggests the model has not learned meaningful patterns by step {checkpoint_data.get('step', 'unknown')}.")
    elif num_unique_preds < task_config.num_classes // 2:
        print(f"\n⚠️  WARNING: Model predicts only {num_unique_preds} out of {task_config.num_classes} classes.")
        print(f"  This may indicate partial collapse or insufficient training.")
    
    # Sample predictions (first 50)
    num_samples = min(50, len(all_predictions))
    print(f"\nSample predictions (first {num_samples}):")
    print(f"  Format: (label, prediction, top1_prob)")
    for i in range(num_samples):
        label = all_labels[i].item()
        pred = all_predictions[i].item()
        prob = top1_probs[i].item()
        match = "✓" if label == pred else "✗"
        print(f"  [{i:3d}] {match} label={label}, pred={pred}, prob={prob:.4f}")
    
    print(f"{'=' * 60}\n")

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
