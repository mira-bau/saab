"""
Training entrypoint script for testing preprocessing.

Usage:
    poetry run python -m saab_v3.train --dataset-name <dataset_name>

Example:
    poetry run python -m saab_v3.train --dataset-name mydataset

The dataset must be located in dataset/raw/{dataset_name}/ directory
and should contain train.csv and val.csv files.

Data locations:
- Raw data: dataset/raw/{dataset_name}/
- Artifacts: data/artifacts/{dataset_name}/
- Preprocessed data: Processed on-the-fly (not cached)
"""

import argparse
from pathlib import Path
from saab_v3.training import (
    PreprocessingConfig,
    Preprocessor,
    StructuredDataset,
    create_dataloader,
)

# ============================================================================
# Command-Line Arguments
# ============================================================================

parser = argparse.ArgumentParser(description="Test preprocessing pipeline on a dataset")
parser.add_argument(
    "--dataset-name",
    "--dataset",
    type=str,
    required=True,
    dest="dataset_name",
    help="Name of the dataset directory in dataset/raw/",
)
args = parser.parse_args()

dataset_name = args.dataset_name

# ============================================================================
# Configuration
# ============================================================================

config = PreprocessingConfig(
    vocab_size=30000,
    max_seq_len=512,
    preserve_original_tags=True,  # Set to True for SAAB Transformer
    device="auto",  # "auto" to auto-detect, or "cpu", "cuda", "mps"
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
preprocessor = Preprocessor(config)

# Fit on training data (builds vocabularies)
print(f"Fitting preprocessor on training data from {train_path}...")
preprocessor.fit(str(train_path))

# Save artifacts for later use
print(f"Saving preprocessing artifacts for '{dataset_name}'...")
preprocessor.save_artifacts(dataset_name)

# Create datasets
print("Creating datasets...")
train_dataset = StructuredDataset(str(train_path), preprocessor, split="train")
val_dataset = StructuredDataset(str(val_path), preprocessor, split="val")

# Create dataloaders
print("Creating dataloaders...")
# Device flows automatically from config - no need to pass it
train_loader = create_dataloader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    preserve_original_tags=config.preserve_original_tags,
)

val_loader = create_dataloader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    preserve_original_tags=config.preserve_original_tags,
)

# ============================================================================
# Verify Preprocessing (Iterate through batches)
# ============================================================================

print("\nVerifying preprocessing by iterating through batches...")
print("=" * 50)

# Check training batches
print("\nTraining batches:")
for batch_idx, batch in enumerate(train_loader):
    print(f"  Batch {batch_idx}:")
    print(f"    - token_ids shape: {batch.token_ids.shape}")
    print(f"    - attention_mask shape: {batch.attention_mask.shape}")
    print(f"    - field_ids shape: {batch.field_ids.shape}")
    print(f"    - entity_ids shape: {batch.entity_ids.shape}")
    print(f"    - time_ids shape: {batch.time_ids.shape}")
    print(f"    - token_type_ids shape: {batch.token_type_ids.shape}")
    if batch.edge_ids is not None:
        print(f"    - edge_ids shape: {batch.edge_ids.shape}")
    if batch.role_ids is not None:
        print(f"    - role_ids shape: {batch.role_ids.shape}")
    if batch.original_tags is not None:
        print(f"    - original_tags: {len(batch.original_tags)} sequences")
    print(f"    - sequence_lengths: {batch.sequence_lengths}")

    # Only show first few batches
    if batch_idx >= 2:
        print("    ... (showing first 3 batches)")
        break

# Check validation batches
print("\nValidation batches:")
for batch_idx, batch in enumerate(val_loader):
    print(
        f"  Batch {batch_idx}: batch_size={batch.token_ids.shape[0]}, "
        f"seq_len={batch.token_ids.shape[1]}"
    )
    if batch_idx >= 2:
        print("  ... (showing first 3 batches)")
        break

print("\n" + "=" * 50)
print("Preprocessing verification complete!")
print(f"\nTotal training batches: {len(train_loader)}")
print(f"Total validation batches: {len(val_loader)}")


if __name__ == "__main__":
    # Script is executed when run directly
    pass
