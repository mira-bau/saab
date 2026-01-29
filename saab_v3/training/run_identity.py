"""Run Identity tracking for reproducibility and comparison.

This module provides utilities to track and verify all deterministic components
of a training run to ensure Scratch and SAAB runs are truly comparable.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_git_commit_hash() -> str | None:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        return result.stdout.strip()[:16]  # First 16 chars
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_hostname() -> str:
    """Get hostname or device identifier."""
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return "unknown"


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    if not file_path.exists():
        return "FILE_NOT_FOUND"
    
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()[:16]  # First 16 chars


def compute_array_hash(arr: np.ndarray | torch.Tensor, algorithm: str = "sha256") -> str:
    """Compute hash of a numpy array or tensor."""
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    
    # Convert to bytes
    arr_bytes = arr.tobytes()
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(arr_bytes)
    return hash_obj.hexdigest()[:16]  # First 16 chars


def fingerprint_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    """Fingerprint preprocessing artifacts.
    
    Args:
        artifacts_dir: Path to artifacts directory
        
    Returns:
        Dictionary with artifact fingerprints
    """
    fingerprints = {
        "artifacts_dir": str(artifacts_dir),
        "exists": artifacts_dir.exists(),
    }
    
    if not artifacts_dir.exists():
        return fingerprints
    
    # Hash config.json
    config_path = artifacts_dir / "config.json"
    if config_path.exists():
        fingerprints["config_hash"] = compute_file_hash(config_path)
    else:
        fingerprints["config_hash"] = "MISSING"
    
    # Hash text tokenizer if present
    text_tokenizer_path = artifacts_dir / "vocabularies" / "text_tokenizer.json"
    if text_tokenizer_path.exists():
        fingerprints["text_tokenizer_hash"] = compute_file_hash(text_tokenizer_path)
    else:
        fingerprints["text_tokenizer_hash"] = "NOT_PRESENT"
    
    # Hash value vocab
    value_vocab_path = artifacts_dir / "vocabularies" / "value_vocab.json"
    if value_vocab_path.exists():
        fingerprints["value_vocab_hash"] = compute_file_hash(value_vocab_path)
    else:
        fingerprints["value_vocab_hash"] = "MISSING"
    
    return fingerprints


def create_run_identity(
    model_type: str,
    task: str,
    seed: int,
    dataset_name: str,
    artifacts_dir: Path,
    subset_sizes: dict[str, int | None],
    subset_hashes: dict[str, str | None],
    permutation_hash: str | None = None,
    num_fields: int | None = None,
    field_emb_size: int | None = None,
    mask_field_id: int | None = None,
    mask_prob: float | None = None,
    determinism_strict: bool = False,
    trainer_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a Run Identity dictionary.
    
    Args:
        model_type: Model type ('scratch', 'saab', 'flat')
        task: Task type ('msm_field', 'classification', etc.)
        seed: Random seed
        dataset_name: Dataset name
        artifacts_dir: Path to artifacts directory
        subset_sizes: Dict with 'train', 'val', 'test' subset sizes (None if full dataset)
        subset_hashes: Dict with 'train', 'val', 'test' subset hashes
        permutation_hash: Hash of training permutation array
        num_fields: Number of fields (for MSM-Field)
        field_emb_size: Field embedding size
        mask_field_id: MASK_FIELD_ID value
        mask_prob: Mask probability
        determinism_strict: Whether strict determinism is enabled
        
    Returns:
        Run Identity dictionary
    """
    identity = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "hostname": get_hostname(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "model_type": model_type,
        "task": task,
        "seed": seed,
        "dataset_name": dataset_name,
        "subset_sizes": subset_sizes,
        "subset_hashes": subset_hashes,
        "artifacts": fingerprint_artifacts(artifacts_dir),
        "determinism_strict": determinism_strict,
    }
    
    if permutation_hash is not None:
        identity["permutation_hash"] = permutation_hash
    
    if num_fields is not None:
        identity["num_fields"] = num_fields
    if field_emb_size is not None:
        identity["field_emb_size"] = field_emb_size
    if mask_field_id is not None:
        identity["mask_field_id"] = mask_field_id
    if mask_prob is not None:
        identity["mask_prob"] = mask_prob
    
    # Add trainer data (permutation and mask hashes)
    if trainer_data is not None:
        if "train_permutation_hash_16" in trainer_data:
            identity["train_permutation_hash_16"] = trainer_data["train_permutation_hash_16"]
            identity["train_permutation_len"] = trainer_data["train_permutation_len"]
            identity["train_permutation_seed"] = trainer_data["train_permutation_seed"]
        if "train_mask_step0" in trainer_data:
            identity["train_mask_step0"] = trainer_data["train_mask_step0"]
        if "val_mask_step0" in trainer_data:
            identity["val_mask_step0"] = trainer_data["val_mask_step0"]
        if "val_eval_index_step0" in trainer_data:
            identity["val_eval_index_step0"] = trainer_data["val_eval_index_step0"]
    
    return identity


def save_run_identity(identity: dict[str, Any], log_dir: Path) -> Path:
    """Save Run Identity to JSON file.
    
    Args:
        identity: Run Identity dictionary
        log_dir: Log directory where to save
        
    Returns:
        Path to saved JSON file
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    identity_path = log_dir / "run_identity.json"
    
    with open(identity_path, "w") as f:
        json.dump(identity, f, indent=2)
    
    return identity_path


def update_run_identity(log_dir: Path, trainer_data: dict[str, Any]) -> None:
    """Update Run Identity JSON file with trainer data (permutation and mask hashes).
    
    Args:
        log_dir: Log directory where Run Identity JSON is saved
        trainer_data: Dictionary with trainer data from get_run_identity_data()
    """
    identity_path = log_dir / "run_identity.json"
    if not identity_path.exists():
        return  # Run Identity doesn't exist yet
    
    # Load existing identity
    with open(identity_path, "r") as f:
        identity = json.load(f)
    
    # Update with trainer data
    if "train_permutation_hash_16" in trainer_data:
        identity["train_permutation_hash_16"] = trainer_data["train_permutation_hash_16"]
        identity["train_permutation_len"] = trainer_data["train_permutation_len"]
        identity["train_permutation_seed"] = trainer_data["train_permutation_seed"]
    if "train_mask_step0" in trainer_data:
        identity["train_mask_step0"] = trainer_data["train_mask_step0"]
    if "val_mask_step0" in trainer_data:
        identity["val_mask_step0"] = trainer_data["val_mask_step0"]
    if "val_eval_index_step0" in trainer_data:
        identity["val_eval_index_step0"] = trainer_data["val_eval_index_step0"]
    
    # Save updated identity
    with open(identity_path, "w") as f:
        json.dump(identity, f, indent=2)


def print_run_identity(identity: dict[str, Any]) -> None:
    """Print Run Identity in a readable format.
    
    Args:
        identity: Run Identity dictionary
    """
    print("\n" + "=" * 80)
    print("RUN IDENTITY")
    print("=" * 80)
    print(f"Timestamp: {identity['timestamp']}")
    print(f"Git Commit: {identity.get('git_commit', 'N/A')}")
    print(f"Hostname: {identity['hostname']}")
    print(f"Python: {identity['python_version']}")
    print(f"PyTorch: {identity['pytorch_version']}")
    print(f"\nModel: {identity['model_type']}")
    print(f"Task: {identity['task']}")
    print(f"Seed: {identity['seed']}")
    print(f"Dataset: {identity['dataset_name']}")
    
    print(f"\nSubset Sizes:")
    for split, size in identity['subset_sizes'].items():
        print(f"  {split}: {size if size is not None else 'full'}")
    
    print(f"\nSubset Hashes:")
    for split, hash_val in identity['subset_hashes'].items():
        if hash_val:
            print(f"  {split}: {hash_val}")
    
    print(f"\nArtifacts:")
    artifacts = identity['artifacts']
    print(f"  Directory: {artifacts['artifacts_dir']}")
    print(f"  Exists: {artifacts['exists']}")
    if artifacts['exists']:
        print(f"  Config Hash: {artifacts.get('config_hash', 'N/A')}")
        print(f"  Text Tokenizer Hash: {artifacts.get('text_tokenizer_hash', 'N/A')}")
        print(f"  Value Vocab Hash: {artifacts.get('value_vocab_hash', 'N/A')}")
    
    if 'permutation_hash' in identity:
        print(f"\nPermutation Hash: {identity['permutation_hash']}")
    
    if 'train_permutation_hash_16' in identity:
        print(f"\nTrain Permutation:")
        print(f"  hash_16: {identity['train_permutation_hash_16']}")
        print(f"  length: {identity['train_permutation_len']}")
        print(f"  seed: {identity['train_permutation_seed']}")
    
    if 'num_fields' in identity:
        print(f"\nMSM-Field Config:")
        print(f"  num_fields: {identity.get('num_fields')}")
        print(f"  field_emb_size: {identity.get('field_emb_size')}")
        print(f"  MASK_FIELD_ID: {identity.get('mask_field_id')}")
        print(f"  mask_prob: {identity.get('mask_prob')}")
    
    if 'train_mask_step0' in identity:
        train_mask = identity['train_mask_step0']
        print(f"\nTrain Mask (Step 0):")
        print(f"  mask_hash_16: {train_mask.get('mask_hash_16')}")
        print(f"  mask_count: {train_mask.get('mask_count')}")
        print(f"  nonpad_count: {train_mask.get('nonpad_count')}")
        print(f"  mask_ratio: {train_mask.get('mask_ratio', 0):.4f}")
    
    if 'val_mask_step0' in identity:
        val_mask = identity['val_mask_step0']
        print(f"\nVal Mask (Step 0):")
        print(f"  mask_hash_16: {val_mask.get('mask_hash_16')}")
        print(f"  mask_count: {val_mask.get('mask_count')}")
        print(f"  nonpad_count: {val_mask.get('nonpad_count')}")
        print(f"  mask_ratio: {val_mask.get('mask_ratio', 0):.4f}")
    
    print(f"\nDeterminism Strict: {identity.get('determinism_strict', False)}")
    print("=" * 80 + "\n")
