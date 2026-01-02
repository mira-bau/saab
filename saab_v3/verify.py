"""
Verification script for preprocessing pipeline.

Usage:
    poetry run python -m saab_v3.verify_preprocessing --dataset-name <dataset_name>

This script checks:
- Input data files exist and are readable
- Artifact files are created correctly
- Vocabulary and config files are valid
- File structure matches expectations
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def verify_input_data(
    dataset_dir: Path, sample_rows: int = 5
) -> tuple[bool, list[str]]:
    """Verify input data files exist and are readable.

    Args:
        dataset_dir: Path to dataset directory
        sample_rows: Number of rows to sample for verification

    Returns:
        Tuple of (success: bool, messages: list[str])
    """
    messages = []
    success = True

    print("\n" + "=" * 60)
    print("INPUT DATA VERIFICATION")
    print("=" * 60)

    # Check train.csv
    train_path = dataset_dir / "train.csv"
    if not train_path.exists():
        messages.append(f"❌ Training file not found: {train_path}")
        success = False
    else:
        try:
            # Sample first few rows
            df_train = pd.read_csv(train_path, nrows=sample_rows)
            messages.append(f"✅ Training file exists: {train_path}")
            messages.append(f"   - Columns: {list(df_train.columns)}")
            messages.append(f"   - Sampled {len(df_train)} rows")
            if len(df_train) > 0:
                messages.append(f"   - First row sample: {dict(df_train.iloc[0])}")
        except Exception as e:
            messages.append(f"❌ Error reading training file: {e}")
            success = False

    # Check val.csv
    val_path = dataset_dir / "val.csv"
    if not val_path.exists():
        messages.append(f"❌ Validation file not found: {val_path}")
        success = False
    else:
        try:
            df_val = pd.read_csv(val_path, nrows=sample_rows)
            messages.append(f"✅ Validation file exists: {val_path}")
            messages.append(f"   - Columns: {list(df_val.columns)}")
            messages.append(f"   - Sampled {len(df_val)} rows")
        except Exception as e:
            messages.append(f"❌ Error reading validation file: {e}")
            success = False

    # Check test.csv (optional)
    test_path = dataset_dir / "test.csv"
    if test_path.exists():
        try:
            df_test = pd.read_csv(test_path, nrows=sample_rows)
            messages.append(f"✅ Test file exists (optional): {test_path}")
            messages.append(f"   - Sampled {len(df_test)} rows")
        except Exception as e:
            messages.append(f"⚠️  Warning: Error reading test file: {e}")

    for msg in messages:
        print(msg)

    return success, messages


def verify_artifacts(artifacts_dir: Path) -> tuple[bool, list[str]]:
    """Verify artifact files exist and are valid.

    Args:
        artifacts_dir: Path to artifacts directory

    Returns:
        Tuple of (success: bool, messages: list[str])
    """
    messages = []
    success = True

    print("\n" + "=" * 60)
    print("ARTIFACTS VERIFICATION")
    print("=" * 60)

    # Check artifacts directory exists
    if not artifacts_dir.exists():
        messages.append(f"❌ Artifacts directory not found: {artifacts_dir}")
        return False, messages

    messages.append(f"✅ Artifacts directory exists: {artifacts_dir}")

    # Check config.json
    config_path = artifacts_dir / "config.json"
    if not config_path.exists():
        messages.append(f"❌ Config file not found: {config_path}")
        success = False
    else:
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            messages.append(f"✅ Config file exists and is valid JSON")
            messages.append(f"   - Fields: {list(config_data.keys())}")
            messages.append(f"   - Sample values:")
            for key, value in list(config_data.items())[:3]:
                messages.append(f"     {key}: {value}")
        except json.JSONDecodeError as e:
            messages.append(f"❌ Config file is not valid JSON: {e}")
            success = False
        except Exception as e:
            messages.append(f"❌ Error reading config file: {e}")
            success = False

    # Check vocabularies directory
    vocab_dir = artifacts_dir / "vocabularies"
    if not vocab_dir.exists():
        messages.append(f"❌ Vocabularies directory not found: {vocab_dir}")
        success = False
    else:
        messages.append(f"✅ Vocabularies directory exists: {vocab_dir}")

        # Check value vocabulary
        value_vocab_path = vocab_dir / "value_vocab.json"
        if not value_vocab_path.exists():
            messages.append(f"❌ Value vocabulary not found: {value_vocab_path}")
            success = False
        else:
            try:
                with open(value_vocab_path, "r") as f:
                    value_vocab = json.load(f)
                messages.append(f"✅ Value vocabulary exists and is valid JSON")
                messages.append(
                    f"   - Has token_to_idx: {'token_to_idx' in value_vocab}"
                )
                messages.append(
                    f"   - Has idx_to_token: {'idx_to_token' in value_vocab}"
                )
                if "token_to_idx" in value_vocab:
                    sample_tokens = list(value_vocab["token_to_idx"].items())[:5]
                    messages.append(f"   - Sample tokens: {sample_tokens}")
            except Exception as e:
                messages.append(f"❌ Error reading value vocabulary: {e}")
                success = False

        # Check tag vocabularies
        tag_vocabs_dir = vocab_dir / "tag_vocabs"
        if not tag_vocabs_dir.exists():
            messages.append(
                f"❌ Tag vocabularies directory not found: {tag_vocabs_dir}"
            )
            success = False
        else:
            messages.append(f"✅ Tag vocabularies directory exists: {tag_vocabs_dir}")

            required_tag_types = [
                "field",
                "entity",
                "time",
                "edge",
                "role",
                "token_type",
            ]
            for tag_type in required_tag_types:
                tag_vocab_path = tag_vocabs_dir / f"{tag_type}.json"
                if not tag_vocab_path.exists():
                    messages.append(
                        f"❌ {tag_type} vocabulary not found: {tag_vocab_path}"
                    )
                    success = False
                else:
                    try:
                        with open(tag_vocab_path, "r") as f:
                            tag_vocab = json.load(f)
                        vocab_size = len(tag_vocab.get("token_to_idx", {}))
                        messages.append(
                            f"✅ {tag_type} vocabulary exists ({vocab_size} tokens)"
                        )
                    except Exception as e:
                        messages.append(f"❌ Error reading {tag_type} vocabulary: {e}")
                        success = False

    for msg in messages:
        print(msg)

    return success, messages


def verify_paths(dataset_name: str) -> tuple[bool, list[str]]:
    """Verify all paths are correct.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple of (success: bool, messages: list[str])
    """
    messages = []
    success = True

    print("\n" + "=" * 60)
    print("PATH VERIFICATION")
    print("=" * 60)

    # Expected paths
    script_path = Path(__file__)
    base_dir = script_path.parent  # saab_v3/
    raw_dir = base_dir / "dataset" / "raw" / dataset_name
    artifacts_dir = base_dir / "dataset" / "artifacts" / dataset_name

    messages.append(f"Base directory: {base_dir}")
    messages.append(f"Expected raw data path: {raw_dir}")
    messages.append(f"Expected artifacts path: {artifacts_dir}")

    # Verify raw data path
    if raw_dir.exists():
        messages.append(f"✅ Raw data directory exists: {raw_dir}")
    else:
        messages.append(f"❌ Raw data directory not found: {raw_dir}")
        success = False

    # Verify artifacts path
    if artifacts_dir.exists():
        messages.append(f"✅ Artifacts directory exists: {artifacts_dir}")
    else:
        messages.append(f"❌ Artifacts directory not found: {artifacts_dir}")
        success = False

    for msg in messages:
        print(msg)

    return success, messages


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(
        description="Verify preprocessing pipeline for a dataset"
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
        "--sample-rows",
        type=int,
        default=5,
        help="Number of rows to sample from input files (default: 5)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name

    print("=" * 60)
    print(f"PREPROCESSING VERIFICATION: {dataset_name}")
    print("=" * 60)

    # Determine paths
    script_path = Path(__file__)
    base_dir = script_path.parent  # saab_v3/
    raw_dir = base_dir / "dataset" / "raw" / dataset_name
    artifacts_dir = base_dir / "dataset" / "artifacts" / dataset_name

    all_success = True
    all_messages = []

    # Run all checks
    path_success, path_messages = verify_paths(dataset_name)
    all_success = all_success and path_success
    all_messages.extend(path_messages)

    input_success, input_messages = verify_input_data(
        raw_dir, sample_rows=args.sample_rows
    )
    all_success = all_success and input_success
    all_messages.extend(input_messages)

    artifacts_success, artifacts_messages = verify_artifacts(artifacts_dir)
    all_success = all_success and artifacts_success
    all_messages.extend(artifacts_messages)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_success:
        print("✅ All checks passed! Preprocessing pipeline is working correctly.")
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
