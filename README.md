# SAAB: Structure-Aware Attention Bias for Transformers

## Overview

Transformers are a powerful architecture, but they still struggle with structured data. They are typically trained as if all tokens are interchangeable, yet many real inputs have known structure: each token belongs to a field (e.g., title or content) and may carry additional structural tags. SAAB makes this structure explicit inside the attention mechanism through an additive bias, similar to how convolutional networks encode pixel locality.

**Key Contribution**: SAAB introduces structure-aware attention bias without altering the Transformer backbone. With the same number of parameters, SAAB outperforms a strong baseline where structure is embedded in the token representations but not explicitly used in attention. When the bias strength (λ) is set to zero, SAAB reduces exactly to standard attention, making it a controlled intervention for studying structural inductive bias.

## Installation

### Prerequisites

- Python 3.12 or higher (but < 3.14.1)
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd saab-v4
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. (Optional) Install tokenizer support for text tokenization:
```bash
poetry install --extras tokenizers
```

4. Verify installation:
```bash
poetry run python -c "import saab_v3; print('SAAB installed successfully')"
```

## Project Structure

```
saab-v4/
├── saab_v3/              # Main code package
│   ├── models/           # Transformer model implementations
│   ├── training/        # Training pipeline and utilities
│   ├── tasks/            # Task heads (MSM-Field, classification, etc.)
│   ├── data/             # Data processing and preprocessing
│   └── config/           # Configuration classes
├── dataset/
│   ├── raw/              # Raw dataset files (CSV format)
│   └── artifacts/        # Preprocessing artifacts (vocabularies, configs)
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs and metrics
└── specs/                # Tests and specifications
```

## Model Types

The project supports two main model variants:

### Baseline

The **baseline** model uses full structural embeddings (token, positional, token-type, field, entity, time) but standard attention mechanism. Structure is encoded only in embeddings; the model must learn to use structure implicitly.

**CLI argument**: `--model scratch`

### SAAB

The **SAAB** (Structure-Aware Attention Bias) model uses the same full structural embeddings as the baseline, plus an additive structural bias in the attention mechanism. The bias encourages attention between tokens that share structural relationships (e.g., same field ID).

**CLI argument**: `--model saab`

**When to use each**:
- Use **baseline** (`scratch`) for controlled comparisons and ablation studies
- Use **SAAB** to study the effect of explicit structural inductive bias in attention

## Dataset Preparation

### Dataset Format

Datasets should be placed in `dataset/raw/{dataset_name}/` and contain CSV files:

- **Required**: `train.csv`, `val.csv`
- **Optional**: `test.csv`

### DBpedia Example

The DBpedia dataset has the following structure:

```csv
label,title,content
12,History Lessons,History Lessons (German: Geschichtsunterricht) is a 1972...
13,The Moon's a Balloon,The Moon's a Balloon is a memoir by British actor...
```

**Fields**:
- `label`: Classification label (integer)
- `title`: Title field (text)
- `content`: Content field (text)

### Field Structure Requirements

The framework automatically extracts structural information from CSV columns:
- Each column (except labels/targets) becomes a **field**
- Field IDs are assigned automatically during preprocessing
- Tokens within the same field share the same field ID

### Text Tokenization

For text-heavy datasets like DBpedia, the framework supports subword tokenization (BPE) for text fields. This is automatically enabled for DBpedia and can be configured via `PreprocessingConfig`.

## Quick Start

Train a SAAB model on DBpedia with MSM-Field task:

```bash
poetry run python -m saab_v3.train \
    --dataset dbpedia \
    --model saab \
    --device cuda \
    --task msm_field
```

Train a baseline model for comparison:

```bash
poetry run python -m saab_v3.train \
    --dataset dbpedia \
    --model scratch \
    --device cuda \
    --task msm_field
```

## Training

### Basic Command

```bash
poetry run python -m saab_v3.train \
    --dataset <dataset_name> \
    --model <model_type> \
    --device <device>
```

### Required Arguments

- `--dataset` (or `--dataset-name`): Name of the dataset directory in `dataset/raw/`
- `--model` (or `--model-type`): Model type (`scratch` for baseline, `saab` for SAAB)
- `--device`: Device to use (`cuda`, `cpu`, `mps`, or `auto`)

### Optional Arguments

- `--task`: Task type (`msm_field`, `classification`, `ranking`, `regression`, `token_classification`). Default: `classification`
- `--max-steps`: Maximum number of training steps (overrides config)
- `--batch-size`: Batch size (overrides config)
- `--seed`: Random seed for reproducibility (default: 42)
- `--mask-prob`: Mask probability for MSM-Field task (default: 0.15)
- `--log-every`: Log metrics every N steps
- `--eval-every`: Evaluate on validation set every N steps
- `--resume`: Path to checkpoint to resume training from
- `--experiment-name`: Custom experiment name (default: `{dataset_name}_{model_type}`)
- `--refit-preprocessor`: Force refit preprocessor even if artifacts exist
- `--determinism-strict`: Enable strict determinism (torch.manual_seed, cudnn.deterministic=True)

### Training Examples

#### MSM-Field Task (Self-Supervised)

Train SAAB on MSM-Field task with custom mask probability:

```bash
poetry run python -m saab_v3.train \
    --dataset dbpedia \
    --model saab \
    --device cuda \
    --task msm_field \
    --mask-prob 0.15 \
    --max-steps 100
```

#### Classification Task

Train for classification with custom batch size:

```bash
poetry run python -m saab_v3.train \
    --dataset dbpedia \
    --model saab \
    --device cuda \
    --task classification \
    --batch-size 64
```

#### Multi-Seed Training

Train with different seeds for reproducibility studies:

```bash
for seed in 0 42 123; do
    poetry run python -m saab_v3.train \
        --dataset dbpedia \
        --model saab \
        --device cuda \
        --task msm_field \
        --seed $seed \
        --experiment-name dbpedia_saab_seed${seed}
done
```

#### Resuming from Checkpoint

Resume training from a saved checkpoint:

```bash
poetry run python -m saab_v3.train \
    --dataset dbpedia \
    --model saab \
    --device cuda \
    --resume checkpoints/dbpedia_saab/checkpoint_epoch_5.pt
```

### Configuration System

All configuration uses Pydantic defaults as a single source of truth. To customize settings, modify the Pydantic config objects in `saab_v3/train.py`:

- **PreprocessingConfig**: Preprocessing settings (vocab sizes, max sequence length, text tokenization)
- **ModelConfig**: Model architecture settings (d_model, num_layers, num_heads, dropout)
- **TrainingConfig**: Training hyperparameters (learning rate, batch size, scheduler, early stopping)
- **TaskConfig**: Task head settings (task-specific parameters)

### Determinism

For reproducible experiments, use the `--determinism-strict` flag:

```bash
poetry run python -m saab_v3.train \
    --dataset dbpedia \
    --model saab \
    --device cuda \
    --determinism-strict \
    --seed 42
```

This enables:
- Fixed random seeds for PyTorch and NumPy
- Deterministic CuDNN operations
- Disabled TF32 for strict determinism

## Evaluation

Evaluate a trained model on test or validation data:

```bash
poetry run python -m saab_v3.evaluate \
    --checkpoint checkpoints/dbpedia_saab/best_model.pt \
    --dataset-name dbpedia \
    --split test \
    --device cuda
```

### Arguments

- `--checkpoint`: Path to checkpoint file (required)
- `--dataset-name`: Name of the dataset directory (required)
- `--split`: Data split to evaluate on (`val` or `test`, default: `test`)
- `--device`: Device to use (`cpu`, `cuda`, `mps`, `auto`)
- `--batch-size`: Batch size for evaluation (default: 64)

### Output

Evaluation results are automatically saved to:
```
dataset/{dataset_name}/evaluation_results_{split}_{experiment_name}.json
```

## Outputs and Artifacts

### Preprocessing Artifacts

Location: `dataset/artifacts/{dataset_name}/`

Contains:
- `config.json`: Preprocessing configuration
- `vocabularies/`: Vocabulary files for tokens and structural tags
- `text_tokenizer.json`: BPE tokenizer (if text tokenization enabled)
- Subset indices (if subset mode enabled)

**Note**: Artifacts are reused across runs for fair comparison. Use `--refit-preprocessor` to regenerate.

### Checkpoints

Location: `checkpoints/{experiment_name}/`

Contains:
- `best_model.pt`: Best model based on validation metric
- `checkpoint_epoch_{N}.pt`: Checkpoints at each epoch
- `training_config.json`: Training configuration snapshot

### Logs and Metrics

Location: `logs/{experiment_name}/`

Contains:
- `metrics.jsonl`: Training metrics (loss, attention diagnostics, etc.)
- `run_identity.json`: Run identity for reproducibility tracking
- Training logs and diagnostics

### Run Identity Tracking

Each training run generates a `run_identity.json` that tracks:
- Model type, task, seed, dataset
- Subset hashes (if subset mode enabled)
- Permutation and mask hashes for determinism verification
- Configuration snapshots

## Utilities and Scripts

### Compare Baseline vs SAAB

Compare training results between baseline and SAAB:

```bash
python compare_scratch_saab.py
```

This script loads metrics from log files and compares:
- Early loss curve slopes
- Loss variance
- Attention diagnostics (entropy, same-field mass)

### Generate Figures

Generate publication-quality figures from results:

```bash
python generate_figures.py
```

Reads from `results_table.csv` and generates comparison plots.

### Parse Logs

Parse training logs and extract metrics:

```bash
python parse_logs.py
```

Extracts metrics from `logs.txt` into a wide table format.

### Additional Tools

Tools in the `tools/` directory:
- `compare_checkpoints.py`: Compare checkpoint contents
- `inspect_checkpoint_predictions.py`: Inspect model predictions

## Configuration

### Configuration Classes

The project uses Pydantic for type-safe configuration:

#### PreprocessingConfig

Controls data preprocessing:
- `vocab_size`: Token vocabulary size
- `max_seq_len`: Maximum sequence length
- `use_text_tokenizer`: Enable BPE tokenization for text fields
- `text_fields`: List of fields to tokenize with BPE
- `field_boundary_token`: Add field boundary tokens

#### ModelConfig

Controls model architecture:
- `d_model`: Model dimension
- `num_layers`: Number of encoder layers
- `num_heads`: Number of attention heads
- `dropout`: Dropout probability
- `lambda_bias`: SAAB bias strength (λ) - only for SAAB model

#### TrainingConfig

Controls training hyperparameters:
- `learning_rate`: Initial learning rate
- `batch_size`: Batch size
- `num_epochs`: Number of epochs (or `max_steps` for step-based training)
- `lr_schedule`: Learning rate schedule (`linear_warmup_cosine`, etc.)
- `warmup_steps` or `warmup_ratio`: Warmup configuration
- `early_stopping_patience`: Early stopping patience
- `subset_size_train/val/test`: Subset sizes for fast experiments

#### Task Configs

Task-specific configurations:
- `MSMFieldTaskConfig`: Mask probability, number of fields
- `ClassificationTaskConfig`: Number of classes, pooling method
- `RankingTaskConfig`: Ranking method, margin
- `RegressionTaskConfig`: Output activation
- `TokenClassificationTaskConfig`: Number of label classes

### Customizing Configuration

To customize configuration, modify the config objects in `saab_v3/train.py` where they are instantiated. The configuration system uses Pydantic defaults, so changes to the default values in the config classes will affect all runs unless overridden via command-line arguments.

## Tasks

### MSM-Field (Masked Structure Modeling)

**Task Type**: `msm_field`

A self-supervised task where field IDs are masked and the model predicts them. This isolates structure learning from downstream-task confounds.

**Configuration**:
- `mask_prob`: Probability of masking a field ID (default: 0.15)
- `num_fields`: Number of field classes (automatically derived from data)

**Example**:
```bash
poetry run python -m saab_v3.train \
    --dataset dbpedia \
    --model saab \
    --device cuda \
    --task msm_field \
    --mask-prob 0.15
```

### Classification

**Task Type**: `classification`

Supervised classification task. Requires a `label` column in CSV files.

**Configuration**:
- `num_classes`: Number of classes
- `multi_label`: Whether to use multi-label classification
- `pooling`: Pooling method (`mean`, `cls`, `max`)

### Other Supported Tasks

- **Ranking** (`ranking`): Pairwise ranking task
- **Regression** (`regression`): Continuous value prediction
- **Token Classification** (`token_classification`): Per-token classification

## Architecture Overview

The SAAB architecture adds a structural bias to the standard Transformer attention mechanism:

```
Attention Score = QK^T / √d + λ · B_struct(tag_i, tag_j)
```

Where:
- `QK^T / √d`: Standard scaled dot-product attention
- `λ`: Bias strength parameter
- `B_struct(tag_i, tag_j)`: Structural bias matrix based on tag relationships

For a detailed architecture diagram, see `architecture.mmd`.

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Dataset directory not found`

**Solution**: Ensure your dataset is in `dataset/raw/{dataset_name}/` with `train.csv` and `val.csv` files.

**Issue**: `CUDA out of memory`

**Solution**: Reduce batch size using `--batch-size` or enable gradient accumulation (configured in `TrainingConfig`).

**Issue**: Artifacts exist but are incomplete

**Solution**: Use `--refit-preprocessor` to regenerate artifacts, or delete the artifacts directory and rerun.

**Issue**: Non-deterministic results across runs

**Solution**: Use `--determinism-strict` flag and ensure the same seed is used.

### Getting Help

- Check log files in `logs/{experiment_name}/` for detailed error messages
- Verify dataset format matches expected structure
- Ensure all dependencies are installed: `poetry install`


