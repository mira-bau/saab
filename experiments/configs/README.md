# Configuration Guide

This directory contains configuration files for experiments. The config-based approach ensures a single source of truth and makes experimentation easy.

## Overview

There are two types of configuration files:

1. **Full Experiment Configs** - Complete configuration including preprocessing, model, training, and task settings (recommended)
2. **Task Configs** - Task-only configuration for backward compatibility

## Full Experiment Configuration (Recommended)

Full experiment configs provide a single source of truth for all experiment settings. Use the `--config` argument when running training:

```bash
poetry run python -m saab_v3.train \
    --dataset-name mydataset \
    --model-type flat \
    --config experiments/configs/examples/stable_training.yaml
```

### Config Structure

Full experiment configs have four main sections:

```yaml
preprocessing:
  vocab_size: 30000
  max_seq_len: 512
  device: "auto"

model:
  d_model: 768
  num_layers: 12
  num_heads: 12
  # ... other model settings

training:
  learning_rate: 1e-6
  batch_size: 16
  lr_schedule: "constant"
  # ... other training settings

task:
  name: classification
  params:
    num_classes: 14
```

### Config Priority

The system uses the following priority order:

1. **`--config`** (full experiment config) - Highest priority, single source of truth
2. **`--task-config`** + code defaults - Backward compatible
3. **Code defaults only** - Fallback

### Examples

- `examples/stable_training.yaml` - Stable training settings (recommended for initial experiments)
- `saab_experiment.yaml` - Complete SAAB experiment example
- `training_default.yaml` - Default training settings
- `model_default.yaml` - Default model settings

### Stable Training Defaults

For stable training, use these recommended settings:

- `learning_rate: 1e-6` (reduced from default 1e-4)
- `lr_schedule: "constant"` (no warmup for stability)
- `max_grad_norm: 0.1` (aggressive gradient clipping)
- `batch_size: 16` (increased for more stable gradients)
- `early_stop_zero_loss_steps: 100` (stop if loss collapses to zero)

See `examples/stable_training.yaml` for a complete example.

### Early Stopping Configuration

The `early_stop_zero_loss_steps` parameter provides a safety mechanism to stop training when the model stops learning:

- **Purpose**: Detects when loss collapses to zero (model has stopped learning)
- **Default**: `null` (disabled)
- **Recommended**: `100` for most experiments
- **How it works**: Stops training if loss is zero (or < 1e-8) for N consecutive steps
- **Warning**: A warning is logged when the streak reaches 50% of the threshold

Example:
```yaml
training:
  early_stop_zero_loss_steps: 100  # Stop after 100 consecutive zero-loss steps
```

To disable early stopping, set it to `null` or omit it from the config.

## Task Configuration (Backward Compatible)

Task configs define only the task head and loss function. Use the `--task-config` argument:

```bash
poetry run python -m saab_v3.train \
    --dataset-name mydataset \
    --model-type flat \
    --task-config experiments/configs/examples/binary_classification.yaml
```

Task heads are created from YAML configuration files. The system validates configs at startup and provides clear error messages if something is wrong.

## Quick Start

1. **Copy a template** from `templates/` directory
2. **Modify parameters** for your experiment
3. **Load and use** in your code:

```python
import yaml
from saab_v3.tasks import create_task_head_from_config

# Load config
with open("experiments/configs/my_task.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create task head
task_head = create_task_head_from_config(config, d_model=768)
```

## Directory Structure

- **`templates/`** - Template configs with extensive inline comments
  - Start here for new experiments
  - Each template explains all parameters
  
- **`examples/`** - Working example configs for common use cases
  - Copy and modify these for quick setup
  - Covers all task types and common configurations

## Task Types

### 1. Classification

**Template:** `templates/classification.yaml`

**Required Parameters:**
- `num_classes`: Number of output classes (must be > 0)

**Optional Parameters:**
- `multi_label`: If true, uses multi-label classification (default: false)
- `hidden_dims`: MLP hidden dimensions (default: null = simple linear)
- `dropout`: Dropout probability (default: 0.1)
- `pooling`: Pooling strategy - "cls", "mean", "max" (default: "cls")

**Examples:**
- `examples/binary_classification.yaml` - Binary classification
- `examples/multi_class_classification.yaml` - Multi-class classification
- `examples/multi_label_classification.yaml` - Multi-label classification
- `examples/classification_with_mlp.yaml` - Classification with MLP head

### 2. Ranking

**Template:** `templates/ranking.yaml`

**Required Parameters:**
- `method`: Ranking method - "dot_product", "cosine", "mlp", "difference"

**Optional Parameters:**
- `hidden_dims`: MLP hidden dimensions (required for "mlp" and "difference")
- `dropout`: Dropout probability (default: 0.1)

**Examples:**
- `examples/ranking_dot_product.yaml` - Dot product method
- `examples/ranking_mlp.yaml` - MLP method

### 3. Regression

**Template:** `templates/regression.yaml`

**Optional Parameters:**
- `num_targets`: Number of target values (default: 1)
- `hidden_dims`: MLP hidden dimensions (default: null = simple linear)
- `dropout`: Dropout probability (default: 0.1)
- `pooling`: Pooling strategy - "cls", "mean", "max" (default: "cls")

**Examples:**
- `examples/regression_single_target.yaml` - Single-target regression
- `examples/regression_multi_target.yaml` - Multi-target regression

### 4. Token Classification

**Template:** `templates/token_classification.yaml`

**Required Parameters:**
- `num_labels`: Number of label classes (must be > 0)

**Optional Parameters:**
- `hidden_dims`: MLP hidden dimensions (default: null = simple linear)
- `dropout`: Dropout probability (default: 0.1)

**Examples:**
- `examples/token_classification.yaml` - Token classification

## Config Structure

All configs follow this structure:

```yaml
task:
  name: <task_type>  # "classification", "ranking", "regression", "token_classification"
  params:
    <task-specific parameters>
```

## Validation

Configs are validated when `create_task_head_from_config()` is called. The system checks:

1. **Structure** - Required keys (`task.name`, `task.params`) are present
2. **Task Name** - Valid task type
3. **Parameters** - Required parameters present, types correct, values in valid ranges
4. **Combinations** - Parameter combinations are valid (e.g., `hidden_dims` required for ranking MLP)

## Error Messages

If validation fails, you'll get clear error messages:

```
KeyError: Classification task requires 'num_classes' parameter
ValueError: num_classes must be a positive integer, got -5
ValueError: Invalid task name: 'classify'. Valid tasks: ['classification', 'ranking', 'regression', 'token_classification']
```

## Common Patterns

### Simple Linear Head (Default)

```yaml
task:
  name: classification
  params:
    num_classes: 10
    # hidden_dims: null (default)
```

### MLP Head

```yaml
task:
  name: classification
  params:
    num_classes: 10
    hidden_dims: [256, 128]  # Two hidden layers
    dropout: 0.1
```

### Multi-label Classification

```yaml
task:
  name: classification
  params:
    num_classes: 10
    multi_label: true
```

### Ranking with MLP

```yaml
task:
  name: ranking
  params:
    method: "mlp"
    hidden_dims: [256, 128]  # Required for MLP method
    dropout: 0.1
```

## Tips

1. **Start with templates** - They have extensive comments explaining each parameter
2. **Use examples** - Copy examples for common use cases
3. **Validate early** - Load config and create task head before training to catch errors
4. **Check error messages** - They point to specific fields and suggest fixes

## Troubleshooting

### "Config must contain 'task' key"
- Make sure your YAML has a `task:` key at the top level

### "Classification task requires 'num_classes' parameter"
- Add `num_classes` to `task.params`

### "Invalid ranking method"
- Check that `method` is one of: "dot_product", "cosine", "mlp", "difference"

### "Ranking method 'mlp' requires 'hidden_dims' parameter"
- MLP and difference methods require `hidden_dims` to be specified

### Negative Loss Warnings

If you see warnings about negative loss values:

1. **Check logit values**: Extreme logit values (>100 or <-100) can cause numerical overflow
2. **Reduce learning rate**: Try lowering `learning_rate` (e.g., from `1e-6` to `5e-7`)
3. **Increase gradient clipping**: Reduce `max_grad_norm` (e.g., from `0.1` to `0.05`)
4. **Check batch size**: Very small batches can lead to unstable gradients

The system automatically clamps logits to `[-50.0, 50.0]` to prevent overflow, but if warnings persist, the above adjustments may help.

### Training Stops with "Loss has been zero"

If training stops with a zero loss error:

1. **This is expected behavior**: The model has stopped learning (loss collapsed to zero)
2. **Check `early_stop_zero_loss_steps`**: This parameter controls when to stop (default: 100 steps)
3. **Review training logs**: Check `warnings.log` for gradient collapse warnings
4. **Possible causes**:
   - Learning rate too high (model overcorrected)
   - Model too small for the task
   - Data preprocessing issues

To disable early stopping, set `early_stop_zero_loss_steps: null` in your config.

## Further Reading

- See `PLAN/TASKS.md` for detailed task head architecture
- See template files for parameter descriptions
- See example files for working configurations

