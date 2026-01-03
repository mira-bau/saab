# Task Configuration Guide

This directory contains configuration files for specifying task heads. The config-based approach ensures a single source of truth and makes experimentation easy.

## Overview

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

## Further Reading

- See `PLAN/TASKS.md` for detailed task head architecture
- See template files for parameter descriptions
- See example files for working configurations

