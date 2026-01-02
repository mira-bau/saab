# Task Heads and Downstream Tasks

## Overview

This document describes the task-specific components that sit on top of the encoder models (Flat, Scratch, SAAB). The task heads convert encoder representations into task-specific outputs (predictions, scores, probabilities).

**Key Principle:** All three encoder models (Flat, Scratch, SAAB) use **identical task head architectures** to ensure fair comparison. The only difference is the encoder itself.

---

## Architecture

### Encoder Output Format

The encoder models produce token-level representations:

- **Shape**: `[batch_size, seq_len, d_model]`
- **Content**: Hidden states for each token in the sequence
- **Usage**: Can be used for both sequence-level and token-level tasks

### Pooling Strategy

**Default: `[CLS]` Token**

- A special `[CLS]` token is prepended to each input sequence
- The encoder output at the `[CLS]` position is used as the sequence representation
- **Shape**: `[batch_size, d_model]`
- **Rationale**: Standard approach (like BERT), learns task-specific representations

**Future Extensions:**
- Mean pooling over all tokens
- Max pooling over all tokens
- Learnable attention-based pooling
- Configurable per task

### Task Head Design

**Base Architecture:**
```
Encoder Output → [CLS] token → Task Head → Task Output
```

**Task Head Complexity:**

1. **Simple (Default):**
   ```
   [CLS] representation → Linear(d_model → output_dim) → Output
   ```
   - Single linear layer
   - Fast, interpretable
   - Good baseline for fair comparison

2. **Configurable (Optional):**
   ```
   [CLS] representation → Dropout → MLP(hidden_dims) → Linear → Output
   ```
   - Configurable via `hidden_dims` parameter
   - `hidden_dims=None`: Simple linear layer
   - `hidden_dims=[256, 128]`: MLP with specified dimensions
   - Optional dropout for regularization

**Design Pattern:**
```python
class TaskHead(ABC):
    def __init__(self, d_model, hidden_dims=None, dropout=0.1):
        self.d_model = d_model
        self.hidden_dims = hidden_dims  # None = simple, [256, 128] = MLP
        self.dropout = dropout
    
    @abstractmethod
    def forward(self, encoder_output, ...):
        # Extract [CLS] token
        # Apply task-specific layers
        # Return task output
        pass
```

---

## Supported Tasks

### 1. Classification

**Types:**
- **Binary Classification**: 2 classes
- **Multi-class Classification**: N classes (mutually exclusive)
- **Multi-label Classification**: Multiple classes per sample (not mutually exclusive)

**Architecture:**
```
Sequence → Encoder → [CLS] → ClassificationHead → Logits → Softmax/Sigmoid → Probabilities
```

**Classification Head:**
- **Input**: `[batch_size, d_model]` (from [CLS] token)
- **Output**: `[batch_size, num_classes]` (logits)
- **Activation**: 
  - Softmax for binary/multi-class
  - Sigmoid for multi-label

**Flexibility:**
- Configurable `num_classes`
- `multi_label` flag to switch between softmax and sigmoid
- Easy to extend to new classification variants

**Usage:**
```python
# Binary/Multi-class
head = ClassificationHead(d_model=768, num_classes=10, multi_label=False)

# Multi-label
head = ClassificationHead(d_model=768, num_classes=10, multi_label=True)
```

---

### 2. Ranking

**Types:**
- **Pairwise Ranking** (Initial): Compare two sequences, determine which is better
- **Listwise Ranking** (Future): Rank a list of sequences
- **Multi-object Ranking** (Future): Rank with multiple objectives

#### Pairwise Ranking

**Architecture:**
```
Sequence A → Encoder → Representation A ┐
                                        ├→ PairwiseRankingHead → Score
Sequence B → Encoder → Representation B ┘
```

**Pairwise Ranking Head:**
- **Input**: Two sequence representations `[batch_size, d_model]` each
- **Output**: Single score (higher = better)
- **Comparison Methods:**
  - Dot product: `score = repr_a @ repr_b`
  - Cosine similarity: `score = cosine(repr_a, repr_b)`
  - MLP: `score = MLP(concat(repr_a, repr_b))`
  - Difference: `score = MLP(repr_a - repr_b)`

**Flexibility:**
- Abstract base class for ranking strategies
- Pairwise implementation (default)
- Easy to add listwise/multi-object ranking later

**Usage:**
```python
head = PairwiseRankingHead(d_model=768, method="dot_product")
score = head(seq_a_repr, seq_b_repr)  # Higher score = seq_a better
```

**Future Extensions:**
- `ListwiseRankingHead`: Rank multiple sequences
- `MultiObjectRankingHead`: Rank with multiple criteria

---

### 3. Regression

**Architecture:**
```
Sequence → Encoder → [CLS] → RegressionHead → Continuous Value
```

**Regression Head:**
- **Input**: `[batch_size, d_model]` (from [CLS] token)
- **Output**: `[batch_size, 1]` or `[batch_size, num_targets]` (continuous values)
- **Activation**: None (linear output)

**Usage:**
```python
head = RegressionHead(d_model=768, num_targets=1)
prediction = head(encoder_output)  # Continuous value(s)
```

---

### 4. Token Classification

**Architecture:**
```
Sequence → Encoder → Token Representations → TokenClassificationHead → Per-token Labels
```

**Token Classification Head:**
- **Input**: `[batch_size, seq_len, d_model]` (all token representations)
- **Output**: `[batch_size, seq_len, num_labels]` (logits per token)
- **Activation**: Softmax per token

**Use Cases:**
- Named Entity Recognition (NER)
- Part-of-Speech Tagging
- Sequence Labeling

**Usage:**
```python
head = TokenClassificationHead(d_model=768, num_labels=10)
logits = head(encoder_output)  # [batch_size, seq_len, num_labels]
```

---

## Loss Functions

### Classification Losses

#### 1. Cross-Entropy Loss (Binary/Multi-class)
- **Formula**: `L = -Σ y_i * log(p_i)`
- **Use Case**: Binary and multi-class classification
- **Implementation**: `torch.nn.CrossEntropyLoss`

#### 2. Binary Cross-Entropy Loss
- **Formula**: `L = -[y * log(p) + (1-y) * log(1-p)]`
- **Use Case**: Binary classification (alternative to cross-entropy)
- **Implementation**: `torch.nn.BCEWithLogitsLoss`

#### 3. Multi-label Cross-Entropy Loss
- **Formula**: `L = -Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]` (per class)
- **Use Case**: Multi-label classification
- **Implementation**: `torch.nn.BCEWithLogitsLoss` with multi-label targets

#### 4. Focal Loss
- **Formula**: `L = -α * (1-p)^γ * log(p)`
- **Use Case**: Handling class imbalance
- **Implementation**: Custom implementation

---

### Ranking Losses

#### 1. Pairwise Ranking Loss (Hinge Loss)
- **Formula**: `L = max(0, margin - (score_positive - score_negative))`
- **Use Case**: Pairwise ranking
- **Implementation**: Custom implementation

#### 2. Pairwise Logistic Loss
- **Formula**: `L = log(1 + exp(-(score_positive - score_negative)))`
- **Use Case**: Pairwise ranking (smooth alternative to hinge)
- **Implementation**: Custom implementation

#### 3. Listwise Ranking Loss (ListNet)
- **Formula**: `L = -Σ P(y_i) * log(P(pred_i))`
- **Use Case**: Listwise ranking (future)
- **Implementation**: Custom implementation (future)

#### 4. Listwise Ranking Loss (ListMLE)
- **Formula**: Maximum likelihood estimation for rankings
- **Use Case**: Listwise ranking (future)
- **Implementation**: Custom implementation (future)

---

### Regression Losses

#### 1. Mean Squared Error (MSE)
- **Formula**: `L = (1/n) * Σ (y_i - ŷ_i)²`
- **Use Case**: Standard regression
- **Implementation**: `torch.nn.MSELoss`

#### 2. Mean Absolute Error (MAE / L1 Loss)
- **Formula**: `L = (1/n) * Σ |y_i - ŷ_i|`
- **Use Case**: Robust regression (less sensitive to outliers)
- **Implementation**: `torch.nn.L1Loss`

#### 3. Huber Loss
- **Formula**: Combines MSE and MAE (quadratic for small errors, linear for large)
- **Use Case**: Robust regression
- **Implementation**: `torch.nn.HuberLoss`

---

### Token Classification Losses

#### 1. Cross-Entropy Loss (Per Token)
- **Formula**: `L = -Σ Σ y_ij * log(p_ij)` (sum over tokens and labels)
- **Use Case**: Token classification
- **Implementation**: `torch.nn.CrossEntropyLoss` with appropriate reshaping

#### 2. CRF Loss (Conditional Random Fields)
- **Formula**: Sequence-level loss considering label transitions
- **Use Case**: Token classification with label dependencies
- **Implementation**: Custom implementation (future)

---

## Training Strategy

### End-to-End Training (Default)

**Approach:**
- Encoder and task head are trained together
- Both updated via backpropagation
- Encoder learns task-relevant representations
- Task head learns task-specific mapping

**Benefits:**
- Encoder adapts to task
- Better task performance
- Standard approach

**Configuration:**
```python
# Default: End-to-end
model = EncoderWithTaskHead(encoder, task_head)
optimizer = torch.optim.Adam(model.parameters())
```

---

### Freeze Encoder (Optional)

**Approach:**
- Encoder weights are frozen (not updated)
- Only task head is trained
- Useful for analysis and transfer learning

**Use Cases:**
- Isolating encoder quality
- Faster training
- Transfer learning scenarios

**Configuration:**
```python
# Freeze encoder, train only task head
for param in encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(task_head.parameters())
```

---

### Training Phases (Optional)

**Phase 1: Pre-training (Optional)**
- Train encoder only (if needed)
- Unsupervised or self-supervised objectives

**Phase 2: Fine-tuning (Primary)**
- Train encoder + task head end-to-end
- Task-specific supervised learning

**Phase 3: Evaluation**
- Encoder + task head frozen
- Evaluate on test set

---

## Evaluation Setup

### Single-Task Evaluation (Primary)

**Approach:**
- One task per experiment
- Same dataset, same train/val/test splits
- Compare Flat/Scratch/SAAB on the same task
- Isolates structural bias effect per task

**Protocol:**
1. **Same Data Splits**: All models use identical train/val/test splits
2. **Same Metrics**: Task-specific evaluation metrics
3. **Multiple Runs**: Report mean ± std across runs (for statistical significance)
4. **Fair Comparison**: Same task head architecture for all encoders

**Example:**
```
Task: Binary Classification
Dataset: MyDataset
Splits: Train (80%), Val (10%), Test (10%)
Models: Flat, Scratch, SAAB
Task Head: Same ClassificationHead for all
Metrics: Accuracy, F1, Precision, Recall
Runs: 5 runs per model
Report: Mean ± Std
```

---

### Multi-Task Evaluation (Future)

**Approach:**
- Multiple tasks on same/different datasets
- Shared encoder, separate task heads
- Compare task transfer effects

**Use Cases:**
- Multi-task learning
- Transfer learning analysis
- Task generalization

**Design:**
```python
# Future: Multi-task setup
encoder = SAABTransformer(...)
task_heads = {
    'classification': ClassificationHead(...),
    'ranking': PairwiseRankingHead(...),
}
```

---

### Evaluation Metrics

#### Classification Metrics
- **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F1 Score**: `2 * (Precision * Recall) / (Precision + Recall)`
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

#### Ranking Metrics
- **NDCG** (Normalized Discounted Cumulative Gain)
- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **Pairwise Accuracy**: Percentage of correctly ranked pairs

#### Regression Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination

#### Token Classification Metrics
- **Token Accuracy**: Per-token accuracy
- **F1 Score**: Per-label F1 scores
- **Exact Match**: Percentage of sequences with all tokens correct

---

## Fair Comparison Requirements

### Critical Invariants

1. **Same Task Head Architecture**: All models (Flat, Scratch, SAAB) use identical task head
   - Same layers, same dimensions
   - Same initialization (same random seed)
   - Same dropout (if applicable)

2. **Same Training Setup**: Identical training conditions
   - Same optimizer, learning rate, batch size
   - Same number of epochs/steps
   - Same data order (same random seed)

3. **Same Evaluation Protocol**: Identical evaluation
   - Same data splits
   - Same metrics
   - Same number of runs

4. **Only Encoder Differs**: The only difference is the encoder model
   - Flat: No structural embeddings
   - Scratch: Structural embeddings (implicit)
   - SAAB: Structural embeddings + attention bias (explicit)

### Implementation Guarantee

```python
# All models use the same task head
task_head = ClassificationHead(d_model=768, num_classes=10)

flat_model = FlatTransformer(...)
scratch_model = ScratchTransformer(...)
saab_model = SAABTransformer(...)

# Same task head for all
flat_with_head = EncoderWithTaskHead(flat_model, task_head)
scratch_with_head = EncoderWithTaskHead(scratch_model, task_head)
saab_with_head = EncoderWithTaskHead(saab_model, task_head)
```

---

## Task Configuration

### Single Source of Truth: Config-Based Approach

**Design Principle:** Config file is the **only** user-facing interface for task specification. This ensures:
- Single source of truth (no multiple ways to specify the same thing)
- Easy experimentation (change config, not code)
- Consistent parameter specification
- Validation at startup

### Architecture

```
User → Config File → Factory (validates & parses) → Direct Instantiation (internal)
```

**Single Source of Truth:** Config file

**Direct instantiation is an internal implementation detail only** (not exposed to users).

---

### User-Facing API

**Single Entry Point:**
```python
from saab.tasks import create_task_head_from_config

# Load config from file
config = load_config("experiments/classification.yaml")
task_head = create_task_head_from_config(config, d_model=768)

# Or pass config dict directly
config = {
    "task": {
        "name": "classification",
        "params": {
            "num_classes": 10,
            "multi_label": False,
            "hidden_dims": None,
            "dropout": 0.1
        }
    }
}
task_head = create_task_head_from_config(config, d_model=768)
```

**No other public API for creating task heads** (avoids multiple sources of truth).

---

### Config Format

**YAML Example (Recommended - supports comments):**
```yaml
task:
  name: "classification"  # Required: "classification", "ranking", "regression", "token_classification"
  params:
    # Task-specific parameters
    num_classes: 10
    multi_label: false
    hidden_dims: null  # null = simple linear layer, [256, 128] = MLP
    dropout: 0.1
```

**JSON Example (No comments supported):**
```json
{
  "task": {
    "name": "ranking",
    "params": {
      "method": "dot_product"
    }
  }
}
```

**Note:** YAML is recommended because it supports inline comments for documentation.

---

### Config Documentation

**Multiple approaches to make configs self-documenting:**

#### 1. Template Config Files with Extensive Comments

**Location:** `configs/templates/` directory

**Example: `configs/templates/classification.yaml`**
```yaml
# ============================================================================
# Classification Task Configuration Template
# ============================================================================
# This config file defines a classification task head.
# Copy this file and modify the parameters for your use case.
#
# Documentation: See PLAN/TASKS.md for detailed parameter descriptions
# ============================================================================

task:
  # Task name - must be one of:
  #   - "classification": Binary/multi-class/multi-label classification
  #   - "ranking": Pairwise ranking
  #   - "regression": Continuous value prediction
  #   - "token_classification": Per-token labeling (NER, POS tagging, etc.)
  name: "classification"
  
  params:
    # ------------------------------------------------------------------------
    # Required Parameters
    # ------------------------------------------------------------------------
    
    # Number of output classes
    # - For binary classification: set to 2
    # - For multi-class: set to number of classes (e.g., 10 for 10 classes)
    # - For multi-label: set to number of possible labels
    # Type: int, Must be: > 0
    num_classes: 10
    
    # ------------------------------------------------------------------------
    # Optional Parameters
    # ------------------------------------------------------------------------
    
    # Whether this is multi-label classification
    # - false: Multi-class (mutually exclusive classes, uses softmax)
    # - true: Multi-label (multiple classes per sample, uses sigmoid)
    # Type: bool, Default: false
    multi_label: false
    
    # Hidden dimensions for task head MLP
    # - null: Simple linear layer (recommended for fair comparison)
    # - [256, 128]: MLP with 2 hidden layers (256 -> 128 -> output)
    # - [512, 256, 128]: MLP with 3 hidden layers
    # Type: null | List[int], Default: null
    # Note: Each dimension must be a positive integer
    hidden_dims: null
    
    # Dropout probability for regularization
    # - Only used if hidden_dims is not null (MLP mode)
    # Type: float, Range: [0, 1), Default: 0.1
    dropout: 0.1
```

**Example: `configs/templates/ranking.yaml`**
```yaml
# ============================================================================
# Ranking Task Configuration Template
# ============================================================================
# This config file defines a pairwise ranking task head.
# The model compares two sequences and determines which is better.
#
# Documentation: See PLAN/TASKS.md for detailed parameter descriptions
# ============================================================================

task:
  name: "ranking"
  
  params:
    # ------------------------------------------------------------------------
    # Required Parameters
    # ------------------------------------------------------------------------
    
    # Comparison method for ranking
    # - "dot_product": Dot product of representations (fast, simple)
    # - "cosine": Cosine similarity (normalized dot product)
    # - "mlp": Multi-layer perceptron (learnable comparison, requires hidden_dims)
    # - "difference": MLP on difference vector (repr_a - repr_b)
    # Type: str, Must be one of: ["dot_product", "cosine", "mlp", "difference"]
    method: "dot_product"
    
    # ------------------------------------------------------------------------
    # Optional Parameters
    # ------------------------------------------------------------------------
    
    # Hidden dimensions for MLP-based methods
    # - Only used if method is "mlp" or "difference"
    # - null: Not applicable for dot_product/cosine methods
    # - [256, 128]: MLP with 2 hidden layers
    # Type: null | List[int], Default: null
    # Note: Required if method is "mlp" or "difference"
    hidden_dims: null
    
    # Dropout probability for regularization
    # - Only used if method is "mlp" or "difference" and hidden_dims is not null
    # Type: float, Range: [0, 1), Default: 0.1
    dropout: 0.1
```

#### 2. Config Schema with Descriptions

**Location:** `tasks/config_schema.py` (enhanced with descriptions)

```python
# tasks/config_schema.py
TASK_SCHEMAS = {
    "classification": {
        "description": "Binary/multi-class/multi-label classification",
        "params": {
            "num_classes": {
                "type": int,
                "required": True,
                "description": "Number of output classes. Must be > 0.",
                "examples": [2, 10, 100],
            },
            "multi_label": {
                "type": bool,
                "required": False,
                "default": False,
                "description": "If true, uses sigmoid for multi-label. If false, uses softmax for multi-class.",
            },
            "hidden_dims": {
                "type": "Optional[List[int]]",
                "required": False,
                "default": None,
                "description": "Hidden dimensions for MLP. null = simple linear layer. [256, 128] = 2-layer MLP.",
                "examples": [None, [256, 128], [512, 256, 128]],
            },
            "dropout": {
                "type": float,
                "required": False,
                "default": 0.1,
                "description": "Dropout probability. Only used if hidden_dims is not null.",
                "range": [0, 1),
            },
        },
    },
    "ranking": {
        "description": "Pairwise ranking - compare two sequences",
        "params": {
            "method": {
                "type": str,
                "required": True,
                "description": "Comparison method for ranking.",
                "choices": ["dot_product", "cosine", "mlp", "difference"],
                "examples": ["dot_product", "mlp"],
            },
            # ... etc
        },
    },
}
```

#### 3. Documentation Generation from Schema

**Function to generate config documentation:**
```python
# tasks/config_schema.py
def generate_config_documentation() -> str:
    """
    Generate markdown documentation from config schemas.
    Can be called to auto-generate docs or print help.
    """
    docs = []
    docs.append("# Task Configuration Documentation\n")
    
    for task_name, schema in TASK_SCHEMAS.items():
        docs.append(f"## {task_name.title()}\n")
        docs.append(f"{schema['description']}\n\n")
        docs.append("### Parameters\n\n")
        
        for param_name, param_spec in schema["params"].items():
            docs.append(f"#### `{param_name}`\n")
            docs.append(f"- **Type**: `{param_spec['type']}`\n")
            docs.append(f"- **Required**: {param_spec['required']}\n")
            if "default" in param_spec:
                docs.append(f"- **Default**: `{param_spec['default']}`\n")
            docs.append(f"- **Description**: {param_spec['description']}\n")
            if "examples" in param_spec:
                docs.append(f"- **Examples**: {param_spec['examples']}\n")
            if "range" in param_spec:
                docs.append(f"- **Range**: {param_spec['range']}\n")
            if "choices" in param_spec:
                docs.append(f"- **Choices**: {param_spec['choices']}\n")
            docs.append("\n")
    
    return "\n".join(docs)

def print_config_help(task_name: str = None):
    """Print help for task configuration."""
    if task_name:
        # Print help for specific task
        if task_name in TASK_SCHEMAS:
            print(generate_task_help(task_name))
        else:
            print(f"Unknown task: {task_name}")
            print(f"Available tasks: {list(TASK_SCHEMAS.keys())}")
    else:
        # Print help for all tasks
        print(generate_config_documentation())
```

#### 4. Example Configs Directory

**Structure:**
```
configs/
├── templates/              # Template configs with extensive comments
│   ├── classification.yaml
│   ├── ranking.yaml
│   ├── regression.yaml
│   └── token_classification.yaml
├── examples/              # Working example configs
│   ├── binary_classification.yaml
│   ├── multi_class_classification.yaml
│   ├── multi_label_classification.yaml
│   ├── ranking_dot_product.yaml
│   └── ranking_mlp.yaml
└── README.md              # Guide on how to use configs
```

#### 5. Config Validation with Helpful Errors

**Enhanced error messages that point to documentation:**
```python
def validate_task_config(config: Dict[str, Any]) -> None:
    """Validate with helpful error messages."""
    try:
        # ... validation logic ...
    except KeyError as e:
        raise KeyError(
            f"Missing required field: {e}. "
            f"See configs/templates/{task_name}.yaml for an example, "
            f"or run: python -m saab.tasks.config_schema --help {task_name}"
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid value: {e}. "
            f"See PLAN/TASKS.md for parameter descriptions, "
            f"or run: python -m saab.tasks.config_schema --help {task_name}"
        )
```

#### 6. CLI Help Command

**Command-line interface for config help:**
```bash
# Show help for all tasks
python -m saab.tasks.config_schema --help

# Show help for specific task
python -m saab.tasks.config_schema --help classification

# Generate example config
python -m saab.tasks.config_schema --example classification > my_config.yaml

# Validate config file
python -m saab.tasks.config_schema --validate my_config.yaml
```

---

### Config Validation at Startup

**Validation occurs when `create_task_head_from_config()` is called:**

1. **Schema Validation:**
   - Required fields present (`task.name`, `task.params`)
   - Task name is valid (exists in registry)
   - Parameter types are correct

2. **Task-Specific Validation:**
   - Classification: `num_classes` must be > 0
   - Ranking: `method` must be valid ("dot_product", "cosine", "mlp", "difference")
   - Regression: `num_targets` must be > 0
   - Token Classification: `num_labels` must be > 0

3. **Value Validation:**
   - `dropout` must be in [0, 1)
   - `hidden_dims` must be None or list of positive integers
   - `multi_label` must be boolean

**Validation Errors:**
- Raise clear, actionable error messages
- Point to specific config fields
- Suggest valid values

---

### Config Schema

**Base Schema:**
```python
{
    "task": {
        "name": str,  # Required: one of ["classification", "ranking", "regression", "token_classification"]
        "params": dict  # Required: task-specific parameters
    }
}
```

**Task-Specific Schemas:**

#### Classification Schema
```python
{
    "task": {
        "name": "classification",
        "params": {
            "num_classes": int,  # Required: > 0
            "multi_label": bool,  # Optional: default False
            "hidden_dims": Optional[List[int]],  # Optional: None or list of positive integers
            "dropout": float  # Optional: [0, 1), default 0.1
        }
    }
}
```

#### Ranking Schema
```python
{
    "task": {
        "name": "ranking",
        "params": {
            "method": str,  # Required: one of ["dot_product", "cosine", "mlp", "difference"]
            "hidden_dims": Optional[List[int]],  # Optional: for MLP method
            "dropout": float  # Optional: [0, 1), default 0.1
        }
    }
}
```

#### Regression Schema
```python
{
    "task": {
        "name": "regression",
        "params": {
            "num_targets": int,  # Optional: default 1, must be > 0
            "hidden_dims": Optional[List[int]],  # Optional: None or list of positive integers
            "dropout": float  # Optional: [0, 1), default 0.1
        }
    }
}
```

#### Token Classification Schema
```python
{
    "task": {
        "name": "token_classification",
        "params": {
            "num_labels": int,  # Required: > 0
            "hidden_dims": Optional[List[int]],  # Optional: None or list of positive integers
            "dropout": float  # Optional: [0, 1), default 0.1
        }
    }
}
```

---

### Implementation

**Factory Function:**
```python
# tasks/factory.py
from typing import Dict, Any
from saab.tasks.config_schema import validate_task_config
from saab.tasks.classification import ClassificationHead
from saab.tasks.ranking import PairwiseRankingHead
from saab.tasks.regression import RegressionHead
from saab.tasks.token_classification import TokenClassificationHead

def create_task_head_from_config(config: Dict[str, Any], d_model: int) -> TaskHead:
    """
    Single entry point for creating task heads.
    Config is the only source of truth.
    
    Args:
        config: Task configuration dict with 'task.name' and 'task.params'
        d_model: Model dimension (from encoder)
    
    Returns:
        TaskHead instance
    
    Raises:
        ValueError: If config is invalid
        KeyError: If required fields are missing
    """
    # Validate config at startup
    validate_task_config(config)
    
    task_name = config["task"]["name"]
    task_params = config["task"]["params"]
    
    # Internal: direct instantiation (not exposed to users)
    task_registry = {
        "classification": _create_classification_head,
        "ranking": _create_ranking_head,
        "regression": _create_regression_head,
        "token_classification": _create_token_classification_head,
    }
    
    if task_name not in task_registry:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Valid tasks: {list(task_registry.keys())}"
        )
    
    return task_registry[task_name](d_model=d_model, **task_params)

# Internal helper functions (not exposed)
def _create_classification_head(d_model: int, **params):
    return ClassificationHead(d_model=d_model, **params)

def _create_ranking_head(d_model: int, **params):
    return PairwiseRankingHead(d_model=d_model, **params)

def _create_regression_head(d_model: int, **params):
    return RegressionHead(d_model=d_model, **params)

def _create_token_classification_head(d_model: int, **params):
    return TokenClassificationHead(d_model=d_model, **params)
```

**Config Validation:**
```python
# tasks/config_schema.py
from typing import Dict, Any, List
from enum import Enum

class TaskName(str, Enum):
    CLASSIFICATION = "classification"
    RANKING = "ranking"
    REGRESSION = "regression"
    TOKEN_CLASSIFICATION = "token_classification"

class RankingMethod(str, Enum):
    DOT_PRODUCT = "dot_product"
    COSINE = "cosine"
    MLP = "mlp"
    DIFFERENCE = "difference"

def validate_task_config(config: Dict[str, Any]) -> None:
    """
    Validate task configuration at startup.
    
    Raises:
        ValueError: If config is invalid
        KeyError: If required fields are missing
    """
    # Validate structure
    if "task" not in config:
        raise KeyError("Config must contain 'task' key")
    
    task_config = config["task"]
    
    if "name" not in task_config:
        raise KeyError("Config must contain 'task.name'")
    
    if "params" not in task_config:
        raise KeyError("Config must contain 'task.params'")
    
    task_name = task_config["name"]
    task_params = task_config["params"]
    
    # Validate task name
    valid_tasks = [t.value for t in TaskName]
    if task_name not in valid_tasks:
        raise ValueError(
            f"Invalid task name: {task_name}. "
            f"Valid tasks: {valid_tasks}"
        )
    
    # Validate task-specific parameters
    if task_name == TaskName.CLASSIFICATION:
        _validate_classification_params(task_params)
    elif task_name == TaskName.RANKING:
        _validate_ranking_params(task_params)
    elif task_name == TaskName.REGRESSION:
        _validate_regression_params(task_params)
    elif task_name == TaskName.TOKEN_CLASSIFICATION:
        _validate_token_classification_params(task_params)

def _validate_classification_params(params: Dict[str, Any]) -> None:
    if "num_classes" not in params:
        raise KeyError("Classification task requires 'num_classes' parameter")
    
    num_classes = params["num_classes"]
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
    
    if "multi_label" in params:
        if not isinstance(params["multi_label"], bool):
            raise ValueError("multi_label must be a boolean")
    
    if "hidden_dims" in params and params["hidden_dims"] is not None:
        if not isinstance(params["hidden_dims"], list):
            raise ValueError("hidden_dims must be None or a list of integers")
        if not all(isinstance(d, int) and d > 0 for d in params["hidden_dims"]):
            raise ValueError("hidden_dims must contain only positive integers")
    
    if "dropout" in params:
        dropout = params["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

def _validate_ranking_params(params: Dict[str, Any]) -> None:
    if "method" not in params:
        raise KeyError("Ranking task requires 'method' parameter")
    
    method = params["method"]
    valid_methods = [m.value for m in RankingMethod]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid ranking method: {method}. "
            f"Valid methods: {valid_methods}"
        )
    
    # Validate hidden_dims and dropout (same as classification)
    if "hidden_dims" in params and params["hidden_dims"] is not None:
        if not isinstance(params["hidden_dims"], list):
            raise ValueError("hidden_dims must be None or a list of integers")
        if not all(isinstance(d, int) and d > 0 for d in params["hidden_dims"]):
            raise ValueError("hidden_dims must contain only positive integers")
    
    if "dropout" in params:
        dropout = params["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

def _validate_regression_params(params: Dict[str, Any]) -> None:
    if "num_targets" in params:
        num_targets = params["num_targets"]
        if not isinstance(num_targets, int) or num_targets <= 0:
            raise ValueError(f"num_targets must be a positive integer, got {num_targets}")
    
    # Validate hidden_dims and dropout (same as classification)
    if "hidden_dims" in params and params["hidden_dims"] is not None:
        if not isinstance(params["hidden_dims"], list):
            raise ValueError("hidden_dims must be None or a list of integers")
        if not all(isinstance(d, int) and d > 0 for d in params["hidden_dims"]):
            raise ValueError("hidden_dims must contain only positive integers")
    
    if "dropout" in params:
        dropout = params["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

def _validate_token_classification_params(params: Dict[str, Any]) -> None:
    if "num_labels" not in params:
        raise KeyError("Token classification task requires 'num_labels' parameter")
    
    num_labels = params["num_labels"]
    if not isinstance(num_labels, int) or num_labels <= 0:
        raise ValueError(f"num_labels must be a positive integer, got {num_labels}")
    
    # Validate hidden_dims and dropout (same as classification)
    if "hidden_dims" in params and params["hidden_dims"] is not None:
        if not isinstance(params["hidden_dims"], list):
            raise ValueError("hidden_dims must be None or a list of integers")
        if not all(isinstance(d, int) and d > 0 for d in params["hidden_dims"]):
            raise ValueError("hidden_dims must contain only positive integers")
    
    if "dropout" in params:
        dropout = params["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
```

---

### Usage Examples

**Example 1: Classification from YAML**
```python
# config.yaml
# task:
#   name: "classification"
#   params:
#     num_classes: 10
#     multi_label: false
#     hidden_dims: null
#     dropout: 0.1

import yaml
from saab.tasks import create_task_head_from_config

with open("config.yaml") as f:
    config = yaml.safe_load(f)

task_head = create_task_head_from_config(config, d_model=768)
# Validation happens automatically at startup
```

**Example 2: Ranking from Dict**
```python
from saab.tasks import create_task_head_from_config

config = {
    "task": {
        "name": "ranking",
        "params": {
            "method": "dot_product"
        }
    }
}

task_head = create_task_head_from_config(config, d_model=768)
# Validation happens automatically at startup
```

**Example 3: Invalid Config (Validation Error)**
```python
config = {
    "task": {
        "name": "classification",
        "params": {
            "num_classes": -5  # Invalid: must be > 0
        }
    }
}

task_head = create_task_head_from_config(config, d_model=768)
# Raises: ValueError: num_classes must be a positive integer, got -5
```

---

### Config Documentation Best Practices

**Recommended approach for maximum clarity:**

1. **Use YAML format** (supports inline comments)
2. **Start from template files** in `configs/templates/` (extensively commented)
3. **Reference schema documentation** when in doubt
4. **Use CLI help** for quick reference: `python -m saab.tasks.config_schema --help <task_name>`
5. **Check example configs** in `configs/examples/` for working examples

**Documentation hierarchy:**
1. **Template configs** (most detailed, inline comments)
2. **Schema documentation** (structured, searchable)
3. **Example configs** (working examples)
4. **CLI help** (quick reference)
5. **This document** (comprehensive reference)

**For users:**
- **New users**: Start with template files, they have extensive inline documentation
- **Quick reference**: Use CLI help command
- **Deep dive**: Read this document (TASKS.md)
- **Examples**: Check `configs/examples/` for working configs

---

### Benefits

1. **Single Source of Truth**: Config file is the only way to specify tasks
2. **Validation at Startup**: Catches errors early, before training starts
3. **Clear Error Messages**: Points to specific config fields and suggests fixes
4. **Easy Experimentation**: Change config, not code
5. **Consistent**: All parameters come from config
6. **Maintainable**: One code path to maintain
7. **Well Documented**: Multiple documentation approaches (templates, schema, examples, CLI)

---

## File Structure

```
tasks/
├── __init__.py
├── base.py                    # Base TaskHead ABC
├── classification.py          # ClassificationHead (internal)
├── ranking.py                 # PairwiseRankingHead (internal)
├── regression.py              # RegressionHead (internal)
├── token_classification.py    # TokenClassificationHead (internal)
├── pooling.py                 # Pooling strategies ([CLS], mean, max, etc.)
├── losses.py                  # Loss functions
├── factory.py                 # create_task_head_from_config() - ONLY public API
└── config_schema.py          # Config validation schema + documentation generation

configs/
├── templates/                 # Template configs with extensive inline comments
│   ├── classification.yaml
│   ├── ranking.yaml
│   ├── regression.yaml
│   └── token_classification.yaml
├── examples/                  # Working example configs
│   ├── binary_classification.yaml
│   ├── multi_class_classification.yaml
│   ├── multi_label_classification.yaml
│   ├── ranking_dot_product.yaml
│   └── ranking_mlp.yaml
└── README.md                  # Guide on how to use configs
```

---

## Usage Examples

### Classification (Config-Based)

```python
import yaml
from saab.models import SAABTransformer
from saab.tasks import create_task_head_from_config

# Load config
with open("configs/classification.yaml") as f:
    config = yaml.safe_load(f)

# Initialize encoder
encoder = SAABTransformer(d_model=768, num_layers=12, ...)

# Create task head from config (validation happens automatically)
task_head = create_task_head_from_config(config, d_model=768)

# Forward pass
encoder_output = encoder(input_ids, attention_mask, ...)  # [batch, seq_len, d_model]
cls_repr = encoder_output[:, 0, :]  # [CLS] token: [batch, d_model]
logits = task_head(cls_repr)  # [batch, num_classes]
```

**Config file (`configs/classification.yaml`):**
```yaml
task:
  name: "classification"
  params:
    num_classes: 10
    multi_label: false
    hidden_dims: null
    dropout: 0.1
```

### Pairwise Ranking (Config-Based)

```python
from saab.models import SAABTransformer
from saab.tasks import create_task_head_from_config

# Config dict
config = {
    "task": {
        "name": "ranking",
        "params": {
            "method": "dot_product"
        }
    }
}

# Initialize encoder
encoder = SAABTransformer(d_model=768, num_layers=12, ...)

# Create task head from config (validation happens automatically)
task_head = create_task_head_from_config(config, d_model=768)

# Forward pass
seq_a_output = encoder(seq_a_ids, seq_a_mask, ...)  # [batch, seq_len, d_model]
seq_b_output = encoder(seq_b_ids, seq_b_mask, ...)  # [batch, seq_len, d_model]

seq_a_repr = seq_a_output[:, 0, :]  # [CLS] token
seq_b_repr = seq_b_output[:, 0, :]  # [CLS] token

score = task_head(seq_a_repr, seq_b_repr)  # [batch] - higher = seq_a better
```

---

## Notes

- **Flexibility First**: Design supports easy extension to new tasks
- **Fair Comparison**: Same task head for all encoders is critical
- **Start Simple**: Default to simple linear layers, add complexity as needed
- **Document Everything**: All tasks/losses documented, implement as needed
- **Pooling Strategy**: `[CLS]` token is default, can be extended later
- **Training Strategy**: End-to-end by default, freezing optional for analysis

---

## Future Enhancements

1. **Additional Pooling Strategies**: Mean, max, attention-based pooling
2. **Listwise Ranking**: Implement listwise ranking heads
3. **Multi-object Ranking**: Ranking with multiple objectives
4. **CRF for Token Classification**: Sequence-level loss with label dependencies
5. **Multi-task Learning**: Shared encoder with multiple task heads
6. **Task-specific Embeddings**: Additional embeddings for specific tasks
7. **Advanced Loss Functions**: Focal loss, label smoothing, etc.

