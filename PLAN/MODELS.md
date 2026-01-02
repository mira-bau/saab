# SAAB: Structure-Aware Attention Bias

## Project Goal

Build and evaluate a general-purpose Transformer encoder for heterogeneous structured data, analogous to BERT for text. The key idea is to control how structure is exposed to the model and demonstrate that **injecting structure explicitly into attention improves learning over implicit approaches**.

## Overview

This project implements and compares three Transformer models to isolate the effect of structural inductive bias:

1. **Flat Transformer** (weak baseline)
2. **Scratch Transformer** (strong baseline)
3. **SAAB Transformer** (proposed method)

All models share the same core architecture and training hyperparameters:
- **Architecture**: Same `d_model`, `num_layers`, `num_heads`, FFN dimensions
- **Training**: Same optimizer, learning rate schedule, batch size, training steps/epochs
- **Initialization**: Same random seed

The only differences are in how structural information is incorporated.

---

## Model Descriptions

### 1. Flat Transformer (Weak Baseline)

**Purpose:** Minimal backbone baseline demonstrating that a generic Transformer alone is insufficient for structured data.

**Important Note:** The Flat Transformer is included only to demonstrate that structure-agnostic sequence modeling is insufficient for structured data. It serves as a **lower bound**, not a competitor. The primary comparison is between Scratch and SAAB.

**Architecture:**
- Standard Transformer encoder
- Input: flat token sequence

**Embeddings:**
- Token embedding
- Positional embedding only
- **No token types, no fields, no entities, no time, no relations**
- Structure is completely ignored

**Attention:**
- Standard: `softmax(QK^T / √d)`

---

### 2. Scratch Transformer (Strong Baseline)

**Purpose:** Strong, fair baseline showing what can be achieved without modifying attention. Structure is encoded only in embeddings; the model must learn to use structure implicitly.

**Architecture:**
- Same Transformer architecture as Flat (depth, width, heads, FFN, optimizer, loss)
- Input: same token sequence

**Embeddings:**
- Token embedding
- Positional embedding
- **Token type embeddings**
- **Field embeddings**
- **Entity embeddings**
- **Time bucket embeddings**

**Attention:**
- Standard: `softmax(QK^T / √d)`
- **Attention mechanism is unchanged**
- Model must learn how to use structure implicitly through embeddings

**Key Point:** Structure is encoded **only in embeddings**, not in attention.

---

### 3. SAAB Transformer (Proposed Method)

**Purpose:** Test whether explicit structural inductive bias improves learning. Isolate the effect of structure-aware attention.

**Architecture:**
- Identical to Scratch Transformer (depth, width, heads, FFN, optimizer, loss)

**Embeddings:**
- **Identical to Scratch Transformer**
  - Token embedding
  - Positional embedding
  - Token type embeddings
  - Field embeddings
  - Entity embeddings
  - Time bucket embeddings
- **All structural information available to SAAB is already available to Scratch via embeddings**
- **SAAB does NOT introduce new structural information**

**Attention:**
- Modified with structural bias: `softmax((QK^T / √d) + λ · B_struct) V`
- `B_struct(i, j)`: scalar bias derived from symbolic structural relations between tokens `i` and `j`
- `λ`: scalar gate controlling bias strength (can be fixed or learned)
- **SAAB bias magnitude is normalized** so it does not dominate dot-product attention

**Key Distinction:**
- **Same embeddings as Scratch** → affects token representations
- **Structural bias in attention** → affects how tokens attend to each other
- **SAAB differs only in how structure influences attention, not in what information is provided**
- This introduces an **explicit structural inductive bias** in attention through a new *pathway*

**Critical Invariant:**
- When `λ = 0`: SAAB is **bitwise-equivalent** to Scratch
- When `λ > 0`: **Only attention behavior changes** (no extra embedding info, no architectural branching)

---

## Key Distinctions

### SAAB is a Mechanism, Not a Fixed Formula

**SAAB Definition:**
- Structural information contributes an **additive bias** to attention logits **before softmax**
- Formally: `Attention = softmax((QK^T / √d) + λ · B_struct) V`
- `B_struct(i, j)` is **any scalar bias** derived from symbolic structural relations (field, entity, role, time, edge, etc.)

**SAAB Does NOT Require:**
- Dot-product bias
- Relation embeddings
- Symmetric bias
- Learned α weights
- Any specific functional form

**Implementation Note:**
- Any concrete formula for `B_struct` (e.g., dot products of relation embeddings) is an **implementation choice**, not part of SAAB's definition
- The mechanism is flexible and can be implemented in various ways

### Structure in Embeddings ≠ Structure in Attention

**Scratch Transformer:**
- Structure encoded **in input embeddings** (token type, field, entity, time, positional)
- Standard attention mechanism
- Model must **learn to use structure implicitly** via embeddings

**SAAB Transformer:**
- Uses **the same embeddings** as Scratch (receives the same structural information)
- **All structural information available to SAAB is already available to Scratch via embeddings**
- Symbolic structure **also affects attention directly** through `B_struct`
- **SAAB differs only in how structure influences attention, not in what information is provided**
- This introduces an **explicit structural inductive bias** through a new *pathway* (additive bias in attention)

**Critical Distinction:**
- **Embeddings** → affect token representations (same for both Scratch and SAAB)
- **SAAB** → affects how tokens attend to each other (new pathway for the same structural information)

---

## Input Format

**Approach: Structural Primitives, Not Format Categories**

Instead of categorizing by input format (Tables, JSON, Graphs, etc.), we focus on **structural primitives** that can appear in any format. This approach is scalable and avoids format-specific edge cases.

### Core Structural Primitives

1. **Fields/Attributes**: Columns, keys, properties (e.g., "name", "date", "price")
2. **Entities/Records**: Rows, objects, nodes, instances (e.g., "user_123", "order_456")
3. **Relationships/Edges**: Foreign keys, links, references, connections (e.g., "parent_of", "contains")
4. **Hierarchy/Nesting**: Parent-child, containment, paths
5. **Temporal Ordering**: Timestamps, sequences, time buckets (e.g., "2023-Q1", "2023-01")
6. **Roles**: Primary keys, foreign keys, labels, types (e.g., "primary_key", "foreign_key")

### Canonical Representation: Tokenized Sequence with Structure Tags

**Pipeline:**
```
Any Input Format → Structural Extractor → Canonical Tokenized Format
```

The structural extractor identifies the primitives present in the data and produces tokens with appropriate structure tags.

**Each token carries:**
- **Value**: The actual data (e.g., "John", "2023-01-15", "42.5")
- **Structure Tags**: Symbolic descriptors derived from structural primitives
  - `field_tag`: Field identifier (from Fields/Attributes primitive)
  - `entity_tag`: Entity identifier (from Entities/Records primitive)
  - `time_tag`: Time bucket (from Temporal Ordering primitive)
  - `edge_tag`: Relationship type (from Relationships/Edges primitive)
  - `role_tag`: Role identifier (from Roles primitive)

**Examples of Format → Primitives → Tokens:**
- **Table**: Extract fields (columns), entities (rows), relationships (FKs) → tokens with field/entity/edge tags
- **JSON**: Extract fields (keys), entities (objects), hierarchy (paths) → tokens with field/entity tags
- **Graph**: Extract entities (nodes), relationships (edges), roles → tokens with entity/edge/role tags
- **Time Series**: Extract fields (series), temporal (timestamps), entities (sensors) → tokens with field/time/entity tags

**Key Benefit:** New formats only require a structural extractor that identifies primitives. The model remains format-agnostic and only sees the canonical tokenized format with structure tags.

---

## Training Strategy

**Approach: Train from Scratch (Option 1)**

All three models are trained under **identical conditions** for fair comparison:

**Architecture Hyperparameters (identical across all models):**
- `d_model`: Same model dimension
- `num_layers`: Same number of Transformer layers
- `num_heads`: Same number of attention heads
- `ffn_dim`: Same feed-forward network dimension
- All other architectural parameters

**Training Hyperparameters (identical across all models):**
- **Optimizer**: Same optimizer (e.g., Adam, AdamW) with same parameters
- **Learning rate schedule**: Same initial LR, same schedule (warmup, decay, etc.)
- **Batch size**: Same batch size
- **Training steps/epochs**: Same number of steps or epochs
- **Data order**: Same data order (same random seed for data shuffling)
- **Random seed**: Same random seed for model initialization
- **Compute budget**: Same computational resources

**Exceptions (only for SAAB):**
- `λ` (bias strength): SAAB-specific parameter. Start with `λ=1.0`, then ablate (0.0, 0.5, 1.0, 2.0)
- Any `B_struct`-specific parameters (if applicable)
- When `λ=0`: SAAB should be bitwise-equivalent to Scratch

**Purpose:**
- Fair comparison that isolates architecture differences
- Ensures any performance differences are due to structural inductive bias, not training variance

---

## Experimental Design

### Model Comparison

| Model | Embeddings | Attention | Structure |
|-------|------------|-----------|-----------|
| **Flat** | Token + Position | Standard | None |
| **Scratch** | Token + Position + Type + Field + Entity + Time | Standard | Implicit (in embeddings) |
| **SAAB** | Token + Position + Type + Field + Entity + Time | Standard + `B_struct` | Explicit (in attention) |

### Key Invariants

1. **Architecture Equivalence**: All models share the same Transformer architecture (`d_model`, `num_layers`, `num_heads`, FFN dimensions)
2. **Training Equivalence**: All models trained with identical hyperparameters (optimizer, LR schedule, batch size, steps/epochs, random seed)
3. **Embedding Equivalence**: SAAB and Scratch use identical embeddings
4. **Information Equivalence**: All structural information available to SAAB is already available to Scratch via embeddings
5. **Bitwise Equivalence**: When `λ=0`, SAAB = Scratch
6. **Fair Comparison**: All models trained under identical conditions to isolate the effect of structural inductive bias

### Additional Safeguards

1. **Dataset-Scoped Vocabularies**: Structural tag vocabularies are dataset-scoped to prevent cross-dataset leakage
2. **Normalized Bias Magnitude**: SAAB bias magnitude is normalized so it does not dominate dot-product attention

### Evaluation Metrics

- Task-specific performance (accuracy, F1, etc.)
- Attention pattern analysis (does SAAB attend more to structurally related tokens?)
- Ablation studies (which structural components matter most?)
- Training dynamics (convergence speed, stability)

---

## Project Structure

```
saab-v3/
├── README.md                 # This file
├── models/                   # Model implementations
│   ├── flat_transformer.py
│   ├── scratch_transformer.py
│   └── saab_transformer.py
├── data/                     # Data processing
│   ├── tokenizer.py
│   └── structure_tags.py
├── training/                 # Training scripts
│   └── train.py
├── evaluation/               # Evaluation scripts
│   └── evaluate.py
└── experiments/              # Experiment configurations
    └── configs/
```

---

## References

- BERT: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- This work extends the Transformer architecture to structured data with explicit structural inductive bias

---

## Notes

- SAAB is a **mechanism**, not a specific formula
- The distinction between implicit (embeddings) and explicit (attention) structure is critical
- **All structural information available to SAAB is already available to Scratch via embeddings**
- **SAAB differs only in how structure influences attention, not in what information is provided**
- Code must preserve the bitwise-equivalence guarantee when `λ=0`
- Structural tag vocabularies are dataset-scoped to prevent cross-dataset leakage
- SAAB bias magnitude is normalized so it does not dominate dot-product attention

