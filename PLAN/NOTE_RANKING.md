# Ranking Tasks: Understanding and Implementation Notes

## Overview

Ranking tasks are fundamentally different from classification, regression, and token classification. Instead of predicting a label or value for a single input, ranking tasks compare **pairs of sequences** to determine which one is "better" according to some criterion.

---

## What is Ranking?

### Conceptual Understanding

**Ranking** answers the question: "Given two items, which one is better?"

Examples:
- **Search Ranking**: Which document is more relevant to a query?
- **Recommendation**: Which product is more likely to be purchased?
- **Quality Assessment**: Which text is higher quality?
- **Preference Learning**: Which option does the user prefer?

### Key Difference from Other Tasks

| Task Type | Input | Output | Question |
|----------|-------|--------|----------|
| Classification | 1 sequence | Class label | "What category is this?" |
| Regression | 1 sequence | Continuous value | "What is the value?" |
| Token Classification | 1 sequence | Label per token | "What is each token?" |
| **Ranking** | **2 sequences** | **Comparison score** | **"Which is better?"** |

---

## Pairwise Ranking Architecture

### How It Works

1. **Two Sequences**: We have sequence A and sequence B
2. **Encode Both**: Both sequences go through the same encoder
3. **Pool Representations**: Extract sequence-level representations (using [CLS] token)
4. **Compare**: Use a comparison function to produce a score
5. **Interpret Score**: Higher score = sequence A is better than sequence B

### Flow Diagram

```
Sequence A → Encoder → [CLS] representation A ──┐
                                                 ├─→ Comparison Function → Score
Sequence B → Encoder → [CLS] representation B ──┘
```

### Implementation Details

**In our codebase:**

1. **Dataset** (`StructuredDataset`):
   - Loads pairs: `(sequence_a, sequence_b, label)`
   - Label: `1` = seq_a better, `-1` = seq_b better (or `0`/`1` binary)

2. **Batcher** (`Batcher`):
   - Creates batches with `_b` fields for sequence B
   - `token_ids`, `attention_mask`, etc. for sequence A
   - `token_ids_b`, `attention_mask_b`, etc. for sequence B

3. **Trainer** (`Trainer._forward_ranking`):
   - Extracts batch A and batch B
   - Encodes both through the model
   - Pools to get representations (uses [CLS] token at position 0)
   - Passes both representations to ranking head

4. **Ranking Head** (`PairwiseRankingHead`):
   - Takes TWO representations (not encoder output)
   - Compares them using one of four methods
   - Returns a score

---

## Comparison Methods

The ranking head supports four methods for comparing sequences:

### 1. Dot Product (`dot_product`)

**Formula**: `score = seq_a · seq_b` (element-wise product, then sum)

**Characteristics**:
- Simple and fast
- No learnable parameters
- Measures alignment between vectors
- Sensitive to magnitude

**Use Case**: When sequences are already well-normalized or magnitude matters

### 2. Cosine Similarity (`cosine`)

**Formula**: `score = cosine(seq_a, seq_b) = (seq_a · seq_b) / (||seq_a|| * ||seq_b||)`

**Characteristics**:
- Normalizes by vector magnitude
- Measures angle/direction similarity
- Range: [-1, 1]
- No learnable parameters

**Use Case**: When you care about direction, not magnitude

### 3. MLP on Concatenation (`mlp`)

**Formula**: `score = MLP([seq_a; seq_b])`

**Characteristics**:
- Learns complex interactions
- Requires `hidden_dims` parameter
- Most expressive but needs more data
- Can capture non-linear relationships

**Use Case**: When relationships are complex and you have enough data

### 4. MLP on Difference (`difference`)

**Formula**: `score = MLP(seq_a - seq_b)`

**Characteristics**:
- Learns from the difference vector
- More parameter-efficient than concatenation
- Still expressive
- Requires `hidden_dims` parameter

**Use Case**: When differences are more informative than absolute values

---

## Loss Functions

### Pairwise Hinge Loss

**Formula**: `loss = max(0, margin - label * score)`

- `label = 1`: seq_a should be better → want `score > margin`
- `label = -1`: seq_b should be better → want `score < -margin`
- Zero loss when margin is satisfied

**Parameters**:
- `margin`: Minimum score difference (default: 1.0)

### Pairwise Logistic Loss

**Formula**: `loss = log(1 + exp(-label * score))`

- Smooth approximation of hinge loss
- Always non-zero (provides gradients)
- More stable for optimization

### Pairwise Margin Loss

**Formula**: Similar to hinge but with different margin handling

---

## Data Format Requirements

### CSV Format

Ranking tasks require a specific CSV format:

```csv
sequence_a,sequence_b,label
"{""title"": ""Document A""}","{""content"": ""Document B""}",1
"{""title"": ""Query""}","{""content"": ""Result 1""}",-1
```

**Columns**:
- `sequence_a`: First sequence (JSON string or dict)
- `sequence_b`: Second sequence (JSON string or dict)
- `label`: `1` = seq_a better, `-1` = seq_b better

### Data Structure

Each sequence can be:
- **Dict/JSON**: `{"title": "text", "content": "more text"}`
- **DataFrame**: Single-row DataFrame
- The preprocessor converts these to `TokenizedSequence` objects

### Label Interpretation

- `label = 1`: Sequence A is better than Sequence B
- `label = -1`: Sequence B is better than Sequence A
- `label = 0`: Sometimes used for "equal" (less common)

---

## Data Preparation Challenges

### Problem: Converting Single-Sequence Data to Pairs

Most datasets have single sequences, not pairs. You need to create pairs.

### Pairing Strategies

#### 1. Same-Row Pairing (Current Script)
- Use two columns from the same row
- Example: `title` vs `content`
- **Pros**: Simple, preserves row-level relationships
- **Cons**: May not be meaningful ranking pairs

#### 2. Consecutive Row Pairing
- Pair row i with row i+1
- **Pros**: Creates actual comparisons
- **Cons**: Order-dependent, may not be meaningful

#### 3. Random Pairing
- Randomly pair rows
- **Pros**: Creates diverse pairs
- **Cons**: Labels are arbitrary (need ground truth)

#### 4. All Pairs (for small datasets)
- Create all possible pairs
- **Pros**: Exhaustive
- **Cons**: O(n²) growth, only feasible for small datasets

#### 5. Ground Truth Pairing
- Use actual ranking data (e.g., search logs, user preferences)
- **Pros**: Real labels
- **Cons**: Requires labeled ranking data

### Label Generation

When creating pairs artificially, you need labels:

1. **Use Original Labels**: If original data has quality scores, use them
2. **Arbitrary**: For testing, use fixed labels (e.g., always 1)
3. **Heuristic**: Use domain knowledge (e.g., longer = better)
4. **External Source**: Use human judgments, click data, etc.

---

## Use Cases and Examples

### 1. Search Ranking

**Goal**: Rank search results by relevance

**Data**:
- `sequence_a`: Query
- `sequence_b`: Document
- `label`: 1 if document is relevant, -1 if not

**Example**:
```
Query: "machine learning"
Doc 1: "Introduction to machine learning algorithms" → label: 1
Doc 2: "Cooking recipes" → label: -1
```

### 2. Quality Assessment

**Goal**: Determine which text is higher quality

**Data**:
- `sequence_a`: High-quality text
- `sequence_b`: Lower-quality text
- `label`: 1 (seq_a is better)

### 3. Recommendation

**Goal**: Which product is more likely to be purchased?

**Data**:
- `sequence_a`: Product A description
- `sequence_b`: Product B description
- `label`: Based on purchase history or ratings

### 4. Preference Learning

**Goal**: Learn user preferences

**Data**:
- `sequence_a`: Option A
- `sequence_b`: Option B
- `label`: User's preference (from clicks, ratings, etc.)

---

## Implementation Flow in Our Codebase

### 1. Data Loading

```python
dataset = StructuredDataset(
    data="ranking_data.csv",
    preprocessor=preprocessor,
    task_type="ranking"  # Key: specifies ranking mode
)
```

**What happens**:
- Loads CSV with `sequence_a`, `sequence_b`, `label` columns
- Stores raw data (doesn't encode immediately)
- On `__getitem__`, encodes both sequences separately

### 2. Batching

```python
batch = batcher.batch(ranking_pairs, task_type="ranking")
```

**What happens**:
- Creates `Batch` object with:
  - `token_ids`, `attention_mask`, etc. (for sequence A)
  - `token_ids_b`, `attention_mask_b`, etc. (for sequence B)
  - `labels` (ranking labels)

### 3. Forward Pass

```python
# In Trainer._forward_ranking()
batch_a = extract_batch_a(batch)  # Sequence A only
batch_b = extract_batch_b(batch)  # Sequence B only

outputs_a = model(batch_a)  # [batch, seq_len, d_model]
outputs_b = model(batch_b)  # [batch, seq_len, d_model]

repr_a = outputs_a[:, 0, :]  # [CLS] token → [batch, d_model]
repr_b = outputs_b[:, 0, :]  # [CLS] token → [batch, d_model]

scores = task_head(repr_a, repr_b)  # [batch]
```

### 4. Loss Computation

```python
loss = loss_fn(scores, labels)  # Pairwise ranking loss
```

---

## Configuration

### Task Config Example

```yaml
task:
  name: ranking
  params:
    method: "dot_product"  # or "cosine", "mlp", "difference"
    # For mlp/difference:
    # hidden_dims: [256, 128]
    # dropout: 0.1
```

### Loss Config

Loss function is created automatically from task params:
- `method`: Used to select loss type (hinge, logistic, margin)
- `margin`: For hinge/margin losses (default: 1.0)

---

## Key Insights

### Why Ranking is Complex

1. **Data Format**: Requires pairs, not single sequences
2. **Data Preparation**: Need to create meaningful pairs from single-sequence data
3. **Label Generation**: Need ground truth for which is better
4. **Architecture**: Different from other tasks (two inputs, comparison function)
5. **Evaluation**: Need ranking metrics (NDCG, MAP, etc.) not just accuracy

### When to Use Ranking

- You have **relative preferences** (A > B) but not absolute scores
- You want to **compare items** rather than classify them
- You have **pairwise comparison data** (e.g., user clicks, A/B test results)
- You're doing **information retrieval** or **recommendation**

### When NOT to Use Ranking

- You have **absolute labels** (use classification/regression instead)
- You only have **single sequences** without comparison data
- You want to **predict properties** of individual items

---

## Future Considerations

### Potential Improvements

1. **Listwise Ranking**: Rank entire lists, not just pairs
2. **Pointwise Ranking**: Predict relevance scores, then rank
3. **Learning to Rank**: More sophisticated ranking algorithms
4. **Multi-Objective Ranking**: Optimize multiple criteria
5. **Interactive Ranking**: Learn from user feedback

### Evaluation Metrics

Currently missing (for future):
- **NDCG** (Normalized Discounted Cumulative Gain)
- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **Pairwise Accuracy**: % of correctly ordered pairs

---

## Summary

Ranking tasks are fundamentally about **comparison**, not prediction. They require:
- **Two sequences** per sample
- **Pairwise labels** (which is better)
- **Comparison functions** (dot product, cosine, MLP, difference)
- **Pairwise loss functions** (hinge, logistic, margin)

The complexity comes from:
- Data preparation (creating pairs)
- Label generation (determining which is better)
- Understanding when ranking is appropriate vs. other tasks

For now, focus on understanding the pairwise ranking approach before exploring more complex ranking methods.

