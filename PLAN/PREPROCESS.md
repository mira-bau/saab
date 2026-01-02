# Data Preprocessing Pipeline

## Overview

This document describes the data preprocessing pipeline for SAAB Transformer models. The pipeline converts heterogeneous structured data (Tables, JSON, Graphs) into a canonical tokenized format with structure tags that all models can consume.

## Pipeline Architecture

```
Input (any format)
    ↓
[Format Detector] → Identify format type
    ↓
[Structural Extractor] → Extract primitives → Raw tokens with structure tags
    ↓
[Token Normalizer] → Normalize values (optional)
    ↓
[Tokenizer] → Convert values to token IDs
    ↓
[Tag Encoder] → Convert symbolic tags to indices
    ↓
[Canonical Format] → Standardized tensor format
    ↓
[Batcher] → Batch sequences with padding
```

---

## Core Data Structures

### StructureTag

Symbolic structure tag for a token, representing structural primitives.

```python
@dataclass
class StructureTag:
    """Symbolic structure tag for a token"""
    field: Optional[str] = None      # e.g., "name", "price"
    entity: Optional[str] = None     # e.g., "user_123", "order_456"
    time: Optional[str] = None       # e.g., "2023-Q1", "2023-01-15"
    edge: Optional[str] = None       # e.g., "parent_of", "contains"
    role: Optional[str] = None       # e.g., "primary_key", "foreign_key"
    token_type: Optional[str] = None  # e.g., "text", "number", "date"
```

**Benefits:**
- Type-safe representation
- Clear structure
- Easy to extend
- Self-documenting

### Token

Single token with value and structure tags.

```python
@dataclass
class Token:
    """Single token with value and structure tags"""
    value: str                    # Raw value: "John", "42.5", "2023-01-15"
    structure_tag: StructureTag   # Structural metadata
    position: int                 # Position in sequence (0-indexed)
```

### TokenizedSequence

Canonical format: tokenized sequence with structure tags.

```python
@dataclass
class TokenizedSequence:
    """Canonical format: tokenized sequence with structure tags"""
    tokens: List[Token]
    sequence_id: Optional[str] = None  # For tracking/debugging
```

---

## Structural Extractors

### Design: Plugin Architecture

Format-specific extractors implement a common interface to extract structural primitives from different input formats.

### Base Interface

```python
class StructuralExtractor(ABC):
    @abstractmethod
    def extract(self, data: Any) -> List[Token]:
        """Extract tokens with structure tags from input data"""
        pass
    
    @abstractmethod
    def can_handle(self, data: Any) -> bool:
        """Check if this extractor can handle the input format"""
        pass
```

### Format-Specific Extractors

#### TableExtractor

Extracts from tabular data (pandas DataFrame, CSV, etc.).

**Extracted Primitives:**
- **Fields**: Column names
- **Entities**: Row identifiers or row indices
- **Relationships**: Foreign key relationships (if schema provided)
- **Roles**: Primary keys, foreign keys (if schema provided)

**Token Ordering:** Row-major (row by row, left to right)

#### JSONExtractor

Extracts from nested JSON objects/arrays.

**Extracted Primitives:**
- **Fields**: Object keys
- **Entities**: Object instances
- **Hierarchy**: Path-based nesting (e.g., "user.address.city")
- **Token Types**: Inferred from value types

**Token Ordering:** Depth-first traversal

#### GraphExtractor

Extracts from graph structures (NetworkX, DGL, etc.).

**Extracted Primitives:**
- **Entities**: Node identifiers
- **Relationships**: Edge types and connections
- **Roles**: Node/edge roles (if provided)
- **Fields**: Node/edge attributes

**Token Ordering:** Configurable (BFS, DFS, or custom)

---

## Tokenization Strategy

### Two-Level Tokenization

#### Level 1: Value Tokenization

Converts raw values ("John", "42.5") to token IDs.

**Features:**
- Vocabulary-based (learned or fixed)
- OOV handling with `[UNK]` token
- Optional subword tokenization for long values
- Special tokens: `[PAD]`, `[MASK]`, `[CLS]`, `[SEP]`

**Implementation:**
```python
class ValueTokenizer:
    def __init__(self, vocab_size: int = 30000):
        self.vocab = Vocabulary()
        self.vocab_size = vocab_size
    
    def encode(self, value: str) -> int:
        """Convert value string to token ID"""
        pass
    
    def decode(self, token_id: int) -> str:
        """Convert token ID back to value string"""
        pass
```

#### Level 2: Structure Tag Encoding

Converts symbolic tags ("field:name") to tag indices.

**Features:**
- Separate vocabularies per tag type
- **Dataset-scoped vocabularies** to prevent cross-dataset leakage
- Special tokens: `[PAD]`, `[NONE]`, `[UNK]` for each tag type
- `[NONE]` used for missing/optional tags

**Implementation:**
```python
class TagEncoder:
    """Encodes symbolic tags to indices, preserves symbols for SAAB"""
    
    def __init__(self):
        self.tag_vocabs = {
            'field': Vocabulary(),
            'entity': Vocabulary(),
            'time': Vocabulary(),
            'edge': Vocabulary(),
            'role': Vocabulary(),
            'token_type': Vocabulary(),
        }
    
    def encode(self, tag: StructureTag) -> EncodedTag:
        """Convert StructureTag to indices, preserve original symbols"""
        return EncodedTag(
            field_idx=self.tag_vocabs['field'][tag.field] if tag.field else None,
            entity_idx=self.tag_vocabs['entity'][tag.entity] if tag.entity else None,
            time_idx=self.tag_vocabs['time'][tag.time] if tag.time else None,
            edge_idx=self.tag_vocabs['edge'][tag.edge] if tag.edge else None,
            role_idx=self.tag_vocabs['role'][tag.role] if tag.role else None,
            token_type_idx=self.tag_vocabs['token_type'][tag.token_type] if tag.token_type else None,
            # Keep original symbols for SAAB bias computation
            original_tag=tag
        )
```

---

## Handling Missing/Optional Tags

### Strategy: Special Token `[NONE]`

**Approach:** Use `[NONE]` token for missing tags.

**Benefits:**
- Simpler implementation
- Consistent tensor shapes
- Model learns to ignore `[NONE]` tags
- No conditional logic needed in model

**Example:**
- Token has `field="name"` but no `entity` → `entity_idx = [NONE]`
- Token has no structure tags → all tag indices = `[NONE]`

---

## Batching Strategy

### Dynamic Padding with Attention Masks

**Approach:** Pad sequences to max length in batch (dynamic batching).

**Benefits:**
- Efficient memory usage
- No wasted computation on excessive padding
- Attention masks handle padding correctly

### Batch Data Structure

```python
@dataclass
class Batch:
    """Batched sequences ready for model input"""
    # Token values
    token_ids: torch.Tensor          # [batch_size, seq_len]
    attention_mask: torch.Tensor     # [batch_size, seq_len] (1 = valid, 0 = pad)
    
    # Structure tag indices
    field_ids: torch.Tensor          # [batch_size, seq_len]
    entity_ids: torch.Tensor         # [batch_size, seq_len]
    time_ids: torch.Tensor           # [batch_size, seq_len]
    edge_ids: Optional[torch.Tensor] # [batch_size, seq_len] or None
    role_ids: Optional[torch.Tensor] # [batch_size, seq_len] or None
    token_type_ids: torch.Tensor     # [batch_size, seq_len]
    
    # Metadata
    sequence_lengths: List[int]      # Actual lengths (for unpacking)
    sequence_ids: Optional[List[str]] # For tracking/debugging
```

### Padding Strategy

1. **Dynamic Padding:** Pad to max length in current batch
2. **Attention Masks:** Use masks to ignore padding tokens
3. **Optional Bucketing:** Group sequences by similar length for efficiency

**Considerations:**
- Sequence length is **configurable** (max_seq_len parameter)
- Sequences longer than max_seq_len are truncated or split
- Padding token ID: `[PAD]` (typically 0)

---

## Tag Encoding for SAAB

### Hybrid Approach: Indices for Embeddings, Symbols for Bias

**Strategy:**
- **Embeddings:** Use encoded indices (efficient lookup)
- **SAAB Bias:** Use original symbolic tags (flexible, no vocabulary constraints)

**Rationale:**
- Embeddings need fast lookup → indices are efficient
- SAAB bias computation is flexible → symbols allow arbitrary relationships
- No vocabulary size constraints for bias computation

**Implementation:**
```python
@dataclass
class EncodedTag:
    """Encoded tag with both indices and original symbols"""
    # Indices for embeddings
    field_idx: Optional[int] = None
    entity_idx: Optional[int] = None
    time_idx: Optional[int] = None
    edge_idx: Optional[int] = None
    role_idx: Optional[int] = None
    token_type_idx: Optional[int] = None
    
    # Original symbols for SAAB bias computation
    original_tag: StructureTag
```

---

## Data Validation

### Validation at Each Stage

**Purpose:** Catch errors early, ensure data quality.

**Validation Points:**

1. **After Extraction:**
   - At least one tag type present per token?
   - Valid tag values?
   - Non-empty sequences?

2. **After Tokenization:**
   - All values tokenized successfully?
   - No invalid token IDs?
   - Sequence length within limits?

3. **After Encoding:**
   - All tags encoded successfully?
   - Valid tag indices?
   - Consistent sequence lengths?

4. **After Batching:**
   - Consistent tensor shapes?
   - Valid indices in all tensors?
   - Attention masks correct?

**Implementation:**
```python
class DataValidator:
    def validate_structure_tags(self, tags: List[StructureTag]) -> bool:
        """Validate structure tags"""
        pass
    
    def validate_tokenized_sequence(self, seq: TokenizedSequence) -> bool:
        """Validate tokenized sequence"""
        pass
    
    def validate_batch(self, batch: Batch) -> bool:
        """Validate batch"""
        pass
```

---

## Efficiency and Performance

### Design Goals

1. **Speed:** Fast preprocessing for training/inference
2. **Memory:** Efficient memory usage, especially for large datasets
3. **Scalability:** Handle large datasets without memory issues

### Optimizations

#### 1. Lazy Loading (Optional)

**Decision:** Not required initially, but architecture supports it.

**Rationale:**
- Focus on efficiency and speed first
- Can add lazy loading later if needed
- Preprocessing can be done offline/cached

**Future Enhancement:**
- Streaming data loader for very large datasets
- Caching preprocessed sequences
- Parallel preprocessing

#### 2. Batch Processing

- Process multiple sequences in parallel
- Vectorized operations where possible
- Efficient tensor operations

#### 3. Caching

- Cache preprocessed sequences to disk
- Avoid re-processing same data
- Fast loading for training

#### 4. Device-Agnostic Design

**MPS/GPU Support:**
- All tensors use PyTorch (device-agnostic)
- Support MPS (Metal Performance Shaders) on Mac
- Easy switching to CUDA GPU if available
- Automatic device detection and fallback

**Implementation:**
```python
def get_device():
    """Get best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
```

---

## Supported Formats

### Initial Support: All Formats

1. **Tables**
   - Pandas DataFrame
   - CSV files
   - Excel files
   - Relational database tables

2. **JSON**
   - Nested objects
   - Arrays
   - Mixed structures

3. **Graphs**
   - NetworkX graphs
   - DGL graphs
   - Custom graph formats

### Extensibility

New formats can be added by implementing the `StructuralExtractor` interface.

---

## File Structure

```
data/
├── __init__.py
├── structures.py          # Data classes (Token, StructureTag, etc.)
├── extractors/
│   ├── __init__.py
│   ├── base.py           # StructuralExtractor ABC
│   ├── table.py          # TableExtractor
│   ├── json.py           # JSONExtractor
│   └── graph.py          # GraphExtractor
├── tokenizer.py          # Value tokenization
├── tag_encoder.py        # Structure tag encoding
├── batcher.py            # Batching logic
├── validator.py          # Data validation
└── pipeline.py           # End-to-end pipeline
```

---

## Key Design Decisions

### Decision 1: Tokenization Granularity

**Choice:** Start with cell-level tokenization (one token per value)

**Rationale:**
- Simpler implementation
- Matches structured data semantics
- Can add subword tokenization later if needed

### Decision 2: Tag Vocabulary Size

**Choice:** Build vocab from training data, with UNK handling

**Rationale:**
- Adapts to data distribution
- Handles OOV tags gracefully
- Configurable vocabulary size

**Important:** Structural tag vocabularies are **dataset-scoped** to prevent cross-dataset leakage. Each dataset builds its own vocabulary from its training data only.

### Decision 3: Sequence Ordering

**Choice:** Format-specific default, configurable

- **Tables:** Row-major
- **JSON:** Depth-first
- **Graphs:** Configurable (BFS/DFS/custom)

### Decision 4: Edge/Relationship Representation

**Choice:** Store edge tags on both tokens

**Rationale:**
- Symmetric representation
- Easy to compute pairwise relationships
- Works well with SAAB bias computation

### Decision 5: Sequence Length

**Choice:** Configurable (max_seq_len parameter)

**Rationale:**
- Flexible for different use cases
- Can be tuned based on data characteristics
- Sequences longer than max are truncated or split

---

## Usage Example

```python
# Initialize pipeline
extractor = TableExtractor()
tokenizer = ValueTokenizer(vocab_size=30000)
tag_encoder = TagEncoder()
batcher = Batcher(max_seq_len=512)

# Process data
tokens = extractor.extract(table_data)
tokenized = tokenizer.encode(tokens)
encoded = tag_encoder.encode(tokenized)
batch = batcher.batch([encoded])

# Move to device (MPS/GPU/CPU)
device = get_device()
batch = batch.to(device)
```

---

## Testing

### Sample Data

Sample data will be provided later for testing and validation.

### Test Coverage

- Format-specific extractors
- Tokenization accuracy
- Tag encoding correctness
- Batching consistency
- Validation logic
- Device compatibility (MPS/GPU/CPU)

---

## Future Enhancements

1. **Streaming/Lazy Loading:** For very large datasets
2. **Parallel Preprocessing:** Multi-process extraction
3. **Subword Tokenization:** For long text values
4. **Advanced Caching:** Smart caching strategies
5. **Format Auto-Detection:** Automatic format identification
6. **Schema Inference:** Automatic schema detection from data

---

## Notes

- All preprocessing should be deterministic (for reproducibility)
- Support for both training and inference pipelines
- Efficient memory usage for large datasets
- Device-agnostic design (MPS/GPU/CPU support)
- **Structural tag vocabularies are dataset-scoped** to prevent cross-dataset leakage
- **SAAB bias magnitude is normalized** so it does not dominate dot-product attention

