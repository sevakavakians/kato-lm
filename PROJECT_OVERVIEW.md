# KATO Hierarchical Concept Learning - Project Overview

## Project Purpose

This project implements **hierarchical concept learning** using multiple KATO nodes to learn abstract representations at different levels of granularity. The system learns patterns in a bottom-up hierarchy where each level's learned patterns become symbolic inputs for the next level, enabling the discovery of increasingly abstract conceptual structures in text data.

## Core Philosophy

**Hierarchical Abstraction Through Pattern Names**

The fundamental insight is that learned patterns can be represented symbolically and used as inputs to higher-level learning systems. When KATO learns a pattern, it returns a unique pattern name (e.g., `PTRN|7f3a2b1c...`). These pattern names become the "vocabulary" for the next level up in the hierarchy.

This creates a cascade of abstraction:
- **Level 0 (Sentences)**: Learns patterns in token sequences → produces sentence pattern names
- **Level 1 (Paragraphs)**: Learns patterns in sentence pattern sequences → produces paragraph pattern names
- **Level 2 (Chapters)**: Learns patterns in paragraph pattern sequences → produces chapter pattern names
- **Level 3 (Books)**: Learns patterns in chapter pattern sequences → produces book pattern names
- **Level N**: Arbitrary depth supported

## Key Concepts

### 1. Single-Pass Hierarchical Training

**NOT multiple iterations.** Pattern names flow up the hierarchy in real-time as structural boundaries complete:

```
Text Sample → Tokenize Sentence → node0.observe_sequence() → node0.learn()
                                                                    ↓
                                                         sentence_pattern_name
                                                                    ↓
                                                         node1.observe(pattern_name)
                                                                    ↓
When paragraph complete → node1.learn() → paragraph_pattern_name → node2.observe()
                                                                         ↓
When chapter complete → node2.learn() → chapter_pattern_name → node3.observe()
                                                                    ↓
When book complete → node3.learn() → book_pattern_name
```

**Structural boundaries trigger learning:**
- Sentence complete → node0 learns
- Paragraph complete (after N sentences) → node1 learns
- Chapter complete (after M paragraphs) → node2 learns
- Book complete (after K chapters) → node3 learns

### 2. Delimiter-Based Segmentation

Text is segmented by linguistic/structural boundaries:
- **sentence**: Full sentences tokenized and learned
- **paragraph**: Groups of sentences
- **chapter**: Groups of paragraphs
- **book**: Groups of chapters

**Important:** Each segment is tokenized using transformers (GPT-2, BERT, RoBERTa, etc.) before being sent to KATO.

### 3. Pattern Names as Symbolic Representations

KATO returns deterministic SHA1-based pattern names:
```
PTRN|7f3a2b1c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a
```

These names:
- Are deterministic (same pattern structure → same name)
- Have frequency counters (how many times seen)
- Carry emotives and metadata (optional contextual information)
- Become first-class symbols for higher levels

### 4. Pattern Storage and Analysis

Patterns are stored in KATO's ClickHouse + Redis backend with frequency counts.

**Note:** Post-training pattern analysis tools are currently being migrated from MongoDB to ClickHouse. Analysis utilities (frequency histograms, cleanup, visualization) will be available once the migration is complete. See `tools/kato_storage/` for the new implementation.

### 5. Arbitrary Node Depth

The system is **not hardcoded to 4 levels**. You can create hierarchies of any depth:

```python
learner = HierarchicalConceptLearner(
    num_nodes=10,  # 10-level hierarchy
    tokenizer_name="gpt2"
)
```

Each level uses the same KATO learning mechanism, just operating on different granularities of input.

### 6. Token Chunking Strategy

**Why Fixed-Length Chunking Instead of Sentence Segmentation?**

The system uses **fixed-length token chunks** (default: 15 tokens) rather than sentence boundaries for node0 learning. This design choice is fundamental to scalability and compression.

#### The Deduplication Argument

**Sentence-based approach:**
```
Sample 1: "The researchers found that machine learning improves accuracy."
Sample 2: "The researchers discovered new methods for deep learning."
Sample 3: "The researchers analyzed data from multiple sources."

→ 3 unique sentence patterns in node0 (minimal reuse)
→ node0 KB size ≈ corpus size
```

**Chunk-based approach (N=15):**
```
Sample 1: ["The researchers found that machine", "learning improves accuracy ."]
Sample 2: ["The researchers discovered new methods", "for deep learning ."]
Sample 3: ["The researchers analyzed data from", "multiple sources ."]

→ Common phrase "The researchers" appears in multiple chunks
→ Frequency counters increment for repeated chunks
→ node0 KB size << corpus size (Zipfian distribution emerges)
```

**Result:** Sentences are mostly unique; chunks repeat frequently.

#### The Hierarchical Composition Argument

Even though node0 chunks have no semantic awareness, meaning emerges through hierarchical composition:

| Level | Chunk Size | Token Coverage | Semantic Granularity |
|-------|-----------|----------------|---------------------|
| **node0** | 15 tokens | 15 tokens | Phrase fragments (no semantics) |
| **node1** | 15 node0 patterns | 15 × 15 = **225 tokens** | ~2-3 sentences worth |
| **node2** | 15 node1 patterns | 15 × 225 = **3,375 tokens** | ~2-3 paragraphs worth |
| **node3** | 15 node2 patterns | 15 × 3,375 = **50,625 tokens** | Full articles/chapters |

**Key Insight:** node0 operates like pixels in an image—individually meaningless, but when composed hierarchically, higher levels build semantic understanding.

#### Configuration

```python
# Create segmenter with custom chunk size
segmenter = CorpusSegmenter(
    tokenizer_name="gpt2",
    chunk_size=15  # Default: 15, recommended: 10-25
)

# Or override per method
book = segmenter.segment_book(text, chunk_size=20)
```

#### Advantages Over Sentence Segmentation

✅ **Scalability**: Deduplication keeps node0 KB manageable
✅ **Robustness**: No fragile boundary detection heuristics
✅ **Simplicity**: Arithmetic splitting always works
✅ **Compression**: Zipfian frequency distribution emerges naturally
✅ **Tokenizer-agnostic**: Works identically for GPT-2, BERT, LLaMA, etc.

#### The Philosophy

Think of the hierarchy like visual processing:
- **Pixels** (node0 chunks): No semantic meaning alone
- **Edges** (node1 patterns): Simple local patterns emerge
- **Shapes** (node2 patterns): More complex structures
- **Objects** (node3 patterns): Full semantic understanding

Semantics don't need to exist at the bottom layer—they **emerge** from composition.

### 7. Hierarchy Sizing Guide

**How to Choose the Right Number of Levels and Chunk Sizes**

The optimal hierarchy configuration depends on your training data characteristics. Mismatched configurations lead to underutilized levels and degraded pattern quality.

#### Key Principle: Match Receptive Fields to Data Scale

Each level's **receptive field** (total token coverage) should align with semantic boundaries in your data:

```
node0: Individual chunks (5-10 tokens) - phrases, word groups
node1: Chunk sequences (50-100 tokens) - sentences, clauses
node2: Higher sequences (300-1000 tokens) - paragraphs, sections
node3: Top sequences (2000-5000 tokens) - articles, chapters
```

#### Receptive Field Calculation

For uniform `chunk_size=N`:
```
node0 receptive field: N tokens
node1 receptive field: N × N tokens
node2 receptive field: N × N × N tokens
node3 receptive field: N × N × N × N tokens
```

#### Recommended Configurations by Dataset

**WikiText (Wikipedia articles, 500-2000 tokens typical):**
```python
# RECOMMENDED: 4 levels, chunk_size=8
nodes = [
    HierarchicalNode('node0', chunk_size=8),  # 8 tokens
    HierarchicalNode('node1', chunk_size=8),  # 64 tokens
    HierarchicalNode('node2', chunk_size=8),  # 512 tokens
    HierarchicalNode('node3', chunk_size=8),  # 4,096 tokens
]
# Rationale: Exponential growth (8→64→512→4K) perfectly covers article scales
# node3 captures full articles without exceeding typical document length
```

**C4/RefinedWeb (Web pages, 300-3000 tokens typical):**
```python
# RECOMMENDED: 4 levels, chunk_size=6
nodes = [
    HierarchicalNode('node0', chunk_size=6),  # 6 tokens
    HierarchicalNode('node1', chunk_size=6),  # 36 tokens
    HierarchicalNode('node2', chunk_size=6),  # 216 tokens
    HierarchicalNode('node3', chunk_size=6),  # 1,296 tokens
]
# Rationale: Smaller chunks for shorter web content
```

**BookCorpus (Books, 50K-150K tokens typical):**
```python
# RECOMMENDED: 5-6 levels, chunk_size=8
nodes = [
    HierarchicalNode('node0', chunk_size=8),  # 8 tokens
    HierarchicalNode('node1', chunk_size=8),  # 64 tokens
    HierarchicalNode('node2', chunk_size=8),  # 512 tokens
    HierarchicalNode('node3', chunk_size=8),  # 4,096 tokens
    HierarchicalNode('node4', chunk_size=8),  # 32,768 tokens
    HierarchicalNode('node5', chunk_size=8),  # 262,144 tokens
]
# Rationale: 6 levels needed to capture book-length structures
```

#### Validation Metrics

The training system automatically validates hierarchy efficiency:

**Good hierarchy:**
- ✅ All levels produce ≥0.5 patterns per document
- ✅ Top level receptive field ≤ 2× average document length
- ✅ <10% of patterns created via force_learn

**Bad hierarchy (needs adjustment):**
- ✗ Higher levels produce <0.5 patterns per document → Too many levels
- ✗ Top level receptive field >> document length → Mismatched scale
- ✗ >50% force_learn patterns → Chunk sizes too large

#### Common Mistakes to Avoid

**❌ Too Many Levels:**
```python
# BAD for WikiText (1K token articles)
nodes = [
    HierarchicalNode('node0', chunk_size=5),   # 5 tokens
    HierarchicalNode('node1', chunk_size=7),   # 35 tokens
    HierarchicalNode('node2', chunk_size=9),   # 315 tokens
    HierarchicalNode('node3', chunk_size=11),  # 3,465 tokens ✓ OK
    HierarchicalNode('node4', chunk_size=13),  # 45,045 tokens ✗ Too large!
    HierarchicalNode('node5', chunk_size=17),  # 765,765 tokens ✗ Way too large!
]
# Problem: node4, node5 have receptive fields far exceeding document length
# Result: These levels rarely learn, patterns are forced and inconsistent
```

**❌ Chunk Sizes Too Large:**
```python
# BAD for short documents
HierarchicalConceptLearner(num_nodes=4, chunk_size=25)
# node3 receptive field: 25^4 = 390,625 tokens (way too large!)
```

**❌ Non-Uniform Chunk Sizes Without Justification:**
```python
# BAD: No clear semantic rationale
nodes = [
    HierarchicalNode('node0', chunk_size=5),
    HierarchicalNode('node1', chunk_size=23),
    HierarchicalNode('node2', chunk_size=7),
    HierarchicalNode('node3', chunk_size=19),
]
# Use uniform sizes unless you have specific semantic goals
```

#### Quick Sizing Formula

For average document length `D` tokens and `N` levels:

```python
# Target: Top level covers 0.5-2× average document length
# Formula: chunk_size = (D / 2) ^ (1/N)

# Example: D=1000 tokens, N=4 levels
chunk_size = (1000 / 2) ** (1/4) = 500 ** 0.25 ≈ 4.73 → use 5

# Example: D=1000 tokens, N=4 levels (targeting 2× coverage)
chunk_size = (1000 * 2) ** (1/4) = 2000 ** 0.25 ≈ 6.69 → use 7 or 8
```

**Rule of thumb:** `chunk_size=8` works well for most text datasets with 4 levels.

#### Debugging Underutilized Hierarchies

If training shows warnings like:
```
⚠️  node3: Low utilization (0.3 patterns/doc)
⚠️  node4: Low utilization (0.1 patterns/doc)
```

**Solutions:**
1. **Reduce hierarchy depth** (remove node4, node5)
2. **Decrease chunk sizes** (use 6 instead of 8)
3. **Concatenate documents** (combine multiple short docs into longer training samples)

**Diagnostic command:**
```python
# Check actual patterns per document
stats = train_hierarchical_single_pass(corpus, learner, verbose=True)
for level in range(num_levels):
    ppd = stats[f'node{level}_patterns_per_doc']
    print(f"node{level}: {ppd:.2f} patterns/doc")
```

## Training API v2.0 - Layer-Based Architecture

**New in v2.0**: Educational, transparent API inspired by TensorFlow/PyTorch.

### Philosophy

The v2.0 training architecture emphasizes:
1. **Explicit KATO API calls** - Show users exactly what's happening (`observe()`, `observe_sequence()`, `learn()`, `get_predictions()`)
2. **Layer-based configuration** - TensorFlow/PyTorch-style `add_layer()` API
3. **Educational transparency** - All configuration visible in notebooks
4. **Flexible metadata** - Configure which layers capture source metadata
5. **Full text processing** - Take complete text → tokenize → chunk → process naturally

### Key Difference from Original Approach

**Original (`training.ipynb`)**:
- Abstracts KATO calls in `HierarchicalConceptLearner` class
- Configuration hidden in function parameters
- Metadata handling fixed
- Good for production, less educational

**v2.0 (`training_v2.ipynb`)**:
- Exposes KATO calls explicitly in notebook
- Layer-by-layer configuration with `HierarchicalBuilder`
- Flexible metadata handling (configure which layers)
- Educational focus - users see the mechanics

### Using HierarchicalBuilder

**TensorFlow/PyTorch-style API:**

```python
from tools.hierarchical_builder import HierarchicalBuilder

# Create builder
hierarchy = HierarchicalBuilder(tokenizer_name='gpt2')

# Add layers (like Keras model.add())
hierarchy.add_layer(
    name='node0',
    chunk_size=15,              # How many inputs per chunk
    max_predictions=10,         # Top N predictions to next layer
    prediction_field='name',    # Which field to extract
    recall_threshold=0.6,       # Pattern matching strictness
    stm_mode='CLEAR',          # 'CLEAR' or 'ROLLING'
    max_pattern_length=0,      # Auto-learning threshold (0 = manual)
    capture_metadata=False     # Don't capture metadata here
)

hierarchy.add_layer(
    name='node1',
    chunk_size=15,
    max_predictions=8,
    prediction_field='name',
    recall_threshold=0.6,
    stm_mode='CLEAR',
    max_pattern_length=0,
    capture_metadata=False     # Don't capture metadata here
)

hierarchy.add_layer(
    name='node2',
    chunk_size=15,
    max_predictions=6,
    prediction_field='name',
    recall_threshold=0.6,
    stm_mode='CLEAR',
    max_pattern_length=0,
    capture_metadata=True      # START capturing metadata at this layer
)

hierarchy.add_layer(
    name='node3',
    chunk_size=15,
    max_predictions=4,
    prediction_field='name',
    recall_threshold=0.6,
    stm_mode='CLEAR',
    max_pattern_length=0,
    capture_metadata=True      # Capture metadata here too
)

# Build the model
model = hierarchy.build()
model.summary()
```

**Output:**
```
================================================================================
HIERARCHICAL MODEL SUMMARY
================================================================================
Tokenizer: gpt2
Total Layers: 4

Layer      Name       Chunk    MaxPred  Recall   STM Mode   Metadata
--------------------------------------------------------------------------------
0          node0      15       10       0.60     CLEAR      No
1          node1      15       8        0.60     CLEAR      No
2          node2      15       6        0.60     CLEAR      Yes
3          node3      15       4        0.60     CLEAR      Yes
================================================================================

Receptive Fields (token coverage):
  node0: 15 tokens
  node1: 225 tokens
  node2: 3,375 tokens
  node3: 50,625 tokens
```

### Explicit Training Loop

The v2.0 API exposes the exact KATO calls:

```python
# Process one sample
def process_text_sample(text, metadata=None):
    # 1. Tokenize full text
    tokens = model.tokenize(text)

    # 2. Chunk tokens for node0
    chunks = model.chunk_tokens(tokens, model.layers[0].chunk_size)

    # 3. Process each chunk at node0
    node0_patterns = []
    for chunk in chunks:
        # EXPLICIT KATO API CALL
        observations = [{'strings': [token]} for token in chunk]
        result = model.layers[0].client.observe_sequence(
            observations=observations,
            learn_at_end=True
        )
        pattern_name = result['final_learned_pattern']
        node0_patterns.append(pattern_name)

    # 4. Send node0 patterns to node1
    # EXPLICIT KATO API CALL
    for pattern in node0_patterns:
        model.layers[1].client.observe(strings=[pattern])

    # Learn at node1 (sample complete)
    # EXPLICIT KATO API CALL
    node1_pattern = model.layers[1].client.learn()

    # Continue for node2, node3...
    # (Users see exact API calls for each step)
```

**Educational value**: Users see:
- How tokens are chunked
- How `observe_sequence()` works
- How pattern names flow between layers
- How metadata is attached
- How `learn()` is triggered

### Metadata Handling

**Simple, clear configuration** - Each layer decides independently:

```python
# Example: Capture source info at top 2 layers
hierarchy.add_layer('node0', ..., capture_metadata=False)  # No metadata
hierarchy.add_layer('node1', ..., capture_metadata=False)  # No metadata
hierarchy.add_layer('node2', ..., capture_metadata=True)   # YES - capture metadata
hierarchy.add_layer('node3', ..., capture_metadata=True)   # YES - capture metadata

# Later during training:
should_attach = model.layers[1].should_capture_metadata()  # False
should_attach = model.layers[2].should_capture_metadata()  # True

if should_attach:
    model.layers[2].client.observe(
        strings=[pattern],
        metadata={'source': 'wikitext', 'doc_id': 123}
    )
```

**Why simple booleans?**
- Clear and obvious - no confusion about which layers capture metadata
- Each layer configured independently - no redundant settings
- Easy to change - modify one layer without affecting others
- Follows single responsibility principle

### Training Workflow (v2.0)

**See `training_v2.ipynb` for complete implementation:**

1. **Configure layers** using `HierarchicalBuilder`
2. **Build model** with `.build()`
3. **Define helper functions** showing explicit KATO calls
4. **Process samples** one-by-one (educational mode)
5. **Batch training** with progress tracking
6. **Analyze patterns** (tools in development for ClickHouse)

**Key files:**
- `training_v2.ipynb` - New educational training notebook
- `tools/hierarchical_builder.py` - Layer-based API module
- `generation.ipynb` - Text generation (also uses explicit API)

### When to Use v2.0 vs Original

**Use v2.0 (`training_v2.ipynb`) when:**
- Learning how KATO hierarchical training works
- Experimenting with different configurations
- Need fine control over metadata
- Want to see exact API calls
- Teaching/demonstrating the system

**Use Original (`training.ipynb`) when:**
- Production training on large datasets
- Parallel worker processing at scale
- Familiar with the system
- Need maximum performance

**Both approaches are valid** - v2.0 focuses on education and transparency, original focuses on performance and convenience.

## Architecture

### Module Structure

```
kato-notebooks/
├── tools/                              # Python package for hierarchical learning
│   ├── __init__.py                     # Exports all components
│   ├── kato_client.py                  # KATO API client
│   ├── streaming_dataset_loader.py     # HuggingFace dataset streaming
│   ├── hardware_analyzer.py            # Performance estimation
│   └── hierarchical_learning.py        # Main hierarchical learning module
├── hierarchical_training.ipynb         # Jupyter notebook for experiments
└── PROJECT_OVERVIEW.md                 # This file
```

### Core Classes

**`HierarchicalConceptLearner`**
- Creates N KATO nodes dynamically
- Manages tokenization (via HuggingFace transformers)
- Coordinates learning across hierarchy
- Tracks statistics per level

**`CorpusSegmenter`**
- Segments text into hierarchical structure
- Creates book/chapter/paragraph/chunk boundaries
- Uses fixed-length token chunking (configurable)
- Handles metadata propagation

**`StreamingDatasetLoader`**
- Streams large datasets from HuggingFace Hub
- No downloads required (streaming mode)
- Time estimation based on hardware
- Supports: C4, RefinedWeb, WikiText, OpenWebText, The Pile, etc.

**`TokenProcessor`**
- Delimiter-based text segmentation
- Tokenization via transformers (GPT-2, BERT, RoBERTa, T5, LLaMA)
- N-gram extraction (bigram, trigram, 4-gram, 5-gram)
- Token decoding utilities

### Training Function

**`train_hierarchical_single_pass()`**

The main training orchestrator that:
1. Iterates through hierarchical corpus structure
2. Tokenizes sentences and sends to node0
3. Collects pattern names and sends to node1 STM
4. Triggers learning at structural boundaries
5. Propagates pattern names up the hierarchy
6. Tracks statistics at each level

## Goals and Objectives

### Primary Goals

1. **Discover Hierarchical Patterns in Natural Language**
   - Learn sentence-level patterns (syntax, semantics, common phrases)
   - Learn paragraph-level patterns (topic structures, argument flows)
   - Learn chapter-level patterns (narrative structures, exposition strategies)
   - Learn book-level patterns (organizational principles, genre characteristics)

2. **Scale to Large Datasets**
   - Train on billions of tokens (C4, RefinedWeb, etc.)
   - Stream data without disk storage requirements
   - Estimate training time based on hardware capabilities
   - Support incremental training and checkpointing

3. **Enable Analysis and Refinement**
   - Analyze learned patterns by frequency
   - Clean up noise (low-frequency patterns)
   - Visualize pattern distributions
   - Understand what the system has learned

4. **Support Research and Experimentation**
   - Jupyter notebook environment for interactive exploration
   - Configurable hierarchy depth (4, 10, 20+ levels)
   - Multiple tokenizer options (GPT-2, BERT, RoBERTa, etc.)
   - Custom modeling functions for prediction transfer

### Secondary Goals

1. **Maintain Flexibility**
   - Not hardcoded to any specific hierarchy depth
   - Pluggable tokenizers
   - Configurable structural boundaries (paragraph size, chapter size, etc.)
   - Support for custom text corpora

2. **Provide Practical Tools**
   - Hardware analysis and time estimation
   - Dataset recommendations based on available time
   - Built-in demo for quick validation
   - Pattern analysis utilities (in development for ClickHouse)

3. **Enable Future Extensions**
   - Prediction-based transfer (use predictions from one level to inform next)
   - Modeling functions (threshold, top-N, weighted transfer)
   - Multi-modal learning (text + images, text + audio)
   - Active learning strategies

## Technical Approach

### KATO Configuration

Each node is configured for **manual learning mode**:
```python
KATOClient(
    node_id=f'node{i}_level{i}',
    max_pattern_length=0,      # No auto-learning
    stm_mode='CLEAR',           # Clear STM after learning
    base_url="http://localhost:8000"
)
```

**Why manual mode?**
- We control exactly when learning happens (at structural boundaries)
- Clear STM after learning to start fresh for next unit
- Predictable behavior aligned with hierarchical structure

### Data Flow

```
HuggingFace Dataset Stream
    ↓
CorpusSegmenter (book/chapter/paragraph/sentence structure)
    ↓
TokenProcessor (GPT-2/BERT tokenization)
    ↓
node0: observe_sequence(tokens) + learn() → sentence_pattern
    ↓
node1: observe(sentence_pattern) ... learn() → paragraph_pattern
    ↓
node2: observe(paragraph_pattern) ... learn() → chapter_pattern
    ↓
node3: observe(chapter_pattern) ... learn() → book_pattern
```

### Pattern Learning Mechanics

**At each level:**
1. **Accumulation Phase**: STM accumulates symbols (tokens or pattern names)
2. **Learning Trigger**: Structural boundary signals completion
3. **Pattern Creation**: `learn()` creates pattern from STM contents
4. **Pattern Name**: Deterministic hash returned as symbolic representation
5. **Frequency Update**: If pattern seen before, frequency increments
6. **STM Clear**: Memory cleared for next unit

**Pattern Structure in KATO Storage:**
```json
{
    "name": "PTRN|7f3a2b1c...",
    "pattern_data": [[sym1, sym2], [sym3], [sym4, sym5]],
    "frequency": 5,
    "emotives": {"confidence": [0.8, 0.7, 0.9, 0.85, 0.88]},
    "metadata": {"book": ["Alice", "Wonderland"], "chapter": ["1", "2"]},
    "length": 5
}
```
*(Stored in ClickHouse + Redis)*

### Performance Optimizations

The system includes multiple optimization strategies for fast training on large datasets:

#### 1. Node0 Batching (4-7x speedup)

Instead of making one API call per token chunk, observations are accumulated and sent in batches:

```python
learner = HierarchicalConceptLearner(
    nodes=nodes,
    tokenizer_name='gpt2',
    node0_batch_size=50  # Batch 50 chunks per API call
)
```

**How it works:**
- Accumulate N observations in memory
- Send all N observations in single API call
- Reduces network overhead and database round-trips
- Typical speedup: 4-7x faster than `node0_batch_size=1`

#### 2. Parallel Worker Processing (2-3x speedup)

Multiple workers process dataset samples concurrently using isolated KATO sessions:

```python
stats = train_from_streaming_dataset_parallel(
    dataset_key='wikitext',
    max_samples=100000,
    learner=learner,
    profiler=profiler,
    num_workers=3,  # 3 concurrent workers
    num_levels=4
)
```

**How it works:**
- Each worker has its own KATOClient instances (one per node)
- Thread-local storage reuses learner/segmenter per thread
- Workers share KATO storage via KATO API (no lock contention)
- **Connection safety**: `workers × nodes ≤ 30` enforced to prevent pool exhaustion
- Typical speedup: 2-3x faster with 75% parallel efficiency

**Combined speedup:** ~10-15x when using both batching + parallelism

#### 3. Checkpoint/Resume System

Long-running training sessions are protected against interruptions:

```python
stats = train_from_streaming_dataset_parallel(
    dataset_key='wikitext',
    max_samples=1000000,
    learner=learner,
    checkpoint_interval=5000,  # Save every 5K samples
    resume_from_checkpoint=True  # Resume if interrupted
)
```

**Features:**
- Auto-saves progress every N samples (default: 5000)
- Configuration validation prevents resuming with mismatched settings
- Validates: num_nodes, chunk_sizes, tokenizer_name, node_ids, segmentation_mode
- Clear error messages if configuration changed
- Learned patterns persist in KATO storage (not lost on crash)

**Why config validation matters:** Different configurations create different storage schemas. Resuming with wrong config would mix incompatible patterns.

#### 4. Connection Pool Management

Database connection exhaustion is prevented through validation:

```python
# System validates before training:
connections_needed = num_workers * num_levels
if connections_needed > 30:
    raise ValueError("Too many connections - reduce workers")
```

**Guidelines:**
- Safe limit: `workers × nodes ≤ 30`
- Example: 3 workers × 5 nodes = 15 connections ✓ SAFE
- Example: 6 workers × 5 nodes = 30 connections ⚠️ AT LIMIT
- Example: 8 workers × 5 nodes = 40 connections ❌ UNSAFE

**Why 30?** Safe default for concurrent KATO API connections with safety margin for other processes.

#### 5. Session Auto-Recovery

KATOClient includes resilience features for long-running training:

- **Auto-recreation**: Sessions auto-recreate on 404 errors (expired/lost)
- **STM recovery**: Attempts to restore STM state after recreation
- **Exponential backoff**: Retries on transient failures (502/503/504)
- **Timeout management**: 120s default timeout for training workloads

This ensures training continues even if KATO sessions expire mid-run.

## Dataset Integration

### Supported Datasets

Large-scale datasets available via streaming:

- **C4 (Common Crawl)**: ~365M samples, cleaned web text
- **RefinedWeb**: ~968M samples, Falcon high-quality corpus
- **The Pile (Uncopyrighted)**: ~210M samples, 22 diverse sources
- **Dolma**: ~3B samples, AI2 open corpus
- **OpenWebText**: ~8M samples, Reddit-curated content
- **WikiText**: ~1.8M samples, Wikipedia articles
- **BookCorpus**: ~74M samples, books dataset
- **RedPajama**: ~1M sample subset, LLaMA reproduction dataset

### Hardware-Aware Training

The system analyzes hardware and provides:
- **Performance tier classification** (low/medium/high/server)
- **Processing rate estimation** (samples/second)
- **Time estimates** for any dataset + sample count combination
- **Recommendations** based on available time

Example output:
```
Hardware Tier: MEDIUM
Processing Rate: ~44 samples/second

Dataset: C4
Samples: 100,000
Estimated Time: 37.8 minutes
```

## Use Cases

### Current Capabilities (Implemented)

#### 1. Hierarchical Pattern Learning ✅
- Train on large-scale datasets (WikiText, C4, RefinedWeb, etc.) via streaming
- Learn patterns at multiple abstraction levels (chunks → paragraphs → chapters → documents)
- Store patterns in KATO storage (ClickHouse + Redis) with frequency statistics
- Single-pass training with real-time pattern name propagation
- Parallel worker processing with 10-15x combined speedup
- Checkpoint/resume for long training sessions

#### 2. Pattern Analysis and Discovery ✅
- Session-independent analysis (works after kernel restarts)
- Frequency distribution visualization and statistics
- Zipfian distribution analysis with calibrated parameters
- Pattern inspection and cleanup (remove low-frequency noise)
- Training run comparison across different configurations
- Hierarchy quality evaluation with 15 metrics across 6 categories

#### 3. Language Model Research ✅
- Study what patterns emerge at different levels of abstraction
- Compare learned structures across different tokenizers (GPT-2, BERT, RoBERTa, etc.)
- Analyze how frequency distributions change by level
- Investigate symbolic vs. continuous representation approaches
- Track training dynamics and resource usage via ProfilingEngine
- Estimate training time from 29 historical runs

#### 4. Document Understanding ✅
- Identify common document structures through pattern analysis
- Learn genre-specific patterns from large corpora
- Discover organizational principles via hierarchical abstraction
- Analyze structural similarity through shared patterns

### Future Capabilities (Planned)

#### 1. Hierarchical Text Generation ⭐ **PRIMARY FUTURE GOAL**
- Generate coherent text at multiple scales (sentence, paragraph, chapter, book)
- Sample from learned Markov chain probabilities at any hierarchy level
- Unravel high-level patterns into lower-level structures
- Create novel compositions with learned stylistic and structural patterns
- Control generation granularity (fine-grained vs. coarse-grained)

**Status:** Basic generation implemented in `generation.ipynb`. The infrastructure for learning patterns is complete. Advanced generation features (multi-scale sampling, controlled generation) require further implementation.

**⚠️ CRITICAL: KATO Prediction Structure for Text Generation**

When implementing text generation, understanding KATO's prediction structure is essential to avoid token repetition:

**KATO Prediction Fields:**
```python
{
    'name': 'hash_id',                  # Pattern identifier (for lookup)
    'present': [['tok1'], ['tok2']],    # Actual matched sequence (KATO events)
    'future': [['tok3']],                # Predicted next tokens/patterns
    'matches': ['tok1', 'tok2'],        # Simplified matched tokens
    'confidence': 0.95,                  # Prediction confidence
    'potential': 0.87,                   # Ranking metric
    # ... other metrics (bayesian_posterior, similarity, etc.)
}
```

**Key Distinction:**
- **`pred['name']`**: Pattern hash (lookup key) - returns **full stored pattern** from training
- **`pred['present']`**: Exact matched tokens - contains **only what was matched**

**The Problem:**

Using `pred['name']` to get present tokens causes repetition because the stored pattern contains the full training sequence:

```python
# ❌ WRONG - Causes repetition:
pattern_name = pred['name']
present_tokens = unravel_pattern(pattern_name, ...)  # API lookup
# Returns full training sequence: ['tok1', 'tok2', ..., 'tokN', 'future_tok']
#                                                                ↑↑↑↑↑↑↑↑↑↑↑
#                                                    Includes tokens from training!

future_tokens = unravel_future_list(pred['future'], ...)
# Returns: ['future_tok']  ← Same token appears twice!

combined = present_tokens + future_tokens
# Result: [..., 'tokN', 'future_tok', 'future_tok']  ❌ REPETITION!
```

**The Solution:**

Use `pred['present']` which contains exactly what was matched:

```python
# ✅ CORRECT - No repetition:
present_events = pred.get('present', [])
present_tokens = extract_tokens_from_present(present_events)
# Returns only matched tokens: ['tok1', 'tok2', ..., 'tokN']
#                                                            ↑
#                                              Stops exactly here!

future_tokens = unravel_future_list(pred['future'], ...)
# Returns: ['future_tok']

combined = present_tokens + future_tokens
# Result: [..., 'tokN', 'future_tok']  ✓ Clean boundary!
```

**Helper Function:**
```python
def extract_tokens_from_present(present_events):
    """Extract tokens from pred['present'] (KATO event format)."""
    if not present_events:
        return []
    return [event[0] for event in present_events if event and len(event) > 0]
```

**Why This Happens:**

When a pattern is learned during training, KATO stores the complete sequence including what comes after. When you later get a prediction that matches this pattern:
1. The `pred['name']` points to the stored pattern (full sequence)
2. The `pred['present']` contains only the matched portion (subset)
3. The `pred['future']` predicts what comes next (may overlap with stored pattern's end)

Using `pred['name']` → API lookup → full sequence → overlap with `pred['future']` → repetition!

**Reference:** See `generation.ipynb` and `GENERATION_FIX_SUMMARY.md` for implementation details.

#### 2. Text Compression (Planned)
- Use pattern names as compressed representations
- Hierarchical encoding of documents
- Efficient storage of repeated structures
- Lossless reconstruction via pattern unraveling

**Status:** Pattern storage exists; decompression/reconstruction not implemented.

#### 3. Transfer Learning (Planned)
- Pre-train hierarchical representations on large corpus
- Transfer learned patterns to specific domains
- Use high-level patterns for document classification
- Fine-tune on domain-specific data

**Status:** Training infrastructure exists; transfer mechanisms not implemented.

#### 4. Anomaly Detection (Planned)
- Identify unusual structures (low-frequency patterns)
- Detect novel narrative forms
- Flag out-of-distribution documents
- Identify style deviations

**Status:** Frequency statistics exist; anomaly detection algorithms not implemented.

## Design Decisions

### Key Choices Made During Development

1. **Single-Pass Training (Not Multi-Pass)**
   - Initially considered iterative training with multiple passes
   - Switched to single-pass with real-time pattern name propagation
   - **Rationale**: Simpler, faster, more aligned with streaming data

2. **Structural Boundaries Trigger Learning (Not Fixed Windows)**
   - Learning happens at paragraph/chapter/book completion
   - Not every N tokens or N patterns
   - **Rationale**: Respects natural linguistic/document structure

3. **Persistent Pattern Storage (Not In-Memory)**
   - Patterns persist in KATO storage (ClickHouse + Redis) with frequency counters
   - Allows post-training analysis and cleanup
   - **Rationale**: Scalability and inspection capabilities

4. **Arbitrary Node Depth (Not Hardcoded)**
   - Initially had hardcoded 4 nodes
   - Refactored to support any number of levels
   - **Rationale**: Research flexibility and experimentation

5. **Streaming Datasets (Not Downloads)**
   - HuggingFace streaming mode (no disk storage)
   - Time estimation before training
   - **Rationale**: Access to massive datasets without infrastructure overhead

6. **Manual Learning Mode (Not Auto-Learning)**
   - MAX_PATTERN_LENGTH=0, explicit learn() calls
   - STM_MODE='CLEAR' to reset after learning
   - **Rationale**: Precise control over learning boundaries

7. **Tools Module Structure (Not Standalone Scripts)**
   - Organized as importable Python package
   - Clean namespace with __all__ exports
   - **Rationale**: Professional structure, easy to extend

## Future Directions

### Near-Term Enhancements

1. **Text Generation Implementation** ⭐ **PRIMARY GOAL**
   - Implement Markov chain probability sampling at top node
   - Build pattern unraveling mechanism (top-down)
   - Create token decoding pipeline
   - Generate text at multiple scales (sentence, paragraph, chapter, book)
   - Evaluate generation quality and coherence

   **Status:** Infrastructure complete (pattern learning, frequency statistics). Needs generation and unraveling algorithms.

2. **Prediction-Based Transfer**
   - Use predictions from node0 to inform node1 input
   - Modeling functions: threshold, top-N, weighted
   - Already implemented, needs testing

3. ✅ **Checkpointing** (IMPLEMENTED)
   - Save/resume training for long runs
   - Configuration validation prevents mismatched resume
   - Auto-save every N samples (default: 5000)

4. ✅ **Parallel Processing** (IMPLEMENTED)
   - Process multiple documents concurrently (2-3x speedup)
   - Separate KATO sessions per worker (no lock contention)
   - Thread-local storage for resource reuse
   - Connection pool management (workers × nodes ≤ 30)

5. **Pattern Content Analysis**
   - Inspect what tokens/patterns make up high-frequency patterns
   - Decode pattern names back to original content
   - Build pattern vocabulary

   **Status:** Pattern data stored in KATO storage; decoding utilities not yet implemented.

### Long-Term Research Directions

1. **Multi-Modal Hierarchies**
   - Text + image patterns at each level
   - Audio + text for speech/podcast learning
   - Video understanding with temporal hierarchies

2. **Active Learning**
   - Identify which data to learn from based on novelty
   - Focus on patterns that increase predictive power
   - Reduce training time by selective learning

3. **Pattern Transfer and Reuse**
   - Transfer learned patterns between domains
   - Compositional pattern libraries
   - Fine-tuning on domain-specific data

4. **Causal and Temporal Patterns**
   - Learn cause-effect relationships in narratives
   - Temporal sequences (before/after patterns)
   - Predict future events based on current patterns

5. **Explainability**
   - Visualize what high-level patterns represent
   - Trace pattern names back to original text
   - Understand abstraction hierarchy

## Getting Started

### Quick Start (Built-in Demo)

```python
from tools import (
    HierarchicalConceptLearner,
    CorpusSegmenter,
    train_hierarchical_single_pass
)

# Segment sample text
segmenter = CorpusSegmenter()
corpus = segmenter.segment_book(sample_text)

# Create 4-level learner
learner = HierarchicalConceptLearner(num_nodes=4, tokenizer_name="gpt2")

# Train single-pass
stats = train_hierarchical_single_pass(corpus, learner, num_levels=4)
```

### Large-Scale Training

```python
from tools import (
    HierarchicalConceptLearner,
    StreamingDatasetLoader,
    train_hierarchical_single_pass
)

# Stream C4 dataset
learner = HierarchicalConceptLearner(num_nodes=4, tokenizer_name="gpt2")

# Build corpus from stream
corpus = build_corpus_from_stream(
    dataset_key='c4',
    max_samples=100000
)

# Train
stats = train_hierarchical_single_pass(corpus, learner, num_levels=4)
```

### Analysis and Cleanup

**Note:** Pattern analysis tools are being migrated to ClickHouse. The following functionality is planned:

```python
# Planned API (not yet implemented):
# from tools.kato_storage import ClickHouseAnalyzer
#
# # Analyze patterns
# analyzer = ClickHouseAnalyzer(learner.nodes['node0'])
# stats = analyzer.get_stats()
# print(f"node0 patterns: {stats['total_patterns']}")
#
# # Visualize
# analyzer.visualize_frequency_distribution(max_freq=50)
```

## Success Metrics

### How to Know If It's Working

1. **Frequency Distributions Make Sense**
   - node0 (sentences): Many unique patterns, Zipfian distribution
   - node1 (paragraphs): Fewer patterns, higher reuse
   - node2 (chapters): Even fewer, clear structural patterns
   - node3 (books): Very few, genre/style-level abstractions

2. **High-Frequency Patterns Are Meaningful**
   - Inspect top 10-20 patterns per level
   - Decode back to original content
   - Should represent common structures, not noise

3. **Cleanup Removes Noise Effectively**
   - Low-frequency patterns (frequency=1) often noise/typos
   - After cleanup, remaining patterns should be robust
   - Frequency distribution should be cleaner

4. **Patterns Are Reused**
   - If every pattern has frequency=1, something is wrong
   - Expect power-law distribution (few very common, many rare)
   - Higher levels should have higher average frequency

5. **Training Completes Successfully**
   - No errors or crashes
   - Statistics match expectations (node0 > node1 > node2 > node3)
   - KATO storage contains patterns at all levels

## Dependencies

### Core Requirements

- **Python 3.8+**
- **KATO Server** (FastAPI service running on http://localhost:8000)
- **ClickHouse + Redis** (KATO's storage backend)
- **HuggingFace Transformers** (tokenizers: GPT-2, BERT, RoBERTa, etc.)
- **HuggingFace Datasets** (streaming large datasets)
- **matplotlib** (visualization)
- **tqdm** (progress bars)
- **numpy** (numerical operations)

### Installation

```bash
pip install datasets transformers clickhouse-connect redis matplotlib tqdm numpy
```

### KATO Server

The KATO FastAPI service must be running:
```bash
# In kato repository
./start.sh
```

Verify at: http://localhost:8000/docs

## Important Notes

### What This Project Is

- A research tool for studying hierarchical concept learning
- A framework for training multi-level pattern recognition systems
- A demonstration of abstraction through symbolic pattern names
- An experiment in bottom-up semantic understanding
- **A pattern learning infrastructure** designed for future hierarchical text generation

### What This Project Is Not

- **Not a transformer** (does not use attention mechanisms)
- **Not a neural network** (uses deterministic pattern matching)
- **Not end-to-end differentiable** (symbolic discrete learning)
- **Not gradient-based** (uses frequency statistics and pattern matching)
- **Not currently a generative model** (generation planned but not yet implemented)

### Text Generation Mechanism (Planned)

**Future Goal: Hierarchical Text Generation**

The infrastructure for learning patterns is complete. Text generation (not yet implemented) will work through a top-down unraveling process:

1. **Top-Level Sampling** (e.g., node3 - book level)
   - Use Markov chain probabilities to sample a book-level pattern
   - Based on learned frequencies and conditional probabilities

2. **Hierarchical Unraveling**
   - node3 pattern → contains sequence of node2 pattern names (chapters)
   - node2 patterns → contain sequences of node1 pattern names (paragraphs)
   - node1 patterns → contain sequences of node0 pattern names (sentences)
   - node0 patterns → contain sequences of actual tokens

3. **Token Decoding**
   - Extract tokenized symbols from node0 patterns
   - Decode tokens back to text using tokenizer vocabulary

4. **Generation Flow**
```
Sample from node3 (Markov probabilities)
    ↓
Retrieve node3 pattern → [node2_ptrn_A, node2_ptrn_B, node2_ptrn_C]
    ↓
For each node2 pattern → [node1_ptrn_X, node1_ptrn_Y, ...]
    ↓
For each node1 pattern → [node0_ptrn_1, node0_ptrn_2, ...]
    ↓
For each node0 pattern → [token_a, token_b, token_c, ...]
    ↓
Decode tokens → Generated Text
```

**Key Insight**: The hierarchy enables generation at multiple scales. You can:
- Generate entire book structures (sample from node3)
- Generate chapter structures (sample from node2)
- Generate paragraph structures (sample from node1)
- Generate sentences (sample from node0)

### Philosophy

This project explores an alternative approach to language generation and understanding:
- **Symbolic** rather than continuous representations
- **Deterministic pattern storage** with probabilistic sampling
- **Hierarchical** rather than flat embeddings
- **Pattern-based** rather than gradient-based
- **Bottom-up learning** with **top-down generation**

It asks: *What happens when you let a system discover its own hierarchy of abstractions purely from the statistics of recurring patterns, then generate new text by sampling and unraveling those learned hierarchies?*

## Conclusion

The KATO Hierarchical Concept Learning system represents a novel approach to understanding and generating structured text data. By learning patterns at multiple levels of abstraction and using pattern names as symbolic representations, the system builds a hierarchy of concepts from the bottom up, then generates new text from the top down through probabilistic sampling and pattern unraveling.

The project is designed for:
- **Generation**: Creating coherent text through hierarchical Markov sampling
- **Research**: Exploring hierarchical learning and abstraction
- **Scalability**: Training on billions of tokens via streaming
- **Flexibility**: Arbitrary hierarchy depth and configuration
- **Practicality**: Analysis, cleanup, and visualization tools

Success is measured not by traditional metrics like perplexity or accuracy, but by the **meaningfulness and reusability** of learned patterns at each level of the hierarchy, and ultimately by the **quality and coherence** of generated text.

---

**Last Updated**: 2025-10-13
**Version**: 1.0
**Project Location**: `/Users/sevakavakians/PROGRAMMING/kato-notebooks/`
