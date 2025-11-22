# KATO Hierarchical Concept Learning

A Python implementation of hierarchical concept-based learning using KATO (Knowledge Abstraction for Traceable Outcomes) cognitive processors. This system learns text at multiple levels of abstraction, creating true hierarchical representations from books down to individual tokens.

## Table of Contents

- [Getting Started - New Simplified Structure!](#getting-started---new-simplified-structure-)
- [Overview](#overview)
- [What is Hierarchical Concept Learning?](#what-is-hierarchical-concept-learning)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Text Segmentation](#text-segmentation)
  - [Hierarchical Learning](#hierarchical-learning)
  - [Progress Tracking](#progress-tracking)
  - [Visualization](#visualization)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)
- [Contributing](#contributing)
- [License](#license)

## Getting Started - New Simplified Structure! 🎉

This project has been streamlined for clarity:

### 📓 **Main Notebooks**

1. **`training_v2.ipynb`** - **NEW! Educational training with explicit KATO API** ⭐
   - TensorFlow/PyTorch-style layer configuration
   - Explicit KATO calls visible (observe, learn, get_predictions)
   - Perfect for learning and experimentation
   - Flexible metadata handling per layer
   - Step-by-step educational demos

2. **`training.ipynb`** - Production training on real datasets
   - Hardware profiling and analysis
   - Real data from HuggingFace (WikiText, C4, RefinedWeb, etc.)
   - Parallel workers for optimal speed (2-3x speedup)
   - Training history tracking
   - Checkpoint/resume support with config validation

3. **`generation.ipynb`** - Text generation with hierarchical predictions
   - Bottom-up activation (input → predictions)
   - Top-down unraveling (patterns → tokens)
   - Explicit KATO API calls (educational)
   - MongoDB pattern retrieval
   - ⚠️ **Critical:** Uses `pred['present']` (not `pred['name']`) to avoid token repetition

4. **`analysis.ipynb`** - Analyze learned patterns
   - Session-independent analysis (works after kernel restarts)
   - Frequency distributions and visualizations
   - Pattern inspection and cleanup
   - Training run comparisons

5. **`hierarchy_metrics.ipynb`** - Comprehensive hierarchy quality analysis
   - 15 metrics across 6 categories (compression, connectivity, information theory, etc.)
   - Graph topology evaluation
   - Training dynamics visualization
   - Detailed interpretation guide

6. **`hierarchy_dashboard.ipynb`** - Quick hierarchy health check
   - 5-tier scoring system
   - At-a-glance quality assessment
   - Actionable recommendations
   - Immediate issue detection

### 🛠️ **Essential Tools** (in `tools/` directory)

- `hierarchical_builder.py` - **NEW**: Layer-based API for educational training (v2.0)
- `hierarchical_learning.py` - Core training engine with batching (4-7x speedup)
- `kato_client.py` - KATO API client with auto-recovery and session management
- `streaming_dataset_loader.py` - HuggingFace dataset streaming with parallel workers
- `training_history.py` - Training run tracking in SQLite
- `training_estimator.py` - Data-driven time predictions from 29 historical runs
- `profiling_engine.py` - Real-time CPU/memory/disk profiling during training
- `hardware_analyzer_v2.py` - Hardware detection and benchmarking
- `storage_estimator.py` - MongoDB storage estimation with Zipfian modeling
- `scaling_analyzer.py` - Scaling analysis
- `hardware_recommender.py` - Hardware recommendations

### 📚 **Documentation**

- **PROJECT_OVERVIEW.md** - Core concepts and philosophy
- **CLAUDE.md** - Development instructions
- **TRAINING_RUN_COMPARISON.md** - How to compare experiments

### 🚀 **Quick Workflow**

```bash
# 1. Train a model

# For learning/experimentation (RECOMMENDED for beginners):
jupyter notebook training_v2.ipynb
# → Educational approach with explicit KATO API calls
# → TensorFlow/PyTorch-style layer configuration
# → Perfect for understanding the mechanics

# For production training:
jupyter notebook training.ipynb
# → Run cells to train on real data with your chosen configuration
# → Supports checkpoint/resume if training is interrupted

# 2. Generate text (optional)
jupyter notebook generation.ipynb
# → Test hierarchical text generation
# → Bottom-up activation + top-down unraveling

# 3. Analyze results
jupyter notebook analysis.ipynb
# → Load training history, analyze patterns, compare runs

# 4. Evaluate hierarchy quality (RECOMMENDED)
jupyter notebook hierarchy_dashboard.ipynb
# → Quick 5-tier health check with actionable recommendations

# Or for detailed analysis:
jupyter notebook hierarchy_metrics.ipynb
# → 15 comprehensive metrics across 6 categories
```

### 🎓 **Learning Path**

**New to KATO hierarchical learning?**
1. Start with `training_v2.ipynb` - See explicit KATO API calls
2. Experiment with `generation.ipynb` - Understand predictions
3. Analyze with `analysis.ipynb` - Inspect learned patterns
4. Scale up with `training.ipynb` - Production training

**Old notebooks moved to `archive/` for reference.**

---

## Overview

Traditional language models process text sequentially without clear concept boundaries. This implementation uses KATO's cognitive architecture to create **true hierarchical abstraction** where:

- **Token Chunks** (15 tokens) are learned as phrase-level patterns
- **Paragraphs** are learned as complete patterns of chunk patterns (~225 tokens)
- **Chapters** are learned as complete patterns of paragraph patterns (~3,375 tokens)
- **Books** are learned as complete patterns of chapter patterns (~50,625 tokens)

Each level learns a **complete pattern** before passing its symbolic representation (pattern name) to the next level. Semantics emerge through hierarchical composition, not from individual chunks.

### Key Features

✨ **True Hierarchical Abstraction** - Each level learns increasingly abstract patterns
🎯 **Concept Boundaries** - Clear separation prevents mixing unrelated concepts
🔗 **Pattern Name Propagation** - Symbolic compression via `PTRN|<hash>` identifiers
📊 **Progress Tracking** - Real-time statistics at all levels
📈 **Visualization** - Built-in matplotlib charts for learning analysis
🚀 **Streaming Support** - Process large corpora efficiently
💾 **Session-Independent Analysis** - Analyze trained models without active sessions
🔬 **Training Run Comparison** - Compare multiple experiments side-by-side
🗄️ **MongoDB Persistence** - Training data persists across kernel restarts

## What is Hierarchical Concept Learning?

### The Problem

Traditional sequential processing mixes concepts together:

```
Traditional Approach (ROLLING STM):
[token1, token2, ..., token10] → learn pattern
[token3, token4, ..., token12] → learn pattern (overlaps!)
[token5, token6, ..., token14] → learn pattern (more overlap!)

Result: Patterns contain mixed concepts from different sentences/paragraphs
```

### The Solution

Hierarchical concept learning maintains clean boundaries:

```
Hierarchical Approach (CLEAR STM):

Level 0 (Token Chunks - 15 tokens each):
[token1, ..., token15] → learn → PTRN|abc123 → CLEAR STM
[token16, ..., token30] → learn → PTRN|def456 → CLEAR STM

Level 1 (Paragraphs - sequences of ~15 chunks = 225 tokens):
[PTRN|abc123, PTRN|def456, ...] → learn → PTRN|ghi789 → CLEAR STM

Level 2 (Chapters - sequences of paragraphs = ~3,375 tokens):
[PTRN|ghi789, PTRN|jkl012, ...] → learn → PTRN|mno345 → CLEAR STM

Level 3 (Books - sequences of chapters = ~50,625 tokens):
[PTRN|mno345, PTRN|pqr678, ...] → learn → PTRN|stu901 → CLEAR STM

Result: Clean hierarchical patterns where semantics emerge from composition
```

### Why Chunking Instead of Sentences?

**Scalability through Deduplication:**
- Sentences are mostly unique → node0 KB size ≈ corpus size
- Chunks repeat frequently → node0 KB size << corpus size (Zipfian distribution)
- Common phrases like "according to the" increment frequency counters

**Hierarchical Composition:**
- node0 chunks have no semantic awareness (like pixels in an image)
- Meaning emerges at higher levels through pattern composition
- node1 with 15 chunks = 225 tokens ≈ 2-3 sentences worth of context

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md#6-token-chunking-strategy) for detailed explanation.

## Architecture

### Four-Level Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│  Book/Article Corpus                                    │
│  (Raw text input)                                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  NODE 3: Book/Article Level                             │
│  ─────────────────────────────────                      │
│  Input:  Chapter pattern names (PTRN|...)              │
│  Output: Book pattern name (PTRN|...)                  │
│  Config: max_pattern_length=0, stm_mode=CLEAR          │
│  KB:     Book patterns (chapter sequences)              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  NODE 2: Chapter/Section Level                          │
│  ─────────────────────────────────                      │
│  Input:  Paragraph pattern names (PTRN|...)            │
│  Output: Chapter pattern name (PTRN|...)               │
│  Config: max_pattern_length=0, stm_mode=CLEAR          │
│  KB:     Chapter patterns (paragraph sequences)         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  NODE 1: Paragraph Level                                │
│  ─────────────────────────────────                      │
│  Input:  Chunk pattern names (PTRN|...)                │
│  Output: Paragraph pattern name (PTRN|...)             │
│  Config: max_pattern_length=0, stm_mode=CLEAR          │
│  KB:     Paragraph patterns (chunk sequences, ~225 tok) │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  NODE 0: Token Chunk Level                              │
│  ─────────────────────────────────                      │
│  Input:  Tokens (strings)                               │
│  Output: Chunk pattern name (PTRN|...)                 │
│  Config: max_pattern_length=0, stm_mode=CLEAR          │
│  KB:     Chunk patterns (15-token sequences)            │
└─────────────────────────────────────────────────────────┘
```

### Configuration Details

| Setting | Value | Purpose |
|---------|-------|---------|
| `max_pattern_length` | `0` | Disables auto-learning; allows manual control of when patterns are learned |
| `stm_mode` | `"CLEAR"` | STM clears after each learn operation, maintaining concept boundaries |
| `node_id` | Unique per node | Isolates knowledgebases per hierarchical level |

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# KATO server running (Docker or local)
# Ensure KATO is accessible at http://kato:8000 or your configured URL
```

### Install Dependencies

```bash
# Clone or navigate to the repository
cd /path/to/kato-notebooks

# Install required packages
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install numpy transformers datasets nltk tqdm matplotlib
```

### Download NLTK Data

```python
import nltk
nltk.download('punkt')
```

### Verify KATO Server

```bash
# Check if KATO server is running
curl http://kato:8000/health

# Or in Python:
import requests
response = requests.get('http://kato:8000/health')
print(response.json())
```

## Quick Start

### 1. Simple Text Processing

```python
from kato_hierarchical_streaming import (
    CorpusSegmenter,
    HierarchicalConceptLearner,
    visualize_hierarchical_stats
)

# Your text
my_text = """
Chapter 1: Introduction

Machine learning is transforming technology. Deep neural networks can
now recognize complex patterns. This has led to breakthroughs in AI.

The field continues to evolve rapidly. New architectures emerge regularly.
Researchers push the boundaries of what's possible.

Chapter 2: Applications

AI systems are deployed worldwide. They power recommendation engines and
search systems. Autonomous vehicles use AI for navigation.

Healthcare benefits from diagnostic AI. Financial systems use it for
fraud detection. The applications are endless.
"""

# Step 1: Segment the text
segmenter = CorpusSegmenter()
book = segmenter.segment_book(
    my_text,
    book_metadata={'title': 'AI Overview', 'author': 'Demo'}
)

# Step 2: Initialize learner
learner = HierarchicalConceptLearner(
    base_url="http://kato:8000",
    tokenizer_name="gpt2"
)

# Step 3: Learn hierarchically
corpus = {'books': [book]}
learner.process_corpus(corpus, verbose=True)

# Step 4: View statistics
learner.tracker.print_summary()
visualize_hierarchical_stats(learner)
```

### 1a. Educational Training (v2.0) - **RECOMMENDED for Beginners** ⭐

**NEW!** Layer-based API with explicit KATO calls (see `training_v2.ipynb`):

```python
from tools.hierarchical_builder import HierarchicalBuilder

# Create hierarchy with TensorFlow/PyTorch-style API
hierarchy = HierarchicalBuilder(tokenizer_name='gpt2')

# Add layers with explicit configuration
hierarchy.add_layer(
    name='node0',
    chunk_size=15,              # Tokens per chunk
    max_predictions=10,         # Top predictions to next layer
    prediction_field='name',    # Field to extract
    recall_threshold=0.6,       # Pattern matching strictness
    capture_metadata=False      # Don't capture metadata here
)

hierarchy.add_layer(
    name='node1',
    chunk_size=15,
    max_predictions=8,
    prediction_field='name',
    capture_metadata=False      # Don't capture metadata here
)

hierarchy.add_layer(name='node2', chunk_size=15, max_predictions=6, prediction_field='name', capture_metadata=True)
hierarchy.add_layer(name='node3', chunk_size=15, max_predictions=4, prediction_field='name', capture_metadata=True)

# Build and see summary
model = hierarchy.build()
model.summary()

# Output:
# ================================================================================
# HIERARCHICAL MODEL SUMMARY
# ================================================================================
# Tokenizer: gpt2
# Total Layers: 4
#
# Layer      Name       Chunk    MaxPred  Recall   STM Mode   Metadata
# --------------------------------------------------------------------------------
# 0          node0      15       10       0.60     CLEAR      No
# 1          node1      15       8        0.60     CLEAR      Yes
# 2          node2      15       6        0.60     CLEAR      Yes
# 3          node3      15       4        0.60     CLEAR      Yes
# ================================================================================
#
# Receptive Fields (token coverage):
#   node0: 15 tokens
#   node1: 225 tokens
#   node2: 3,375 tokens
#   node3: 50,625 tokens

# Process samples (see training_v2.ipynb for complete examples)
# The notebook shows EXPLICIT KATO API calls:
#   - model.layers[0].client.observe_sequence(...)
#   - model.layers[0].client.learn()
#   - model.layers[1].client.observe(...)
# Users see exactly what's happening!
```

**Why v2.0?**
- 🎓 **Educational**: See exact KATO API calls (`observe()`, `learn()`, `get_predictions()`)
- 🔧 **Flexible**: Configure each layer independently
- 🎯 **Familiar**: TensorFlow/PyTorch-style API
- 📊 **Transparent**: All settings visible

**See `training_v2.ipynb` for complete step-by-step examples!**

### 2. Run the Built-in Demo

```python
# Run the demonstration with sample text
from kato_hierarchical_streaming import demonstrate_hierarchical_learning

learner, book = demonstrate_hierarchical_learning()

# Output includes:
# - Segmentation statistics
# - Learning progress by chapter
# - Pattern hierarchy visualization
# - Detailed statistics by node
```

### 3. Run from Command Line

```bash
# Edit kato_hierarchical_streaming.py and uncomment the demo:
# Line 1559: learner, book = demonstrate_hierarchical_learning()

# Then run:
python kato_hierarchical_streaming.py
```

## Session-Independent Analysis

**Analyze trained models without active training sessions!**

Training data persists in MongoDB, allowing you to analyze patterns after kernel restarts, fix analysis bugs without retraining, and work across different notebooks.

### Quick Example: Load and Analyze

```python
from tools import (
    StandaloneMongoDBAnalyzer,
    TrainingManifest,
    load_latest_manifest
)

# Method 1: Load from training manifest (auto-saved after training)
manifest = load_latest_manifest()
print(f"Training: {manifest.training_id}")
print(f"Dataset: {manifest.dataset}, Samples: {manifest.samples_trained:,}")

# Get analyzers for all nodes (NO active sessions needed!)
analyzers = manifest.get_analyzers(mongo_uri="mongodb://localhost:27017/")

# Analyze patterns
for node_name, analyzer in analyzers.items():
    stats = analyzer.get_stats()
    print(f"{node_name}: {stats['total_patterns']:,} patterns")

# Visualize frequency distributions
analyzers['node0'].visualize_frequency_distribution()
```

### Method 2: Direct Database Access

```python
from tools import discover_training_databases, StandaloneMongoDBAnalyzer

# Discover all KATO databases
databases = discover_training_databases(mongo_uri="mongodb://localhost:27017/")
print(f"Found {len(databases)} databases: {databases}")

# Create analyzer for specific database
analyzer = StandaloneMongoDBAnalyzer(
    db_name="node0_level0_kato",
    mongo_uri="mongodb://localhost:27017/"
)

# Get statistics
stats = analyzer.get_stats()
print(f"Total patterns: {stats['total_patterns']:,}")
print(f"Average frequency: {stats['avg_frequency']:.2f}")

# Get high-frequency patterns
high_freq = analyzer.get_patterns_by_frequency(min_freq=10)
print(f"Patterns with frequency >= 10: {len(high_freq)}")
```

### Complete Workflow Example

See [`analysis.ipynb`](analysis.ipynb) for a complete session-independent analysis workflow.

**Benefits:**
- ✅ Analyze after kernel restarts
- ✅ Debug analysis code without retraining
- ✅ Work with historical training data
- ✅ Share analysis across notebooks
- ✅ Fast iteration on visualization code

## Training Run Comparison

**Compare multiple training experiments with isolated databases!**

By default, using the same node IDs overwrites previous training data. Use unique run IDs to preserve separate experiments.

### Quick Example: Create Comparable Runs

```python
from tools import create_training_run_nodes, HierarchicalConceptLearner
from tools import train_from_streaming_dataset_parallel

# Training Run 1: 100 samples
nodes_100 = create_training_run_nodes(run_id='wikitext_100samples')
learner_100 = HierarchicalConceptLearner(nodes=nodes_100, tokenizer_name='gpt2')

train_from_streaming_dataset_parallel(
    dataset_key='wikitext',
    max_samples=100,
    learner=learner_100,
    num_workers=4
)

# Training Run 2: 500 samples (separate databases!)
nodes_500 = create_training_run_nodes(run_id='wikitext_500samples')
learner_500 = HierarchicalConceptLearner(nodes=nodes_500, tokenizer_name='gpt2')

train_from_streaming_dataset_parallel(
    dataset_key='wikitext',
    max_samples=500,
    learner=learner_500,
    num_workers=4
)

# Now both runs have separate databases:
# - node0_wikitext_100samples_kato, node1_wikitext_100samples_kato, ...
# - node0_wikitext_500samples_kato, node1_wikitext_500samples_kato, ...
```

### Compare Training Runs

```python
from tools import list_all_training_runs, StandaloneMongoDBAnalyzer

# List all available training runs
runs = list_all_training_runs(mongo_uri="mongodb://localhost:27017/")
print(f"Available runs: {list(runs.keys())}")

# Compare two runs
for run_id in ['wikitext_100samples', 'wikitext_500samples']:
    print(f"\n{run_id}:")
    for db_name in runs[run_id]:
        analyzer = StandaloneMongoDBAnalyzer(
            db_name=db_name,
            mongo_uri="mongodb://localhost:27017/"
        )
        stats = analyzer.get_stats()
        print(f"  {db_name}: {stats['total_patterns']:,} patterns")
        analyzer.close()
```

### Manage Training Runs

```python
from tools import delete_training_run

# Delete old experiments
delete_training_run('old_experiment')  # Prompts for confirmation

# Delete without confirmation (use carefully!)
delete_training_run('old_experiment', confirm=False)
```

### Common Comparison Scenarios

**Compare Different Sample Sizes:**
```python
for num_samples in [100, 500, 1000, 5000]:
    nodes = create_training_run_nodes(run_id=f'wikitext_{num_samples}samples')
    # ... train with max_samples=num_samples
```

**Compare Different Chunk Sizes:**
```python
for chunk_size in [8, 15, 25]:
    nodes = create_training_run_nodes(
        run_id=f'chunk{chunk_size}',
        chunk_size=chunk_size
    )
    # ... train
```

**Compare Different Datasets:**
```python
for dataset in ['wikitext', 'c4']:
    nodes = create_training_run_nodes(run_id=f'{dataset}_10k')
    # ... train with dataset_key=dataset
```

See [`TRAINING_RUN_COMPARISON.md`](TRAINING_RUN_COMPARISON.md) for complete guide.

## Detailed Usage

### Text Segmentation

#### Option 1: Book with Chapters

```python
from kato_hierarchical_streaming import CorpusSegmenter

segmenter = CorpusSegmenter()

# Read your book text
with open('my_book.txt', 'r') as f:
    book_text = f.read()

# Segment it
book = segmenter.segment_book(
    book_text,
    book_metadata={
        'title': 'My Book Title',
        'author': 'Author Name'
    }
)

# Inspect structure
print(f"Chapters: {len(book['chapters'])}")
for chapter in book['chapters']:
    print(f"  {chapter['title']}: {len(chapter['paragraphs'])} paragraphs")
```

**Chapter Detection Patterns:**
- `Chapter 1`, `CHAPTER I`, `Chapter One`
- Customizable via regex in `segment_book()` method

#### Option 2: Article with Sections

```python
# For articles with section headers
article = segmenter.segment_article(
    article_text,
    article_metadata={
        'title': 'Research Paper',
        'author': 'Researchers'
    }
)

# Section detection patterns:
# - Markdown: "# Section Title", "## Subsection"
# - Numbered: "1. Introduction", "2.1 Background"
# - Labeled: "Section 1: Overview"
```

#### Option 3: Simple Text (No Chapters)

```python
# For text without clear chapters
book = segmenter.segment_simple_text(
    simple_text,
    metadata={'title': 'Notes'}
)

# Creates single-chapter structure
# Automatically detects paragraphs and sentences
```

### Hierarchical Learning

#### Basic Learning Flow

```python
from kato_hierarchical_streaming import HierarchicalConceptLearner

# Initialize
learner = HierarchicalConceptLearner(
    base_url="http://kato:8000",
    tokenizer_name="gpt2"  # or "bert-base-uncased", "roberta-base", etc.
)

# Process corpus
corpus = {'books': [book1, book2, book3]}
learner.process_corpus(corpus, verbose=True)
```

#### Metadata Tracking

The system automatically tracks source information through metadata at each level:

```python
# Metadata flows down the hierarchy:
#
# Book level: {'title': 'My Book', 'author': 'Author', 'chapter_count': 5}
#     ↓
# Chapter level: {...book metadata..., 'chapter_title': 'Chapter 1', 'paragraph_count': 3}
#     ↓
# Paragraph level: {...chapter metadata..., 'paragraph_text': '...', 'sentence_count': 4}
#     ↓
# Sentence level: {...paragraph metadata..., 'sentence': 'The actual sentence text.'}

# This allows patterns to be traced back to their source:
# - Node0 patterns: Track which sentence they came from
# - Node1 patterns: Track which paragraph and chapter
# - Node2 patterns: Track which chapter and book
# - Node3 patterns: Track which book

# Example: Custom metadata at sentence level
custom_metadata = {
    'document_id': '12345',
    'section': 'introduction',
    'page': 42
}

sentence_pattern = learner.learn_sentence(
    "This sentence has custom metadata.",
    metadata=custom_metadata
)
# The metadata is attached to all observations for this sentence
```

#### Level-by-Level Learning

```python
# Learn individual components

# Learn a single sentence
sentence_pattern = learner.learn_sentence(
    "Machine learning is transforming the world."
)
print(f"Sentence pattern: {sentence_pattern}")

# Learn a paragraph from its structure
paragraph = {
    'sentences': [
        "First sentence here.",
        "Second sentence here.",
        "Third sentence here."
    ]
}
paragraph_pattern = learner.learn_paragraph(paragraph)
print(f"Paragraph pattern: {paragraph_pattern}")

# Learn a chapter
chapter = {
    'title': 'Introduction',
    'paragraphs': [paragraph1, paragraph2, paragraph3]
}
chapter_pattern = learner.learn_chapter(chapter)

# Learn a book
book = {
    'title': 'My Book',
    'chapters': [chapter1, chapter2, chapter3]
}
book_pattern = learner.learn_book(book, verbose=True)
```

### Progress Tracking

#### Real-time Statistics

```python
# Get current statistics
stats = learner.get_stats()

print(f"Sentences learned: {stats['sentences_learned']:,}")
print(f"Paragraphs learned: {stats['paragraphs_learned']:,}")
print(f"Chapters learned: {stats['chapters_learned']:,}")
print(f"Books learned: {stats['books_learned']:,}")
print(f"Elapsed time: {stats['elapsed_formatted']}")

# Get pattern names by level
patterns = stats['patterns_by_level']
print(f"\nNode0 patterns: {len(patterns['node0'])}")
print(f"Node1 patterns: {len(patterns['node1'])}")
print(f"Node2 patterns: {len(patterns['node2'])}")
print(f"Node3 patterns: {len(patterns['node3'])}")
```

#### Node-Level Statistics

```python
# Get detailed node statistics
node_stats = learner.get_node_stats()

for level, stats in node_stats.items():
    print(f"\n{level.upper()}:")
    print(f"  Observations: {stats['observations']:,}")
    print(f"  Patterns learned: {stats['patterns_learned']:,}")
    print(f"  Tokens processed: {stats['tokens_processed']:,}")
```

#### Custom Progress Tracking

```python
from kato_hierarchical_streaming import LearningTracker

# Create custom tracker
tracker = LearningTracker()

# Manual tracking
tracker.record_pattern('node0', 'PTRN|abc123')
tracker.record_pattern('node1', 'PTRN|def456')

# Print summary
tracker.print_summary()
```

### Visualization

#### Standard Visualization

```python
from kato_hierarchical_streaming import visualize_hierarchical_stats

# After learning, visualize results
visualize_hierarchical_stats(learner)

# Creates 4 charts:
# 1. Concepts Learned by Level (sentences, paragraphs, chapters, books)
# 2. Patterns Learned by Node (node0, node1, node2, node3)
# 3. Observations by Node
# 4. Tokens Processed by Node
```

#### Custom Visualization

```python
import matplotlib.pyplot as plt

stats = learner.get_stats()

# Example: Pattern count comparison
levels = ['Sentences', 'Paragraphs', 'Chapters', 'Books']
counts = [
    stats['sentences_learned'],
    stats['paragraphs_learned'],
    stats['chapters_learned'],
    stats['books_learned']
]

plt.figure(figsize=(10, 6))
plt.bar(levels, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.title('Hierarchical Learning Progress')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(counts):
    plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
plt.show()
```

## API Reference

### CorpusSegmenter

Segment raw text into hierarchical structure.

```python
class CorpusSegmenter:
    def __init__(self)
```

**Methods:**

#### `segment_book(book_text: str, book_metadata: dict = None) -> dict`

Segment book text into hierarchical structure.

**Parameters:**
- `book_text` (str): Raw text of the book
- `book_metadata` (dict, optional): Metadata like title, author

**Returns:**
- `dict`: `{'title': str, 'author': str, 'chapters': [...]}`

**Example:**
```python
book = segmenter.segment_book(
    book_text,
    book_metadata={'title': 'My Book', 'author': 'Me'}
)
```

#### `segment_article(article_text: str, article_metadata: dict = None) -> dict`

Segment article text (uses sections instead of chapters).

**Parameters:**
- `article_text` (str): Raw text of the article
- `article_metadata` (dict, optional): Metadata

**Returns:**
- `dict`: Same structure as `segment_book()` but with sections

#### `segment_simple_text(text: str, metadata: dict = None) -> dict`

Segment text without chapter markers into single-chapter structure.

**Parameters:**
- `text` (str): Raw text
- `metadata` (dict, optional): Metadata

**Returns:**
- `dict`: Single-chapter hierarchical structure

---

### HierarchicalConceptLearner

Manages hierarchical learning across 4 KATO nodes.

```python
class HierarchicalConceptLearner:
    def __init__(self, base_url: str = "http://kato:8000", tokenizer_name: str = "gpt2")
```

**Parameters:**
- `base_url` (str): KATO server URL
- `tokenizer_name` (str): HuggingFace tokenizer name (gpt2, bert-base-uncased, etc.)

**Attributes:**
- `nodes` (dict): Dictionary of KATO nodes by level (node0, node1, node2, node3)
- `tokenizer` (TokenProcessor): Tokenizer instance
- `tracker` (LearningTracker): Progress tracker

**Methods:**

#### `learn_sentence(sentence: str, metadata: dict = None) -> str`

Learn a sentence at Node0.

**Parameters:**
- `sentence` (str): Text of the sentence
- `metadata` (dict, optional): Additional metadata to attach to observations

**Returns:**
- `str`: Pattern name (e.g., "PTRN|abc123...")

**Metadata Fields Added Automatically:**
- `sentence`: The sentence text (truncated to 100 chars)

**Example:**
```python
# Basic usage
pattern = learner.learn_sentence("AI is transforming technology.")
print(pattern)  # PTRN|a1b2c3d4...

# With custom metadata
pattern = learner.learn_sentence(
    "AI is transforming technology.",
    metadata={'source': 'research_paper', 'page': 5}
)
```

#### `learn_paragraph(paragraph: dict, metadata: dict = None) -> str`

Learn a paragraph at Node1.

**Parameters:**
- `paragraph` (dict): Dictionary with 'sentences' list
- `metadata` (dict, optional): Additional metadata (e.g., chapter info)

**Returns:**
- `str`: Pattern name for the paragraph

**Metadata Fields Added Automatically:**
- `paragraph_text`: First 100 chars of paragraph text
- `sentence_count`: Number of sentences in paragraph

**Example:**
```python
paragraph = {
    'sentences': [
        "First sentence.",
        "Second sentence.",
        "Third sentence."
    ]
}
pattern = learner.learn_paragraph(paragraph)

# With metadata from higher level (chapter)
pattern = learner.learn_paragraph(
    paragraph,
    metadata={'chapter_title': 'Introduction', 'book_title': 'My Book'}
)
```

#### `learn_chapter(chapter: dict, metadata: dict = None) -> str`

Learn a chapter at Node2.

**Parameters:**
- `chapter` (dict): Dictionary with 'paragraphs' list
- `metadata` (dict, optional): Additional metadata (e.g., book info)

**Returns:**
- `str`: Pattern name for the chapter

**Metadata Fields Added Automatically:**
- `chapter_title`: Title of the chapter
- `paragraph_count`: Number of paragraphs in chapter

#### `learn_book(book: dict, verbose: bool = True) -> str`

Learn a complete book at Node3.

**Parameters:**
- `book` (dict): Dictionary with 'title' and 'chapters' list
- `verbose` (bool): Print progress information

**Returns:**
- `str`: Pattern name for the book

#### `process_corpus(corpus: dict, verbose: bool = True)`

Process entire corpus hierarchically.

**Parameters:**
- `corpus` (dict): Dictionary with 'books' list
- `verbose` (bool): Print progress information

**Example:**
```python
corpus = {'books': [book1, book2, book3]}
learner.process_corpus(corpus, verbose=True)
```

#### `get_stats() -> dict`

Get learning statistics.

**Returns:**
- `dict`: Statistics including counts, patterns, and elapsed time

#### `get_node_stats() -> dict`

Get statistics from all nodes.

**Returns:**
- `dict`: Node-level statistics

---

### LearningTracker

Track learning progress across hierarchical levels.

```python
class LearningTracker:
    def __init__(self)
```

**Attributes:**
- `stats` (dict): Statistics dictionary

**Methods:**

#### `record_pattern(level: str, pattern_name: str)`

Record a learned pattern.

**Parameters:**
- `level` (str): Node level (node0, node1, node2, node3)
- `pattern_name` (str): Pattern identifier

#### `get_stats() -> dict`

Get current statistics.

**Returns:**
- `dict`: Current statistics with elapsed time

#### `print_summary()`

Print formatted learning summary.

---

### StandaloneMongoDBAnalyzer

Session-independent MongoDB analyzer for post-training analysis.

```python
class StandaloneMongoDBAnalyzer:
    def __init__(
        self,
        db_name: str,
        mongo_uri: str = "mongodb://localhost:27017/",
        timeout_ms: int = 30000
    )
```

**Parameters:**
- `db_name` (str): MongoDB database name (e.g., "node0_level0_kato")
- `mongo_uri` (str): MongoDB connection URI
- `timeout_ms` (int): Connection timeout in milliseconds

**Methods:**

#### `get_stats() -> dict`

Get database statistics.

**Returns:**
- `dict`: Statistics with keys:
  - `total_patterns` (int): Total pattern count
  - `avg_frequency` (float): Average pattern frequency
  - `max_frequency` (int): Maximum pattern frequency
  - `min_frequency` (int): Minimum pattern frequency

#### `get_frequency_histogram() -> dict`

Get pattern frequency distribution.

**Returns:**
- `dict`: Mapping of frequency → count (e.g., `{1: 120000, 2: 1500, 3: 200}`)

#### `get_patterns_by_frequency(min_freq: int = 1, max_freq: int = None) -> list`

Get patterns within frequency range.

**Parameters:**
- `min_freq` (int): Minimum frequency (inclusive)
- `max_freq` (int): Maximum frequency (inclusive), None for no limit

**Returns:**
- `list`: List of pattern dicts with keys: `name`, `frequency`, `length`

#### `visualize_frequency_distribution(max_freq: int = None, use_log_scale: bool = True)`

Visualize pattern frequency distribution.

**Parameters:**
- `max_freq` (int): Maximum frequency to show (None = show all)
- `use_log_scale` (bool): Use logarithmic y-axis

#### `delete_patterns_below_threshold(threshold: int) -> int`

Delete low-frequency patterns (noise cleanup).

**Parameters:**
- `threshold` (int): Delete patterns with frequency < threshold

**Returns:**
- `int`: Number of patterns deleted

#### `close()`

Close MongoDB connection.

---

### TrainingManifest

Training metadata manager for session-independent analysis.

```python
class TrainingManifest:
    def __init__(self, training_id, timestamp, nodes, tokenizer, dataset, samples_trained)
```

**Attributes:**
- `training_id` (str): Unique training identifier
- `timestamp` (str): ISO format timestamp
- `nodes` (dict): Node configurations with database names
- `tokenizer` (str): Tokenizer name used
- `dataset` (str): Dataset name
- `samples_trained` (int): Number of samples processed

**Class Methods:**

#### `create_from_learner(learner, dataset=None, samples_trained=None, training_id=None) -> TrainingManifest`

Create manifest from HierarchicalConceptLearner.

**Parameters:**
- `learner` (HierarchicalConceptLearner): The learner instance
- `dataset` (str): Dataset name
- `samples_trained` (int): Sample count
- `training_id` (str): Custom ID (auto-generated if None)

**Returns:**
- `TrainingManifest`: New manifest instance

#### `load(filepath: str) -> TrainingManifest`

Load manifest from JSON file.

**Parameters:**
- `filepath` (str): Path to manifest JSON file

**Returns:**
- `TrainingManifest`: Loaded manifest

**Instance Methods:**

#### `save(filepath: str)`

Save manifest to JSON file.

**Parameters:**
- `filepath` (str): Destination file path

#### `get_analyzers(mongo_uri: str = "mongodb://localhost:27017/", timeout_ms: int = 2000) -> dict`

Create StandaloneMongoDBAnalyzer instances for all nodes.

**Parameters:**
- `mongo_uri` (str): MongoDB connection URI
- `timeout_ms` (int): Connection timeout

**Returns:**
- `dict`: Mapping of node_name → StandaloneMongoDBAnalyzer

---

### Helper Functions

#### `discover_training_databases(mongo_uri: str = "mongodb://localhost:27017/", timeout_ms: int = 2000) -> list`

Discover all KATO databases in MongoDB.

**Parameters:**
- `mongo_uri` (str): MongoDB connection URI
- `timeout_ms` (int): Connection timeout

**Returns:**
- `list`: Sorted list of database names ending with "_kato"

#### `list_available_manifests(manifests_dir: str = 'manifests') -> list`

List all saved training manifest files.

**Parameters:**
- `manifests_dir` (str): Directory containing manifest files

**Returns:**
- `list`: List of manifest file paths

#### `load_latest_manifest(manifests_dir: str = 'manifests') -> TrainingManifest`

Load the most recent training manifest.

**Parameters:**
- `manifests_dir` (str): Directory containing manifest files

**Returns:**
- `TrainingManifest`: Most recent manifest

#### `create_training_run_nodes(run_id: str = None, num_nodes: int = 4, chunk_size: int = 8, mode: str = 'chunking', base_url: str = 'http://kato:8000') -> list`

Create nodes with unique IDs for a training run.

**Parameters:**
- `run_id` (str): Unique run identifier (auto-generated if None)
- `num_nodes` (int): Number of hierarchical nodes
- `chunk_size` (int): Token chunk size for node0
- `mode` (str): Segmentation mode
- `base_url` (str): KATO server URL

**Returns:**
- `list`: List of HierarchicalNode instances

**Example:**
```python
# Create nodes for a specific training run
nodes = create_training_run_nodes(run_id='wikitext_100k')
# Creates: node0_wikitext_100k, node1_wikitext_100k, etc.

learner = HierarchicalConceptLearner(nodes=nodes, tokenizer_name='gpt2')
```

#### `list_all_training_runs(mongo_uri: str = "mongodb://localhost:27017/", timeout_ms: int = 2000) -> dict`

Group discovered databases by training run ID.

**Parameters:**
- `mongo_uri` (str): MongoDB connection URI
- `timeout_ms` (int): Connection timeout

**Returns:**
- `dict`: Mapping of run_id → list of database names

**Example:**
```python
runs = list_all_training_runs()
# Returns: {
#   'wikitext_100k': ['node0_wikitext_100k_kato', 'node1_wikitext_100k_kato', ...],
#   'c4_50k': ['node0_c4_50k_kato', 'node1_c4_50k_kato', ...]
# }
```

#### `delete_training_run(run_id: str, mongo_uri: str = "mongodb://localhost:27017/", confirm: bool = True, timeout_ms: int = 2000)`

Delete all databases for a specific training run.

**Parameters:**
- `run_id` (str): Training run identifier
- `mongo_uri` (str): MongoDB connection URI
- `confirm` (bool): Prompt for confirmation before deletion
- `timeout_ms` (int): Connection timeout

**Example:**
```python
# Delete with confirmation prompt
delete_training_run('old_experiment')

# Delete without confirmation (use carefully!)
delete_training_run('old_experiment', confirm=False)
```

## Examples

### Example 1: Learning a Research Paper

```python
from kato_hierarchical_streaming import CorpusSegmenter, HierarchicalConceptLearner

# Read paper
with open('research_paper.txt', 'r') as f:
    paper_text = f.read()

# Segment as article
segmenter = CorpusSegmenter()
article = segmenter.segment_article(
    paper_text,
    article_metadata={
        'title': 'Deep Learning for NLP',
        'author': 'Smith et al.'
    }
)

# Learn hierarchically
learner = HierarchicalConceptLearner()
corpus = {'books': [article]}  # Articles use same structure as books
learner.process_corpus(corpus)

# Get results
stats = learner.get_stats()
print(f"Learned {stats['books_learned']} paper")
print(f"Total patterns: {sum(len(p) for p in stats['patterns_by_level'].values())}")
```

### Example 2: Batch Processing Multiple Documents

```python
from pathlib import Path

# Directory of text files
text_dir = Path('./documents')

# Segment all files
books = []
segmenter = CorpusSegmenter()

for text_file in text_dir.glob('*.txt'):
    with open(text_file, 'r') as f:
        text = f.read()

    book = segmenter.segment_simple_text(
        text,
        metadata={'title': text_file.stem}
    )
    books.append(book)

print(f"Segmented {len(books)} documents")

# Learn all documents
learner = HierarchicalConceptLearner()
corpus = {'books': books}
learner.process_corpus(corpus, verbose=True)

# Save statistics
import json
stats = learner.get_stats()
with open('learning_stats.json', 'w') as f:
    json.dump(stats, f, indent=2, default=str)
```

### Example 3: Custom Text Processing

```python
# Process text with custom structure
custom_text = """
Introduction

This is the introduction paragraph. It sets the stage.
It provides context for what follows.

Main Content

Here is the main content. It contains the core information.
Multiple sentences make up this section.

Conclusion

Finally, we conclude. This wraps up the discussion.
"""

# Manual segmentation for custom structure
book = {
    'title': 'Custom Document',
    'chapters': [
        {
            'title': 'Content',
            'paragraphs': [
                {
                    'text': 'Introduction paragraph',
                    'sentences': [
                        'This is the introduction paragraph.',
                        'It sets the stage.',
                        'It provides context for what follows.'
                    ]
                },
                {
                    'text': 'Main content',
                    'sentences': [
                        'Here is the main content.',
                        'It contains the core information.',
                        'Multiple sentences make up this section.'
                    ]
                },
                {
                    'text': 'Conclusion',
                    'sentences': [
                        'Finally, we conclude.',
                        'This wraps up the discussion.'
                    ]
                }
            ]
        }
    ]
}

# Learn it
learner = HierarchicalConceptLearner()
pattern = learner.learn_book(book)
print(f"Document learned as: {pattern}")
```

### Example 4: Analyzing Pattern Hierarchy

```python
# After learning, analyze the pattern hierarchy
stats = learner.get_stats()

# Show pattern hierarchy
print("\nPattern Hierarchy:")
print("=" * 80)

# Get a book pattern from Node3
book_patterns = stats['patterns_by_level']['node3']
if book_patterns:
    book_pattern = book_patterns[0]
    print(f"Book Pattern: {book_pattern}")

    # Get the pattern's content from Node3
    # (You'd need to query the KATO node for this)
    node3 = learner.nodes['node3']
    # ... query logic here
```

### Example 5: Different Tokenizers

```python
# Compare different tokenizers

tokenizers = ['gpt2', 'bert-base-uncased', 'roberta-base']

for tokenizer_name in tokenizers:
    print(f"\nTesting with {tokenizer_name}:")

    learner = HierarchicalConceptLearner(
        tokenizer_name=tokenizer_name
    )

    # Learn same sentence with different tokenizers
    sentence = "Machine learning is revolutionizing artificial intelligence."
    pattern = learner.learn_sentence(sentence)

    stats = learner.get_node_stats()
    print(f"  Pattern: {pattern}")
    print(f"  Tokens processed: {stats['node0']['tokens_processed']}")
```

### Example 6: Metadata Tracking for Source Attribution

```python
# Track metadata through hierarchical learning for source attribution

# Process multiple books with rich metadata
books = []

for book_file in Path('./library').glob('*.txt'):
    with open(book_file, 'r') as f:
        text = f.read()

    # Extract metadata from filename or frontmatter
    metadata = {
        'title': book_file.stem,
        'author': 'Various',
        'source_file': str(book_file),
        'year': 2024,
        'collection': 'Technical Papers'
    }

    book = segmenter.segment_simple_text(text, metadata=metadata)
    books.append(book)

# Learn with metadata
learner = HierarchicalConceptLearner()

for book in books:
    # Metadata automatically flows down:
    # Book → Chapter → Paragraph → Sentence
    pattern = learner.learn_book(book)

    print(f"Learned: {book['title']}")
    print(f"  Book pattern: {pattern}")
    print(f"  Metadata attached at all levels:")
    print(f"    - Title: {book['title']}")
    print(f"    - Author: {book['author']}")
    print(f"    - Source: {book.get('source_file', 'N/A')}")

# Now when you query predictions, the metadata helps trace patterns back to source
# For example, if Node1 generates a prediction, you can trace it back to:
#   - Which paragraph it came from
#   - Which chapter that paragraph was in
#   - Which book that chapter was from
#   - The original source file
```

### Example 7: Session-Independent Analysis Workflow

```python
from tools import (
    StandaloneMongoDBAnalyzer,
    load_latest_manifest,
    discover_training_databases
)

# Method 1: Load from auto-saved manifest
print("Loading latest training manifest...")
manifest = load_latest_manifest()

print(f"\nTraining Info:")
print(f"  ID: {manifest.training_id}")
print(f"  Dataset: {manifest.dataset}")
print(f"  Samples: {manifest.samples_trained:,}")
print(f"  Timestamp: {manifest.timestamp}")

# Get analyzers for all nodes
analyzers = manifest.get_analyzers(mongo_uri="mongodb://localhost:27017/")

# Analyze patterns at each level
print(f"\nPattern Statistics:")
for node_name, analyzer in analyzers.items():
    stats = analyzer.get_stats()
    print(f"{node_name}:")
    print(f"  Total patterns: {stats['total_patterns']:,}")
    print(f"  Avg frequency: {stats['avg_frequency']:.2f}")
    print(f"  Max frequency: {stats['max_frequency']:,}")

# Get high-frequency patterns from node0
high_freq = analyzers['node0'].get_patterns_by_frequency(min_freq=10)
print(f"\nHigh-frequency patterns (freq >= 10): {len(high_freq)}")

# Visualize frequency distribution
print("\nGenerating frequency distribution charts...")
for node_name in ['node0', 'node1', 'node2', 'node3']:
    analyzers[node_name].visualize_frequency_distribution(
        max_freq=None,  # Show all frequencies
        use_log_scale=True
    )

# Clean up connections
for analyzer in analyzers.values():
    analyzer.close()

print("\n✓ Analysis complete!")
```

### Example 8: Comparing Training Runs

```python
from tools import (
    create_training_run_nodes,
    HierarchicalConceptLearner,
    list_all_training_runs,
    StandaloneMongoDBAnalyzer,
    train_from_streaming_dataset_parallel
)

# Experiment 1: Small dataset
print("Training Run 1: 100 samples...")
nodes_small = create_training_run_nodes(run_id='experiment_100samples')
learner_small = HierarchicalConceptLearner(nodes=nodes_small, tokenizer_name='gpt2')

train_from_streaming_dataset_parallel(
    dataset_key='wikitext',
    max_samples=100,
    learner=learner_small,
    num_workers=4
)

# Experiment 2: Larger dataset
print("\nTraining Run 2: 1000 samples...")
nodes_large = create_training_run_nodes(run_id='experiment_1000samples')
learner_large = HierarchicalConceptLearner(nodes=nodes_large, tokenizer_name='gpt2')

train_from_streaming_dataset_parallel(
    dataset_key='wikitext',
    max_samples=1000,
    learner=learner_large,
    num_workers=4
)

# Compare the results
print("\n" + "="*80)
print("COMPARING TRAINING RUNS")
print("="*80 + "\n")

runs = list_all_training_runs(mongo_uri="mongodb://localhost:27017/")

for run_id in ['experiment_100samples', 'experiment_1000samples']:
    print(f"\nRun: {run_id}")
    print("-" * 60)

    # Analyze each node
    for db_name in sorted(runs[run_id]):
        analyzer = StandaloneMongoDBAnalyzer(
            db_name=db_name,
            mongo_uri="mongodb://localhost:27017/",
            timeout_ms=2000
        )

        stats = analyzer.get_stats()
        node_name = db_name.split('_')[0]

        print(f"  {node_name}: {stats['total_patterns']:,} patterns "
              f"(avg freq: {stats['avg_frequency']:.2f})")

        analyzer.close()

print("\n✓ Comparison complete!")
```

### Example 9: Pattern Cleanup and Re-analysis

```python
from tools import StandaloneMongoDBAnalyzer

# Analyze patterns before cleanup
analyzer = StandaloneMongoDBAnalyzer(
    db_name="node0_level0_kato",
    mongo_uri="mongodb://localhost:27017/"
)

print("Before cleanup:")
stats_before = analyzer.get_stats()
print(f"  Total patterns: {stats_before['total_patterns']:,}")
print(f"  Average frequency: {stats_before['avg_frequency']:.2f}")

# Get frequency histogram
histogram = analyzer.get_frequency_histogram()
low_freq_count = histogram.get(1, 0)
print(f"  Patterns with frequency = 1: {low_freq_count:,}")

# Delete low-frequency patterns (noise)
threshold = 2
print(f"\nDeleting patterns with frequency < {threshold}...")
deleted = analyzer.delete_patterns_below_threshold(threshold)
print(f"  Deleted: {deleted:,} patterns")

# Re-analyze
print("\nAfter cleanup:")
stats_after = analyzer.get_stats()
print(f"  Total patterns: {stats_after['total_patterns']:,}")
print(f"  Average frequency: {stats_after['avg_frequency']:.2f}")

# Visualize improvement
print("\nGenerating comparison visualization...")
analyzer.visualize_frequency_distribution(max_freq=None, use_log_scale=True)

analyzer.close()
print("\n✓ Cleanup complete!")
```

## Configuration

### KATO Server Configuration

```python
# Default configuration
learner = HierarchicalConceptLearner(
    base_url="http://kato:8000"
)

# Custom server
learner = HierarchicalConceptLearner(
    base_url="http://localhost:8080"
)

# Remote server
learner = HierarchicalConceptLearner(
    base_url="http://kato-server.example.com:8000"
)
```

### Tokenizer Configuration

```python
# Available tokenizers (via HuggingFace)
tokenizers = [
    "gpt2",                    # GPT-2 BPE (default)
    "bert-base-uncased",       # BERT WordPiece
    "roberta-base",            # RoBERTa byte-level BPE
    "t5-small",                # T5 seq2seq
    "albert-base-v2",          # ALBERT
    "distilbert-base-uncased", # Distilled BERT
    "xlnet-base-cased",        # XLNet
    "facebook/bart-base",      # BART
]

learner = HierarchicalConceptLearner(
    tokenizer_name="bert-base-uncased"
)
```

### Node Configuration

The nodes are pre-configured with optimal settings for hierarchical learning:

```python
# Configuration applied to all 4 nodes:
{
    'max_pattern_length': 0,     # Manual learning only
    'stm_mode': 'CLEAR',          # Clear STM after each learn
    'node_id': 'node{N}_{level}', # Unique identifier
}
```

**Why these settings?**

- `max_pattern_length=0`: Prevents auto-learning, giving you full control over when patterns are learned (at concept boundaries)
- `stm_mode='CLEAR'`: Ensures STM is cleared after each learn operation, maintaining clean concept boundaries

### Performance Optimization Configuration

**⚡ Batching (4-7x speedup)**

The `node0_batch_size` parameter accumulates multiple observations before making a single API call:

```python
# Default (no batching, slow)
learner = HierarchicalConceptLearner(
    nodes=nodes,
    tokenizer_name='gpt2',
    node0_batch_size=1  # One API call per chunk
)

# RECOMMENDED: Enable batching (4-7x faster)
learner = HierarchicalConceptLearner(
    nodes=nodes,
    tokenizer_name='gpt2',
    node0_batch_size=50  # Batch 50 chunks per API call
)

# Aggressive batching (for very large datasets)
learner = HierarchicalConceptLearner(
    nodes=nodes,
    tokenizer_name='gpt2',
    node0_batch_size=100  # Maximum batching
)
```

**⚡ Parallel Workers (2-3x speedup)**

```python
from tools import train_from_streaming_dataset_parallel

# Train with parallel workers (combines with batching!)
stats = train_from_streaming_dataset_parallel(
    dataset_key='wikitext',
    max_samples=10000,
    learner=learner,  # With node0_batch_size=50
    profiler=profiler,
    num_workers=3,  # RECOMMENDED: 2-4 workers
    num_levels=4
)

# Total speedup: ~10-15x (batching × parallelism)
```

**Connection pool safety:** `workers × nodes ≤ 30` (to prevent database connection exhaustion)

**📌 Checkpoint/Resume Configuration**

```python
# Training with checkpoints (auto-save progress every 5K samples)
stats = train_from_streaming_dataset_parallel(
    dataset_key='wikitext',
    max_samples=100000,
    learner=learner,
    profiler=profiler,
    checkpoint_interval=5000,  # Save every 5K samples
    checkpoint_dir='./checkpoints',
    resume_from_checkpoint=False  # Set True to resume interrupted training
)

# Resume after crash/interruption
# 1. Set resume_from_checkpoint=True
# 2. Use EXACT same learner configuration (validated automatically)
# 3. Training skips already-processed samples
stats = train_from_streaming_dataset_parallel(
    dataset_key='wikitext',
    max_samples=100000,
    learner=learner,  # Must match checkpoint config!
    profiler=profiler,
    resume_from_checkpoint=True  # Resume from last checkpoint
)
```

**Config validation:** System validates that your current learner configuration matches the checkpoint to prevent data corruption. Mismatches result in clear error messages with actionable fixes.

### Segmentation Configuration

```python
# Customize chapter detection pattern
import re

segmenter = CorpusSegmenter()

# Modify the chapter pattern in segment_book
custom_pattern = r'\n\s*(?:Part|PART)\s+\d+\s*[:\n]'

# You can also create custom segmentation logic
def custom_segment(text):
    # Your custom logic here
    chapters = []
    # ... segmentation logic
    return {
        'title': 'Custom',
        'chapters': chapters
    }
```

## Troubleshooting

### KATO Server Connection Issues

**Problem:** `ConnectionError: Failed to connect to KATO server`

**Solutions:**
```python
# 1. Check if server is running
import requests
try:
    response = requests.get('http://kato:8000/health')
    print("Server is running:", response.json())
except:
    print("Server is not accessible")

# 2. Verify server URL
learner = HierarchicalConceptLearner(
    base_url="http://localhost:8000"  # Try different URL
)

# 3. Check Docker/server logs
# docker logs kato-container
```

### NLTK Download Issues

**Problem:** `LookupError: punkt tokenizer not found`

**Solution:**
```python
import nltk
nltk.download('punkt')

# Or download to specific directory
nltk.download('punkt', download_dir='/path/to/nltk_data')
```

### Memory Issues with Large Texts

**Problem:** Out of memory when processing large books

**Solution:**
```python
# Process in chunks
def process_large_book_in_chunks(book, chunk_size=5):
    chapters = book['chapters']

    for i in range(0, len(chapters), chunk_size):
        chunk_chapters = chapters[i:i+chunk_size]
        chunk_book = {
            'title': f"{book['title']} (Part {i//chunk_size + 1})",
            'chapters': chunk_chapters
        }

        pattern = learner.learn_book(chunk_book)
        print(f"Processed chunk: {pattern}")
```

### Tokenizer Loading Issues

**Problem:** `OSError: Can't load tokenizer`

**Solution:**
```python
# 1. Install transformers
pip install --upgrade transformers

# 2. Use different tokenizer
learner = HierarchicalConceptLearner(
    tokenizer_name="gpt2"  # More widely available
)

# 3. Pre-download tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Then use it
```

### Pattern Name Not Returning

**Problem:** `learn()` returns empty or 'UNKNOWN' pattern name

**Solution:**
```python
# Check KATO response format
result = learner.nodes['node0'].learn()
print("Full result:", result)

# Try both possible keys
pattern_name = result.get('pattern_name', result.get('name', 'UNKNOWN'))

# Verify STM has content before learning
stm = learner.nodes['node0'].get_stm()
print(f"STM length: {len(stm)}")
if len(stm) < 2:
    print("Warning: STM has less than 2 events")
```

### Visualization Not Displaying

**Problem:** `visualize_hierarchical_stats()` doesn't show chart

**Solution:**
```python
import matplotlib.pyplot as plt

# 1. Use interactive backend
plt.ion()

# 2. Explicitly show plot
visualize_hierarchical_stats(learner)
plt.show()

# 3. Save to file instead
def save_visualization(learner, filename='learning_stats.png'):
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    visualize_hierarchical_stats(learner)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved to {filename}")
```

## Advanced Topics

### Multi-Scale Prediction

Once patterns are learned, you can query predictions at any level:

```python
# After learning, query predictions

# Sentence-level prediction
node0 = learner.nodes['node0']
sentence_predictions = node0.get_predictions()

# Paragraph-level prediction
node1 = learner.nodes['node1']
paragraph_predictions = node1.get_predictions()

# Analyze prediction quality
for pred in sentence_predictions[:3]:
    print(f"Pattern: {pred['name'][:30]}...")
    print(f"  Confidence: {pred['confidence']:.3f}")
    print(f"  Similarity: {pred['similarity']:.3f}")
    print(f"  Predictive Information: {pred['predictive_information']:.3f}")
```

### Pattern Inspection

```python
# Inspect learned patterns
stats = learner.get_stats()

# Get specific pattern
pattern_name = stats['patterns_by_level']['node0'][0]

# Query KATO for pattern details
# (Requires KATO API call - implementation depends on KATO version)
```

### Custom Learning Workflows

```python
# Create custom learning workflow

class CustomHierarchicalLearner(HierarchicalConceptLearner):
    def learn_section(self, section_paragraphs):
        """Learn a custom 'section' between paragraph and chapter level."""

        # Learn paragraphs
        para_patterns = []
        for para in section_paragraphs:
            pattern = self.learn_paragraph(para)
            para_patterns.append(pattern)

        # Observe in a temporary node
        # (You'd need to create an additional node for this)

        return para_patterns
```

### Integration with Streaming Datasets

```python
# Process streaming data hierarchically
from datasets import load_dataset

# Load streaming dataset
dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train', streaming=True)

segmenter = CorpusSegmenter()
learner = HierarchicalConceptLearner()

# Process first N articles
for i, example in enumerate(dataset.take(100)):
    text = example['text']

    if len(text) < 100:  # Skip short texts
        continue

    # Segment
    article = segmenter.segment_simple_text(
        text,
        metadata={'title': f'Article {i}'}
    )

    # Learn
    pattern = learner.learn_book(article, verbose=False)

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} articles")
        learner.tracker.print_summary()
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def learn_book_wrapper(args):
    learner, book = args
    return learner.learn_book(book, verbose=False)

# Create multiple learners (one per thread)
# Note: Each needs unique node IDs to avoid conflicts
learners = [
    HierarchicalConceptLearner() for _ in range(4)
]

# Process books in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(
        learn_book_wrapper,
        [(learners[i % 4], book) for i, book in enumerate(books)]
    )

    pattern_names = list(results)
    print(f"Learned {len(pattern_names)} books in parallel")
```

### Export and Analysis

```python
import json
import pandas as pd

# Export learning statistics
stats = learner.get_stats()
node_stats = learner.get_node_stats()

# Create comprehensive report
report = {
    'summary': stats,
    'nodes': node_stats,
    'patterns': {
        level: patterns
        for level, patterns in stats['patterns_by_level'].items()
    }
}

# Save as JSON
with open('learning_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

# Create DataFrame for analysis
df = pd.DataFrame({
    'Level': ['Sentences', 'Paragraphs', 'Chapters', 'Books'],
    'Count': [
        stats['sentences_learned'],
        stats['paragraphs_learned'],
        stats['chapters_learned'],
        stats['books_learned']
    ],
    'Patterns': [
        len(stats['patterns_by_level']['node0']),
        len(stats['patterns_by_level']['node1']),
        len(stats['patterns_by_level']['node2']),
        len(stats['patterns_by_level']['node3'])
    ]
})

print(df.to_markdown())
df.to_csv('learning_stats.csv', index=False)
```

## Documentation

### Related Documentation

- **KATO Documentation**: `/Users/sevakavakians/PROGRAMMING/kato/docs/`
- **Specification**: `HIERARCHICAL_CONCEPT_LEARNING.md`
- **Pattern Matching**: `../kato/docs/PATTERN_MATCHING.md`
- **Prediction Objects**: `../kato/docs/technical/PREDICTION_OBJECT_REFERENCE.md`
- **API Reference**: `../kato/docs/API_REFERENCE.md`
- **Glossary**: `../kato/docs/GLOSSARY.md`

### Key Concepts

- **Pattern**: Learned structure identified by `PTRN|<hash>`
- **STM (Short-Term Memory)**: Temporary storage for observations
- **LTM (Long-Term Memory)**: Persistent pattern storage (MongoDB)
- **Prediction Object**: Output from pattern matching with temporal segmentation
- **max_pattern_length**: Controls auto-learning behavior
- **stm_mode**: Controls STM behavior (ROLLING vs CLEAR)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd kato-notebooks

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests (if available)
pytest tests/
```

## License

[Specify your license here]

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{kato_hierarchical_learning,
  title={KATO Hierarchical Concept Learning},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

## Support

For questions and support:

- **Issues**: [GitHub Issues](your-repo-url/issues)
- **Documentation**: See `HIERARCHICAL_CONCEPT_LEARNING.md`
- **KATO Documentation**: `/Users/sevakavakians/PROGRAMMING/kato/docs/`

## Acknowledgments

- Built on KATO (Knowledge Abstraction for Traceable Outcomes) by Intelligent Artifacts
- Based on GAIuS (General Autonomous Intelligence using Symbols) architecture
- Uses HuggingFace Transformers for tokenization
- NLTK for sentence segmentation

---

**Version**: 1.0.0
**Last Updated**: 2025-10-09
**Status**: Production Ready
