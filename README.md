# KATO Hierarchical Concept Learning

A Python implementation of hierarchical concept-based learning using KATO (Knowledge Abstraction for Traceable Outcomes) cognitive processors. This system learns text at multiple levels of abstraction, creating true hierarchical representations from books down to individual tokens.

## Table of Contents

- [Getting Started](#getting-started)
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

## Getting Started 🎉

Read PROJECT_OVERVIEW.md.

### 📓 **Main Notebooks**

1. **`training.ipynb`** - Production training on real datasets
   - Hardware profiling and analysis
   - Real data from HuggingFace (WikiText, C4, RefinedWeb, etc.)
   - Parallel workers for optimal speed (2-3x speedup)
   - Training history tracking
   - Checkpoint/resume support with config validation

2. **`generation.ipynb`** - Text generation with hierarchical predictions
   - Bottom-up activation (input → predictions)
   - Top-down unraveling (patterns → tokens)
   - Explicit KATO API calls (educational)
   - Pattern retrieval from KATO storage
   - ⚠️ **Critical:** Uses `pred['present']` (not `pred['name']`) to avoid token repetition

### 🛠️ **Essential Tools** (in `tools/` directory)

- `hierarchical_builder.py` - Layer-based API for educational training (v2.0)
- `hierarchical_learning.py` - Core training engine with batching (4-7x speedup)
- `kato_client.py` - KATO API client with auto-recovery and session management
- `streaming_dataset_loader.py` - HuggingFace dataset streaming with parallel workers

### 📚 **Documentation**

- **PROJECT_OVERVIEW.md** - Core concepts and philosophy
- **CLAUDE.md** - Development instructions
- **frequency_analysis/Pattern_Frequency_Analysis.md** - KATO Pattern Frequency Analysis Results

### 🚀 **Quick Workflow**

```bash
# 1. Train a model
jupyter notebook training.ipynb
# → Run cells to train on real data with your chosen configuration
# → Supports checkpoint/resume if training is interrupted

# 2. Generate text
jupyter notebook generation.ipynb
# → Test hierarchical text generation
# → Bottom-up activation + top-down unraveling
```
---

## Overview

Traditional language models process text sequentially without clear concept boundaries. This implementation uses KATO's cognitive architecture to create **true hierarchical abstraction** where:

- **Token Chunks** (15 tokens) are learned as phrase-level patterns
- **Paragraphs** are learned as complete patterns of chunk patterns (~225 tokens)
- **Chapters** are learned as complete patterns of paragraph patterns (~3,375 tokens)
- **Books** are learned as complete patterns of chapter patterns (~50,625 tokens)

Each level learns a **complete pattern** before passing its symbolic representation (pattern name) to the next level. Semantics emerge through hierarchical composition, not from individual chunks.

### Key Features

- ✨ **True Hierarchical Abstraction** - Each level learns increasingly abstract patterns
- 🎯 **Concept Boundaries** - Clear separation prevents mixing unrelated concepts
- 🔗 **Pattern Name Propagation** - Symbolic compression via `PTRN|<hash>` identifiers
- 📊 **Progress Tracking** - Real-time statistics at all levels
- 🚀 **Streaming Support** - Process large corpora efficiently
- 💾 **Session-Independent Analysis** - Analyze trained models without active sessions
- 🗄️ **Persistent Storage** - Training data persists in KATO storage across kernel restarts

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


## License

Apache

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{kato_hierarchical_learning,
  title={KATO Hierarchical Concept Learning},
  author={Sevak Avakians},
  year={2025},
  url={https://github.com/sevakavakians/kato-lm}
}
```

## Support

For questions and support:

- **Issues**: [GitHub Issues](https://github.com/sevakavakians/kato-lm/issues)
- **Documentation**: See `HIERARCHICAL_CONCEPT_LEARNING.md`
- **KATO Documentation**: `https://github.com/sevakavakians/kato/tree/main/docs`

## Acknowledgments

- Built on KATO (Knowledge Abstraction for Traceable Outcomes) by Sevak Avakians (2025) https://github.com/sevakavakians/kato
- Based on GAIuS (General Artificial Intelligence using Symbols) neuro-symbolic cognitive architecture created by Sevak Avakians (2008)
- Uses HuggingFace Transformers for tokenization
- NLTK for sentence segmentation
- KATO Dashboard https://github.com/sevakavakians/kato-dashboard

---

**Version**: 1.0.0
**Last Updated**: 2025-10-09
**Status**: Production Ready
