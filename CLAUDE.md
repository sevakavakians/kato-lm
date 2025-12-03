# Claude Instructions for KATO Hierarchical Learning Project

## Project Context

**IMPORTANT**: Before working on any task in this project, read the comprehensive project documentation:

📖 **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**

This document contains:
- Project goals and philosophy
- Architecture and design decisions
- Key concepts (single-pass training, hierarchical abstraction, pattern names)
- Technical approach and data flow
- Use cases and future directions
- Getting started examples

## Quick Reference

### Project Structure

```
kato-notebooks/
├── tools/                              # Python package (hierarchical learning)
│   ├── __init__.py
│   ├── kato_client.py                 # KATO API client
│   ├── streaming_dataset_loader.py    # HuggingFace datasets
│   ├── hardware_analyzer.py           # Performance estimation
│   └── hierarchical_learning.py       # Main module
├── hierarchical_training.ipynb        # Jupyter notebook
├── PROJECT_OVERVIEW.md                # **READ THIS FIRST**
└── CLAUDE.md                          # This file
```

### Key Principles

1. **Single-Pass Training**: Pattern names flow up hierarchy in real-time, NOT multiple iterations
2. **Structural Boundaries Trigger Learning**: Learn at paragraph/chapter/book completion
3. **Arbitrary Depth**: System supports any number of hierarchical levels (not hardcoded)
4. **Token Chunking**: Fixed-length chunks (default: 15 tokens) for scalable pattern learning
   - node0: CHUNK-level patterns (15-token sequences)
   - node1: PARAGRAPH-level patterns (sequences of chunk pattern names, ~225 tokens)
   - node2: CHAPTER-level patterns (sequences of paragraph pattern names, ~3,375 tokens)
   - node3: BOOK-level patterns (sequences of chapter pattern names, ~50,625 tokens)
5. **Streaming Data**: Large datasets (C4, RefinedWeb) via HuggingFace streaming
6. **ClickHouse/Redis Storage**: KATO now uses ClickHouse + Redis (MongoDB removed)

## CRITICAL: KATO Event Ordering Behavior

**⚠️ KATO does NOT preserve symbol order within events:**
- ✓ **Order IS preserved**: Between events in a sequence (event1 → event2 → event3)
- ✗ **Order is NOT preserved**: Within a single event (symbols are sorted alphanumerically)

**Implication for token storage:**

✓ **CORRECT** - Each token is its own event:
```python
[{'strings': ["The"]}, {'strings': ["field"]}, {'strings': ["of"]}]
```
Order preserved between events.

✗ **WRONG** - Multiple tokens in one event:
```python
[{'strings': ["rap", "id", "ly"]}]  # "rapidly" tokenized
```
KATO sorts to `["id", "ly", "rap"]` → **Order destroyed!**

**Golden Rule: ONE token = ONE event**

This is why our observation structures look like:
```python
observations = [{'strings': [token]} for token in tokens]  # ✓ Correct
```

Not:
```python
observations = [{'strings': tokens}]  # ✗ Wrong - loses order
```

## CRITICAL: KATO Prediction Structure (Text Generation)

**⚠️ When working with predictions, use `pred['present']` NOT `pred['name']`:**

KATO predictions contain multiple fields with distinct purposes:

```python
{
    'name': 'hash_id',                  # Pattern identifier (lookup key)
    'present': [['tok1'], ['tok2']],    # Actual matched sequence (KATO events)
    'future': [['tok3']],                # Predicted next tokens/patterns
    'matches': ['tok1', 'tok2'],        # Simplified matched tokens
    'confidence': 0.95,                  # Prediction metrics
    'potential': 0.87,
    # ... other metrics
}
```

### The Critical Distinction

**`pred['name']`** (Pattern Hash):
- Just an identifier like `"6850d8ef6abf023e778693c4d5d9986db464e5cd"`
- Used to look up the **full stored pattern** via API
- The stored pattern contains the **entire training sequence**, including future tokens
- ❌ **DO NOT use for text generation** - causes repetition!

**`pred['present']`** (Matched Sequence):
- Contains the **exact matched tokens** in KATO event format
- Represents only what was actually matched, no more, no less
- No API lookup needed - direct extraction
- ✅ **USE THIS for text generation** - prevents repetition!

### Why Using `pred['name']` Causes Repetition

When you look up a pattern by `pred['name']`, KATO returns the **full stored pattern from training time**, which includes tokens beyond what was actually matched:

```python
# ❌ WRONG APPROACH:
pattern_name = pred['name']  # "6850d8ef6abf..."
present_tokens = unravel_pattern(pattern_name, ...)  # API lookup
# Returns: ['Among', 'fl', 'ukes', ',', 'the', 'most', 'common', 'in']
#                                                                  ↑↑
#                                              From training sequence!

future_tokens = unravel_future_list(pred['future'], ...)
# Returns: ['in']  ← Same token!

combined = present_tokens + future_tokens
# Result: "...common in in"  ❌ REPETITION!
```

### Correct Approach - Use `pred['present']`

```python
# ✅ CORRECT APPROACH:
present_events = pred.get('present', [])  # Matched sequence
present_tokens = extract_tokens_from_present(present_events)
# Returns: ['Among', 'fl', 'ukes', ',', 'the', 'most', 'common']
#                                                              ↑
#                                              Stops exactly here!

future_tokens = unravel_future_list(pred['future'], ...)
# Returns: ['in']

combined = present_tokens + future_tokens
# Result: "...common in"  ✓ NO REPETITION!
```

### Helper Function

```python
def extract_tokens_from_present(present_events):
    """Extract tokens directly from pred['present'] field."""
    if not present_events:
        return []

    tokens = []
    for event in present_events:
        if event and len(event) > 0:
            tokens.append(event[0])  # Take first string from event

    return tokens
```

### When This Matters

This distinction is **critical** when:
- Implementing text generation (combining present + future)
- Unraveling predictions to tokens
- Decoding hierarchical patterns for output

This distinction is **not relevant** when:
- Just storing pattern names for training
- Analyzing pattern frequencies
- Working with pattern identifiers

### Reference

See `GENERATION_FIX_SUMMARY.md` for complete details on the bug fix that discovered this issue.

## Token Chunking Strategy

**Why Fixed-Length Chunking:**

The system uses fixed-length token chunking (default N=15) instead of sentence segmentation because:

1. **Deduplication**: Sentences are unique, chunks repeat
   - 1M sentences → ~1M unique patterns (no compression)
   - 1M samples chunked → ~5-10M chunks with high reuse (compression via frequency)

2. **Scalability**: node0 knowledge base remains manageable
   - With sentences: KB size ≈ corpus size
   - With chunking: KB size << corpus size (Zipfian distribution)

3. **Hierarchical Composition**: Semantics emerge from hierarchy
   - node0 (15 tokens): Phrase fragments, no semantic awareness
   - node1 (15 × 15 = 225 tokens): ~2-3 sentences worth of context
   - node2 (15 × 225 = 3,375 tokens): ~2-3 paragraphs
   - node3 (15 × 3,375 = 50,625 tokens): Full articles/chapters

4. **Robustness**: No fragile boundary detection heuristics
   - Arithmetic splitting always works
   - No tokenizer-specific logic needed
   - Predictable behavior

**Configuration:**
```python
segmenter = CorpusSegmenter(chunk_size=15)  # 10-25 recommended
```

**Example:**
```python
tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "then",
          "the", "dog", "ran", "away", "very", "quickly", "today"]
# With chunk_size=5:
chunks = [
    ["The", "cat", "sat", "on", "the"],
    ["mat", "and", "then", "the", "dog"],
    ["ran", "away", "very", "quickly", "today"]
]
# Each chunk → one node0 pattern
# Sequence of chunk pattern names → one node1 pattern
```

**How Granularity Emerges:**

| Level | Pattern Granularity | Learned From | Token Coverage |
|-------|-------------------|--------------|----------------|
| node0 | Token chunks | 15-token sequences | 15 tokens |
| node1 | Paragraphs | Sequences of chunk pattern names | ~225 tokens |
| node2 | Chapters | Sequences of paragraph pattern names | ~3,375 tokens |
| node3 | Books | Sequences of chapter pattern names | ~50,625 tokens |

**Key Insight:** Like visual processing—pixels (chunks) → edges (paragraphs) → shapes (chapters) → objects (books). Semantics emerge from composition, not from the bottom layer.

### Working with This Project

**Before making changes:**
- Read PROJECT_OVERVIEW.md to understand design decisions
- Check if similar functionality already exists
- Understand the single-pass training flow
- Remember: pattern names are first-class symbolic representations

**Common Tasks:**
- Adding new datasets → `tools/streaming_dataset_loader.py`
- Modifying training logic → `tools/hierarchical_learning.py`
- Analysis utilities → `tools/kato_storage/` (ClickHouse/Redis)
- Jupyter experiments → `hierarchical_training.ipynb`

**Important Constraints:**
- Do NOT change to multi-pass training (by design)
- Do NOT hardcode hierarchy depth (support arbitrary N)
- Do NOT modify KATO server configuration without understanding implications
- Do NOT create documentation files unless explicitly requested

### Dependencies

This project requires:
- **KATO Server**: http://localhost:8000 (must be running)
- **ClickHouse + Redis**: KATO's storage backend (replaces MongoDB)
- **Python packages**: datasets, transformers, clickhouse-connect, redis, matplotlib, tqdm

### Related Projects

- **Main KATO Project**: `/Users/sevakavakians/PROGRAMMING/kato/` (the KATO server)
- **This Project**: `/Users/sevakavakians/PROGRAMMING/kato-notebooks/` (hierarchical learning)

These are separate projects. Changes to KATO server happen in `/kato/`, changes to hierarchical learning happen here in `/kato-notebooks/`.

## Session Initialization

When starting a new session in this directory:

1. ✅ Read **PROJECT_OVERVIEW.md** for complete context
2. ✅ Check recent git commits for latest changes
3. ✅ Understand current working state
4. ✅ Ask clarifying questions if goals are unclear

## Development Philosophy

- **Research-Oriented**: This is an experimental framework, not production code
- **Flexibility First**: Support different tokenizers, datasets, hierarchy depths
- **Symbolic Learning**: Pattern names are symbolic, not continuous embeddings
- **Deterministic**: KATO uses deterministic pattern matching, not neural networks

## Success Criteria

The system is working correctly when:
- ✅ Frequency distributions show Zipfian patterns
- ✅ High-frequency patterns are meaningful when inspected
- ✅ Pattern counts decrease at higher levels (node0 > node1 > node2 > node3)
- ✅ Cleanup removes noise effectively
- ✅ Training completes without errors

---

**Remember**: This project explores hierarchical concept learning through symbolic pattern abstraction. Read PROJECT_OVERVIEW.md for the complete picture.
