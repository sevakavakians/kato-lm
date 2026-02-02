# KATO Hierarchical Learning: Architecture Guide

**Purpose**: This document explains the correct hierarchical architecture used in KATO training and generation. It captures the multi-timescale processing paradigm and clarifies intentional differences between training and generation modes.

---

## Table of Contents

1. [Core Concept: Multi-Timescale Processing](#core-concept-multi-timescale-processing)
2. [Training Architecture](#training-architecture)
3. [Generation Architecture](#generation-architecture)
4. [Intentional Differences (NOT Bugs)](#intentional-differences-not-bugs)
5. [Fuzzy Matching & Signal Emergence](#fuzzy-matching--signal-emergence)
6. [STM Clearing Behavior](#stm-clearing-behavior)
7. [Example Walkthrough](#example-walkthrough)
8. [Common Misconceptions](#common-misconceptions)
9. [Debug Checklist](#debug-checklist)

---

## Core Concept: Multi-Timescale Processing

The hierarchical architecture operates on **different timescales** at each level. This creates natural hierarchical abstraction without explicit semantic understanding at the lowest level.

### Timescales by Level

| Level | STM Size | Fills Every | Token Coverage | Clears Every |
|-------|----------|-------------|----------------|--------------|
| node0 | 8 events | 8 tokens | 8 tokens | 1 chunk (8 tokens) |
| node1 | 8 events | 8 chunks | 64 tokens | 8 chunks |
| node2 | 8 events | 8 node1 patterns | 512 tokens | 64 chunks |
| node3 | 8 events | 8 node2 patterns | 4,096 tokens | 512 chunks |

**Key Insight**:
- node0 processes tokens rapidly (clears every 8 tokens)
- node1 accumulates context across 8 chunks before clearing
- node2 accumulates context across 64 chunks before clearing
- node3 accumulates context across 512 chunks before clearing

This multi-timescale behavior creates hierarchical compression naturally.

---

## Training Architecture

### Purpose
Learn clean, unambiguous patterns from text at multiple abstraction levels.

### Processing Mode
**Batch / Level-by-Level**

### Input
Full documents or long text samples, processed completely before moving to the next sample.

### Event Structure
**1 symbol per event** - Clean, deterministic sequences

```python
# Example at node1
Event 1: {pattern_A}    # 1 symbol
Event 2: {pattern_B}    # 1 symbol
Event 3: {pattern_C}    # 1 symbol
...
Event 8: {pattern_H}    # 1 symbol
```

### Processing Flow

```
Text → Tokenize → [tok1, tok2, ..., tok400]

Chunk tokens (8 per chunk):
  [[tok1-8], [tok9-16], [tok17-24], ..., [tok393-400]]  # 50 chunks

Process ALL chunks at node0:
  chunk1 → observe 8 tokens → learn → pattern_A
  chunk2 → observe 8 tokens → learn → pattern_B
  ...
  chunk50 → observe 8 tokens → learn → pattern_Z''

  Result: [pattern_A, pattern_B, ..., pattern_Z'']  # 50 patterns

Chunk node0 patterns (8 per chunk):
  [[pattern_A-H], [pattern_I-P], ..., [pattern_zz-Z'']]  # 7 chunks (50/8 rounded up)

Process ALL chunks at node1:
  chunk1 → observe 8 patterns → learn → pattern1_node1
  chunk2 → observe 8 patterns → learn → pattern2_node1
  ...
  chunk7 → observe 2 patterns → learn → pattern7_node1

  Result: [pattern1_node1, ..., pattern7_node1]  # 7 patterns

Continue for node2 and node3...
```

### Characteristics
- ✅ Full document, batch processing
- ✅ Clean sequences (1 symbol per event)
- ✅ Deterministic pattern learning
- ✅ Level-by-level: collect ALL patterns at level N → process at level N+1
- ✅ Each level learns from **learned patterns** of previous level (top-1)

---

## Generation Architecture

### Purpose
Generate text incrementally using learned patterns, with fuzzy matching to handle uncertainty.

### Processing Mode
**Streaming / Per-Chunk Cascade**

### Input
Streaming/incremental input (may be short prompts: 1 chunk, 2 chunks, or any length).

### Event Structure
**N symbols per event** - Noisy, probabilistic ensembles

```python
# Example at node1
Event 1: {pred1, pred2, pred3, ..., pred10}      # 10 symbols (ensemble from chunk1)
Event 2: {pred1', pred2', pred3', ..., pred10'}  # 10 symbols (ensemble from chunk2)
Event 3: {pred1'', pred2'', ..., pred10''}       # 10 symbols (ensemble from chunk3)
...
Event 8: {pred1''', ..., pred10'''}              # 10 symbols (ensemble from chunk8)
```

**Why N symbols per event?**
- Each event represents a **prediction ensemble** from the previous level
- All predictions in the ensemble are hypotheses for the same chunk position
- They belong together as alternative explanations
- KATO uses fuzzy matching to find patterns despite the noise

### Processing Flow

```
Text → Tokenize → [tok1, tok2, ..., tokN]

Chunk tokens (8 per chunk):
  [[tok1-8], [tok9-16], [tok17-24], ...]

For EACH chunk incrementally:

  1. Process at node0:
     - Clear node0 STM
     - Observe 8 tokens (8 events, 1 symbol each)
     - Get predictions → ensemble of 10 patterns
     - Example: [pred_A, pred_B, pred_C, ..., pred_J]

  2. Cascade to node1:
     - Send ensemble as 1 event: {pred_A, pred_B, ..., pred_J}
     - node1 STM accumulates (now has N events, where N = chunks processed)
     - Get predictions from node1 → ensemble of 10 patterns
     - DON'T clear STM (unless it just reached 8 events)

  3. Cascade to node2:
     - Send node1 ensemble as 1 event
     - node2 STM accumulates
     - Get predictions from node2
     - DON'T clear STM (unless it just reached 8 events)

  4. Cascade to node3:
     - Send node2 ensemble as 1 event
     - node3 STM accumulates
     - Get predictions from node3
     - DON'T clear STM (unless it just reached 8 events)

Process next chunk (repeat cascade)...
```

### Characteristics
- ✅ Incremental/streaming processing (chunk by chunk)
- ✅ Noisy sequences (N symbols per event - prediction ensembles)
- ✅ Probabilistic pattern matching (fuzzy)
- ✅ Immediate cascade through all levels after each chunk
- ✅ Higher levels accumulate context across many chunks
- ✅ Each level observes **prediction ensembles** from previous level (top-K)

---

## Intentional Differences (NOT Bugs)

### These Are By Design

| Aspect | Training | Generation | Why Different? |
|--------|----------|------------|----------------|
| **Processing Mode** | Batch (level-by-level) | Streaming (per-chunk cascade) | Training has full documents, generation is incremental |
| **Event Structure** | 1 symbol per event | N symbols per event (ensembles) | Training needs clean patterns, generation needs fuzzy matching |
| **Pattern Source** | Learned patterns (top-1) | Prediction ensembles (top-K) | Training creates ground truth, generation explores possibilities |
| **STM Behavior** | Clear after each chunk of patterns | Accumulate across chunks, clear when full | Batch vs incremental context |
| **Input** | Full documents | Short prompts or streaming text | Different use cases |
| **Goal** | Learn clean patterns | Match patterns despite noise | Training vs inference |

### Why the Mismatch Works

**Training learns clean patterns:**
```
node1 pattern in KB: [{pattern_A}, {pattern_B}, {pattern_C}, {pattern_D}, {pattern_E}, {pattern_F}, {pattern_G}, {pattern_H}]
```

**Generation observes noisy sequences:**
```
node1 STM: [{A, X, Y, Z, ...}, {B, X, Y, ...}, {C, X, ...}, {D, X, ...}, {E, ...}, {F, ...}, {G, ...}, {H, ...}]
                ↑                    ↑               ↑            ↑          ↑        ↑        ↑        ↑
            signal              signal          signal       signal    signal   signal   signal   signal
```

**KATO's fuzzy matching (recall_threshold=0.6):**
- Detects signal (A, B, C, D, E, F, G, H) despite noise
- Finds the pattern learned during training
- Returns it as a prediction with confidence metrics

This is **intentional** - not a bug!

---

## Fuzzy Matching & Signal Emergence

### The Challenge

Generation doesn't have perfect information. When node0 produces predictions, we don't know which is correct. So we send **all top predictions** to the next level.

### How Signal Emerges

**Short Context (Few Chunks)**:
- Much noise: Many irrelevant patterns in ensembles
- Weak signal: Correct patterns present but hard to identify
- Result: Many predictions, low confidence

**Long Context (Many Chunks)**:
- Less noise (relatively): Incorrect patterns don't match longer sequences
- Strong signal: Correct patterns accumulate evidence across events
- Result: Fewer predictions, high confidence

### Metrics for Signal Detection

KATO predictions include metrics to identify signal:
- **Potential**: How likely the pattern continues
- **Evidence**: How much data supports this pattern
- **Confidence**: Match quality (recall_threshold)
- **Bayesian/TF-IDF**: Statistical relevance

**Top predictions with high metrics = signal, not noise**

### Why This Works

As more chunks are processed:
1. Correct patterns consistently appear in ensembles
2. KATO recognizes the recurring pattern (fuzzy match)
3. Incorrect patterns fail to match longer sequences
4. Signal-to-noise ratio improves at higher levels

**The hierarchy naturally filters noise through fuzzy matching across timescales.**

---

## STM Clearing Behavior

### Critical Rule

**Clear STM only when it fills to capacity (8 events), not before!**

### node0: Clear After Every Chunk

```python
Chunk 1:
  - Observe 8 tokens → STM full (8/8)
  - Get predictions
  - CLEAR STM

Chunk 2:
  - Observe 8 tokens → STM full (8/8)
  - Get predictions
  - CLEAR STM

# Continues for every chunk...
```

**Why?** node0 processes individual chunks. Each chunk is independent at this level.

### node1: Clear After 8 Chunks

```python
Chunk 1:  Observe 1 event (ensemble) → STM: 1/8 → DON'T CLEAR
Chunk 2:  Observe 1 event → STM: 2/8 → DON'T CLEAR
Chunk 3:  Observe 1 event → STM: 3/8 → DON'T CLEAR
Chunk 4:  Observe 1 event → STM: 4/8 → DON'T CLEAR
Chunk 5:  Observe 1 event → STM: 5/8 → DON'T CLEAR
Chunk 6:  Observe 1 event → STM: 6/8 → DON'T CLEAR
Chunk 7:  Observe 1 event → STM: 7/8 → DON'T CLEAR
Chunk 8:  Observe 1 event → STM: 8/8 → CLEAR ← Only now!
Chunk 9:  Observe 1 event → STM: 1/8 → DON'T CLEAR
# Continues...
```

**Why?** node1 builds context across multiple chunks. Clearing too early destroys context.

### node2: Clear After 64 Chunks

```python
Chunks 1-7:   node2 STM: 0/8 (node1 hasn't sent anything yet)
Chunk 8:      node1 clears → sends pattern to node2 → node2 STM: 1/8
Chunks 9-15:  node2 STM: 1/8
Chunk 16:     node1 clears → sends pattern to node2 → node2 STM: 2/8
...
Chunk 64:     node1 clears → sends pattern to node2 → node2 STM: 8/8 → CLEAR
```

**Why?** node2 operates at even longer timescales, accumulating context across many more chunks.

### node3: Clear After 512 Chunks

```python
# Similar pattern, but even slower timescale
Chunk 512:    node2 STM: 8/8 → CLEAR → sends pattern to node3 → node3 STM: 8/8 → CLEAR
```

**Why?** Highest level operates at the longest timescale, capturing document-level patterns.

### Summary Table

| Event at node0 (chunks) | node0 STM | node1 STM | node2 STM | node3 STM | Actions |
|-------------------------|-----------|-----------|-----------|-----------|---------|
| 1-7 | Varies | 1-7/8 | 0/8 | 0/8 | - |
| 8 | 8/8→CLEAR | 8/8→CLEAR | 1/8 | 0/8 | node0 clears, node1 clears + sends to node2 |
| 9-15 | Varies | 1-7/8 | 1/8 | 0/8 | - |
| 16 | 8/8→CLEAR | 8/8→CLEAR | 2/8 | 0/8 | node0 clears, node1 clears + sends to node2 |
| ... | ... | ... | ... | ... | ... |
| 64 | 8/8→CLEAR | 8/8→CLEAR | 8/8→CLEAR | 1/8 | All clear, node2 sends to node3 |

---

## Example Walkthrough

### Input: "Among flukes, the most common in"

**Tokens**: `[' Among', ' flu', 'kes', ',', ' the', ' most', ' common', ' in']` (8 tokens)

### Chunk 1 Processing

**node0**:
```
- Clear STM
- Observe 8 tokens: [' Among', ' flu', 'kes', ',', ' the', ' most', ' common', ' in']
- Get predictions: 10 patterns
  [pred_A, pred_B, pred_C, pred_D, pred_E, pred_F, pred_G, pred_H, pred_I, pred_J]
```

**node1**:
```
- Observe 1 event: {pred_A, pred_B, pred_C, pred_D, pred_E, pred_F, pred_G, pred_H, pred_I, pred_J}
- STM now has 1 event (1/8 full)
- Get predictions: Weak predictions (only 1 event of context)
  Maybe returns 10 patterns, but low confidence
```

**node2**:
```
- Observe 1 event: {node1's 10 predictions}
- STM now has 1 event (1/8 full)
- Get predictions: Very weak (only 1 event)
```

**node3**:
```
- Observe 1 event: {node2's predictions}
- STM now has 1 event (1/8 full)
- Get predictions: Very weak (only 1 event)
```

**Result after 1 chunk**: Predictions exist but are weak due to limited context.

---

### Append: " North American wolves is Alaria, which"

**Tokens**: `[' North', ' American', ' wolves', ' is', ' Al', 'aria', ',', ' which']` (8 more tokens)

### Chunk 2 Processing

**node0**:
```
- Clear STM (forget previous chunk)
- Observe 8 new tokens: [' North', ' American', ' wolves', ' is', ' Al', 'aria', ',', ' which']
- Get predictions: 10 patterns
  [pred_A', pred_B', pred_C', pred_D', pred_E', pred_F', pred_G', pred_H', pred_I', pred_J']
```

**node1**:
```
- Observe 1 event: {pred_A', pred_B', ..., pred_J'}
- STM now has 2 events (2/8 full):
  Event 1: {pred_A, pred_B, ..., pred_J}        ← From chunk 1
  Event 2: {pred_A', pred_B', ..., pred_J'}     ← From chunk 2

- CRITICAL: Among the noise (18 other patterns), 2 patterns should match a trained sequence:
  - Signal: pred_A from event 1, pred_B' from event 2
  - If during training, node1 learned a pattern [pred_A, pred_B', ...],
  - KATO should fuzzy-match and return that pattern as a prediction!

- Get predictions: Stronger predictions now (2 events of context)
  The matched pattern should have HIGH metrics:
    - potential: high
    - evidence: high
    - confidence: above recall_threshold
```

**node2**:
```
- Observe 1 event: {node1's predictions from chunk 2}
- STM now has 2 events (2/8 full)
- Get predictions: Slightly stronger (2 events)
```

**node3**:
```
- Observe 1 event: {node2's predictions}
- STM now has 2 events (2/8 full)
- Get predictions: Slightly stronger (2 events)
```

**Result after 2 chunks**:
- node1 predictions are significantly stronger
- The correct pattern should emerge with high confidence metrics
- Higher levels still need more context (only 2/8 full)

---

### As More Chunks Arrive

**Chunk 3-8**:
- node0 continues processing and clearing
- node1 accumulates more events (3/8, 4/8, ..., 8/8)
- Signal strengthens at node1 as more evidence accumulates
- At chunk 8: node1 STM is FULL (8/8) → clear STM

**Chunk 9-64**:
- node1 patterns sent to node2
- node2 accumulates events (1/8, 2/8, ..., 8/8)
- Signal strengthens at node2
- At chunk 64: node2 STM is FULL → clear STM

**Chunk 65-512**:
- node2 patterns sent to node3
- node3 accumulates events
- Signal strengthens at node3 (document-level patterns)
- At chunk 512: node3 STM is FULL → clear STM

---

## Common Misconceptions

### ❌ "Generation should send 1 symbol per event like training"

**Wrong!** Generation needs to send prediction ensembles (N symbols per event) because:
- We don't know which prediction is correct
- Fuzzy matching requires exploring multiple hypotheses
- Signal emerges through accumulation across chunks

### ❌ "Training and generation should use the same architecture"

**Wrong!** They have different goals:
- Training: Learn clean patterns (batch processing)
- Generation: Match patterns despite uncertainty (streaming processing)

The architectures are intentionally different.

### ❌ "Higher levels should clear STM after every chunk like node0"

**Wrong!** Higher levels need to accumulate context across many chunks:
- node1: 8 chunks worth of context
- node2: 64 chunks worth of context
- node3: 512 chunks worth of context

Clearing too early destroys the hierarchical context.

### ❌ "If patterns don't match exactly, it's a bug"

**Wrong!** Fuzzy matching is the design:
- Training creates clean patterns
- Generation observes noisy patterns
- KATO's recall_threshold enables matching despite noise

This mismatch is intentional and necessary.

### ❌ "Node1 should only observe from node0 after 8 chunks"

**Wrong!** In generation, the cascade happens immediately after each chunk:
- node0 processes chunk 1 → node1 observes (STM: 1/8)
- node0 processes chunk 2 → node1 observes (STM: 2/8)
- ...

Higher levels accumulate, but observation happens per chunk.

---

## Debug Checklist

When debugging hierarchical behavior, verify:

### STM Clearing
- [ ] node0 clears STM after every chunk
- [ ] node1 clears STM only after 8 events (8 chunks from node0)
- [ ] node2 clears STM only after 8 events (when node1 has sent 8 patterns)
- [ ] node3 clears STM only after 8 events (when node2 has sent 8 patterns)

### Event Structure
- [ ] Training: 1 symbol per event (pattern names)
- [ ] Generation: N symbols per event (prediction ensembles)
- [ ] Ensembles contain all predictions from same chunk/position

### Pattern Flow
- [ ] Training: Collect all patterns at level N → chunk → send to level N+1
- [ ] Generation: Immediate cascade per chunk (node0 → node1 → node2 → node3)

### Processing Paradigm
- [ ] Training: Batch, level-by-level
- [ ] Generation: Streaming, per-chunk cascade
- [ ] These are different BY DESIGN

### Signal Quality
- [ ] Short context: Many predictions, low confidence
- [ ] Long context: Fewer predictions, high confidence
- [ ] Metrics (potential, evidence) increase with more chunks

### Timescales
- [ ] node0 operates on 8-token timescale
- [ ] node1 operates on 64-token timescale
- [ ] node2 operates on 512-token timescale
- [ ] node3 operates on 4096-token timescale

---

## Summary

The KATO hierarchical architecture uses **multi-timescale processing** where each level operates at exponentially slower rates. This creates natural hierarchical abstraction.

**Training and generation are intentionally different**:
- Training: Clean, batch processing for learning patterns
- Generation: Noisy, streaming processing for inference with fuzzy matching

**The key insight**: Signal emerges from noise over time as correct patterns accumulate evidence across multiple chunks while incorrect patterns fail to match longer sequences.

This architecture enables:
- Scalable pattern learning (Zipfian compression)
- Robust inference despite uncertainty
- Natural hierarchical abstraction without explicit semantics
- Multi-scale context (tokens → paragraphs → chapters → documents)

**When in doubt, remember**: The mismatch between training and generation is not a bug—it's a feature that enables fuzzy pattern matching and robust inference.
