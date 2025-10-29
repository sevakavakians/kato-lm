# Hierarchical Learning & Text Generation Architecture

**Document Purpose:** Comprehensive reference for KATO hierarchical learning architecture, mental models, data flow, and future text generation implementation.

**Last Updated:** 2025-10-22

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Mental Model: Directed Graph with Cascading Filters](#core-mental-model-directed-graph-with-cascading-filters)
3. [Training Data Flow](#training-data-flow)
4. [Frequency Distribution Across Hierarchy](#frequency-distribution-across-hierarchy)
5. [Text Generation Architecture (Future)](#text-generation-architecture-future)
6. [Architectural Decisions](#architectural-decisions)
7. [KATO API Clarifications](#kato-api-clarifications)
8. [Implementation Notes](#implementation-notes)
9. [Open Questions & Future Work](#open-questions--future-work)

---

## Executive Summary

### Key Insights

The KATO hierarchical learning system can be conceptualized as:

1. **Directed Graph Structure**: Learned patterns form a directed graph where:
   - Nodes = learned patterns at each hierarchical level
   - Edges = "composed of" relationships between levels
   - Traversal = text generation process

2. **Cascading Constraint Satisfaction**: Each hierarchical level acts as a filter:
   - Bottom-up: Input tokens activate node0 patterns
   - Upward propagation: Pattern activations constrain higher levels
   - Top-down: High-level context (book/chapter) constrains lower-level predictions
   - Result: Exponential search space reduction

3. **Hierarchical Compression**: Pattern frequency decreases up the hierarchy:
   - node0: High-frequency building blocks (common phrases)
   - node1-3: Progressively unique compositions
   - Zipfian distribution: Power-law at each level

### Architecture Philosophy

- **Hierarchy is flexible**: Not rigidly bound to linguistic units (sentence/paragraph/chapter)
- **Metadata anchoring**: High-level patterns tagged with source metadata (book name, author, etc.)
- **Symbolic composition**: Pattern names as first-class symbolic representations
- **Deterministic pattern matching**: No neural networks, no embedding drift

---

## Core Mental Model: Directed Graph with Cascading Filters

### Graph Structure

```
                     node3: Book-level patterns
                     (100-500 unique patterns)
                            ↓
                    ┌───────┼───────┐
                    ↓       ↓       ↓
            node2: Chapter-level patterns
            (1K-5K unique patterns)
                ↓           ↓           ↓
            (15 node1    (15 node1    (15 node1
             patterns)    patterns)    patterns)
                ↓           ↓           ↓
            node1: Paragraph-level patterns
            (10K-50K unique patterns)
                ↓           ↓           ↓
            (15 node0    (15 node0    (15 node0
             patterns)    patterns)    patterns)
                ↓           ↓           ↓
            node0: Chunk-level patterns
            (100K-500K unique patterns)
                ↓           ↓           ↓
            [tokens]    [tokens]    [tokens]
```

**Graph Properties:**
- **Branching factor**: ~chunk_size (typically 3-8) at each level
- **Depth**: Number of hierarchical levels (typically 4-5)
- **Edge weights**: Pattern frequency (higher = more common)
- **Node attributes**: Pattern name, frequency, metadata, emotives

### Cascading Filter Mechanism

**During Training (Bottom-Up):**
```
Input: "The cat sat on the mat and then the dog ran away"

1. Tokenize → ["The", " cat", " sat", " on", " the", " mat", ...]
2. Chunk (size=5) → 3 chunks
3. node0 learns 3 patterns → [P0_001, P0_002, P0_003]
4. node1 observes sequence [P0_001, P0_002, P0_003]
5. When paragraph complete → node1 learns → P1_042
6. node2 observes P1_042 (waits for more paragraphs)
7. When chapter complete → node2 learns → P2_015
8. When book complete → node3 learns → P3_007
```

**During Generation (Top-Down + Bottom-Up):**
```
Input: "The cat"

1. node0 receives ["The", " cat"] (2 tokens minimum for prediction)
2. node0 predicts likely next patterns → [P0_123, P0_456, P0_789]
3. These pattern names activate node1 predictions
4. node1 returns patterns containing P0_123/456/789 → [P1_050, P1_091]
5. These activate node2 predictions → [P2_020, P2_031]
6. These activate node3 predictions → [P3_007 (metadata: book="Moby_Dick")]

Constraint satisfaction:
- node3 says "we're in Moby_Dick context"
- node2 says "we're in chapter about whales"
- node1 says "we're in descriptive paragraph"
- node0 says "next chunk likely continues cat description"

Result: Predicted tokens are constrained by ALL hierarchical levels
```

### Why This Works: Exponential Search Space Reduction

**Without hierarchy:**
- Vocabulary: 50K tokens
- Predict next 10 tokens: 50K^10 ≈ 10^47 possibilities (intractable)

**With hierarchy:**
- node0: 100K patterns → filtered to ~100 by recall_threshold (~0.1% selected)
- node1: 10K patterns → filtered to ~10 by node0 activations (~0.1% selected)
- node2: 1K patterns → filtered to ~5 by node1 activations (~0.5% selected)
- node3: 100 patterns → filtered to ~2 by node2 activations (~2% selected)

**Effective search space: ~100-500 possibilities (manageable!)**

---

## Training Data Flow

### Complete Example: Paragraph Processing

**Configuration:**
- chunk_size = 5
- tokenizer = GPT-2
- segment_method = 'simple'

**Input Text (one paragraph):**
```
Machine learning is a field of artificial intelligence. It enables computers to learn from data.
```

### Step-by-Step Processing

#### Step 1: Paragraph Splitting
```python
# segment_simple_text() splits by double newlines
paragraphs = re.split(r'\n\s*\n', text)

# Result: 1 paragraph
paragraph = "Machine learning is a field of artificial intelligence. It enables computers to learn from data."
```

#### Step 2: Tokenize Entire Paragraph
```python
# tokenize_segment() using GPT-2
tokens = tokenizer.tokenize(paragraph)

# Result (17 tokens):
tokens = [
    "Machine", " learning", " is", " a", " field", " of", " artificial",
    " intelligence", ".", " It", " enables", " computers", " to",
    " learn", " from", " data", "."
]
```

**Key Point:** GPT-2 uses subword tokenization with space prefixes:
- `" learning"` = token with leading space (mid-word boundary)
- `"Machine"` = token without space (start of text)

#### Step 3: Split into Fixed-Length Chunks
```python
# chunk_tokens() with chunk_size=5
chunks = [
    ["Machine", " learning", " is", " a", " field"],           # Chunk 1 (5 tokens)
    [" of", " artificial", " intelligence", ".", " It"],       # Chunk 2 (5 tokens)
    [" enables", " computers", " to", " learn", " from"],      # Chunk 3 (5 tokens)
    [" data", "."]                                             # Chunk 4 (2 tokens, remainder)
]
```

**Critical:** These are **non-overlapping, fixed-length** splits using arithmetic division.

#### Step 4: Send Each Chunk via observe_sequence()
```python
# For each chunk, ONE API call to KATO
for chunk in chunks:
    # Build observations (one event per token)
    observations = [
        {'strings': ["Machine"]},
        {'strings': [" learning"]},
        {'strings': [" is"]},
        {'strings': [" a"]},
        {'strings': [" field"]}
    ]

    # ONE API call per chunk (NOT 5 calls!)
    result = node0.observe_sequence(
        observations=observations,
        learn_at_end=True  # Learn once after all observations in STM
    )

    # Returns: {'final_learned_pattern': 'pattern_0001', ...}
    pattern_name = result['final_learned_pattern']
    pattern_names.append(pattern_name)

# Result: 4 API calls (one per chunk), not 17 (one per token)
# pattern_names = ['pattern_0001', 'pattern_0002', 'pattern_0003', 'pattern_0004']
```

**Why Each Token is Its Own Event:**

From CLAUDE.md critical section:
> **⚠️ KATO does NOT preserve symbol order within events**
> - Order IS preserved: Between events in a sequence
> - Order is NOT preserved: Within a single event (symbols sorted alphanumerically)

Therefore:
```python
# ✓ CORRECT - each token is its own event
[{'strings': ["Machine"]}, {'strings': [" learning"]}, ...]

# ✗ WRONG - multiple tokens in one event
[{'strings': ["Machine", " learning", " is", " a", " field"]}]
# KATO would sort alphabetically → [" a", " field", " is", " learning", "Machine"]
# Order destroyed!
```

#### Step 5: Pattern Names Flow to node1
```python
# node1 receives the sequence of node0 pattern names
node1_observations = [
    {'strings': ['pattern_0001']},
    {'strings': ['pattern_0002']},
    {'strings': ['pattern_0003']},
    {'strings': ['pattern_0004']}
]

# When paragraph complete, node1 learns
result = node1.observe_sequence(
    observations=node1_observations,
    learn_at_end=True
)

# Returns: {'final_learned_pattern': 'paragraph_pattern_042', ...}
```

#### Step 6: Hierarchy Continues Upward
```python
# node2 observes paragraph pattern names
# When chapter complete → node2 learns
# node3 observes chapter pattern names
# When book complete → node3 learns
```

### Performance Characteristics

**API Calls for One 17-Token Paragraph:**
- node0: 4 calls to `observe_sequence()` (one per chunk)
- node1: 1 call to `observe_sequence()` (paragraph complete)
- **Total: 5 API calls**

**Without `observe_sequence()` (old approach):**
- node0: 17 calls to `observe()` + 4 calls to `learn()`
- **Total: 21 API calls**
- **Speedup: ~4x with batching!**

---

## Frequency Distribution Across Hierarchy

### Why Frequency Decreases Upward

**Fundamental Principle:** Common building blocks compose into unique structures.

#### node0: Chunk-Level (High Frequency, High Reuse)
```
Pattern: ["the", " cat", " sat"]
Appears in:
  - "The cat sat on the mat"
  - "The cat sat by the window"
  - "When the cat sat down..."
  - ... (1000+ occurrences across corpus)

Frequency: 1,247
Total node0 patterns: ~100K-500K
```

**Why high frequency?**
- Fixed-length chunks create many overlapping phrases
- Common phrases like "according to the", "in the study", "as a result"
- Zipfian distribution: Few patterns dominate (power-law)

#### node1: Paragraph-Level (Medium Frequency)
```
Pattern: [P0_123, P0_456, P0_789, P0_012, P0_345, ...]  (15 node0 patterns)

Represents: Paragraph about "cat sitting behavior"

Appears in:
  - Chapter 1 of "Cat Behavior Book"
  - Article about feline habits
  - Children's story about cats
  - ... (50 occurrences)

Frequency: 50
Total node1 patterns: ~10K-50K
```

**Why medium frequency?**
- Paragraphs have more unique structure than chunks
- But similar topics create reusable patterns
- Less reuse than node0, but not fully unique

#### node2: Chapter-Level (Low Frequency)
```
Pattern: [P1_042, P1_087, P1_091, ...]  (15 node1 patterns)

Represents: Chapter about "Introduction to Animal Behavior"

Appears in:
  - Multiple books about animal behavior
  - ... (5 occurrences)

Frequency: 5
Total node2 patterns: ~1K-5K
```

**Why low frequency?**
- Chapters are distinctive
- Similar structure only in closely related texts
- Topic-specific content

#### node3: Book-Level (Very Low Frequency)
```
Pattern: [P2_015, P2_020, P2_031, ...]  (15 node2 patterns)

Represents: Entire book "Understanding Feline Behavior"

Appears: Unique to this book

Frequency: 1
Total node3 patterns: ~100-500
```

**Why very low frequency?**
- Each book is essentially unique
- Only reused if training on multiple editions of same book
- Top-level context differentiator

### Zipfian Distribution at Each Level

**Expected alpha (α) values:**
- **node0**: α ≈ 1.0-1.5 (classic Zipfian for natural language chunks)
- **node1**: α ≈ 0.8-1.2 (moderate power-law)
- **node2**: α ≈ 0.5-1.0 (weaker power-law, more uniform)
- **node3**: α ≈ 0.2-0.5 (nearly uniform, most patterns unique)

**Current experimental results (10K samples, WikiText):**
- node0: α ≈ 0.01-0.26 (lower than ideal, needs more data or larger chunk_size)

**Interpretation:** Current low alpha suggests insufficient pattern reuse at node0. Solutions:
1. Train on 100K+ samples (more data → better Zipfian emergence)
2. Try intermediate chunk_size (4-6) for sweet spot
3. Use larger datasets (C4, RefinedWeb)

---

## Text Generation Architecture (Future)

### Generation as Graph Traversal

**Core Idea:** Generation = walking the directed graph from node3 → node0 → tokens

```
Input: "The cat sat"

1. Activate node0 with input tokens
2. Get node0 predictions (filtering by recall_threshold)
3. Use predicted pattern names to query node1
4. Propagate up to node3 (get book/chapter context)
5. Use top-level context to constrain lower-level predictions
6. Select next pattern at each level (using strategy)
7. Unravel selected path down to tokens
8. Output: Generated token sequence
```

### Generation Strategies (All Configurable)

#### 1. Greedy Top-Down
```python
def generate_greedy(input_tokens, max_length=100):
    """
    Greedy generation: Always pick highest-confidence pattern at each level.

    Pros: Fast, deterministic, coherent
    Cons: Boring, repetitive, no diversity
    """
    # Activate hierarchy with input
    node0_predictions = node0.predict(input_tokens)
    node1_predictions = node1.predict([p['pattern_name'] for p in node0_predictions])
    node2_predictions = node2.predict([p['pattern_name'] for p in node1_predictions])
    node3_predictions = node3.predict([p['pattern_name'] for p in node2_predictions])

    # Pick best node3 pattern (highest confidence)
    best_node3 = max(node3_predictions, key=lambda x: x['confidence'])

    # Unravel deterministically
    node2_patterns = unravel_pattern(best_node3, deterministic=True)
    node1_patterns = [unravel_pattern(p, deterministic=True) for p in node2_patterns]
    node0_patterns = [[unravel_pattern(p, deterministic=True) for p in seq] for seq in node1_patterns]
    tokens = [[decode_pattern(p) for p in seq] for seq in node0_patterns]

    return flatten(tokens)
```

**Use case:** Predictable, safe generation (documentation, templates)

#### 2. Beam Search
```python
def generate_beam_search(input_tokens, beam_width=5, max_length=100):
    """
    Beam search: Maintain top-K hypotheses, prune low-scoring paths.

    Pros: Good balance of quality and diversity
    Cons: Slower than greedy, more complex
    """
    # Initialize beams with top-K node3 patterns
    beams = [
        (node3_pred, score=node3_pred['confidence'])
        for node3_pred in node3_predictions[:beam_width]
    ]

    generated_sequences = []

    while len(generated_sequences) < beam_width:
        # Expand each beam
        candidates = []
        for beam, score in beams:
            # Unravel one level
            node2_patterns = unravel_pattern(beam, stochastic=False)
            for node2_pat in node2_patterns:
                new_score = score * get_transition_probability(beam, node2_pat)
                candidates.append((node2_pat, new_score))

        # Keep top-K candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Continue until tokens generated
        # ... (unravel to node1, node0, tokens)

    return generated_sequences
```

**Use case:** High-quality generation (summaries, creative text)

#### 3. Sampling with Temperature
```python
def generate_sampling(input_tokens, temperature=1.0, max_length=100):
    """
    Stochastic sampling: Sample from distribution at each level.

    Pros: Diverse, creative, exploratory
    Cons: Less coherent, may diverge from topic

    Temperature:
        - 0.1: Nearly deterministic (sharp distribution)
        - 1.0: True distribution (balanced)
        - 2.0: Flattened distribution (more random)
    """
    # Get predictions at each level
    node3_predictions = get_hierarchical_predictions(input_tokens)

    # Apply temperature to confidence scores
    probs = softmax([p['confidence'] / temperature for p in node3_predictions])

    # Sample from distribution
    selected_node3 = np.random.choice(node3_predictions, p=probs)

    # Unravel stochastically
    node2_patterns = unravel_pattern(selected_node3, stochastic=True, temperature=temperature)
    # ... continue down hierarchy

    return generated_tokens
```

**Use case:** Creative writing, diverse outputs, exploration

### Unraveling Strategies (Both Configurable)

#### Deterministic Unraveling
```python
def unravel_pattern_deterministic(pattern, node_level):
    """
    Deterministic: Always return same constituents for a pattern.

    Example:
        P2_chapter1 = [P1_intro, P1_methods, P1_results, ...]
        Every call returns: [P1_intro, P1_methods, P1_results, ...]

    Pros: Reproducible, exact structure preservation
    Cons: Repetitive if same pattern appears multiple times
    """
    # Query KATO knowledgebase for pattern composition
    constituents = db.query(
        "SELECT constituent_patterns FROM patterns WHERE name = ?",
        (pattern,)
    )
    return constituents  # Always same order
```

**Use case:** Reproducing exact training data, deterministic generation

#### Stochastic Unraveling
```python
def unravel_pattern_stochastic(pattern, node_level, temperature=1.0):
    """
    Stochastic: Sample from pattern's possible constituents based on co-occurrence.

    Example:
        P2_chapter1 historically composed of:
            - [P1_intro, P1_methods, P1_results] (3 times)
            - [P1_intro, P1_background, P1_methods, P1_results] (2 times)

        Sample from these alternatives based on frequency.

    Pros: Variation, creative recombination
    Cons: May lose exact structure
    """
    # Get all historical compositions of this pattern
    compositions = db.query_all_compositions(pattern)

    # Weight by frequency
    probs = softmax([comp.frequency / temperature for comp in compositions])

    # Sample
    selected_composition = np.random.choice(compositions, p=probs)
    return selected_composition.constituents
```

**Use case:** Creative generation, variation, avoiding repetition

### Frequency-Based Priors

**Already Implemented in KATO:**

KATO's `potential` metric incorporates pattern frequency:
```python
# KATO prediction scoring (simplified)
potential = confidence * log(frequency + 1) * other_factors

# High-frequency patterns get boosted (more "natural")
# Low-frequency patterns get penalized (less common in training data)
```

**Why this works:**
- Zipfian distribution = natural language prior
- High-frequency patterns = common phrases, grammatical structures
- Low-frequency patterns = rare, potentially noisy

**Configuration:**
```python
# Adjust frequency weighting in generation
generate(
    input_tokens=["The", "cat"],
    frequency_weight=1.0,  # Default: trust Zipfian distribution
    # frequency_weight=0.5,  # Flatten: more exploration
    # frequency_weight=2.0,  # Sharpen: very conservative
)
```

### Metadata-Conditioned Generation

**Example: Generate text in specific book's style**

```python
# Query: "Generate text about whales"
input_tokens = tokenize("Generate text about whales")

# Activate hierarchy
predictions = get_hierarchical_predictions(input_tokens)

# Filter node3 predictions by metadata
whale_books = [
    p for p in node3_predictions
    if 'whale' in p.get('metadata', {}).get('keywords', [])
]

# Select: "Moby Dick" (highest confidence whale book)
selected_node3 = whale_books[0]

# Unravel → generates text in Moby Dick's style with whale vocabulary
```

**Use case:**
- Style transfer (generate like author X)
- Domain-specific generation (medical, legal, scientific)
- Source-faithful generation (avoid hallucinations)

---

## Architectural Decisions

### 1. Cold Start Handling: Empty Predictions Acceptable

**Decision:** If node0 receives completely novel input (out-of-vocabulary chunks) and recall_threshold filtering returns ZERO matches, return empty predictions.

**Rationale:**
- Empty predictions = KATO saying "I don't know"
- This is a **good result** (honest, no hallucination)
- Alternative approaches (fallback, interpolation) mask ignorance

**Example:**
```python
# Trained on English Wikipedia
node0.predict(["量子", "力学"])  # Chinese tokens

# Returns: [] (no matches)
# Interpretation: "I was not trained on Chinese text"
```

**Implication for generation:**
```python
if len(predictions) == 0:
    # Option A: Stop generation (safest)
    return "[UNKNOWN INPUT]"

    # Option B: Fallback to high-frequency patterns (unsafe, may hallucinate)
    # NOT RECOMMENDED
```

### 2. Partial Input: Works with 2+ Tokens

**Decision:** KATO starts making predictions with 2+ tokens, regardless of chunk_size or STM size.

**Rationale:**
- KATO's minimum pattern length = 2
- chunk_size is for **learning**, not prediction
- Prediction works on **partial sequences**

**Example:**
```python
# chunk_size = 5 (patterns learned from 5-token chunks)
# Input: 2 tokens
node0.predict(["The", " cat"])

# KATO finds patterns that START with ["The", " cat"]
# Returns predictions for likely 3rd, 4th, 5th tokens
```

**No special handling needed** — KATO's prediction API naturally handles partial input.

### 3. Hybrid Sliding Windows: Not Beneficial

**Decision:** Do NOT use sliding windows at any hierarchical level.

**Rationale:**
- **Storage explosion**: 2-3x more patterns at node0
- **Training time increase**: 2-3x slower
- **Artificial frequency inflation**: Not genuine reuse
- **Marginal benefit**: Overlap doesn't significantly improve pattern quality

**Exception:** If future experiments show clear benefit at higher levels (node1+), revisit.

### 4. Evaluation Priority: Faithfulness to Source

**Decision:** Primary evaluation metric = faithfulness to training data (anti-hallucination).

**Rationale:**
- KATO is **symbolic**, not generative
- Goal: Reproduce learned structures, not create new ones
- Hallucination = composition of patterns that never co-occurred in training

**Evaluation Metrics:**

1. **Pattern Provenance** (primary):
   - Every generated token sequence must trace back to training data
   - Log: Which training documents contributed to this generation?

2. **Composition Validity** (secondary):
   - Are adjacent patterns compatible? (Did they co-occur in training?)
   - Check: node1 pattern → valid node0 constituent patterns?

3. **Metadata Consistency** (tertiary):
   - Does generated text match requested metadata? (book, author, topic)

**Anti-Metrics** (DO NOT optimize for):
- Perplexity (irrelevant for symbolic system)
- BLEU score (measures similarity, not faithfulness)
- Human preference (may favor creative hallucination)

---

## KATO API Clarifications

### metadata vs emotives: Critical Distinction

#### metadata: Governance & Data Lineage

**Purpose:** Store information **about** the pattern for traceability, governance, compliance.

**Stored with pattern:** Yes (in MongoDB)

**Used during prediction:** No (does not influence scores)

**Query support:** Yes (can filter predictions by metadata)

**Example:**
```python
# During training
observations = [
    {
        'strings': ['Chapter', ' One'],
        'metadata': {
            'book': 'Moby_Dick',
            'author': 'Herman_Melville',
            'year': 1851,
            'chapter': 1,
            'dataset': 'gutenberg',
            'license': 'public_domain'
        }
    }
]

node3.observe_sequence(observations, learn_at_end=True)

# Later: Query for Melville books
melville_patterns = node3.query_patterns(
    metadata_filter={'author': 'Herman_Melville'}
)
```

**Use cases:**
- Data lineage ("Where did this pattern come from?")
- Copyright compliance ("Which books are public domain?")
- Dataset attribution ("Show me patterns from WikiText")
- Temporal filtering ("Only use recent data")
- Source control ("Exclude specific sources")

#### emotives: Prediction Steering

**Purpose:** Influence prediction scores to steer generation toward desired **characteristics**.

**Stored with pattern:** Yes (with persistence window)

**Used during prediction:** Yes (affects potential scores)

**Query support:** Yes (can request specific emotive profiles)

**Example:**
```python
# During training - tag patterns with sentiment
observations = [
    {
        'strings': ['The', 'cat', 'purred', 'contentedly'],
        'emotives': {
            'happy': 0.9,
            'calm': 0.8,
            'anxiety': 0.1,
            'excitement': 0.3
        }
    },
    {
        'strings': ['The', 'dog', 'barked', 'aggressively'],
        'emotives': {
            'happy': 0.2,
            'calm': 0.1,
            'anxiety': 0.8,
            'excitement': 0.9
        }
    }
]

# During generation - steer toward happy/calm text
predictions = node0.predict(
    input_tokens=['The'],
    emotive_target={'happy': 0.9, 'calm': 0.8}
)

# Result: Predictions biased toward "cat purred" over "dog barked"
# Because cat pattern has emotive profile closer to target
```

**Use cases:**
- **Sentiment control**: Generate positive/negative/neutral text
- **Mood steering**: Calm vs anxious vs excited tone
- **Formality**: Formal vs casual language
- **Complexity**: Technical vs accessible explanations
- **Safety**: Avoid toxic/harmful patterns (tag with 'safe': 0.0)

**How it works (simplified):**
```python
# KATO prediction scoring with emotives
base_score = confidence * frequency_prior

# Compute emotive distance
emotive_distance = euclidean_distance(
    pattern.emotives,
    requested_emotives
)

# Boost patterns with similar emotives
final_score = base_score * exp(-emotive_distance)
```

### API Usage Summary

```python
# Training with both metadata and emotives
node.observe_sequence(
    observations=[
        {
            'strings': [token],
            'metadata': {  # For governance/lineage
                'book': 'Moby_Dick',
                'chapter': 1
            },
            'emotives': {  # For prediction steering
                'excitement': 0.7,
                'calm': 0.3
            }
        }
    ],
    learn_at_end=True
)

# Query by metadata (does NOT affect scores)
patterns = node.get_patterns(
    metadata_filter={'book': 'Moby_Dick'}
)

# Predict with emotive steering (DOES affect scores)
predictions = node.predict(
    input_tokens=['The', 'whale'],
    emotive_target={'excitement': 0.8, 'danger': 0.6}
)
```

---

## Implementation Notes

### Current Status (2025-10-22)

**Implemented:**
- ✅ Hierarchical learning (4-5 levels)
- ✅ Fixed-length token chunking
- ✅ `observe_sequence()` batching (4-7x speedup)
- ✅ Parallel training with 6 workers (2-3x speedup)
- ✅ Training snapshot system (MongoDB statistics)
- ✅ Multi-run comparison (analysis.ipynb)
- ✅ Frequency distribution analysis (Zipfian fitting)
- ✅ Metadata storage with patterns

**Not Yet Implemented:**
- ❌ Text generation module
- ❌ Graph traversal algorithms
- ❌ Unraveling strategies (deterministic/stochastic)
- ❌ Generation strategy configuration (greedy/beam/sampling)
- ❌ Emotive-conditioned generation
- ❌ Pattern provenance tracking
- ❌ Generation evaluation metrics

### Future Implementation: Text Generation Module

**Suggested structure:**
```
tools/
├── text_generation.py        # Main generation module
│   ├── GenerationEngine      # Core generation class
│   ├── GreedyGenerator       # Strategy: greedy top-down
│   ├── BeamSearchGenerator   # Strategy: beam search
│   ├── SamplingGenerator     # Strategy: sampling
│   ├── PatternUnraveler      # Deterministic/stochastic unraveling
│   └── ProvenanceTracker     # Track source patterns
│
├── generation_config.py      # Configuration dataclasses
│   ├── GenerationConfig
│   ├── UnravelingConfig
│   └── EmotiveConfig
│
└── generation_evaluation.py  # Evaluation metrics
    ├── FaithfulnessScorer
    ├── CompositionValidator
    └── MetadataConsistencyChecker
```

**Key interfaces:**
```python
# Configuration
config = GenerationConfig(
    strategy='beam_search',  # 'greedy' | 'beam_search' | 'sampling'
    beam_width=5,
    temperature=1.0,
    unraveling='stochastic',  # 'deterministic' | 'stochastic'
    frequency_weight=1.0,
    emotive_target={'happy': 0.8, 'calm': 0.6},
    metadata_filter={'book': 'Moby_Dick'},
    max_length=500,
    track_provenance=True
)

# Generation
engine = GenerationEngine(learner, config)
result = engine.generate(
    prompt="The whale emerged from",
    return_provenance=True
)

# Result
print(result.text)  # Generated text
print(result.provenance)  # Which training patterns contributed
print(result.faithfulness_score)  # 0.0-1.0 (1.0 = perfect faithfulness)
```

---

## Open Questions & Future Work

### Research Questions

1. **Optimal hierarchy depth**:
   - Current: 4-5 levels
   - Question: Does 6-7 levels improve long-range coherence?
   - Experiment: Train with varying depths, evaluate generation quality

2. **Zipfian distribution evolution**:
   - Current: α ≈ 0.01-0.26 at node0 (low)
   - Question: How does α change with:
     - More training data (10K → 100K → 1M samples)?
     - Different chunk sizes (3 vs 5 vs 8)?
     - Different datasets (WikiText vs C4 vs books)?
   - Experiment: Track α across configurations in analysis.ipynb

3. **Emotive-conditioned generation**:
   - Current: Emotives stored but not used for generation
   - Question: How well can emotives steer generation?
   - Experiment: Train with sentiment-tagged data, generate with emotive targets

4. **Metadata-driven faithfulness**:
   - Current: Metadata stored but not used for validation
   - Question: Can metadata prevent cross-contamination (mixing books)?
   - Experiment: Generate from mixed-source training, check metadata consistency

### Engineering Tasks

1. **Implement generation module** (see Implementation Notes above)
2. **Add provenance tracking** to link generated text → training patterns
3. **Build evaluation framework** for faithfulness, composition validity
4. **Create generation examples notebook** (generation.ipynb)
5. **Optimize unraveling performance** (caching, lazy evaluation)
6. **Add generation to multi-run comparison** (compare gen quality across configs)

### Experiments to Run

1. **Scaling study**: Train on 100K samples, measure α improvement
2. **Chunk size sweep**: Test chunk_size ∈ [3,4,5,6,7,8], find optimal
3. **Generation diversity**: Compare greedy vs beam vs sampling outputs
4. **Emotive steering**: Generate happy vs sad text from same prompt
5. **Cross-book contamination**: Detect if generation mixes sources incorrectly
6. **Pattern reuse analysis**: Which node0 patterns appear in most node1 patterns?

---

## References

- **PROJECT_OVERVIEW.md**: Core concepts, philosophy, technical approach
- **CLAUDE.md**: Project-specific instructions, token chunking rationale
- **training.ipynb**: Hierarchical training implementation
- **analysis.ipynb**: Multi-run comparison and Zipfian analysis
- **tools/hierarchical_learning.py**: Core training module
- **tools/training_comparison.py**: Visualization and optimization

---

**Document Maintenance:**
- Update after major architectural changes
- Add new sections for implemented features
- Document experimental results in "Open Questions" section
- Version control: Track changes via git commits
