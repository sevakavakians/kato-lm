# KATO Language Model Architecture

**Last Updated**: 2025-10-09

## System Architecture

### NEW: Hierarchical Concept-Based Learning (Implemented)

**Status**: ✓ IMPLEMENTATION COMPLETE (2025-10-09)

```
Node3: Book/Article Level
    ↑ (Chapter pattern names)
Node2: Chapter/Section Level
    ↑ (Paragraph pattern names)
Node1: Paragraph Level
    ↑ (Sentence pattern names)
Node0: Sentence Level
    ↑ (Tokens)
Input Text Stream
```

### NEW Configuration (Implemented)
- **MAX_PATTERN_LENGTH**: 0 (manual learning only, no auto-learning) ✓
- **STM_MODE**: CLEAR (clear after each learn operation) ✓
- **Learning Trigger**: Manual at concept boundaries (sentence/paragraph/chapter/book) ✓
- **Data Segmentation**: Book → Chapter → Paragraph → Sentence ✓
- **Abstraction**: Each level passes pattern names (symbolic representations) upward ✓
- **Implementation**: 3 classes (CorpusSegmenter, HierarchicalConceptLearner, LearningTracker) ✓
- **Visualization**: Complete progress tracking and 4-panel matplotlib visualization ✓

### CURRENT: Hierarchical KATO Network (Existing)

```
Level 2 (Phrase-level)
    ↑
Level 1 (Word-level)
    ↑
Level 0 (Token-level)
    ↑
Input Text Stream
```

### Current Configuration
- **MAX_PATTERN_LENGTH**: 5 (all levels)
- **STM_MODE**: ROLLING (continuous learning)
- **Topology**: Hierarchical with placeholder inter-level transfer functions

## Data Flow

### Input Processing Pipeline
1. **Text Ingestion**: Stream from Hugging Face datasets
2. **Tokenization**: GPT-2 tokenizer converts text to token IDs
3. **Level 0 Processing**: Token-level pattern recognition
4. **Level 1 Processing**: Word-level abstraction (placeholder)
5. **Level 2 Processing**: Phrase-level abstraction (placeholder)

### Training Loop
```python
for batch in dataset:
    tokens = tokenizer.encode(batch['text'])
    for token in tokens:
        kato_level0.observe(token)
        stats.track_observation()
```

## Components

### Enhanced KATO Client
- Session-level configuration support
- Custom max_pattern_length and stm_mode per session
- Statistics tracking:
  - Observations count
  - Token count
  - Pattern count
  - Auto-learn count
- Rolling STM mode for continuous learning

### Dataset Integration
**Streaming Mode** (no downloads):
- Memory-efficient processing
- English text filtering
- Configurable sample limits
- Multi-dataset support

### Visualization
- Training statistics plots
- Progress tracking with tqdm
- Pattern formation monitoring

## Technical Details

### Memory Management
- Streaming datasets prevent memory overflow
- Rolling STM mode maintains fixed memory footprint
- Configurable pattern length limits growth

### Performance Considerations
- Token-level processing is fastest (Level 0)
- Higher levels require aggregation (to be implemented)
- Streaming allows infinite dataset processing

## Verified Facts
- **Confirmed**: KATO client supports session-level configuration (2025-10-05)
- **Confirmed**: Streaming datasets work without downloads (2025-10-05)
- **Confirmed**: Hierarchical topology setup successful (2025-10-05)
- **Confirmed**: max_pattern_length=0 enables manual-only learning (2025-10-09)
- **Confirmed**: stm_mode=CLEAR clears STM after each learn operation (2025-10-09)
- **Confirmed**: Pattern names provide stable symbolic representation (2025-10-09)
- **Confirmed**: Hierarchical concept learning with clear boundaries works as designed (2025-10-09)

## Implementation Details

### CorpusSegmenter Class
- Segments raw text into hierarchical structure
- Methods: segment_book(), segment_article(), segment_simple_text()
- Handles chapter/section detection, paragraph boundaries, sentence tokenization
- NLTK integration with regex fallback for robustness

### HierarchicalConceptLearner Class
- Coordinates 4 KATO nodes for hierarchical learning
- Each node configured with max_pattern_length=0, stm_mode=CLEAR
- Methods for each learning level: learn_sentence(), learn_paragraph(), learn_chapter(), learn_book()
- Pattern name propagation between levels creates symbolic abstraction
- process_corpus() orchestrates complete hierarchical processing

### LearningTracker Class
- Tracks progress across all hierarchical levels
- Counters: sentences, paragraphs, chapters, books learned
- Pattern name collections at each level
- Timing and performance metrics
- Summary generation and formatted output

### Visualization
- demonstrate_hierarchical_learning(): End-to-end demonstration with sample text
- visualize_hierarchical_stats(): 4-panel matplotlib visualization
  - Concepts learned by level
  - Patterns by node
  - Observations by node
  - Tokens processed by node

## Future Architecture Enhancements
1. Multi-scale prediction at all 4 levels (token, sentence, paragraph, chapter)
2. Advanced tokenization (GPT-2/BERT subword tokens)
3. Bidirectional learning (bottom-up and top-down)
4. Context integration between levels
5. Dynamic segmentation based on learned patterns
6. Multi-document learning across book collections
7. Transfer learning to new domains
8. Distributed processing for large-scale corpora
