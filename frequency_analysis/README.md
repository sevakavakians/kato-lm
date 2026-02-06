# KATO Pattern Frequency Analysis Results

Generated on: 2026-02-06

## Overview

This directory contains frequency distribution analysis for all KATO hierarchical learning nodes. The analysis examines how often patterns repeat in the knowledge base, which is a key indicator of the effectiveness of the hierarchical abstraction.

## Files

- **frequency_analysis_summary.txt** - Detailed statistical summary for all nodes
- **node0_kato_frequency_histogram.png** - Token-level pattern frequencies
- **node1_kato_frequency_histogram.png** - Paragraph-level pattern frequencies
- **node2_kato_frequency_histogram.png** - Chapter-level pattern frequencies
- **node3_kato_frequency_histogram.png** - Book-level pattern frequencies

## Key Findings

### Node0 (Token-level patterns)
- **Total Patterns**: 804,286
- **Max Frequency**: 8,950 (highly repeated chunk pattern)
- **Distribution**: Strong Zipfian distribution with heavy tail
- **Mean**: 1.04, **Median**: 1.00
- **Interpretation**: Most patterns appear once, but a few common chunks (e.g., "the", "and", "in") repeat thousands of times

### Node1 (Paragraph-level patterns)
- **Total Patterns**: 126,552
- **Max Frequency**: 50
- **Distribution**: Much flatter than node0
- **Mean**: 1.02, **Median**: 1.00
- **Interpretation**: Paragraph-level patterns show less repetition, indicating diverse content at this level

### Node2 (Chapter-level patterns)
- **Total Patterns**: 36,494
- **Max Frequency**: 3
- **Distribution**: Almost entirely unique patterns
- **Mean**: 1.00, **Median**: 1.00
- **Interpretation**: Very little repetition at chapter level - most patterns unique to their context

### Node3 (Book-level patterns)
- **Total Patterns**: 61
- **Max Frequency**: 1
- **Distribution**: All patterns unique
- **Mean**: 1.00, **Median**: 1.00
- **Interpretation**: Every book-level pattern is unique (as expected with limited training data)

## Zipfian Analysis

The histograms include two visualizations:

1. **Linear Scale (Left)** - Shows the raw frequency distribution
2. **Log-Log Scale (Right)** - Shows Zipfian behavior (rank vs frequency)

A true Zipfian distribution follows the power law: `f(r) = k/r`, where:
- `f(r)` is the frequency of the r-th ranked item
- `k` is a constant
- In log-log space, this appears as a straight line

**Observation**: Node0 shows strong Zipfian behavior (common in natural language), while higher nodes show increasing uniqueness.

## Implications for KATO Learning

✅ **Good Signs**:
- Node0 shows excellent compression (Zipfian distribution means common patterns reused)
- Pattern counts decrease at higher levels (804k → 126k → 36k → 61)
- No zero-frequency patterns (all stored patterns have been observed)

⚠️ **Training Stage Indicators**:
- Node2/Node3 have very low repetition → Need more training data to learn reusable high-level patterns
- Node3 only has 61 patterns → Early in training (expected with current dataset size)

## Reproducing This Analysis

```bash
# Activate virtual environment
source .venv/bin/activate

# Run full analysis
python pattern_frequency_analysis.py

# Run on specific nodes only
python pattern_frequency_analysis.py --nodes node0_kato node1_kato

# Quick test with limited patterns
python pattern_frequency_analysis.py --max-patterns 1000 --output-dir ./test_output
```

## Technical Details

- **Data Source**: Redis (frequency values) + ClickHouse (pattern names)
- **Analysis Time**: ~5-10 minutes for full dataset (~967k patterns)
- **Libraries**: matplotlib, numpy, tqdm
- **Script**: `/pattern_frequency_analysis.py`
