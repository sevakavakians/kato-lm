# Pattern Frequency Analysis

## Quick Start

Analyze pattern repetition frequencies across all KATO nodes:

```bash
source .venv/bin/activate
python pattern_frequency_analysis.py
```

This will generate:
- Histogram visualizations for each node (PNG files)
- Statistical summary report (TXT file)
- Results saved to `./frequency_analysis/` directory

## What It Does

The script:
1. **Discovers nodes** - Queries ClickHouse for all KATO knowledge bases
2. **Collects pattern names** - Gets all stored pattern identifiers per node
3. **Queries frequencies** - Retrieves frequency counts from Redis
4. **Analyzes distributions** - Calculates statistics and Zipfian properties
5. **Generates visualizations** - Creates histograms (linear + log-log scales)

## Output

### Histograms (PNG files)
Each histogram contains two plots:
- **Left**: Linear scale frequency distribution
- **Right**: Log-log rank-frequency plot (Zipfian analysis)

### Summary Report (TXT file)
Statistics for each node:
- Total pattern count
- Mean, median, standard deviation
- Percentiles (25th, 50th, 75th, 90th, 95th, 99th)
- Top 10 most frequent patterns

## Usage Options

```bash
# Basic - analyze all nodes
python pattern_frequency_analysis.py

# Custom output directory
python pattern_frequency_analysis.py --output-dir ./my_analysis

# Specific nodes only
python pattern_frequency_analysis.py --nodes node0_kato node1_kato

# Quick test (limit patterns)
python pattern_frequency_analysis.py --max-patterns 1000
```

## Expected Results

For a well-trained hierarchical learning system:

✅ **Node0 (Token-level)**: Strong Zipfian distribution
  - A few patterns with very high frequency (common chunks)
  - Long tail of rare patterns
  - Mean frequency >> 1

✅ **Node1-Node3**: Decreasing repetition at higher levels
  - Node1: Moderate repetition
  - Node2: Low repetition
  - Node3: Minimal repetition (unique book patterns)

## Performance

Approximate runtime for full dataset:
- node0_kato (804k patterns): 2-3 minutes
- node1_kato (126k patterns): 20-30 seconds
- node2_kato (36k patterns): 6-10 seconds
- node3_kato (61 patterns): <1 second
- **Total**: ~3-5 minutes

Progress bars show real-time status.

## Interpreting Zipfian Distributions

A Zipfian distribution follows: `frequency ∝ 1/rank`

In natural language and hierarchical learning:
- **High-frequency patterns**: Common building blocks (e.g., "the", "and")
- **Low-frequency patterns**: Rare/unique combinations
- **Log-log linearity**: Indicates power-law behavior (good compression)

## Requirements

- **KATO Server**: Must be running (http://localhost:8000)
- **ClickHouse**: Pattern metadata storage
- **Redis**: Frequency value storage
- **Python packages**: matplotlib, numpy, tqdm (included in .venv)

## See Also

- `frequency_analysis/README.md` - Detailed analysis results
- `frequency_analysis_summary.txt` - Latest statistical summary
- `storage_comparison.py` - Related storage analysis tool
