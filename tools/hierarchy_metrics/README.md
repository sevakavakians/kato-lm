# Hierarchy Metrics - Comprehensive Graph-Based Evaluation

A complete metrics framework for analyzing hierarchical concept learning systems using graph-centric evaluation instead of traditional Zipfian distributions.

## Overview

This package provides **15 comprehensive metrics** across 6 categories to evaluate hierarchical learning systems:

### 📊 Metric Categories

1. **Compression Metrics (3)**: Compression ratios, pattern counts, effectiveness
2. **Connectivity Metrics (3)**: Reusability, coverage, branching factors
3. **Information-Theoretic (3)**: Mutual information, conditional entropy, constraint effectiveness
4. **Prediction Readiness (1)**: Fan-out analysis
5. **Context & Coherence (2)**: Alignment, diversity, co-occurrence validation
6. **Training Dynamics (2)**: Growth exponent, reusability trends

## Quick Start

### Installation

```python
# No installation needed - already part of kato-notebooks
from tools.hierarchy_metrics import MetricsReport
```

### Basic Usage

```python
# After training, generate comprehensive report
report = MetricsReport.generate(
    graph_db_path='./metrics/hierarchy_graph.db',
    learner=learner  # Optional, for prediction analysis
)

# Display summary
print(report.summary())

# Export results
report.export_json('metrics_report.json')
report.export_csv('./metrics/csv')
```

### Jupyter Notebooks

Two notebooks are provided:

1. **`hierarchy_metrics.ipynb`**: Comprehensive analysis (all 15 metrics, detailed visualizations)
2. **`hierarchy_dashboard.ipynb`**: Quick dashboard (health check, actionable insights)

## Architecture

```
hierarchy_metrics/
├── __init__.py              # Public API
├── config.py                # Dataclasses, thresholds, health scoring
├── storage.py               # SQLite graph database
├── collectors.py            # Training-time data collection
├── graph_analyzer.py        # Graph topology metrics (10 metrics)
├── information_theory.py    # Information-theoretic metrics (3 metrics)
├── prediction_analyzer.py   # Prediction fan-out (1 metric)
├── report.py                # Comprehensive report generator
├── visualization.py         # Matplotlib plotting functions
├── export.py                # Web-ready export (Plotly, Parquet)
└── TRAINING_INTEGRATION.md  # Integration guide
```

## Workflow

### 1. During Training (Optional)

```python
from hierarchy_metrics import HierarchyMetricsCollector

# Initialize collector
collector = HierarchyMetricsCollector(
    learner=learner,
    checkpoint_interval=1000
)

# Training happens...
# (With hooks, collector captures patterns automatically)

# After training
collector.save(
    graph_db_path='./metrics/hierarchy_graph.db',
    dynamics_path='./metrics/training_dynamics.jsonl'
)
```

### 2. After Training (Analysis)

```python
# Generate comprehensive report
report = MetricsReport.generate(
    graph_db_path='./metrics/hierarchy_graph.db',
    learner=learner
)

# View results
print(report.summary())
```

### 3. Visualization

```python
from hierarchy_metrics.visualization import plot_full_dashboard

# Plot all metrics
plot_full_dashboard(report)
```

### 4. Export

```python
from hierarchy_metrics.export import export_for_web

# Export for web dashboards
export_for_web(
    report=report,
    output_dir='./metrics/dashboard',
    formats=['plotly', 'json', 'csv']
)
```

## Detailed Metrics

### Compression Metrics (1-3)

**Compression Ratio**: Pattern count reduction at each level
- **Target**: ~chunk_size (e.g., 5-15x)
- **Health Thresholds**:
  - Excellent: ±20% of chunk_size
  - Warning: <50% or >150% of chunk_size

**Pattern Counts**: Number of unique patterns per level
- Should decrease exponentially up the hierarchy
- node0 > node1 > node2 > node3

**Effective Compression**: Observations / unique patterns
- Higher values = more pattern reuse

### Connectivity Metrics (4-6)

**Reusability**: How many parents use each child pattern
- **Orphan Rate**: % patterns with 0 parents
- **Target**: <10% orphans
- **Health Thresholds**:
  - Excellent: <5%
  - Critical: >40%

**Coverage**: % of lower patterns used in upper level
- **Target**: >70%
- **Interpretation**: Low coverage = wasted patterns

**Branching Factor**: Children per parent pattern
- **Target**: ~chunk_size with low variance
- **CV (Coefficient of Variation)**:
  - Excellent: <0.2
  - Warning: >0.6

### Information-Theoretic Metrics (7-9)

**Mutual Information I(X;Y)**: Dependency between levels (bits)
- Measures how much knowing upper level tells you about lower level

**Conditional Entropy H(X|Y)**: Remaining uncertainty
- Lower = upper level constrains lower level more

**Constraint Effectiveness**: I(X;Y) / H(X) (normalized)
- **Target**: 50-80%
- **Interpretation**:
  - <30%: Weak constraints
  - >90%: Over-deterministic

**Entropy Progression**: H(X) at each level
- Generally decreases up hierarchy (more concentrated)

### Prediction Metrics (10)

**Fan-Out**: Number of predictions at each level
- **Target**: 20-200 predictions
- **Too low**: Over-constrained
- **Too high**: Under-constrained

### Context Metrics (11-12)

**Context Window Alignment**: Branching vs chunk_size
- Score: 1 - |mean_branching - chunk_size| / chunk_size
- Higher = better alignment

**Pattern Diversity**: Jaccard similarity between patterns
- Sampled pairs, mean similarity
- Lower = more diverse patterns

### Training Dynamics (13-14)

**Growth Exponent**: Power-law fit N ∝ S^β
- **Target**: 0.5-0.7 (sublinear growth)
- **Interpretation**:
  - β = 1.0: Linear (no compression)
  - β < 0.7: Good compression
  - β > 1.0: Problematic

**Reusability Trend**: Slope of mean_parents over time
- Positive slope = patterns becoming more reusable
- Good sign for learning

### Co-occurrence Validation (15)

**Coherence Rate**: % patterns with valid co-occurrence
- Validates that patterns represent real linguistic chunks
- **Target**: >80%

## Health Scoring

Each metric has health thresholds:

- 🟢 **EXCELLENT**: Optimal performance
- 🟢 **GOOD**: Healthy, minor improvements possible
- 🟡 **WARNING**: Review recommended
- 🟠 **POOR**: Action needed
- 🔴 **CRITICAL**: Immediate attention required

Overall health = worst category health.

## API Reference

### MetricsReport

```python
# Generate report
report = MetricsReport.generate(
    graph_db_path: str,              # Path to SQLite graph DB
    learner: Optional[Any] = None,   # For prediction analysis
    config: Optional[MetricsConfig] = None,
    test_inputs: Optional[List[List[str]]] = None,
    verbose: bool = False
)

# Access metrics
report.compression      # CompressionMetrics
report.connectivity     # ConnectivityMetrics
report.information      # InformationMetrics
report.prediction       # PredictionMetrics (if learner provided)
report.training_dynamics # TrainingDynamicsMetrics (if checkpoints)
report.metrics_summary  # MetricsSummary (overall health)

# Display
print(report.summary())

# Export
report.export_json('report.json')
report.export_csv('./csv_dir')
```

### Individual Analyzers

```python
# Graph analysis
from hierarchy_metrics import GraphAnalyzer

analyzer = GraphAnalyzer('hierarchy_graph.db')
compression, connectivity, context = analyzer.analyze_all()
analyzer.close()

# Information theory
from hierarchy_metrics import InformationTheoryAnalyzer

info_analyzer = InformationTheoryAnalyzer('hierarchy_graph.db')
info_metrics = info_analyzer.analyze_all()
info_analyzer.close()

# Prediction analysis
from hierarchy_metrics import PredictionAnalyzer

pred_analyzer = PredictionAnalyzer(learner, 'hierarchy_graph.db')
pred_metrics = pred_analyzer.analyze_all()
pred_analyzer.close()
```

### Visualization

```python
from hierarchy_metrics.visualization import (
    plot_health_summary,
    plot_compression_ratios,
    plot_pattern_counts,
    plot_reusability_distribution,
    plot_coverage_heatmap,
    plot_mutual_information,
    plot_entropy_progression,
    plot_training_dynamics,
    plot_prediction_fanout,
    plot_full_dashboard
)

# Plot all metrics
plot_full_dashboard(report)

# Individual plots
plot_compression_ratios(report)
plot_training_dynamics(report)
```

### Export

```python
from hierarchy_metrics.export import DashboardExporter, export_for_web

# Full export
export_for_web(
    report=report,
    output_dir='./dashboard',
    formats=['plotly', 'json', 'parquet', 'csv']
)

# Custom export
exporter = DashboardExporter(report)
exporter.export_all_plotly('./plotly_charts')
exporter.export_dashboard_json('./config.json')
exporter.export_time_series_json('./timeseries.json')
exporter.export_parquet('./parquet')
```

## Configuration

### MetricsConfig

```python
from hierarchy_metrics import MetricsConfig

config = MetricsConfig(
    # Collection
    enable_training_collection=True,
    checkpoint_interval=1000,
    max_graph_size_mb=5000,

    # Analysis
    compute_mutual_information=True,
    compute_pattern_diversity=True,
    diversity_sample_size=1000,

    # Prediction
    prediction_test_size=100,
    prediction_timeout_seconds=60.0,

    # Storage
    graph_db_path='./hierarchy_graph.db',
    dynamics_log_path='./training_dynamics.jsonl',
)
```

### ThresholdConfig

```python
from hierarchy_metrics import ThresholdConfig

thresholds = ThresholdConfig(
    # Compression
    compression_ratio_excellent=(0.8, 1.2),
    compression_ratio_good=(0.5, 1.5),

    # Orphan rate
    orphan_rate_excellent=0.05,
    orphan_rate_good=0.10,
    orphan_rate_warning=0.20,

    # Coverage
    coverage_excellent=0.85,
    coverage_good=0.70,

    # Constraint effectiveness
    constraint_effectiveness_excellent=(0.5, 0.8),

    # Growth exponent
    growth_exponent_excellent=(0.5, 0.7),
)
```

## Common Workflows

### Workflow 1: Quick Health Check

```python
# After training
from hierarchy_metrics import MetricsReport
from hierarchy_metrics.visualization import plot_health_summary

report = MetricsReport.generate('./metrics/hierarchy_graph.db')
plot_health_summary(report)
print(report.summary())
```

### Workflow 2: Compare Multiple Runs

```python
runs = [
    './metrics/run1_graph.db',
    './metrics/run2_graph.db',
    './metrics/run3_graph.db',
]

reports = []
for path in runs:
    r = MetricsReport.generate(path, verbose=False)
    reports.append(r)

# Compare compression ratios
for i, r in enumerate(reports):
    print(f"Run {i+1}: {r.compression.compression_ratios}")
```

### Workflow 3: Export for Web Dashboard

```python
from hierarchy_metrics import MetricsReport
from hierarchy_metrics.export import export_for_web

report = MetricsReport.generate('./metrics/hierarchy_graph.db')

export_for_web(
    report=report,
    output_dir='./web_dashboard',
    formats=['plotly', 'json']
)

# Files created:
# ./web_dashboard/plotly/*.json  (Plotly charts)
# ./web_dashboard/dashboard_config.json  (Full config)
# ./web_dashboard/time_series.json  (Training dynamics)
```

## Troubleshooting

### Issue: No checkpoints in training dynamics

**Solution**: Enable checkpoints during training:

```python
collector = HierarchyMetricsCollector(
    learner=learner,
    checkpoint_interval=1000  # Checkpoint every 1000 samples
)
```

### Issue: Prediction analysis unavailable

**Solution**: Provide learner to report generator:

```python
report = MetricsReport.generate(
    graph_db_path='./metrics/hierarchy_graph.db',
    learner=learner  # Add this
)
```

### Issue: High orphan rate

**Solutions**:
1. Increase training data
2. Reduce chunk_size
3. Review pattern composition logic

### Issue: Poor compression ratios

**Solutions**:
1. Adjust chunk_size (try 5-15)
2. Add/remove hierarchy levels
3. Increase training data quality

### Issue: Low constraint effectiveness

**Solutions**:
1. Use different chunk_size per level
2. Review hierarchy architecture
3. Increase training data diversity

## Performance

- **Collection Overhead**: <5% with batching
- **Memory Usage**: ~1-2% increase for buffers
- **Analysis Time**: ~10-30 seconds for typical graphs
- **Storage**: ~0.1-1 MB per 1000 patterns

## Citation

If you use this metrics framework, please cite:

```
Hierarchy Metrics Framework for Graph-Based Concept Learning Evaluation
KATO Hierarchical Learning Project, 2025
```

## License

Part of the KATO Hierarchical Learning project.

## Support

- **Issues**: Create GitHub issue
- **Notebooks**: `hierarchy_metrics.ipynb` (comprehensive), `hierarchy_dashboard.ipynb` (quick)
- **Documentation**: `TRAINING_INTEGRATION.md`, `PROJECT_OVERVIEW.md`

---

**Version**: 1.0.0
**Last Updated**: October 2025
