"""
Hierarchy Metrics - Comprehensive Graph-Based Metrics for KATO Hierarchical Learning

This package provides tools to measure and analyze hierarchical concept learning systems
using graph-centric metrics instead of traditional Zipfian distributions.

Key Components:
- HierarchyMetricsCollector: Training-time data collection
- GraphAnalyzer: Post-training graph topology metrics
- InformationTheoryAnalyzer: Entropy, mutual information calculations
- PredictionAnalyzer: Prediction cascade metrics
- MetricsReport: Comprehensive reporting and health scoring

Usage:
    from hierarchy_metrics import HierarchyMetricsCollector, MetricsReport

    # During training
    collector = HierarchyMetricsCollector(learner)
    # ... training happens ...
    collector.save('hierarchy_graph.db')

    # After training
    report = MetricsReport.generate(
        graph_db_path='hierarchy_graph.db',
        learner=learner
    )

    print(report.summary())
    report.export_json('metrics_report.json')
"""

from .config import (
    MetricsConfig,
    CollectorConfig,
    AnalyzerConfig,
    ThresholdConfig,
)

from .collectors import (
    HierarchyMetricsCollector,
    TrainingDynamicsTracker,
)

# Lazy imports (only when needed to avoid heavy dependencies)
def _get_graph_analyzer():
    from .graph_analyzer import GraphAnalyzer
    return GraphAnalyzer

def _get_information_theory_analyzer():
    from .information_theory import InformationTheoryAnalyzer
    return InformationTheoryAnalyzer

def _get_prediction_analyzer():
    from .prediction_analyzer import PredictionAnalyzer
    return PredictionAnalyzer

def _get_metrics_report():
    from .report import MetricsReport
    return MetricsReport

# Public API
__all__ = [
    # Configuration
    'MetricsConfig',
    'CollectorConfig',
    'AnalyzerConfig',
    'ThresholdConfig',

    # Collectors
    'HierarchyMetricsCollector',
    'TrainingDynamicsTracker',

    # Analyzers (lazy-loaded)
    'GraphAnalyzer',
    'InformationTheoryAnalyzer',
    'PredictionAnalyzer',

    # Reporting
    'MetricsReport',
]

# Lazy-load heavy dependencies
def __getattr__(name):
    if name == 'GraphAnalyzer':
        return _get_graph_analyzer()
    elif name == 'InformationTheoryAnalyzer':
        return _get_information_theory_analyzer()
    elif name == 'PredictionAnalyzer':
        return _get_prediction_analyzer()
    elif name == 'MetricsReport':
        return _get_metrics_report()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = '1.0.0'
