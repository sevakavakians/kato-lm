"""
Configuration dataclasses for hierarchy metrics system.

Defines all configuration options, thresholds, and result structures.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json


class HealthStatus(Enum):
    """Health status indicators for metrics"""
    EXCELLENT = "excellent"  # Green
    GOOD = "good"            # Light green
    WARNING = "warning"      # Yellow
    POOR = "poor"            # Orange
    CRITICAL = "critical"    # Red
    UNKNOWN = "unknown"      # Gray


@dataclass
class MetricsConfig:
    """Global configuration for metrics collection and analysis"""

    # Collection settings
    enable_training_collection: bool = True
    checkpoint_interval: int = 1000  # Collect data every N samples
    max_graph_size_mb: int = 5000    # Stop collecting if graph > this size

    # Analysis settings
    compute_mutual_information: bool = True
    compute_pattern_diversity: bool = True  # Expensive for large graphs
    diversity_sample_size: int = 1000       # Sample pairs for diversity

    # Prediction analysis
    prediction_test_size: int = 100         # Test inputs for fan-out
    prediction_timeout_seconds: float = 60.0

    # Storage
    graph_db_path: str = './hierarchy_graph.db'
    dynamics_log_path: str = './training_dynamics.jsonl'

    # Thresholds
    thresholds: 'ThresholdConfig' = field(default_factory=lambda: ThresholdConfig())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['thresholds'] = asdict(self.thresholds)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsConfig':
        """Create from dictionary"""
        if 'thresholds' in data:
            data['thresholds'] = ThresholdConfig(**data['thresholds'])
        return cls(**data)


@dataclass
class CollectorConfig:
    """Configuration for training-time data collection"""

    # What to collect
    collect_pattern_relationships: bool = True  # Parent-child mappings
    collect_timestamps: bool = True            # When patterns created
    collect_frequencies: bool = True           # Pattern frequencies
    collect_metadata: bool = True              # Pattern metadata

    # Performance limits
    max_patterns_in_memory: int = 100_000     # Flush to disk after this
    batch_size: int = 1000                    # Batch inserts for performance

    # Checkpointing
    enable_checkpoints: bool = True
    checkpoint_interval: int = 1000           # Samples between checkpoints


@dataclass
class AnalyzerConfig:
    """Configuration for post-training analysis"""

    # Sampling (for large graphs)
    use_sampling: bool = True                 # Sample instead of full analysis
    sample_size: int = 10_000                 # Patterns to sample

    # Information theory
    entropy_bins: int = 100                   # Bins for entropy estimation
    mi_estimator: str = 'discrete'            # 'discrete' or 'kraskov'

    # Diversity computation
    diversity_metric: str = 'levenshtein'     # 'levenshtein' or 'jaccard'
    diversity_sample_pairs: int = 1000

    # Co-occurrence validation
    cooccurrence_window: int = 5              # Tokens within N positions


@dataclass
class ThresholdConfig:
    """Thresholds for health scoring (green/yellow/red)"""

    # Compression Ratio (ratio = num_patterns[i] / num_patterns[i+1])
    compression_ratio_excellent: Tuple[float, float] = (0.8, 1.2)  # ±20% of chunk_size
    compression_ratio_good: Tuple[float, float] = (0.5, 1.5)
    compression_ratio_warning: Tuple[float, float] = (0.3, 2.0)
    # Below warning = poor/critical

    # Orphan Rate (% of patterns with 0 parents)
    orphan_rate_excellent: float = 0.05  # <5%
    orphan_rate_good: float = 0.10       # <10%
    orphan_rate_warning: float = 0.20    # <20%
    orphan_rate_poor: float = 0.40       # <40%
    # Above poor = critical

    # Coverage (% of lower patterns used in upper)
    coverage_excellent: float = 0.85     # >85%
    coverage_good: float = 0.70          # >70%
    coverage_warning: float = 0.50       # >50%
    coverage_poor: float = 0.30          # >30%

    # Conditional Entropy (constraint effectiveness)
    constraint_effectiveness_excellent: Tuple[float, float] = (0.5, 0.8)
    constraint_effectiveness_good: Tuple[float, float] = (0.3, 0.9)
    constraint_effectiveness_warning: Tuple[float, float] = (0.1, 0.95)

    # Mutual Information
    mi_excellent: Tuple[float, float] = (0.4, 0.7)
    mi_good: Tuple[float, float] = (0.2, 0.8)
    mi_warning: Tuple[float, float] = (0.1, 0.9)

    # Branching Factor (should be ~chunk_size)
    branching_cv_excellent: float = 0.2  # Coefficient of variation <0.2
    branching_cv_good: float = 0.4
    branching_cv_warning: float = 0.6

    # Prediction Fan-Out (number of predictions per level)
    fanout_excellent: Tuple[int, int] = (20, 200)
    fanout_good: Tuple[int, int] = (10, 500)
    fanout_warning: Tuple[int, int] = (5, 1000)

    # Pattern Growth Exponent (sublinear growth)
    growth_exponent_excellent: Tuple[float, float] = (0.5, 0.7)
    growth_exponent_good: Tuple[float, float] = (0.4, 0.8)
    growth_exponent_warning: Tuple[float, float] = (0.3, 0.9)

    # Co-occurrence Validation (coherence rate)
    coherence_excellent: float = 0.9
    coherence_good: float = 0.8
    coherence_warning: float = 0.6
    coherence_poor: float = 0.4

    def evaluate_metric(
        self,
        metric_name: str,
        value: float,
        chunk_size: Optional[int] = None
    ) -> HealthStatus:
        """
        Evaluate a metric value and return health status.

        Args:
            metric_name: Name of metric (e.g., 'orphan_rate', 'mi')
            value: Measured value
            chunk_size: Optional chunk size for ratio normalization

        Returns:
            HealthStatus enum
        """
        if metric_name == 'compression_ratio':
            # Normalize by chunk_size if provided
            if chunk_size:
                normalized = value / chunk_size
                if self.compression_ratio_excellent[0] <= normalized <= self.compression_ratio_excellent[1]:
                    return HealthStatus.EXCELLENT
                elif self.compression_ratio_good[0] <= normalized <= self.compression_ratio_good[1]:
                    return HealthStatus.GOOD
                elif self.compression_ratio_warning[0] <= normalized <= self.compression_ratio_warning[1]:
                    return HealthStatus.WARNING
                else:
                    return HealthStatus.POOR
            return HealthStatus.UNKNOWN

        elif metric_name == 'orphan_rate':
            if value <= self.orphan_rate_excellent:
                return HealthStatus.EXCELLENT
            elif value <= self.orphan_rate_good:
                return HealthStatus.GOOD
            elif value <= self.orphan_rate_warning:
                return HealthStatus.WARNING
            elif value <= self.orphan_rate_poor:
                return HealthStatus.POOR
            else:
                return HealthStatus.CRITICAL

        elif metric_name == 'coverage':
            if value >= self.coverage_excellent:
                return HealthStatus.EXCELLENT
            elif value >= self.coverage_good:
                return HealthStatus.GOOD
            elif value >= self.coverage_warning:
                return HealthStatus.WARNING
            elif value >= self.coverage_poor:
                return HealthStatus.POOR
            else:
                return HealthStatus.CRITICAL

        elif metric_name == 'constraint_effectiveness':
            low, high = self.constraint_effectiveness_excellent
            if low <= value <= high:
                return HealthStatus.EXCELLENT
            low, high = self.constraint_effectiveness_good
            if low <= value <= high:
                return HealthStatus.GOOD
            low, high = self.constraint_effectiveness_warning
            if low <= value <= high:
                return HealthStatus.WARNING
            else:
                return HealthStatus.POOR

        elif metric_name == 'coherence':
            if value >= self.coherence_excellent:
                return HealthStatus.EXCELLENT
            elif value >= self.coherence_good:
                return HealthStatus.GOOD
            elif value >= self.coherence_warning:
                return HealthStatus.WARNING
            elif value >= self.coherence_poor:
                return HealthStatus.POOR
            else:
                return HealthStatus.CRITICAL

        else:
            return HealthStatus.UNKNOWN


# Result dataclasses

@dataclass
class CompressionMetrics:
    """Compression-related metrics (metrics 1-3)"""
    compression_ratios: Dict[str, float]  # e.g., {'node0->node1': 5.2}
    pattern_counts: Dict[str, int]        # e.g., {'node0': 150000, 'node1': 30000}
    effective_compression_rates: Dict[str, float]  # observations / unique patterns
    health_status: Dict[str, HealthStatus]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['health_status'] = {k: v.value for k, v in self.health_status.items()}
        return d


@dataclass
class ConnectivityMetrics:
    """Graph connectivity metrics (metrics 4-6)"""
    reusability: Dict[str, Dict[str, float]]  # {node: {mean_parents, median, orphan_rate}}
    coverage: Dict[str, float]                # {level_pair: coverage_rate}
    branching_factors: Dict[str, Dict[str, float]]  # {node: {mean, std, cv}}
    health_status: Dict[str, HealthStatus]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['health_status'] = {k: v.value for k, v in self.health_status.items()}
        return d


@dataclass
class InformationMetrics:
    """Information-theoretic metrics (metrics 7-9)"""
    mutual_information: Dict[str, float]      # {'node0-node1': 0.42}
    conditional_entropy: Dict[str, float]     # {'H(node0|node1)': 2.3}
    constraint_effectiveness: Dict[str, float]  # {level_pair: 0.58}
    entropy_progression: Dict[str, float]     # {node: entropy}
    health_status: Dict[str, HealthStatus]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['health_status'] = {k: v.value for k, v in self.health_status.items()}
        return d


@dataclass
class PredictionMetrics:
    """Prediction-related metrics (metric 10)"""
    fanout_by_level: Dict[str, Dict[str, float]]  # {node: {mean, std, min, max}}
    confidence_distributions: Optional[Dict[str, List[float]]] = None
    health_status: Dict[str, HealthStatus] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['health_status'] = {k: v.value for k, v in self.health_status.items()}
        return d


@dataclass
class ContextMetrics:
    """Context and coherence metrics (metrics 11-12, 15)"""
    context_window_alignment: Dict[str, float]  # {node: alignment_score}
    pattern_diversity: Dict[str, float]         # {node: mean_similarity}
    cooccurrence_validation: Dict[str, float]   # {node: coherence_rate}
    health_status: Dict[str, HealthStatus]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['health_status'] = {k: v.value for k, v in self.health_status.items()}
        return d


@dataclass
class TrainingDynamicsMetrics:
    """Training dynamics metrics (metrics 13-14)"""
    growth_exponent: float                      # Power-law fit exponent
    growth_r_squared: float                     # Goodness of fit
    reusability_trend: List[float]              # Over time
    reusability_trend_slope: float              # Linear fit slope
    checkpoints: List[Dict[str, Any]]           # Checkpoint data
    health_status: Dict[str, HealthStatus]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['health_status'] = {k: v.value for k, v in self.health_status.items()}
        return d


@dataclass
class MetricsSummary:
    """Overall summary with health scores"""
    overall_health: HealthStatus
    compression: HealthStatus
    connectivity: HealthStatus
    information: HealthStatus
    prediction: HealthStatus
    training_dynamics: HealthStatus

    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_health': self.overall_health.value,
            'compression': self.compression.value,
            'connectivity': self.connectivity.value,
            'information': self.information.value,
            'prediction': self.prediction.value,
            'training_dynamics': self.training_dynamics.value,
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
        }
