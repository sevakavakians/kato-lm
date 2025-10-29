"""
Comprehensive metrics report generator.

Aggregates all metrics, computes health scores, and generates reports.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np

from .storage import HierarchyGraphStorage
from .graph_analyzer import GraphAnalyzer
from .information_theory import InformationTheoryAnalyzer
from .prediction_analyzer import PredictionAnalyzer
from .collectors import TrainingDynamicsTracker
from .config import (
    MetricsConfig,
    ThresholdConfig,
    MetricsSummary,
    CompressionMetrics,
    ConnectivityMetrics,
    InformationMetrics,
    PredictionMetrics,
    TrainingDynamicsMetrics,
    HealthStatus,
)


class MetricsReport:
    """
    Comprehensive metrics report for hierarchy analysis.

    Aggregates all metrics from:
    - Graph topology (compression, connectivity)
    - Information theory (MI, entropy, constraints)
    - Prediction behavior (fan-out)
    - Training dynamics (growth, reusability trends)

    Usage:
        report = MetricsReport.generate(
            graph_db_path='hierarchy_graph.db',
            learner=learner
        )

        print(report.summary())
        report.export_json('metrics_report.json')
    """

    def __init__(
        self,
        compression: CompressionMetrics,
        connectivity: ConnectivityMetrics,
        information: InformationMetrics,
        prediction: Optional[PredictionMetrics] = None,
        training_dynamics: Optional[TrainingDynamicsMetrics] = None,
        thresholds: Optional[ThresholdConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize report with metric results.

        Args:
            compression: Compression metrics
            connectivity: Connectivity metrics
            information: Information-theoretic metrics
            prediction: Optional prediction metrics
            training_dynamics: Optional training dynamics
            thresholds: Optional threshold configuration
            verbose: Print debug information
        """
        self.compression = compression
        self.connectivity = connectivity
        self.information = information
        self.prediction = prediction
        self.training_dynamics = training_dynamics
        self.thresholds = thresholds or ThresholdConfig()
        self.verbose = verbose

        # Compute overall summary
        self.metrics_summary = self._compute_summary()

    @classmethod
    def generate(
        cls,
        graph_db_path: str,
        learner: Optional[Any] = None,
        config: Optional[MetricsConfig] = None,
        test_inputs: Optional[List[List[str]]] = None,
        verbose: bool = False
    ) -> 'MetricsReport':
        """
        Generate comprehensive metrics report.

        Args:
            graph_db_path: Path to hierarchy graph database
            learner: Optional HierarchicalConceptLearner (for prediction analysis)
            config: Optional metrics configuration
            test_inputs: Optional test inputs for prediction analysis
            verbose: Print debug information

        Returns:
            MetricsReport instance
        """
        config = config or MetricsConfig()
        thresholds = config.thresholds

        if verbose:
            print(f"\n{'='*80}")
            print("GENERATING COMPREHENSIVE METRICS REPORT")
            print(f"{'='*80}\n")

        # Graph analysis
        if verbose:
            print("Running graph analysis...")

        graph_analyzer = GraphAnalyzer(graph_db_path, thresholds=thresholds, verbose=verbose)
        compression, connectivity, context = graph_analyzer.analyze_all()

        # Information theory analysis
        if verbose:
            print("\nRunning information-theoretic analysis...")

        info_analyzer = InformationTheoryAnalyzer(graph_db_path, thresholds=thresholds, verbose=verbose)
        information = info_analyzer.analyze_all()

        # Prediction analysis (if learner provided)
        prediction = None
        if learner is not None:
            if verbose:
                print("\nRunning prediction analysis...")

            pred_analyzer = PredictionAnalyzer(learner, graph_db_path, thresholds=thresholds, verbose=verbose)
            prediction = pred_analyzer.analyze_all(test_inputs=test_inputs, num_samples=config.prediction_test_size)

        # Training dynamics analysis
        training_dynamics = None
        storage = HierarchyGraphStorage(graph_db_path, verbose=False)
        checkpoints = storage.get_checkpoints()

        if checkpoints:
            if verbose:
                print("\nAnalyzing training dynamics...")

            training_dynamics = cls._analyze_training_dynamics(checkpoints, thresholds)

        storage.close()

        if verbose:
            print(f"\n{'='*80}")
            print("REPORT GENERATION COMPLETE")
            print(f"{'='*80}\n")

        return cls(
            compression=compression,
            connectivity=connectivity,
            information=information,
            prediction=prediction,
            training_dynamics=training_dynamics,
            thresholds=thresholds,
            verbose=verbose
        )

    @staticmethod
    def _analyze_training_dynamics(
        checkpoints: List[Any],
        thresholds: ThresholdConfig
    ) -> TrainingDynamicsMetrics:
        """
        Analyze training dynamics from checkpoints.

        Computes:
        - Growth exponent (power-law fit)
        - Reusability trend over time

        Args:
            checkpoints: List of GraphCheckpoint objects
            thresholds: Threshold configuration

        Returns:
            TrainingDynamicsMetrics
        """
        if not checkpoints:
            return TrainingDynamicsMetrics(
                growth_exponent=0.0,
                growth_r_squared=0.0,
                reusability_trend=[],
                reusability_trend_slope=0.0,
                checkpoints=[],
                health_status={}
            )

        # Extract growth data (samples vs total patterns)
        samples = []
        total_patterns = []

        for cp in checkpoints:
            samples.append(cp.samples_processed)
            total_patterns.append(sum(cp.pattern_counts.values()))

        # Fit power-law: patterns = a * samples^b
        # log(patterns) = log(a) + b * log(samples)
        if len(samples) >= 3:
            log_samples = np.log(samples)
            log_patterns = np.log(total_patterns)

            # Linear regression in log-space
            coeffs = np.polyfit(log_samples, log_patterns, deg=1)
            growth_exponent = coeffs[0]  # b (exponent)

            # R-squared
            predicted = coeffs[0] * log_samples + coeffs[1]
            ss_res = np.sum((log_patterns - predicted) ** 2)
            ss_tot = np.sum((log_patterns - np.mean(log_patterns)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            growth_exponent = 0.0
            r_squared = 0.0

        # Extract reusability trend (mean_parents over time for node0)
        reusability_trend = []
        for cp in checkpoints:
            if 'reusability' in cp.metrics_snapshot:
                node0_reuse = cp.metrics_snapshot['reusability'].get('node0', {})
                mean_parents = node0_reuse.get('mean_parents', 0)
                reusability_trend.append(mean_parents)
            else:
                reusability_trend.append(0.0)

        # Compute trend slope
        if len(reusability_trend) >= 2:
            x = np.arange(len(reusability_trend))
            slope, _ = np.polyfit(x, reusability_trend, deg=1)
        else:
            slope = 0.0

        # Health scoring
        health_status = {}

        # Growth exponent health
        low, high = thresholds.growth_exponent_excellent
        if low <= growth_exponent <= high:
            health_status['growth_exponent'] = HealthStatus.EXCELLENT
        else:
            low, high = thresholds.growth_exponent_good
            if low <= growth_exponent <= high:
                health_status['growth_exponent'] = HealthStatus.GOOD
            else:
                low, high = thresholds.growth_exponent_warning
                if low <= growth_exponent <= high:
                    health_status['growth_exponent'] = HealthStatus.WARNING
                else:
                    health_status['growth_exponent'] = HealthStatus.POOR

        # Reusability trend health (positive slope is good)
        if slope > 0.01:
            health_status['reusability_trend'] = HealthStatus.EXCELLENT
        elif slope > 0:
            health_status['reusability_trend'] = HealthStatus.GOOD
        elif slope > -0.01:
            health_status['reusability_trend'] = HealthStatus.WARNING
        else:
            health_status['reusability_trend'] = HealthStatus.POOR

        # Convert checkpoints to dicts
        checkpoint_dicts = []
        for cp in checkpoints:
            checkpoint_dicts.append({
                'checkpoint_id': cp.checkpoint_id,
                'samples_processed': cp.samples_processed,
                'timestamp': cp.timestamp,
                'pattern_counts': cp.pattern_counts,
                'metrics_snapshot': cp.metrics_snapshot,
            })

        return TrainingDynamicsMetrics(
            growth_exponent=float(growth_exponent),
            growth_r_squared=float(r_squared),
            reusability_trend=reusability_trend,
            reusability_trend_slope=float(slope),
            checkpoints=checkpoint_dicts,
            health_status=health_status
        )

    def _compute_summary(self) -> MetricsSummary:
        """
        Compute overall health summary and recommendations.

        Returns:
            MetricsSummary with overall health and issues
        """
        # Aggregate health statuses
        all_statuses = []

        all_statuses.extend(self.compression.health_status.values())
        all_statuses.extend(self.connectivity.health_status.values())
        all_statuses.extend(self.information.health_status.values())

        if self.prediction:
            all_statuses.extend(self.prediction.health_status.values())

        if self.training_dynamics:
            all_statuses.extend(self.training_dynamics.health_status.values())

        # Count status frequencies
        status_counts = defaultdict(int)
        for status in all_statuses:
            status_counts[status] += 1

        # Determine overall health (worst status wins)
        if status_counts[HealthStatus.CRITICAL] > 0:
            overall_health = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.POOR] > 0:
            overall_health = HealthStatus.POOR
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_health = HealthStatus.WARNING
        elif status_counts[HealthStatus.GOOD] > 0:
            overall_health = HealthStatus.GOOD
        else:
            overall_health = HealthStatus.EXCELLENT

        # Category health (majority vote per category)
        def majority_health(statuses: List[HealthStatus]) -> HealthStatus:
            if not statuses:
                return HealthStatus.UNKNOWN

            counts = defaultdict(int)
            for s in statuses:
                counts[s] += 1

            # Return worst status
            for status in [HealthStatus.CRITICAL, HealthStatus.POOR, HealthStatus.WARNING,
                          HealthStatus.GOOD, HealthStatus.EXCELLENT]:
                if counts[status] > 0:
                    return status

            return HealthStatus.UNKNOWN

        compression_health = majority_health(list(self.compression.health_status.values()))
        connectivity_health = majority_health(list(self.connectivity.health_status.values()))
        information_health = majority_health(list(self.information.health_status.values()))

        prediction_health = HealthStatus.UNKNOWN
        if self.prediction:
            prediction_health = majority_health(list(self.prediction.health_status.values()))

        training_dynamics_health = HealthStatus.UNKNOWN
        if self.training_dynamics:
            training_dynamics_health = majority_health(list(self.training_dynamics.health_status.values()))

        # Identify critical issues
        critical_issues = []
        warnings = []
        recommendations = []

        # Compression issues
        for key, status in self.compression.health_status.items():
            if status == HealthStatus.CRITICAL:
                critical_issues.append(f"Critical compression issue: {key}")
            elif status == HealthStatus.POOR:
                critical_issues.append(f"Poor compression: {key}")
            elif status == HealthStatus.WARNING:
                warnings.append(f"Compression warning: {key}")

        # Connectivity issues
        for key, status in self.connectivity.health_status.items():
            if status == HealthStatus.CRITICAL:
                if 'orphan' in key:
                    critical_issues.append(f"Critical orphan rate: {key}")
                    recommendations.append("Increase training data or adjust chunk size")
                elif 'coverage' in key:
                    critical_issues.append(f"Critical coverage: {key}")
                    recommendations.append("Review pattern composition logic")
            elif status == HealthStatus.POOR:
                warnings.append(f"Poor connectivity: {key}")

        # Information theory issues
        for key, status in self.information.health_status.items():
            if status == HealthStatus.CRITICAL or status == HealthStatus.POOR:
                if 'effectiveness' in key:
                    critical_issues.append(f"Poor constraint effectiveness: {key}")
                    recommendations.append("Upper levels may not be constraining lower levels effectively")

        # Prediction issues
        if self.prediction:
            for key, status in self.prediction.health_status.items():
                if status == HealthStatus.POOR:
                    warnings.append(f"Prediction fan-out issue: {key}")
                    recommendations.append("Review prediction settings or increase training data")

        # Training dynamics issues
        if self.training_dynamics:
            if self.training_dynamics.health_status.get('growth_exponent') in [HealthStatus.POOR, HealthStatus.CRITICAL]:
                warnings.append("Growth exponent suggests suboptimal pattern learning")
                recommendations.append("Review chunk size and training data quality")

        # General recommendations
        if overall_health in [HealthStatus.EXCELLENT, HealthStatus.GOOD]:
            recommendations.append("Hierarchy is healthy. Consider scaling up training data.")
        elif overall_health == HealthStatus.WARNING:
            recommendations.append("Minor issues detected. Review warnings and adjust hyperparameters.")
        else:
            recommendations.append("Significant issues detected. Consider adjusting chunk size, increasing data, or reviewing architecture.")

        return MetricsSummary(
            overall_health=overall_health,
            compression=compression_health,
            connectivity=connectivity_health,
            information=information_health,
            prediction=prediction_health,
            training_dynamics=training_dynamics_health,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=list(set(recommendations)),  # Deduplicate
        )

    # ========================================================================
    # EXPORT & DISPLAY
    # ========================================================================

    def summary(self) -> str:
        """
        Generate text summary of metrics.

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("HIERARCHY METRICS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Overall health
        lines.append(f"Overall Health: {self.metrics_summary.overall_health.value.upper()}")
        lines.append("")

        # Category health
        lines.append("Category Health:")
        lines.append(f"  Compression:      {self.metrics_summary.compression.value}")
        lines.append(f"  Connectivity:     {self.metrics_summary.connectivity.value}")
        lines.append(f"  Information:      {self.metrics_summary.information.value}")
        if self.prediction:
            lines.append(f"  Prediction:       {self.metrics_summary.prediction.value}")
        if self.training_dynamics:
            lines.append(f"  Training Dynamics: {self.metrics_summary.training_dynamics.value}")
        lines.append("")

        # Critical issues
        if self.metrics_summary.critical_issues:
            lines.append("Critical Issues:")
            for issue in self.metrics_summary.critical_issues:
                lines.append(f"  ⚠️  {issue}")
            lines.append("")

        # Warnings
        if self.metrics_summary.warnings:
            lines.append("Warnings:")
            for warning in self.metrics_summary.warnings:
                lines.append(f"  ⚠  {warning}")
            lines.append("")

        # Recommendations
        if self.metrics_summary.recommendations:
            lines.append("Recommendations:")
            for rec in self.metrics_summary.recommendations:
                lines.append(f"  →  {rec}")
            lines.append("")

        # Key metrics
        lines.append("Key Metrics:")
        lines.append("")

        lines.append("  Compression Ratios:")
        for pair, ratio in self.compression.compression_ratios.items():
            lines.append(f"    {pair}: {ratio:.2f}x")
        lines.append("")

        lines.append("  Pattern Counts:")
        for level, count in sorted(self.compression.pattern_counts.items()):
            lines.append(f"    {level}: {count:,}")
        lines.append("")

        lines.append("  Reusability (mean parents):")
        for level, stats in sorted(self.connectivity.reusability.items()):
            lines.append(f"    {level}: {stats['mean_parents']:.2f} (orphan rate: {stats['orphan_rate']:.1%})")
        lines.append("")

        lines.append("  Coverage:")
        for pair, rate in sorted(self.connectivity.coverage.items()):
            lines.append(f"    {pair}: {rate:.1%}")
        lines.append("")

        lines.append("  Constraint Effectiveness:")
        for pair, eff in sorted(self.information.constraint_effectiveness.items()):
            lines.append(f"    {pair}: {eff:.1%}")
        lines.append("")

        if self.training_dynamics:
            lines.append("  Training Dynamics:")
            lines.append(f"    Growth exponent: {self.training_dynamics.growth_exponent:.3f} (R²={self.training_dynamics.growth_r_squared:.3f})")
            lines.append(f"    Reusability trend slope: {self.training_dynamics.reusability_trend_slope:.4f}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def export_json(self, output_path: str):
        """
        Export report as JSON.

        Args:
            output_path: Path to output JSON file
        """
        report_dict = {
            'summary': self.metrics_summary.to_dict(),
            'compression': self.compression.to_dict(),
            'connectivity': self.connectivity.to_dict(),
            'information': self.information.to_dict(),
        }

        if self.prediction:
            report_dict['prediction'] = self.prediction.to_dict()

        if self.training_dynamics:
            report_dict['training_dynamics'] = self.training_dynamics.to_dict()

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        if self.verbose:
            print(f"✓ Report exported to {output_path}")

    def export_csv(self, output_dir: str):
        """
        Export metrics as multiple CSV files.

        Args:
            output_dir: Directory for CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compression metrics
        with open(output_dir / 'compression.csv', 'w') as f:
            f.write("metric,level_pair,value,health\n")
            for pair, ratio in self.compression.compression_ratios.items():
                health = self.compression.health_status.get(f"compression_{pair}", HealthStatus.UNKNOWN).value
                f.write(f"compression_ratio,{pair},{ratio},{health}\n")

        # Pattern counts
        with open(output_dir / 'pattern_counts.csv', 'w') as f:
            f.write("level,count\n")
            for level, count in sorted(self.compression.pattern_counts.items()):
                f.write(f"{level},{count}\n")

        # Reusability
        with open(output_dir / 'reusability.csv', 'w') as f:
            f.write("level,mean_parents,median_parents,orphan_rate,p90_parents\n")
            for level, stats in sorted(self.connectivity.reusability.items()):
                f.write(f"{level},{stats['mean_parents']},{stats['median_parents']},"
                       f"{stats['orphan_rate']},{stats['p90_parents']}\n")

        if self.verbose:
            print(f"✓ CSV files exported to {output_dir}")

    def __str__(self) -> str:
        return self.summary()
