#!/usr/bin/env python3
"""
Training Run Snapshot - Capture Complete KB Statistics

NOTE: MongoDB support has been removed. KATO now uses ClickHouse + Redis.
This module needs to be reimplemented to query ClickHouse instead of MongoDB.

TODO: Reimplement using ClickHouse queries via clickhouse-connect library.

Original functionality captured:
- Database sizes (per node)
- Frequency histograms (complete distributions)
- Shannon entropy distributions
- Pattern statistics (counts, means, medians)
- Zipfian power-law fits
- Top-N most frequent patterns
- Hierarchy utilization metrics

Planned usage (once reimplemented):
    snapshot = TrainingRunSnapshot.create_from_learner(
        learner=learner,
        run_id='run_12345',
        clickhouse_uri='clickhouse://localhost:9000/'
    )

    snapshot.save('snapshots/run_12345_snapshot.json')

    # Later analysis
    snapshot = TrainingRunSnapshot.load('snapshots/run_12345_snapshot.json')
    print(f"Total patterns: {snapshot.total_patterns:,}")
    snapshot.plot_frequency_distributions()
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class NodeSnapshot:
    """Statistics for a single node"""
    node_name: str
    db_name: str
    db_size_mb: float
    total_patterns: int
    total_observations: int

    # Frequency distribution
    frequency_histogram: Dict[int, int]  # {frequency: count}

    # Statistics
    mean_frequency: float
    median_frequency: float
    max_frequency: int
    min_frequency: int
    std_frequency: float

    # Zipfian analysis
    zipf_alpha: Optional[float] = None  # Power law exponent
    zipf_r_squared: Optional[float] = None  # Goodness of fit

    # Shannon entropy (if available)
    shannon_histogram: Optional[Dict[str, int]] = None  # {"0.0-0.1": count, ...}
    mean_shannon: Optional[float] = None
    median_shannon: Optional[float] = None

    # Top patterns
    top_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Frequency range breakdowns
    freq_1: int = 0  # Patterns with frequency == 1
    freq_2_5: int = 0
    freq_6_10: int = 0
    freq_11_50: int = 0
    freq_50_plus: int = 0

    # NEW: Graph topology (for composition analysis and post-pruning validation)
    parent_child_edges: Optional[List[Tuple[str, str]]] = None  # [(parent_id, child_id), ...]
    reusability_histogram: Optional[Dict[int, int]] = None  # {num_parents: count}
    orphan_count: Optional[int] = None  # Patterns with 0 parents
    orphan_rate: Optional[float] = None  # orphan_count / total_patterns
    coverage_to_parent: Optional[float] = None  # % used by parent level

    # NEW: Prediction sample statistics
    prediction_samples: Optional[Dict[str, Dict[str, float]]] = None
    # {
    #   'predictive_information': {'mean': 0.5, 'std': 0.2, 'median': 0.55, ...},
    #   'potential': {'mean': 0.3, 'std': 0.15, ...},
    #   'confidence': {'mean': 0.6, ...},
    #   'similarity': {'mean': 0.7, ...},
    #   'fanout': {'mean': 15, 'std': 8, ...}
    # }

    # NEW: Hierarchical frequency validation
    freq_correlation_to_children: Optional[float] = None  # Spearman correlation
    freq_compression_ratio: Optional[float] = None  # mean(this_freq) / mean(child_freq)


@dataclass
class TrainingRunSnapshot:
    """Complete snapshot of a training run's knowledge bases

    NOTE: MongoDB support removed. Needs ClickHouse implementation.
    """
    run_id: str
    timestamp: float
    config: Dict[str, Any]
    storage_uri: str  # Changed from mongo_uri to storage_uri

    # Node-level snapshots
    nodes: Dict[str, NodeSnapshot]

    # Summary statistics
    total_patterns: int = 0
    total_storage_mb: float = 0.0
    total_observations: int = 0
    hierarchy_utilization: List[float] = field(default_factory=list)  # Ratio per level

    # Snapshot metadata
    snapshot_duration_seconds: float = 0.0
    snapshot_path: Optional[str] = None

    @classmethod
    def create_from_learner(
        cls,
        learner: Any,  # HierarchicalConceptLearner
        run_id: str,
        storage_uri: str = 'clickhouse://localhost:9000/',
        timeout_ms: int = 5000,
        top_n_patterns: int = 10,
        capture_graph_topology: bool = True,
        capture_prediction_samples: bool = True,
        num_prediction_samples: int = 100,
        validate_hierarchy: bool = True,
        verbose: bool = True
    ) -> 'TrainingRunSnapshot':
        """
        Create snapshot by querying ClickHouse for all node pattern data.

        NOTE: MongoDB support removed. This method needs to be reimplemented
        to query ClickHouse instead of MongoDB.

        Args:
            learner: HierarchicalConceptLearner instance
            run_id: Unique run identifier
            storage_uri: ClickHouse connection string
            timeout_ms: Connection timeout
            top_n_patterns: Number of top patterns to capture
            capture_graph_topology: Capture parent-child relationships and orphan rates
            capture_prediction_samples: Sample predictions to measure quality
            num_prediction_samples: Number of prediction samples to take
            validate_hierarchy: Validate frequency correlation between levels
            verbose: Print progress

        Returns:
            TrainingRunSnapshot with complete statistics

        Raises:
            NotImplementedError: MongoDB support removed, needs ClickHouse implementation
        """
        raise NotImplementedError(
            "TrainingRunSnapshot.create_from_learner() requires ClickHouse implementation. "
            "MongoDB support has been removed from KATO. "
            "TODO: Reimplement using clickhouse-connect library to query pattern statistics."
        )

    @staticmethod
    def _capture_node(*args, **kwargs) -> NodeSnapshot:
        """MongoDB support removed. Needs ClickHouse implementation."""
        raise NotImplementedError("MongoDB support removed. Needs ClickHouse implementation.")

    @staticmethod
    def _fit_zipf(frequency_histogram: Dict[int, int]) -> Tuple[Optional[float], Optional[float]]:
        """
        Fit Zipfian power law to frequency distribution.

        Returns:
            (alpha, r_squared) where y = x^(-alpha)
        """
        if not frequency_histogram or len(frequency_histogram) < 3:
            return None, None

        try:
            # Create rank-frequency data
            sorted_freqs = sorted(frequency_histogram.items(), key=lambda x: x[0], reverse=True)

            ranks = []
            frequencies = []
            for freq, count in sorted_freqs:
                start_rank = len(ranks) + 1
                for i in range(count):
                    ranks.append(start_rank + i)
                    frequencies.append(freq)

            # Log-log fit: log(freq) = -alpha * log(rank) + c
            log_ranks = np.log(ranks)
            log_freqs = np.log(frequencies)

            # Linear regression in log-log space
            A = np.vstack([log_ranks, np.ones(len(log_ranks))]).T
            alpha_neg, c = np.linalg.lstsq(A, log_freqs, rcond=None)[0]
            alpha = -alpha_neg  # Convert to positive exponent

            # Compute R-squared
            predicted = -alpha * log_ranks + c
            ss_res = np.sum((log_freqs - predicted) ** 2)
            ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            return float(alpha), float(r_squared)

        except Exception:
            return None, None

    @staticmethod
    def _extract_shannon(*args, **kwargs) -> Tuple[Optional[Dict], Optional[float], Optional[float]]:
        """MongoDB support removed. Needs ClickHouse implementation."""
        return None, None, None

    @staticmethod
    def _capture_graph_topology(*args, **kwargs) -> Dict[str, Any]:
        """MongoDB support removed. Needs ClickHouse implementation."""
        raise NotImplementedError("MongoDB support removed. Needs ClickHouse implementation.")

    @staticmethod
    def _capture_prediction_samples(
        node_client: Any,
        num_samples: int = 100,
        test_sequence_length: int = 10
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Sample predictions and extract quality metrics.

        NOTE: This method doesn't require MongoDB and can still be used.

        Args:
            node_client: KATO node client
            num_samples: Number of test samples
            test_sequence_length: Length of test sequences

        Returns:
            Aggregated statistics for:
                - predictive_information
                - potential
                - confidence
                - similarity
                - fanout (number of predictions)
        """
        import random
        from collections import defaultdict

        # Generate test vocabulary (common words)
        vocab = ["the", "a", "is", "was", "in", "on", "to", "of", "and", "for",
                 "it", "with", "as", "at", "by", "from", "an", "be", "this", "that"]

        all_metrics = defaultdict(list)

        for _ in range(num_samples):
            # Generate random test sequence
            test_tokens = random.choices(vocab, k=test_sequence_length)
            test_obs = [{'strings': [tok]} for tok in test_tokens]

            try:
                # Observe sequence (without learning)
                for obs in test_obs:
                    node_client.observe(**obs)

                # Get predictions
                result = node_client.get_predictions()

                if result and 'predictions' in result:
                    predictions = result['predictions']

                    # Record fan-out
                    all_metrics['fanout'].append(len(predictions))

                    # Extract metrics from each prediction
                    for pred in predictions:
                        if 'predictive_information' in pred:
                            all_metrics['predictive_information'].append(pred['predictive_information'])
                        if 'potential' in pred:
                            all_metrics['potential'].append(pred['potential'])
                        if 'confidence' in pred:
                            all_metrics['confidence'].append(pred['confidence'])
                        if 'similarity' in pred:
                            all_metrics['similarity'].append(pred['similarity'])
                else:
                    all_metrics['fanout'].append(0)

                # Clear STM for next sample
                node_client.clear_stm()

            except Exception:
                # Prediction failed, record as zero
                all_metrics['fanout'].append(0)
                try:
                    node_client.clear_stm()
                except:
                    pass

        # Compute summary statistics
        if not all_metrics:
            return None

        summary = {}
        for metric_name, values in all_metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        return summary if summary else None

    @staticmethod
    def _validate_hierarchical_frequencies(*args, **kwargs) -> Dict[str, Optional[float]]:
        """MongoDB support removed. Needs ClickHouse implementation."""
        return {
            'freq_correlation': None,
            'freq_compression_ratio': None
        }

    def save(self, filepath: str):
        """Save snapshot to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle nested dataclasses
        data = {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'config': self.config,
            'storage_uri': self.storage_uri,
            'nodes': {
                node_name: asdict(node_snapshot)
                for node_name, node_snapshot in self.nodes.items()
            },
            'total_patterns': self.total_patterns,
            'total_storage_mb': self.total_storage_mb,
            'total_observations': self.total_observations,
            'hierarchy_utilization': self.hierarchy_utilization,
            'snapshot_duration_seconds': self.snapshot_duration_seconds
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.snapshot_path = filepath

    @staticmethod
    def load(filepath: str) -> 'TrainingRunSnapshot':
        """Load snapshot from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct NodeSnapshot objects
        nodes = {
            node_name: NodeSnapshot(**node_data)
            for node_name, node_data in data['nodes'].items()
        }

        # Support old format with mongo_uri
        storage_uri = data.get('storage_uri', data.get('mongo_uri', 'unknown'))

        snapshot = TrainingRunSnapshot(
            run_id=data['run_id'],
            timestamp=data['timestamp'],
            config=data['config'],
            storage_uri=storage_uri,
            nodes=nodes,
            total_patterns=data['total_patterns'],
            total_storage_mb=data['total_storage_mb'],
            total_observations=data.get('total_observations', 0),
            hierarchy_utilization=data.get('hierarchy_utilization', []),
            snapshot_duration_seconds=data.get('snapshot_duration_seconds', 0.0),
            snapshot_path=filepath
        )

        return snapshot

    def get_summary(self) -> Dict[str, Any]:
        """Get summary dict for storage in TrainingHistory SQLite"""
        summary = {
            'snapshot_path': self.snapshot_path,
            'total_patterns': self.total_patterns,
            'total_storage_mb': self.total_storage_mb,
            'total_observations': self.total_observations,
        }

        # Add per-node summaries
        for node_name, node_snapshot in self.nodes.items():
            summary[f'{node_name}_patterns'] = node_snapshot.total_patterns
            summary[f'{node_name}_storage_mb'] = node_snapshot.db_size_mb
            summary[f'{node_name}_mean_freq'] = node_snapshot.mean_frequency
            summary[f'{node_name}_zipf_alpha'] = node_snapshot.zipf_alpha
            summary[f'{node_name}_mean_shannon'] = node_snapshot.mean_shannon

        return summary

    def print_summary(self):
        """Print human-readable summary"""
        print(f"\n{'='*80}")
        print(f"TRAINING SNAPSHOT: {self.run_id}")
        print(f"{'='*80}")

        print(f"\n📊 SUMMARY")
        print(f"  Total patterns: {self.total_patterns:,}")
        print(f"  Total storage: {self.total_storage_mb:.2f} MB")
        print(f"  Total observations: {self.total_observations:,}")
        print(f"  Snapshot duration: {self.snapshot_duration_seconds:.2f}s")

        print(f"\n📁 NODE STATISTICS")
        for node_name in sorted(self.nodes.keys()):
            ns = self.nodes[node_name]
            print(f"\n  {node_name}:")
            print(f"    Patterns: {ns.total_patterns:,}")
            print(f"    Storage: {ns.db_size_mb:.2f} MB")
            print(f"    Mean frequency: {ns.mean_frequency:.2f}")
            print(f"    Frequency range: {ns.min_frequency}-{ns.max_frequency}")
            if ns.zipf_alpha:
                print(f"    Zipf alpha: {ns.zipf_alpha:.3f} (R²={ns.zipf_r_squared:.3f})")
            if ns.mean_shannon:
                print(f"    Mean Shannon: {ns.mean_shannon:.3f}")
            print(f"    Freq breakdown: 1:{ns.freq_1} | 2-5:{ns.freq_2_5} | "
                  f"6-10:{ns.freq_6_10} | 11-50:{ns.freq_11_50} | 50+:{ns.freq_50_plus}")

        print(f"\n{'='*80}\n")


def main():
    """Example usage"""
    # This would typically be called after training
    print("TrainingRunSnapshot module loaded")
    print("Use: TrainingRunSnapshot.create_from_learner(learner, run_id)")


if __name__ == '__main__':
    main()
