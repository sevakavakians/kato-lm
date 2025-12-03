#!/usr/bin/env python3
"""
Training Run Snapshot - Capture Complete KB Statistics

Captures comprehensive post-training statistics from MongoDB before knowledgebases
are cleared for the next training run. Enables detailed comparison between
different training configurations.

Captured Data:
- Database sizes (per node)
- Frequency histograms (complete distributions)
- Shannon entropy distributions
- Pattern statistics (counts, means, medians)
- Zipfian power-law fits
- Top-N most frequent patterns
- Hierarchy utilization metrics

Usage:
    snapshot = TrainingRunSnapshot.create_from_learner(
        learner=learner,
        run_id='run_12345',
        mongo_uri='mongodb://localhost:27017/'
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
from pymongo import MongoClient


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
    """Complete snapshot of a training run's knowledge bases"""
    run_id: str
    timestamp: float
    config: Dict[str, Any]
    mongo_uri: str

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
        mongo_uri: str = 'mongodb://localhost:27017/',
        timeout_ms: int = 5000,
        top_n_patterns: int = 10,
        capture_graph_topology: bool = True,
        capture_prediction_samples: bool = True,
        num_prediction_samples: int = 100,
        validate_hierarchy: bool = True,
        verbose: bool = True
    ) -> 'TrainingRunSnapshot':
        """
        Create snapshot by querying MongoDB for all node databases.

        Args:
            learner: HierarchicalConceptLearner instance
            run_id: Unique run identifier
            mongo_uri: MongoDB connection string
            timeout_ms: Connection timeout
            top_n_patterns: Number of top patterns to capture
            capture_graph_topology: Capture parent-child relationships and orphan rates
            capture_prediction_samples: Sample predictions to measure quality
            num_prediction_samples: Number of prediction samples to take
            validate_hierarchy: Validate frequency correlation between levels
            verbose: Print progress

        Returns:
            TrainingRunSnapshot with complete statistics
        """
        start_time = time.time()

        if verbose:
            print(f"\n📸 Creating training snapshot for {run_id}...")

        # Extract configuration
        config = {
            'num_levels': learner.num_nodes,
            'chunk_sizes': [n.chunk_size for n in learner.node_configs],
            'tokenizer': learner.tokenizer_name if hasattr(learner, 'tokenizer_name') else 'unknown',
            'batch_size': learner.node0_batch_size if hasattr(learner, 'node0_batch_size') else None,
        }

        # Connect to MongoDB
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=timeout_ms)

        try:
            # Test connection
            client.server_info()

            # Capture each node
            node_snapshots = {}

            for node_name, node_client in learner.nodes.items():
                if verbose:
                    print(f"  Capturing {node_name}...", end='', flush=True)

                try:
                    node_snapshot = cls._capture_node(
                        client=client,
                        node_name=node_name,
                        node_client=node_client,
                        top_n=top_n_patterns
                    )
                    node_snapshots[node_name] = node_snapshot

                    if verbose:
                        print(f" ✓ {node_snapshot.total_patterns:,} patterns, {node_snapshot.db_size_mb:.2f} MB")

                except Exception as e:
                    if verbose:
                        print(f" ⚠️  Failed: {e}")

            # Second pass: Capture graph topology, predictions, and hierarchy validation
            if node_snapshots:
                # Get node names in order (node0, node1, node2, node3)
                node_names_sorted = sorted(node_snapshots.keys())

                for i, node_name in enumerate(node_names_sorted):
                    node_snapshot = node_snapshots[node_name]
                    node_client = learner.nodes[node_name]

                    # Determine parent node (next level up)
                    parent_node_name = node_names_sorted[i + 1] if i + 1 < len(node_names_sorted) else None

                    if verbose:
                        print(f"  Analyzing {node_name}...", end='', flush=True)

                    try:
                        # Capture graph topology
                        if capture_graph_topology:
                            parent_db_name = f"{parent_node_name}_kato" if parent_node_name else None
                            topology = cls._capture_graph_topology(
                                client=client,
                                node_name=node_name,
                                node_db_name=node_snapshot.db_name,
                                parent_db_name=parent_db_name
                            )
                            node_snapshot.parent_child_edges = topology['parent_child_edges']
                            node_snapshot.reusability_histogram = topology['reusability_histogram']
                            node_snapshot.orphan_count = topology['orphan_count']
                            node_snapshot.orphan_rate = topology['orphan_rate']
                            node_snapshot.coverage_to_parent = topology['coverage_to_parent']

                        # Capture prediction samples
                        if capture_prediction_samples:
                            pred_stats = cls._capture_prediction_samples(
                                node_client=node_client,
                                num_samples=num_prediction_samples
                            )
                            node_snapshot.prediction_samples = pred_stats

                        # Validate hierarchical frequencies (with child level below)
                        if validate_hierarchy and i > 0:
                            child_node_name = node_names_sorted[i - 1]
                            child_snapshot = node_snapshots[child_node_name]
                            freq_validation = cls._validate_hierarchical_frequencies(
                                client=client,
                                child_db_name=child_snapshot.db_name,
                                parent_db_name=node_snapshot.db_name
                            )
                            node_snapshot.freq_correlation_to_children = freq_validation['freq_correlation']
                            node_snapshot.freq_compression_ratio = freq_validation['freq_compression_ratio']

                        if verbose:
                            metrics_captured = []
                            if capture_graph_topology:
                                metrics_captured.append(f"orphan={node_snapshot.orphan_rate:.1%}")
                            if capture_prediction_samples and node_snapshot.prediction_samples:
                                fanout = node_snapshot.prediction_samples.get('fanout', {}).get('mean', 0)
                                metrics_captured.append(f"fanout={fanout:.1f}")
                            print(f" ✓ {', '.join(metrics_captured) if metrics_captured else 'done'}")

                    except Exception as e:
                        if verbose:
                            print(f" ⚠️  Analysis failed: {e}")

            # Compute summary statistics
            total_patterns = sum(ns.total_patterns for ns in node_snapshots.values())
            total_storage_mb = sum(ns.db_size_mb for ns in node_snapshots.values())
            total_observations = sum(ns.total_observations for ns in node_snapshots.values())

            # Hierarchy utilization (pattern count ratio per level)
            if total_patterns > 0:
                hierarchy_utilization = [
                    ns.total_patterns / total_patterns
                    for ns in node_snapshots.values()
                ]
            else:
                hierarchy_utilization = [0.0] * len(node_snapshots)

            snapshot = cls(
                run_id=run_id,
                timestamp=time.time(),
                config=config,
                mongo_uri=mongo_uri,
                nodes=node_snapshots,
                total_patterns=total_patterns,
                total_storage_mb=total_storage_mb,
                total_observations=total_observations,
                hierarchy_utilization=hierarchy_utilization,
                snapshot_duration_seconds=time.time() - start_time
            )

            if verbose:
                print(f"\n✓ Snapshot complete ({snapshot.snapshot_duration_seconds:.2f}s)")
                print(f"  Total patterns: {total_patterns:,}")
                print(f"  Total storage: {total_storage_mb:.2f} MB")

            return snapshot

        finally:
            client.close()

    @staticmethod
    def _capture_node(
        client: MongoClient,
        node_name: str,
        node_client: Any,
        top_n: int = 10
    ) -> NodeSnapshot:
        """Capture statistics for a single node"""
        # Get database name from node client
        db_name = node_client.database_name if hasattr(node_client, 'database_name') else f"{node_name}_kato"

        db = client[db_name]
        patterns_collection = db['patterns_kb']

        # Get database size
        stats = db.command('dbStats')
        db_size_mb = stats['dataSize'] / (1024 * 1024)

        # Get pattern count
        total_patterns = patterns_collection.count_documents({})

        # Get total observations (sum of frequencies)
        total_obs_result = list(patterns_collection.aggregate([
            {'$group': {'_id': None, 'total': {'$sum': '$frequency'}}}
        ]))
        total_observations = total_obs_result[0]['total'] if total_obs_result else 0

        # Get frequency histogram
        freq_histogram_result = patterns_collection.aggregate([
            {'$group': {'_id': '$frequency', 'count': {'$sum': 1}}},
            {'$sort': {'_id': 1}}
        ])
        frequency_histogram = {doc['_id']: doc['count'] for doc in freq_histogram_result}

        # Compute frequency statistics
        if frequency_histogram:
            all_frequencies = []
            for freq, count in frequency_histogram.items():
                all_frequencies.extend([freq] * count)

            mean_frequency = np.mean(all_frequencies)
            median_frequency = np.median(all_frequencies)
            std_frequency = np.std(all_frequencies)
            max_frequency = max(frequency_histogram.keys())
            min_frequency = min(frequency_histogram.keys())
        else:
            mean_frequency = median_frequency = std_frequency = 0.0
            max_frequency = min_frequency = 0

        # Frequency range breakdowns
        freq_1 = frequency_histogram.get(1, 0)
        freq_2_5 = sum(frequency_histogram.get(f, 0) for f in range(2, 6))
        freq_6_10 = sum(frequency_histogram.get(f, 0) for f in range(6, 11))
        freq_11_50 = sum(frequency_histogram.get(f, 0) for f in range(11, 51))
        freq_50_plus = sum(count for freq, count in frequency_histogram.items() if freq > 50)

        # Fit Zipf distribution (power law)
        zipf_alpha, zipf_r_squared = TrainingRunSnapshot._fit_zipf(frequency_histogram)

        # Try to get Shannon entropy distribution (may not be stored)
        shannon_histogram, mean_shannon, median_shannon = TrainingRunSnapshot._extract_shannon(
            patterns_collection
        )

        # Get top N patterns
        top_patterns_cursor = patterns_collection.find(
            {},
            {'name': 1, 'frequency': 1, 'pattern_data': 1, '_id': 0}
        ).sort('frequency', -1).limit(top_n)

        top_patterns = []
        for pattern in top_patterns_cursor:
            top_patterns.append({
                'name': pattern['name'][:50] + '...' if len(pattern['name']) > 50 else pattern['name'],
                'frequency': pattern['frequency'],
                'length': len(pattern.get('pattern_data', []))
            })

        return NodeSnapshot(
            node_name=node_name,
            db_name=db_name,
            db_size_mb=db_size_mb,
            total_patterns=total_patterns,
            total_observations=total_observations,
            frequency_histogram=frequency_histogram,
            mean_frequency=float(mean_frequency),
            median_frequency=float(median_frequency),
            max_frequency=int(max_frequency),
            min_frequency=int(min_frequency),
            std_frequency=float(std_frequency),
            zipf_alpha=zipf_alpha,
            zipf_r_squared=zipf_r_squared,
            shannon_histogram=shannon_histogram,
            mean_shannon=mean_shannon,
            median_shannon=median_shannon,
            top_patterns=top_patterns,
            freq_1=freq_1,
            freq_2_5=freq_2_5,
            freq_6_10=freq_6_10,
            freq_11_50=freq_11_50,
            freq_50_plus=freq_50_plus
        )

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
            # Rank 1 = highest frequency, rank N = lowest frequency
            sorted_freqs = sorted(frequency_histogram.items(), key=lambda x: x[0], reverse=True)

            ranks = []
            frequencies = []
            for freq, count in sorted_freqs:
                # Each pattern with this frequency gets consecutive ranks
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
    def _extract_shannon(patterns_collection) -> Tuple[Optional[Dict], Optional[float], Optional[float]]:
        """
        Try to extract Shannon entropy distribution from patterns.

        Note: Shannon values may not be stored in pattern KB. This attempts to
        find them if they exist in emotives or other fields.

        Returns:
            (histogram, mean, median) or (None, None, None)
        """
        try:
            # Try to extract from emotives.normalized_entropy if it exists
            pipeline = [
                {'$match': {'emotives.normalized_entropy': {'$exists': True}}},
                {'$project': {'shannon': '$emotives.normalized_entropy'}}
            ]

            results = list(patterns_collection.aggregate(pipeline))

            if not results:
                return None, None, None

            # Create histogram with 0.1 bins
            shannon_values = [r['shannon'] for r in results if 'shannon' in r]

            if not shannon_values:
                return None, None, None

            # Create bins: 0.0-0.1, 0.1-0.2, etc.
            histogram = {}
            for val in shannon_values:
                bin_key = f"{int(val * 10) / 10:.1f}-{int(val * 10) / 10 + 0.1:.1f}"
                histogram[bin_key] = histogram.get(bin_key, 0) + 1

            mean_shannon = float(np.mean(shannon_values))
            median_shannon = float(np.median(shannon_values))

            return histogram, mean_shannon, median_shannon

        except Exception:
            # Shannon values not available
            return None, None, None

    @staticmethod
    def _capture_graph_topology(
        client: MongoClient,
        node_name: str,
        node_db_name: str,
        parent_db_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Capture parent-child relationships and composition statistics.

        Args:
            client: MongoDB client
            node_name: Node name (e.g., 'node0')
            node_db_name: Database name for this node
            parent_db_name: Database name for parent node (None for top level)

        Returns:
            Dict with:
                - parent_child_edges: List of (parent_id, child_id) tuples
                - reusability_histogram: {num_parents: count}
                - orphan_count: Number of patterns with 0 parents
                - orphan_rate: Percentage of orphans
                - coverage_to_parent: % used by parent level
        """
        from collections import defaultdict

        # Query this level's patterns
        db = client[node_db_name]
        patterns = db['patterns_kb'].find({}, {'name': 1, '_id': 0})
        pattern_ids = {p['name'] for p in patterns}

        if not parent_db_name:
            # Top level (node3) has no parents - all are orphans
            return {
                'parent_child_edges': [],
                'reusability_histogram': {0: len(pattern_ids)},
                'orphan_count': len(pattern_ids),
                'orphan_rate': 1.0 if pattern_ids else 0.0,
                'coverage_to_parent': 0.0
            }

        # Query parent level patterns and extract child references
        parent_db = client[parent_db_name]
        parent_patterns = parent_db['patterns_kb'].find(
            {}, {'name': 1, 'pattern_data': 1, '_id': 0}
        )

        edges = []
        child_parent_count = defaultdict(int)

        for parent_pat in parent_patterns:
            parent_id = parent_pat['name']
            pattern_data = parent_pat.get('pattern_data', [])

            # Extract child pattern names from pattern_data
            # pattern_data = [[child1], [child2], ...]
            for event in pattern_data:
                if event and len(event) > 0:
                    child_id = event[0]  # Each event has one pattern name
                    if child_id in pattern_ids:  # Validate reference exists
                        edges.append((parent_id, child_id))
                        child_parent_count[child_id] += 1

        # Compute reusability statistics
        reusability_histogram = defaultdict(int)
        for pattern_id in pattern_ids:
            num_parents = child_parent_count.get(pattern_id, 0)
            reusability_histogram[num_parents] += 1

        orphan_count = reusability_histogram.get(0, 0)
        orphan_rate = orphan_count / len(pattern_ids) if pattern_ids else 0.0
        coverage = (len(pattern_ids) - orphan_count) / len(pattern_ids) if pattern_ids else 0.0

        return {
            'parent_child_edges': edges[:1000],  # Limit to 1000 edges for storage
            'reusability_histogram': dict(reusability_histogram),
            'orphan_count': orphan_count,
            'orphan_rate': float(orphan_rate),
            'coverage_to_parent': float(coverage)
        }

    @staticmethod
    def _capture_prediction_samples(
        node_client: Any,
        num_samples: int = 100,
        test_sequence_length: int = 10
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Sample predictions and extract quality metrics.

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

            except Exception as e:
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
    def _validate_hierarchical_frequencies(
        client: MongoClient,
        child_db_name: str,
        parent_db_name: str
    ) -> Dict[str, Optional[float]]:
        """
        Validate frequency alignment between parent and child levels.

        For each parent pattern:
          - Sum frequencies of its child patterns
          - Compare parent_freq vs sum(child_freqs)

        Args:
            client: MongoDB client
            child_db_name: Database name for child level
            parent_db_name: Database name for parent level

        Returns:
            Dict with:
                - freq_correlation: Spearman correlation
                - freq_compression_ratio: mean(parent_freq) / mean(child_freq)
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            # scipy not available
            return {
                'freq_correlation': None,
                'freq_compression_ratio': None
            }

        # Get parent patterns with their children
        parent_db = client[parent_db_name]
        parent_patterns = parent_db['patterns_kb'].find(
            {}, {'name': 1, 'frequency': 1, 'pattern_data': 1, '_id': 0}
        )

        # Get child frequencies
        child_db = client[child_db_name]
        child_freq_map = {}
        for child_pat in child_db['patterns_kb'].find({}, {'name': 1, 'frequency': 1, '_id': 0}):
            child_freq_map[child_pat['name']] = child_pat.get('frequency', 0)

        parent_freqs = []
        child_freq_sums = []

        for parent_pat in parent_patterns:
            parent_freq = parent_pat.get('frequency', 0)
            pattern_data = parent_pat.get('pattern_data', [])

            # Sum child frequencies
            child_sum = 0
            for event in pattern_data:
                if event and len(event) > 0:
                    child_id = event[0]
                    child_sum += child_freq_map.get(child_id, 0)

            if child_sum > 0:  # Only include if children found
                parent_freqs.append(parent_freq)
                child_freq_sums.append(child_sum)

        # Compute statistics
        if len(parent_freqs) >= 3:
            correlation, _ = spearmanr(parent_freqs, child_freq_sums)

            mean_parent = np.mean(parent_freqs)
            mean_child_sum = np.mean(child_freq_sums)
            compression_ratio = mean_parent / mean_child_sum if mean_child_sum > 0 else 0.0

            return {
                'freq_correlation': float(correlation) if not np.isnan(correlation) else None,
                'freq_compression_ratio': float(compression_ratio)
            }
        else:
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
            'mongo_uri': self.mongo_uri,
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

        snapshot = TrainingRunSnapshot(
            run_id=data['run_id'],
            timestamp=data['timestamp'],
            config=data['config'],
            mongo_uri=data['mongo_uri'],
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
