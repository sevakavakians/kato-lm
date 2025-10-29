"""
Training-time data collectors for hierarchy metrics.

Lightweight collectors that capture graph structure during training with minimal overhead.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .storage import HierarchyGraphStorage, PatternNode
from .config import CollectorConfig


@dataclass
class TrainingDynamicsTracker:
    """
    Track training dynamics over time for metrics 13-14.

    Captures pattern growth, reusability trends at checkpoints.
    """
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    samples_processed: int = 0
    start_time: float = field(default_factory=time.time)

    def add_checkpoint(
        self,
        samples: int,
        pattern_counts: Dict[str, int],
        reusability_stats: Optional[Dict[str, float]] = None
    ):
        """Add checkpoint snapshot"""
        self.samples_processed = samples
        self.checkpoints.append({
            'samples': samples,
            'timestamp': time.time(),
            'elapsed_seconds': time.time() - self.start_time,
            'pattern_counts': pattern_counts.copy(),
            'reusability': reusability_stats or {},
        })

    def get_growth_data(self) -> tuple:
        """
        Get (samples, patterns) arrays for power-law fitting.

        Returns:
            (samples_array, patterns_array) for curve fitting
        """
        if not self.checkpoints:
            return [], []

        samples = [cp['samples'] for cp in self.checkpoints]
        patterns = [sum(cp['pattern_counts'].values()) for cp in self.checkpoints]
        return samples, patterns

    def get_reusability_trend(self, node_level: str = 'node0') -> List[float]:
        """
        Get reusability trend over time for a specific level.

        Returns:
            List of mean_parents values at each checkpoint
        """
        return [
            cp['reusability'].get(node_level, {}).get('mean_parents', 0)
            for cp in self.checkpoints
            if 'reusability' in cp and node_level in cp['reusability']
        ]


class HierarchyMetricsCollector:
    """
    Lightweight collector for training-time graph data.

    Captures:
    - Pattern creation (IDs, levels, timestamps)
    - Parent-child relationships (graph edges)
    - Checkpoints at intervals
    - Training dynamics

    Usage:
        collector = HierarchyMetricsCollector(learner, checkpoint_interval=1000)

        # During training (automatic if integrated)
        # ... training happens ...

        # After training
        collector.save('hierarchy_graph.db')
    """

    def __init__(
        self,
        learner: Any,  # HierarchicalConceptLearner
        config: Optional[CollectorConfig] = None,
        checkpoint_interval: int = 1000,
        verbose: bool = False
    ):
        """
        Initialize collector.

        Args:
            learner: HierarchicalConceptLearner instance
            config: Optional collector configuration
            checkpoint_interval: Samples between checkpoints
            verbose: Print debug information
        """
        self.learner = learner
        self.config = config or CollectorConfig()
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose

        # In-memory buffers (flush to storage periodically)
        self._patterns: Dict[str, PatternNode] = {}  # pattern_id → PatternNode
        self._edges_buffer: List[tuple] = []         # (parent, child, position, weight)
        self._pattern_frequencies: Dict[str, int] = defaultdict(int)

        # Node level metadata
        self._levels = [f"node{i}" for i in range(learner.num_nodes)]

        # Training dynamics tracker
        self.dynamics = TrainingDynamicsTracker()

        # Samples counter
        self._samples_processed = 0
        self._last_checkpoint = 0

        # Reusability tracking
        self._parent_counts: Dict[str, Set[str]] = defaultdict(set)  # child → {parents}

        if self.verbose:
            print(f"✓ HierarchyMetricsCollector initialized")
            print(f"  Levels: {self._levels}")
            print(f"  Checkpoint interval: {checkpoint_interval}")

    def record_pattern(
        self,
        pattern_id: str,
        node_level: str,
        frequency: int = 1,
        metadata: Optional[Dict] = None
    ):
        """
        Record a pattern (called during training).

        Args:
            pattern_id: Unique pattern identifier
            node_level: Level (node0, node1, etc.)
            frequency: Pattern frequency
            metadata: Optional metadata
        """
        if not self.config.collect_pattern_relationships:
            return

        if pattern_id not in self._patterns:
            self._patterns[pattern_id] = PatternNode(
                pattern_id=pattern_id,
                node_level=node_level,
                frequency=frequency,
                created_at=time.time(),
                parent_ids=[],
                child_ids=[],
                metadata=metadata if self.config.collect_metadata else None
            )
        else:
            # Update frequency
            self._patterns[pattern_id].frequency += frequency

        self._pattern_frequencies[pattern_id] += frequency

    def record_composition(
        self,
        parent_id: str,
        child_ids: List[str],
        parent_level: str,
        child_level: str
    ):
        """
        Record that parent pattern is composed of child patterns.

        Args:
            parent_id: Parent pattern ID
            child_ids: List of child pattern IDs (in sequence order)
            parent_level: Parent level
            child_level: Child level
        """
        if not self.config.collect_pattern_relationships:
            return

        # Record edges with positions
        for position, child_id in enumerate(child_ids):
            self._edges_buffer.append((parent_id, child_id, position, 1.0))

            # Track for reusability metrics
            self._parent_counts[child_id].add(parent_id)

            # Update pattern nodes
            if parent_id in self._patterns:
                if child_id not in self._patterns[parent_id].child_ids:
                    self._patterns[parent_id].child_ids.append(child_id)

            if child_id in self._patterns:
                if parent_id not in self._patterns[child_id].parent_ids:
                    self._patterns[child_id].parent_ids.append(parent_id)

    def record_sample_processed(self):
        """
        Increment samples counter and trigger checkpoint if needed.
        """
        self._samples_processed += 1

        # Check if checkpoint needed
        if self.config.enable_checkpoints and \
           self._samples_processed - self._last_checkpoint >= self.checkpoint_interval:
            self.create_checkpoint()
            self._last_checkpoint = self._samples_processed

    def create_checkpoint(self):
        """
        Create a checkpoint snapshot of current state.
        """
        # Compute pattern counts by level
        pattern_counts = defaultdict(int)
        for pattern in self._patterns.values():
            pattern_counts[pattern.node_level] += 1

        # Compute reusability stats
        reusability_stats = {}
        for level in self._levels:
            level_patterns = [p for p in self._patterns.values() if p.node_level == level]
            if level_patterns:
                parent_counts = [len(self._parent_counts.get(p.pattern_id, set())) for p in level_patterns]
                reusability_stats[level] = {
                    'mean_parents': sum(parent_counts) / len(parent_counts) if parent_counts else 0,
                    'total_patterns': len(level_patterns),
                }

        self.dynamics.add_checkpoint(
            samples=self._samples_processed,
            pattern_counts=dict(pattern_counts),
            reusability_stats=reusability_stats
        )

        if self.verbose:
            print(f"\n✓ Checkpoint at {self._samples_processed:,} samples")
            for level, count in sorted(pattern_counts.items()):
                print(f"  {level}: {count:,} patterns")

    def save(
        self,
        graph_db_path: str,
        dynamics_path: Optional[str] = None
    ):
        """
        Save collected data to persistent storage.

        Args:
            graph_db_path: Path to SQLite database
            dynamics_path: Optional path for dynamics JSONL file
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("SAVING HIERARCHY METRICS DATA")
            print(f"{'='*80}\n")

        # Initialize storage
        storage = HierarchyGraphStorage(graph_db_path, verbose=self.verbose)

        # Save patterns in batches
        if self._patterns:
            pattern_list = list(self._patterns.values())
            batch_size = self.config.batch_size

            for i in range(0, len(pattern_list), batch_size):
                batch = pattern_list[i:i+batch_size]
                storage.add_patterns_batch(batch)

            if self.verbose:
                print(f"✓ Saved {len(pattern_list):,} patterns")

        # Save edges in batches
        if self._edges_buffer:
            batch_size = self.config.batch_size

            for i in range(0, len(self._edges_buffer), batch_size):
                batch = self._edges_buffer[i:i+batch_size]
                storage.add_edges_batch(batch)

            if self.verbose:
                print(f"✓ Saved {len(self._edges_buffer):,} edges")

        # Save checkpoints
        if self.dynamics.checkpoints:
            for checkpoint in self.dynamics.checkpoints:
                storage.add_checkpoint(
                    samples_processed=checkpoint['samples'],
                    pattern_counts=checkpoint['pattern_counts'],
                    metrics_snapshot=checkpoint
                )

            if self.verbose:
                print(f"✓ Saved {len(self.dynamics.checkpoints)} checkpoints")

        # Save metadata
        storage.set_metadata('collector_config', self.config.__dict__)
        storage.set_metadata('num_nodes', self.learner.num_nodes)
        storage.set_metadata('node_levels', self._levels)
        storage.set_metadata('samples_processed', self._samples_processed)
        storage.set_metadata('collection_timestamp', time.time())

        # Save chunk sizes if available
        if hasattr(self.learner, 'node_configs'):
            chunk_sizes = [config.chunk_size for config in self.learner.node_configs]
            storage.set_metadata('chunk_sizes', chunk_sizes)

        # Get statistics
        stats = storage.get_statistics()

        if self.verbose:
            print(f"\n{'='*80}")
            print("STORAGE STATISTICS")
            print(f"{'='*80}")
            print(f"\n  Total patterns: {stats['total_patterns']:,}")
            print(f"  Total edges: {stats['total_edges']:,}")
            print(f"  Database size: {stats['database_size_mb']:.2f} MB")
            print(f"\n  Patterns by level:")
            for level, count in sorted(stats['pattern_counts_by_level'].items()):
                print(f"    {level}: {count:,}")
            print(f"\n{'='*80}\n")

        # Optionally save dynamics to JSONL
        if dynamics_path and self.dynamics.checkpoints:
            dynamics_file = Path(dynamics_path)
            with open(dynamics_file, 'w') as f:
                for checkpoint in self.dynamics.checkpoints:
                    f.write(json.dumps(checkpoint) + '\n')

            if self.verbose:
                print(f"✓ Saved training dynamics to {dynamics_path}")

        storage.close()

        if self.verbose:
            print(f"✓ All data saved successfully")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current collector statistics"""
        pattern_counts = defaultdict(int)
        for pattern in self._patterns.values():
            pattern_counts[pattern.node_level] += 1

        # Compute reusability
        reusability_by_level = {}
        for level in self._levels:
            level_patterns = [p for p in self._patterns.values() if p.node_level == level]
            if level_patterns:
                parent_counts = [len(self._parent_counts.get(p.pattern_id, set())) for p in level_patterns]
                orphan_count = sum(1 for pc in parent_counts if pc == 0)

                reusability_by_level[level] = {
                    'total_patterns': len(level_patterns),
                    'mean_parents': sum(parent_counts) / len(parent_counts) if parent_counts else 0,
                    'orphan_count': orphan_count,
                    'orphan_rate': orphan_count / len(level_patterns) if level_patterns else 0,
                }

        return {
            'samples_processed': self._samples_processed,
            'total_patterns': len(self._patterns),
            'total_edges': len(self._edges_buffer),
            'pattern_counts': dict(pattern_counts),
            'reusability': reusability_by_level,
            'checkpoints': len(self.dynamics.checkpoints),
        }

    def print_summary(self):
        """Print collection summary"""
        stats = self.get_statistics()

        print(f"\n{'='*80}")
        print("HIERARCHY METRICS COLLECTOR SUMMARY")
        print(f"{'='*80}\n")
        print(f"Samples processed: {stats['samples_processed']:,}")
        print(f"Total patterns: {stats['total_patterns']:,}")
        print(f"Total edges: {stats['total_edges']:,}")
        print(f"Checkpoints: {stats['checkpoints']}")

        print(f"\nPatterns by level:")
        for level, count in sorted(stats['pattern_counts'].items()):
            print(f"  {level}: {count:,}")

        if stats['reusability']:
            print(f"\nReusability:")
            for level, reuse_stats in sorted(stats['reusability'].items()):
                print(f"  {level}:")
                print(f"    Mean parents: {reuse_stats['mean_parents']:.2f}")
                print(f"    Orphan rate: {reuse_stats['orphan_rate']:.1%}")

        print(f"\n{'='*80}\n")
