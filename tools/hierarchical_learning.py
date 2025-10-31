"""
KATO Hierarchical Learning - Single-Pass Training with Structural Abstraction

This module provides a complete framework for hierarchical KATO learning with N nodes,
designed as an importable module for Jupyter notebooks and Python scripts.

Architecture:
- N KATO Nodes (default 4, configurable to any depth)
- Manual learning mode (max_pattern_length=0, stm_mode=CLEAR)
- Single-pass hierarchical training where pattern names flow up immediately
- Structural boundaries trigger learning at each level

Learning Flow (Single-Pass):
1. Text → CorpusSegmenter → hierarchical structure (book/chapter/paragraph/sentence)
2. node0 learns each sentence → pattern_name → send to node1's STM
3. When paragraph complete → node1 learns → pattern_name → send to node2's STM
4. When chapter complete → node2 learns → pattern_name → send to node3's STM
5. When book complete → node3 learns → pattern_name

Key Features:
- Single-pass hierarchical training (train_hierarchical_single_pass)
- MongoDB analysis and cleanup (MongoDBAnalyzer, cleanup_all_nodes, analyze_all_nodes)
- Modeling functions for prediction transfer (transfer_threshold, transfer_top_n, etc.)
- Delimiter-based streaming for flat learning (learn_from_stream)
- Support for arbitrary node depth (5, 10, 20+ nodes)
- Comprehensive pattern frequency analysis and visualization
- Configurable tokenizers (GPT-2, BERT, RoBERTa, T5, LLaMA, etc.)

Module Exports for Jupyter:
    from kato_hierarchical_streaming import (
        HierarchicalConceptLearner,
        CorpusSegmenter,
        MongoDBAnalyzer,
        train_hierarchical_single_pass,
        cleanup_all_nodes,
        analyze_all_nodes,
        transfer_threshold,
        transfer_top_n,
        transfer_weighted,
        transfer_predictions,
    )

Example Usage:
    # Segment text
    segmenter = CorpusSegmenter()
    corpus = segmenter.segment_book(text, metadata={'title': 'My Book'})

    # Create learner
    learner = HierarchicalConceptLearner(num_nodes=4, tokenizer_name="gpt2")

    # Train single-pass (pattern names flow up hierarchy)
    stats = train_hierarchical_single_pass(corpus, learner, num_levels=4)

    # Analyze results
    all_stats = analyze_all_nodes(learner)

    # Cleanup low-frequency patterns
    deleted = cleanup_all_nodes(learner, threshold=3)

    # Visualize frequency distribution
    analyzer = MongoDBAnalyzer(learner.nodes['node0'])
    analyzer.visualize_frequency_distribution(max_freq=50)
"""

import json
import time
import numpy as np
import re
import pickle
import os
from typing import List, Dict, Any, Optional, Iterator, Tuple
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Import tools from same package
from tools.kato_client import KATOClient
from tools.streaming_dataset_loader import StreamingDatasetLoader, recommend_dataset_configuration

# Module exports for Jupyter notebook usage
__all__ = [
    # Core classes
    'HierarchicalConceptLearner',
    'CorpusSegmenter',
    'TokenProcessor',
    'TokenDecoder',
    'MongoDBAnalyzer',
    'LearningTracker',
    'TrainingCheckpoint',

    # Main training function
    'train_hierarchical_single_pass',

    # Analysis & cleanup functions
    'cleanup_all_nodes',
    'analyze_all_nodes',

    # Modeling functions (for transfer_predictions)
    'transfer_all_names',
    'transfer_threshold',
    'transfer_top_n',
    'transfer_weighted',

    # Utility functions
    'transfer_predictions',
]

# ============================================================================
# MONGODB ANALYSIS & CLEANUP
# ============================================================================

class MongoDBAnalyzer:
    """
    Direct MongoDB access for pattern analysis and cleanup.

    This class provides direct access to the MongoDB knowledge base used by a KATO node,
    enabling frequency analysis, pattern statistics, and cleanup operations.
    """

    def __init__(self, node, mongo_uri: str = "mongodb://localhost:27017/", timeout_ms: int = 30000):
        """
        Initialize analyzer with node's database name for pattern access.

        Args:
            node: KATOClient instance to analyze
            mongo_uri: MongoDB connection URI
            timeout_ms: Server selection timeout in milliseconds (default: 30000)
        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("pymongo is required for MongoDB analysis. Install with: pip install pymongo")

        self.node = node
        # KATO creates databases with pattern: {node_id}_kato
        # where node_id comes from KATOClient initialization
        self.db_name = f"{node.node_id}_kato"
        self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=timeout_ms)
        self.db = self.client[self.db_name]
        self.patterns_collection = self.db['patterns_kb']

    def get_frequency_histogram(self) -> Dict[int, int]:
        """
        Get histogram of patterns by frequency.

        Returns:
            Dict mapping frequency → count of patterns with that frequency
            Example: {1: 500, 2: 200, 3: 100, 5: 50, 10: 20}
        """
        pipeline = [
            {'$group': {
                '_id': '$frequency',
                'count': {'$sum': 1}
            }},
            {'$sort': {'_id': 1}}
        ]

        results = self.patterns_collection.aggregate(pipeline)
        histogram = {doc['_id']: doc['count'] for doc in results}
        return histogram

    def get_patterns_by_frequency(self, min_freq: int = 1, max_freq: int = None) -> List[Dict]:
        """
        Get patterns within frequency range.

        Args:
            min_freq: Minimum frequency (inclusive)
            max_freq: Maximum frequency (inclusive), None = no upper limit

        Returns:
            List of pattern dicts with name, frequency, pattern_data, length
        """
        query = {'frequency': {'$gte': min_freq}}
        if max_freq is not None:
            query['frequency']['$lte'] = max_freq

        patterns = list(self.patterns_collection.find(
            query,
            {'name': 1, 'frequency': 1, 'pattern_data': 1, '_id': 0}
        ))

        # Add length field
        for pattern in patterns:
            pattern['length'] = len(pattern.get('pattern_data', []))

        return patterns

    def delete_patterns_below_threshold(self, threshold: int) -> int:
        """
        Delete all patterns with frequency < threshold.

        Args:
            threshold: Minimum frequency to keep (patterns with freq < threshold are deleted)

        Returns:
            Number of patterns deleted
        """
        result = self.patterns_collection.delete_many({'frequency': {'$lt': threshold}})
        return result.deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the knowledge base.

        Returns:
            Dict with: total_patterns, avg_frequency, max_frequency, min_frequency,
                      median_frequency, patterns_by_frequency_range
        """
        # Get all frequencies
        frequencies = list(self.patterns_collection.aggregate([
            {'$project': {'frequency': 1}},
            {'$group': {
                '_id': None,
                'total': {'$sum': 1},
                'avg': {'$avg': '$frequency'},
                'max': {'$max': '$frequency'},
                'min': {'$min': '$frequency'}
            }}
        ]))

        if not frequencies:
            return {
                'total_patterns': 0,
                'avg_frequency': 0,
                'max_frequency': 0,
                'min_frequency': 0,
                'median_frequency': 0
            }

        stats = frequencies[0]

        # Get frequency ranges
        freq_ranges = {
            'freq_1': self.patterns_collection.count_documents({'frequency': 1}),
            'freq_2_5': self.patterns_collection.count_documents({'frequency': {'$gte': 2, '$lte': 5}}),
            'freq_6_10': self.patterns_collection.count_documents({'frequency': {'$gte': 6, '$lte': 10}}),
            'freq_11_50': self.patterns_collection.count_documents({'frequency': {'$gte': 11, '$lte': 50}}),
            'freq_50_plus': self.patterns_collection.count_documents({'frequency': {'$gt': 50}})
        }

        return {
            'total_patterns': stats['total'],
            'avg_frequency': round(stats['avg'], 2),
            'max_frequency': stats['max'],
            'min_frequency': stats['min'],
            'frequency_ranges': freq_ranges
        }

    def visualize_frequency_distribution(self, max_freq: int = None, use_log_scale: bool = True):
        """
        Display histogram of pattern frequencies using matplotlib.

        Args:
            max_freq: Maximum frequency to display (None = show all)
            use_log_scale: Use logarithmic y-axis scale (recommended for large ranges)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
        except ImportError:
            print("matplotlib is required for visualization. Install with: pip install matplotlib")
            return

        histogram = self.get_frequency_histogram()

        if not histogram:
            print("No patterns found")
            return

        # Auto-detect max_freq if not specified
        actual_max_freq = max(histogram.keys())
        if max_freq is None:
            max_freq = actual_max_freq
            show_all = True
        else:
            show_all = (max_freq >= actual_max_freq)

        # Filter data
        filtered = {k: v for k, v in histogram.items() if k <= max_freq}
        frequencies = sorted(filtered.keys())
        counts = [filtered[f] for f in frequencies]

        # Print summary
        print(f"Frequency range: {min(frequencies)} to {max(frequencies)}")
        print(f"Pattern count range: {min(counts):,} to {max(counts):,}")

        # Show excluded data summary if any
        if not show_all:
            excluded = {k: v for k, v in histogram.items() if k > max_freq}
            if excluded:
                print(f"⚠️  {len(excluded)} frequency values > {max_freq} not shown")
                highest_excluded = max(excluded.keys())
                print(f"   Highest excluded: freq={highest_excluded}, count={excluded[highest_excluded]:,}")

        # Create plot
        plt.figure(figsize=(14, 7))
        plt.bar(frequencies, counts, width=0.8, edgecolor='black', linewidth=0.5)

        # Force integer x-axis ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

        # Use log scale for y-axis if there's a huge range
        if use_log_scale and len(counts) > 1 and max(counts) / min(counts) > 100:
            plt.yscale('log')
            plt.ylabel('Number of Patterns (log scale)', fontsize=12)
        else:
            plt.ylabel('Number of Patterns', fontsize=12)

        plt.xlabel('Pattern Frequency', fontsize=12)

        # Title
        node_id = self.node.node_id if hasattr(self, 'node') else 'Unknown'
        if show_all:
            title = f'{node_id} - Complete Frequency Distribution'
        else:
            title = f'{node_id} - Frequency Distribution (up to {max_freq})'
        plt.title(title, fontsize=14, fontweight='bold')

        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    def close(self):
        """Close MongoDB connection."""
        self.client.close()


class StandaloneMongoDBAnalyzer:
    """
    Session-independent MongoDB analyzer for post-training analysis.

    Unlike MongoDBAnalyzer, this class doesn't require active KATOClient sessions.
    Initialize with just the database name to analyze any trained model, even after
    kernel restarts or session expiration.

    Perfect for:
    - Post-training analysis without retraining
    - Analysis-only notebooks separate from training
    - Debugging/exploration after bugs fixed
    - Working with historical training runs

    Example:
        # Analyze without needing active sessions!
        analyzer = StandaloneMongoDBAnalyzer("node0_kato")
        stats = analyzer.get_stats()
        analyzer.visualize_frequency_distribution()
        analyzer.close()
    """

    def __init__(self, db_name: str, mongo_uri: str = "mongodb://localhost:27017/", timeout_ms: int = 30000):
        """
        Initialize analyzer with database name (no KATOClient required).

        Args:
            db_name: MongoDB database name (e.g., "node0_kato")
            mongo_uri: MongoDB connection URI
            timeout_ms: Server selection timeout in milliseconds (default: 30000)
        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("pymongo is required for MongoDB analysis. Install with: pip install pymongo")

        self.db_name = db_name
        self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=timeout_ms)
        self.db = self.client[self.db_name]
        self.patterns_collection = self.db['patterns_kb']
        self.symbols_collection = self.db['symbols_kb']
        self.metadata_collection = self.db['metadata']

    def get_frequency_histogram(self) -> Dict[int, int]:
        """Get histogram of patterns by frequency."""
        pipeline = [
            {'$group': {
                '_id': '$frequency',
                'count': {'$sum': 1}
            }},
            {'$sort': {'_id': 1}}
        ]
        results = self.patterns_collection.aggregate(pipeline)
        histogram = {doc['_id']: doc['count'] for doc in results}
        return histogram

    def get_patterns_by_frequency(self, min_freq: int = 1, max_freq: int = None) -> List[Dict]:
        """Get patterns within frequency range."""
        query = {'frequency': {'$gte': min_freq}}
        if max_freq is not None:
            query['frequency']['$lte'] = max_freq

        patterns = list(self.patterns_collection.find(
            query,
            {'name': 1, 'frequency': 1, 'pattern_data': 1, '_id': 0}
        ))

        for pattern in patterns:
            pattern['length'] = len(pattern.get('pattern_data', []))

        return patterns

    def delete_patterns_below_threshold(self, threshold: int) -> int:
        """Delete all patterns with frequency < threshold."""
        result = self.patterns_collection.delete_many({'frequency': {'$lt': threshold}})
        return result.deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base."""
        frequencies = list(self.patterns_collection.aggregate([
            {'$project': {'frequency': 1}},
            {'$group': {
                '_id': None,
                'total': {'$sum': 1},
                'avg': {'$avg': '$frequency'},
                'max': {'$max': '$frequency'},
                'min': {'$min': '$frequency'}
            }}
        ]))

        if not frequencies:
            return {
                'total_patterns': 0,
                'avg_frequency': 0,
                'max_frequency': 0,
                'min_frequency': 0,
                'median_frequency': 0,
                'total_frequency': 0
            }

        stats = frequencies[0]

        # Get frequency ranges
        freq_ranges = {
            'freq_1': self.patterns_collection.count_documents({'frequency': 1}),
            'freq_2_5': self.patterns_collection.count_documents({'frequency': {'$gte': 2, '$lte': 5}}),
            'freq_6_10': self.patterns_collection.count_documents({'frequency': {'$gte': 6, '$lte': 10}}),
            'freq_11_50': self.patterns_collection.count_documents({'frequency': {'$gte': 11, '$lte': 50}}),
            'freq_50_plus': self.patterns_collection.count_documents({'frequency': {'$gt': 50}})
        }

        # Get total frequency sum from metadata if available
        total_freq = 0
        try:
            metadata = self.metadata_collection.find_one({'class': 'totals'})
            if metadata:
                total_freq = metadata.get('total_pattern_frequencies', 0)
        except Exception:
            pass  # Metadata collection might not exist or be accessible

        return {
            'total_patterns': stats['total'],
            'avg_frequency': round(stats['avg'], 2) if stats['avg'] else 0,
            'max_frequency': stats['max'] if stats['max'] else 0,
            'min_frequency': stats['min'] if stats['min'] else 0,
            'frequency_ranges': freq_ranges,
            'total_frequency': total_freq,
            'average_frequency': round(stats['avg'], 2) if stats['avg'] else 0
        }

    def visualize_frequency_distribution(self, max_freq: int = None, use_log_scale: bool = True):
        """
        Display histogram of pattern frequencies.

        Args:
            max_freq: Maximum frequency to display (None = show all)
            use_log_scale: Use logarithmic y-axis scale (recommended for large ranges)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
        except ImportError:
            print("matplotlib is required for visualization. Install with: pip install matplotlib")
            return

        histogram = self.get_frequency_histogram()

        if not histogram:
            print("No patterns found")
            return

        # Auto-detect max_freq if not specified
        actual_max_freq = max(histogram.keys())
        if max_freq is None:
            max_freq = actual_max_freq
            show_all = True
        else:
            show_all = (max_freq >= actual_max_freq)

        # Filter data
        filtered = {k: v for k, v in histogram.items() if k <= max_freq}
        frequencies = sorted(filtered.keys())
        counts = [filtered[f] for f in frequencies]

        # Print summary
        print(f"Frequency range: {min(frequencies)} to {max(frequencies)}")
        print(f"Pattern count range: {min(counts):,} to {max(counts):,}")

        # Show excluded data summary if any
        if not show_all:
            excluded = {k: v for k, v in histogram.items() if k > max_freq}
            if excluded:
                print(f"⚠️  {len(excluded)} frequency values > {max_freq} not shown")
                highest_excluded = max(excluded.keys())
                print(f"   Highest excluded: freq={highest_excluded}, count={excluded[highest_excluded]:,}")

        # Create plot
        plt.figure(figsize=(14, 7))
        plt.bar(frequencies, counts, width=0.8, edgecolor='black', linewidth=0.5)

        # Force integer x-axis ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

        # Use log scale for y-axis if there's a huge range
        if use_log_scale and len(counts) > 1 and max(counts) / min(counts) > 100:
            plt.yscale('log')
            plt.ylabel('Number of Patterns (log scale)', fontsize=12)
        else:
            plt.ylabel('Number of Patterns', fontsize=12)

        plt.xlabel('Pattern Frequency', fontsize=12)

        # Title
        if show_all:
            title = f'{self.db_name} - Complete Frequency Distribution'
        else:
            title = f'{self.db_name} - Frequency Distribution (up to {max_freq})'
        plt.title(title, fontsize=14, fontweight='bold')

        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    def close(self):
        """Close MongoDB connection."""
        self.client.close()


# ============================================================================
# TRAINING MANIFEST SYSTEM - Session-Independent Metadata
# ============================================================================

class TrainingManifest:
    """
    Training metadata for session-independent analysis.

    Saves essential information about training runs so you can analyze them later
    without needing to maintain active KATOClient sessions or retrain from scratch.

    Storage: JSON files in manifests/ directory + optional MongoDB collection

    Example:
        # Auto-saved after training
        manifest = TrainingManifest.create_from_learner(learner, dataset='wikitext', samples=100)
        manifest.save('manifests/training_20250118.json')

        # Load later for analysis
        manifest = TrainingManifest.load('manifests/training_20250118.json')
        analyzers = manifest.get_analyzers()  # Dict of StandaloneMongoDBAnalyzer instances
    """

    def __init__(self,
                 training_id: str,
                 timestamp: str,
                 nodes: Dict[str, Dict[str, Any]],
                 dataset: str = None,
                 samples_trained: int = None,
                 tokenizer: str = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize training manifest.

        Args:
            training_id: Unique identifier for this training run
            timestamp: ISO format timestamp
            nodes: Dict mapping node_name → {db_name, chunk_size, mode, ...}
            dataset: Dataset name used for training
            samples_trained: Number of samples processed
            tokenizer: Tokenizer name used
            metadata: Additional custom metadata
        """
        self.training_id = training_id
        self.timestamp = timestamp
        self.nodes = nodes
        self.dataset = dataset
        self.samples_trained = samples_trained
        self.tokenizer = tokenizer
        self.metadata = metadata or {}

    @classmethod
    def create_from_learner(cls,
                           learner: 'HierarchicalConceptLearner',
                           dataset: str = None,
                           samples_trained: int = None,
                           training_id: str = None) -> 'TrainingManifest':
        """
        Create manifest from HierarchicalConceptLearner instance.

        Args:
            learner: Trained HierarchicalConceptLearner
            dataset: Dataset name
            samples_trained: Number of samples processed
            training_id: Optional custom ID (auto-generated if None)

        Returns:
            TrainingManifest instance
        """
        import datetime

        # Generate training ID if not provided
        if training_id is None:
            timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            training_id = f"training_{timestamp_str}"

        # Extract node information
        nodes = {}
        for node_name, node in learner.nodes.items():
            node_config = {
                'db_name': f"{node.node_id}_kato",
                'node_id': node.node_id
            }
            # Try to get configuration from learner's node_configs
            for config in learner.node_configs:
                if config.name == node_name:
                    node_config['chunk_size'] = config.chunk_size
                    node_config['mode'] = config.mode
                    break
            nodes[node_name] = node_config

        return cls(
            training_id=training_id,
            timestamp=datetime.datetime.now().isoformat(),
            nodes=nodes,
            dataset=dataset,
            samples_trained=samples_trained,
            tokenizer=learner.token_processor.tokenizer_name if hasattr(learner, 'token_processor') else None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for serialization."""
        return {
            'training_id': self.training_id,
            'timestamp': self.timestamp,
            'nodes': self.nodes,
            'dataset': self.dataset,
            'samples_trained': self.samples_trained,
            'tokenizer': self.tokenizer,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingManifest':
        """Create manifest from dictionary."""
        return cls(
            training_id=data['training_id'],
            timestamp=data['timestamp'],
            nodes=data['nodes'],
            dataset=data.get('dataset'),
            samples_trained=data.get('samples_trained'),
            tokenizer=data.get('tokenizer'),
            metadata=data.get('metadata', {})
        )

    def save(self, filepath: str):
        """
        Save manifest to JSON file.

        Args:
            filepath: Path to save manifest (e.g., 'manifests/training_123.json')
        """
        import json
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'TrainingManifest':
        """
        Load manifest from JSON file.

        Args:
            filepath: Path to manifest file

        Returns:
            TrainingManifest instance
        """
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_analyzers(self,
                     mongo_uri: str = "mongodb://localhost:27017/",
                     timeout_ms: int = 2000) -> Dict[str, StandaloneMongoDBAnalyzer]:
        """
        Get StandaloneMongoDBAnalyzer instances for all nodes.

        Args:
            mongo_uri: MongoDB connection URI
            timeout_ms: Connection timeout

        Returns:
            Dict mapping node_name → StandaloneMongoDBAnalyzer instance

        Example:
            manifest = TrainingManifest.load('manifests/training_123.json')
            analyzers = manifest.get_analyzers()
            stats = analyzers['node0'].get_stats()
        """
        analyzers = {}
        for node_name, node_info in self.nodes.items():
            analyzers[node_name] = StandaloneMongoDBAnalyzer(
                db_name=node_info['db_name'],
                mongo_uri=mongo_uri,
                timeout_ms=timeout_ms
            )
        return analyzers

    def __repr__(self):
        return (f"TrainingManifest(id={self.training_id}, "
                f"nodes={len(self.nodes)}, dataset={self.dataset}, "
                f"samples={self.samples_trained})")


def discover_training_databases(mongo_uri: str = "mongodb://localhost:27017/",
                                timeout_ms: int = 2000) -> List[str]:
    """
    Discover all KATO training databases in MongoDB.

    Scans MongoDB for databases matching the pattern *_kato and returns their names.

    Args:
        mongo_uri: MongoDB connection URI
        timeout_ms: Connection timeout

    Returns:
        List of database names (empty list if connection fails)

    Raises:
        ConnectionError: If MongoDB is not accessible with helpful diagnostics

    Example:
        >>> discover_training_databases()
        ['node0_kato', 'node1_kato', 'node2_kato', 'node3_kato']

        # Docker environment
        >>> discover_training_databases(mongo_uri="mongodb://mongo:27017/")
    """
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=timeout_ms)
        all_dbs = client.list_database_names()
        training_dbs = [db for db in all_dbs if db.endswith('_kato')]
        client.close()
        return sorted(training_dbs)
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        # Provide helpful error message
        error_msg = f"""
MongoDB connection failed: {mongo_uri}

Common solutions:
1. Check if MongoDB is running:
   - Docker: docker ps | grep mongo
   - Local: pgrep mongod

2. If running in Docker, try:
   - mongo_uri="mongodb://mongo:27017/"  (Docker Compose service name)
   - mongo_uri="mongodb://host.docker.internal:27017/"  (Docker Desktop)

3. If MongoDB is on a different port:
   - Check with: lsof -i :27017
   - Update mongo_uri accordingly

4. Verify network connectivity:
   - Ping the host
   - Check firewall rules

Original error: {str(e)}
"""
        raise ConnectionError(error_msg) from e


def list_available_manifests(manifests_dir: str = 'manifests') -> List[str]:
    """
    List all available training manifest files.

    Args:
        manifests_dir: Directory containing manifest files

    Returns:
        List of manifest file paths

    Example:
        >>> list_available_manifests()
        ['manifests/training_20250118_123456.json', 'manifests/training_20250118_234567.json']
    """
    import os
    import glob

    if not os.path.exists(manifests_dir):
        return []

    manifest_files = glob.glob(os.path.join(manifests_dir, '*.json'))
    return sorted(manifest_files)


def load_latest_manifest(manifests_dir: str = 'manifests') -> TrainingManifest:
    """
    Load the most recent training manifest.

    Args:
        manifests_dir: Directory containing manifest files

    Returns:
        TrainingManifest instance

    Raises:
        FileNotFoundError: If no manifests found

    Example:
        >>> manifest = load_latest_manifest()
        >>> analyzers = manifest.get_analyzers()
        >>> analyzers['node0'].get_stats()
    """
    manifests = list_available_manifests(manifests_dir)
    if not manifests:
        raise FileNotFoundError(f"No training manifests found in {manifests_dir}/")

    return TrainingManifest.load(manifests[-1])


def create_training_run_nodes(
    run_id: str = None,
    num_nodes: int = 4,
    chunk_size: int = 8,
    mode: str = 'chunking',
    base_url: str = 'http://kato:8000'
) -> List['HierarchicalNode']:
    """
    Create nodes with unique IDs for a training run to preserve separate databases.

    **IMPORTANT:** Using the same node IDs across training runs will cause database
    conflicts where new training overwrites or appends to previous data. This function
    ensures each training run gets its own unique databases for proper comparison.

    Args:
        run_id: Unique identifier for this training run (auto-generated if None)
                Examples: 'wikitext_100k', 'experiment_a', 'baseline'
        num_nodes: Number of hierarchical nodes (default: 4)
        chunk_size: Tokens per chunk for all nodes (default: 8)
        mode: Segmentation mode: 'chunking' or 'sentences' (default: 'chunking')
        base_url: KATO server URL (default: 'http://kato:8000')

    Returns:
        List of HierarchicalNode instances with unique node IDs

    Example:
        # Training Run 1: WikiText 100k samples
        nodes_run1 = create_training_run_nodes(run_id='wikitext_100k')
        # Creates databases: node0_wikitext_100k_kato, node1_wikitext_100k_kato, ...

        # Training Run 2: WikiText 500k samples
        nodes_run2 = create_training_run_nodes(run_id='wikitext_500k')
        # Creates databases: node0_wikitext_500k_kato, node1_wikitext_500k_kato, ...

        # Now you can compare both runs without data conflicts!

        # Without run_id (databases will be reused/overwritten):
        nodes = create_training_run_nodes()  # Auto-generates timestamp-based ID
    """
    import datetime

    # Auto-generate run_id if not provided
    if run_id is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"run_{timestamp}"

    # Create nodes with unique IDs
    nodes = []
    for i in range(num_nodes):
        node_id = f"node{i}_{run_id}"
        nodes.append(HierarchicalNode(
            name=f'node{i}',
            node_id=node_id,
            chunk_size=chunk_size,
            mode=mode,
            base_url=base_url
        ))

    return nodes


def list_all_training_runs(mongo_uri: str = "mongodb://localhost:27017/",
                           timeout_ms: int = 2000) -> Dict[str, List[str]]:
    """
    Group discovered databases by training run ID.

    Returns:
        Dict mapping run_id → list of database names

    Example:
        >>> runs = list_all_training_runs()
        >>> print(runs)
        {
            'wikitext_100k': ['node0_wikitext_100k_kato', 'node1_wikitext_100k_kato'],
            'wikitext_500k': ['node0_wikitext_500k_kato', 'node1_wikitext_500k_kato']
        }
    """
    from collections import defaultdict

    databases = discover_training_databases(mongo_uri, timeout_ms)

    # Group by run_id (everything after last underscore before _kato)
    runs = defaultdict(list)

    for db in databases:
        # Remove _kato suffix
        if db.endswith('_kato'):
            base = db[:-5]  # Remove '_kato'

            # Extract run_id: everything after the first underscore
            # e.g., "node0_wikitext_100k" -> run_id = "wikitext_100k"
            parts = base.split('_', 1)
            if len(parts) == 2:
                run_id = parts[1]
            else:
                run_id = 'unknown'

            runs[run_id].append(db)

    return dict(runs)


def delete_training_run(run_id: str,
                       mongo_uri: str = "mongodb://localhost:27017/",
                       timeout_ms: int = 2000,
                       confirm: bool = True) -> int:
    """
    Delete all databases associated with a training run.

    **CAUTION:** This permanently deletes data!

    Args:
        run_id: Training run identifier
        mongo_uri: MongoDB connection URI
        timeout_ms: Connection timeout
        confirm: Require confirmation (default: True)

    Returns:
        Number of databases deleted

    Example:
        >>> delete_training_run('old_experiment', confirm=False)
        Deleted 4 databases for run 'old_experiment'
    """
    from pymongo import MongoClient

    runs = list_all_training_runs(mongo_uri, timeout_ms)

    if run_id not in runs:
        print(f"Training run '{run_id}' not found")
        return 0

    databases_to_delete = runs[run_id]

    if confirm:
        print(f"About to delete {len(databases_to_delete)} databases:")
        for db in databases_to_delete:
            print(f"  - {db}")
        response = input("Type 'DELETE' to confirm: ")
        if response != 'DELETE':
            print("Cancelled")
            return 0

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=timeout_ms)

    deleted = 0
    for db_name in databases_to_delete:
        client.drop_database(db_name)
        deleted += 1
        print(f"Deleted: {db_name}")

    client.close()

    print(f"\n✓ Deleted {deleted} databases for run '{run_id}'")
    return deleted


# ============================================================================
# TOKENIZATION & DATA PROCESSING
# ============================================================================

class CorpusSegmenter:
    """
    Segment raw text into hierarchical structure: book → chapter → paragraph → chunk.

    This class provides methods to take raw text and break it down into the hierarchical
    structure required for hierarchical concept learning.

    Uses fixed-length token chunking for scalable pattern learning.
    """

    def __init__(
        self,
        token_processor: 'TokenProcessor' = None,
        tokenizer_name: str = "gpt2",
        mode: str = 'chunking',
        chunk_size: int = 15,
        min_sentence_tokens: int = 3
    ):
        """
        Initialize the corpus segmenter with a tokenizer.

        Args:
            token_processor: Optional TokenProcessor instance. If None, creates one with tokenizer_name.
            tokenizer_name: Name of tokenizer to use (only if token_processor is None)
            mode: Segmentation mode - 'chunking' or 'sentences'
                  'chunking': Fixed-length token chunks (scalable, better compression)
                  'sentences': Sentence boundary detection (semantic coherence)
            chunk_size: Number of tokens per chunk (default: 15, recommended: 10-25)
                       Only used if mode='chunking'
            min_sentence_tokens: Minimum tokens per sentence (default: 3)
                                Only used if mode='sentences'
        """
        if mode not in ['chunking', 'sentences']:
            raise ValueError(f"mode must be 'chunking' or 'sentences', got: {mode}")

        if token_processor is not None:
            self.token_processor = token_processor
        else:
            self.token_processor = TokenProcessor(tokenizer_name=tokenizer_name)

        self.mode = mode
        self.chunk_size = chunk_size
        self.min_sentence_tokens = min_sentence_tokens

    def segment_book(self, book_text: str, book_metadata: dict = None, chunk_size: int = None) -> dict:
        """
        Segment book text into hierarchical structure using fixed-length token chunking.

        Args:
            book_text: Raw text of the book
            book_metadata: Optional metadata (title, author, etc.)
            chunk_size: Override instance chunk_size (default: use self.chunk_size)

        Returns:
            Dictionary with hierarchical structure: {title, chapters: [{title, paragraphs: [...]}]}
            Paragraphs contain 'chunks' (lists of token sequences), not 'sentences'
        """
        if book_metadata is None:
            book_metadata = {}

        # Detect chapters using common patterns
        # Patterns: "Chapter 1", "CHAPTER I", "Chapter One", etc.
        chapter_pattern = r'\n\s*(?:Chapter|CHAPTER)\s+(?:\d+|[IVX]+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\s*[:\n]'
        chapter_splits = re.split(chapter_pattern, book_text)

        chapters = []

        # Process each chapter
        for i, chapter_text in enumerate(chapter_splits):
            if not chapter_text.strip():
                continue

            # Split into paragraphs (double newline or single newline for some formats)
            paragraph_texts = re.split(r'\n\s*\n', chapter_text)
            paragraphs = []

            for para_text in paragraph_texts:
                para_text = para_text.strip()
                if not para_text or len(para_text) < 10:  # Skip very short paragraphs
                    continue

                # Tokenize entire paragraph
                para_tokens = self.token_processor.tokenize_segment(para_text, max_tokens=2048)

                # Split tokens based on mode
                if self.mode == 'chunking':
                    effective_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
                    segments = self.token_processor.chunk_tokens(
                        para_tokens, chunk_size=effective_chunk_size, min_chunk_size=3
                    )
                elif self.mode == 'sentences':
                    segments = self.token_processor.split_tokens_into_sentences(
                        para_tokens, min_tokens=self.min_sentence_tokens
                    )
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

                if segments:
                    paragraphs.append({
                        'text': para_text,
                        'chunks': segments  # List of token segments (chunks or sentences)
                    })

            if paragraphs:
                chapters.append({
                    'title': f'Chapter {i}',
                    'paragraphs': paragraphs
                })

        return {
            'title': book_metadata.get('title', 'Untitled'),
            'author': book_metadata.get('author', 'Unknown'),
            'chapters': chapters
        }

    def segment_article(self, article_text: str, article_metadata: dict = None, chunk_size: int = None) -> dict:
        """
        Segment article text into hierarchical structure using fixed-length token chunking.

        Args:
            article_text: Raw text of the article
            article_metadata: Optional metadata (title, author, etc.)
            chunk_size: Override instance chunk_size (default: use self.chunk_size)

        Returns:
            Dictionary with hierarchical structure: {title, chapters: [{title, paragraphs: [...]}]}
            Paragraphs contain 'chunks' (lists of token sequences), not 'sentences'
        """
        if article_metadata is None:
            article_metadata = {}

        # Detect sections using markdown-style headers or numbered sections
        section_pattern = r'\n\s*(?:#+\s+|\d+\.\s+|Section\s+\d+[:\n])'
        section_splits = re.split(section_pattern, article_text)

        sections = []

        for i, section_text in enumerate(section_splits):
            if not section_text.strip():
                continue

            # Split into paragraphs
            paragraph_texts = re.split(r'\n\s*\n', section_text)
            paragraphs = []

            for para_text in paragraph_texts:
                para_text = para_text.strip()
                if not para_text or len(para_text) < 10:
                    continue

                # Tokenize entire paragraph
                para_tokens = self.token_processor.tokenize_segment(para_text, max_tokens=2048)

                # Split tokens based on mode
                if self.mode == 'chunking':
                    effective_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
                    segments = self.token_processor.chunk_tokens(
                        para_tokens, chunk_size=effective_chunk_size, min_chunk_size=3
                    )
                elif self.mode == 'sentences':
                    segments = self.token_processor.split_tokens_into_sentences(
                        para_tokens, min_tokens=self.min_sentence_tokens
                    )
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

                if segments:
                    paragraphs.append({
                        'text': para_text,
                        'chunks': segments  # List of token segments (chunks or sentences)
                    })

            if paragraphs:
                sections.append({
                    'title': f'Section {i}',
                    'paragraphs': paragraphs
                })

        return {
            'title': article_metadata.get('title', 'Untitled Article'),
            'author': article_metadata.get('author', 'Unknown'),
            'chapters': sections  # Use 'chapters' key for consistency with book structure
        }

    def segment_simple_text(self, text: str, metadata: dict = None, chunk_size: int = None) -> dict:
        """
        Segment simple text without chapter markers using fixed-length token chunking.

        Args:
            text: Raw text
            metadata: Optional metadata
            chunk_size: Override instance chunk_size (default: use self.chunk_size)

        Returns:
            Dictionary with single-chapter hierarchical structure
            Paragraphs contain 'chunks' (lists of token sequences), not 'sentences'
        """
        if metadata is None:
            metadata = {}

        # Split into paragraphs
        paragraph_texts = re.split(r'\n\s*\n', text)
        paragraphs = []

        for para_text in paragraph_texts:
            para_text = para_text.strip()
            if not para_text or len(para_text) < 10:
                continue

            # Tokenize entire paragraph
            para_tokens = self.token_processor.tokenize_segment(para_text, max_tokens=2048)

            # Split tokens based on mode
            if self.mode == 'chunking':
                effective_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
                segments = self.token_processor.chunk_tokens(
                    para_tokens, chunk_size=effective_chunk_size, min_chunk_size=3
                )
            elif self.mode == 'sentences':
                segments = self.token_processor.split_tokens_into_sentences(
                    para_tokens, min_tokens=self.min_sentence_tokens
                )
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            if segments:
                paragraphs.append({
                    'text': para_text,
                    'chunks': segments  # List of token segments (chunks or sentences)
                })

        return {
            'title': metadata.get('title', 'Untitled Text'),
            'chapters': [{
                'title': 'Main Content',
                'paragraphs': paragraphs
            }]
        }


class TokenProcessor:
    """
    Process text with delimiter-based segmentation and tokenization for KATO nodes.

    Supported tokenizers (via HuggingFace AutoTokenizer):
    - gpt2 (default): GPT-2 tokenizer using BPE
    - bert-base-uncased/cased: BERT WordPiece tokenizer
    - roberta-base: RoBERTa byte-level BPE
    - t5-small/base/large: T5 seq2seq tokenizer
    - albert-base-v2: ALBERT factorized embeddings
    - distilbert-base-uncased: Compressed BERT
    - xlnet-base-cased: XLNet permutation-based
    - electra-base-discriminator: ELECTRA discriminator
    - deberta-base/v3: DeBERTa disentangled attention
    - facebook/bart-base: BART denoising autoencoder
    - microsoft/phi-2: Phi-2 small language model
    - meta-llama/Llama-2-7b-hf: LLaMA 2 (requires auth)

    Supported delimiters:
    - 'sentence': Segment by sentences
    - 'word': Segment by words
    - 'bigram': Segment by 2-word sequences
    - 'trigram': Segment by 3-word sequences
    - '4-gram': Segment by 4-word sequences
    - '5-gram': Segment by 5-word sequences
    - 'paragraph': Segment by paragraphs
    """

    def __init__(self, tokenizer_name: str = "gpt2"):
        """Initialize with a tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            model_max_length=4096  # Support long paragraphs without warnings
        )
        self.tokenizer_name = tokenizer_name

    def segment_text_by_delimiter(self, text: str, delimiter: str) -> List[str]:
        """
        Segment text into units based on delimiter type.

        Args:
            text: Input text to segment
            delimiter: One of 'sentence', 'word', 'bigram', 'trigram', '4-gram', '5-gram', 'paragraph'

        Returns:
            List of text segments
        """
        if delimiter == 'sentence':
            return self._split_into_sentences(text)
        elif delimiter == 'word':
            return self._split_into_words(text)
        elif delimiter in ['bigram', 'trigram', '4-gram', '5-gram']:
            n = {'bigram': 2, 'trigram': 3, '4-gram': 4, '5-gram': 5}[delimiter]
            return self._extract_ngrams(text, n)
        elif delimiter == 'paragraph':
            return self._split_into_paragraphs(text)
        else:
            raise ValueError(f"Unknown delimiter: {delimiter}. Must be one of: sentence, word, bigram, trigram, 4-gram, 5-gram, paragraph")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            import nltk
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback: simple regex splitting
            sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words."""
        words = text.split()
        return [w.strip() for w in words if w.strip()]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text."""
        words = self._split_into_words(text)
        if len(words) < n:
            return []
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def tokenize_segment(self, segment: str, max_tokens: int = 512) -> List[str]:
        """
        Tokenize a single text segment using the configured tokenizer.

        Args:
            segment: Text segment to tokenize
            max_tokens: Maximum number of tokens to return

        Returns:
            List of token strings
        """
        tokens = self.tokenizer.tokenize(segment)[:max_tokens]
        return tokens

    def split_tokens_into_sentences(self, tokens: List[str], min_tokens: int = 3) -> List[List[str]]:
        """
        Split a token sequence into sentence-level token sequences using tokenizer-specific
        boundary detection logic.

        This method identifies sentence boundaries by looking for sentence-ending punctuation
        tokens (., !, ?) followed by indicators that a new sentence is starting. The logic
        is tailored to different tokenizer types:

        - GPT-2/RoBERTa: Look for space-prefixed tokens (Ġ) after punctuation
        - BERT: Look for non-continuation tokens (not ##) after punctuation
        - Other: Fallback heuristics

        Args:
            tokens: List of token strings from tokenizer
            min_tokens: Minimum tokens per sentence (filters out degenerate short sentences)

        Returns:
            List of sentence-token-sequences: [[tok1, tok2], [tok3, tok4, tok5], ...]
        """
        if not tokens:
            return []

        sentence_endings = ['.', '!', '?']
        sentences = []
        current_sentence = []

        # Detect tokenizer type from tokenizer name
        tokenizer_lower = self.tokenizer_name.lower()
        is_gpt2_like = any(x in tokenizer_lower for x in ['gpt2', 'gpt-2', 'roberta', 'bart', 'llama', 'phi'])
        is_bert_like = any(x in tokenizer_lower for x in ['bert', 'albert', 'electra', 'deberta'])

        for i, token in enumerate(tokens):
            current_sentence.append(token)

            # Check if this is a sentence-ending punctuation
            if token in sentence_endings:
                is_sentence_boundary = False

                # Check if there's a next token
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]

                    if is_gpt2_like:
                        # GPT-2/RoBERTa: Space-prefixed token indicates new sentence
                        # Ġ (U+0120) is GPT-2's space marker
                        if next_token.startswith('Ġ') or next_token.startswith('Â'):
                            is_sentence_boundary = True
                    elif is_bert_like:
                        # BERT: Non-continuation token (not ##) indicates word boundary
                        if not next_token.startswith('##'):
                            is_sentence_boundary = True
                    else:
                        # Fallback: Assume space or capital letter in next token
                        if next_token.startswith((' ', 'Ġ', 'Â')) or (len(next_token) > 0 and next_token[0].isupper()):
                            is_sentence_boundary = True
                else:
                    # End of token sequence - this is definitely a sentence boundary
                    is_sentence_boundary = True

                if is_sentence_boundary:
                    # Save current sentence if it meets minimum length
                    if len(current_sentence) >= min_tokens:
                        sentences.append(current_sentence)
                    current_sentence = []

        # Add any remaining tokens as final sentence
        if current_sentence and len(current_sentence) >= min_tokens:
            sentences.append(current_sentence)

        return sentences

    def chunk_tokens(
        self,
        tokens: List[str],
        chunk_size: int = 15,
        min_chunk_size: int = 3
    ) -> List[List[str]]:
        """
        Split tokens into fixed-length chunks for pattern learning.

        This method implements the chunking strategy for hierarchical learning:
        - Each chunk is exactly chunk_size tokens (except possibly the last)
        - Chunks repeat more than full sentences → better compression
        - No semantic awareness needed at node0 (emerges from hierarchy)
        - Hierarchical composition: node1 learns sequences of chunks

        Why chunking instead of sentences:
        1. Deduplication: Chunks repeat, sentences are unique
        2. Scalability: node0 KB size << corpus size (Zipfian distribution)
        3. Composition: 15 chunks @ node1 = 225 tokens (2-3 sentences worth)
        4. Robustness: No fragile boundary detection

        Args:
            tokens: List of token strings from tokenizer
            chunk_size: Target size for each chunk (default: 15)
                       Recommended range: 10-25 tokens
            min_chunk_size: Minimum chunk size (merge small remainder if below this)

        Returns:
            List of token chunks: [[tok1, tok2, ...], [tok16, tok17, ...], ...]

        Example:
            tokens = ["The", "cat", "sat", "on", "the", "mat", ...]  # 50 tokens
            chunk_size = 15
            → [[tok0-14], [tok15-29], [tok30-44], [tok45-49]]  # 4 chunks
        """
        if not tokens:
            return []

        if len(tokens) <= chunk_size:
            # Entire token list fits in one chunk
            return [tokens] if len(tokens) >= min_chunk_size else []

        chunks = []
        i = 0

        while i < len(tokens):
            # Take next chunk_size tokens
            chunk = tokens[i:i + chunk_size]

            # Check if this is the last chunk and it's too small
            remaining_after = len(tokens) - (i + chunk_size)

            if remaining_after > 0 and remaining_after < min_chunk_size:
                # Last chunk would be too small - include it in current chunk
                chunk = tokens[i:i + chunk_size + remaining_after]
                chunks.append(chunk)
                break
            else:
                chunks.append(chunk)
                i += chunk_size

        return chunks

    def process_with_delimiter(self, text: str, delimiter: str, max_tokens_per_segment: int = 512) -> List[List[str]]:
        """
        Complete pipeline: segment text by delimiter, then tokenize each segment.

        Args:
            text: Input text
            delimiter: Delimiter type
            max_tokens_per_segment: Max tokens per segment

        Returns:
            List of tokenized segments, where each segment is a list of tokens
        """
        segments = self.segment_text_by_delimiter(text, delimiter)
        tokenized_segments = []

        for segment in segments:
            tokens = self.tokenize_segment(segment, max_tokens=max_tokens_per_segment)
            if tokens:  # Only include non-empty
                tokenized_segments.append(tokens)

        return tokenized_segments


class TokenDecoder:
    """Decode tokenized sequences back into human-readable text."""

    def __init__(self, tokenizer_name: str = "gpt2"):
        """
        Initialize decoder with a tokenizer.

        Args:
            tokenizer_name: Name of the tokenizer (same options as AutoTokenizer)
                          e.g., "gpt2", "bert-base-uncased", "roberta-base", etc.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name
        print(f"✓ Loaded decoder with tokenizer: {tokenizer_name}")

    def decode_ids(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a sequence of token IDs back to text.

        Args:
            token_ids: List of token IDs (integers)
            skip_special_tokens: Whether to remove special tokens like [PAD], [CLS], etc.

        Returns:
            Decoded string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_tokens(self, tokens: List[str]) -> str:
        """
        Decode a sequence of token strings back to text.

        This is useful when you have tokenized strings (like from tokenize_for_level0)
        but need to convert them back to readable text.

        Args:
            tokens: List of token strings

        Returns:
            Decoded string
        """
        # Convert tokens to IDs first, then decode
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.decode_ids(token_ids, skip_special_tokens=True)

    def decode_batch(self, batch_token_ids: List[List[int]],
                     skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            batch_token_ids: List of token ID sequences
            skip_special_tokens: Whether to remove special tokens

        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(batch_token_ids,
                                          skip_special_tokens=skip_special_tokens)


# ============================================================================
# HIERARCHICAL CONCEPT LEARNING ENGINE
# ============================================================================

class LearningTracker:
    """Track learning progress across hierarchical levels with dynamic node support."""

    def __init__(self):
        self.stats = {
            'patterns_by_level': defaultdict(list)
        }
        self.start_time = time.time()

    def record_pattern(self, level: str, pattern_name: str):
        """Record learned pattern for any level."""
        self.stats['patterns_by_level'][level].append(pattern_name)

    def get_level_count(self, level: str) -> int:
        """Get pattern count for a specific level."""
        return len(self.stats['patterns_by_level'][level])

    def get_stats(self) -> dict:
        """Get current statistics with backward compatibility."""
        elapsed = time.time() - self.start_time

        # Create stats dict with counts per level
        level_counts = {
            f'{level}_patterns': len(patterns)
            for level, patterns in self.stats['patterns_by_level'].items()
        }

        # Backward compatibility: maintain old field names for first 4 levels
        backward_compat = {
            'sentences_learned': self.get_level_count('node0'),
            'paragraphs_learned': self.get_level_count('node1'),
            'chapters_learned': self.get_level_count('node2'),
            'books_learned': self.get_level_count('node3')
        }

        return {
            **self.stats,
            **level_counts,
            **backward_compat,
            'elapsed_time': elapsed,
            'elapsed_formatted': self._format_time(elapsed)
        }

    def print_summary(self):
        """Print learning summary for all levels."""
        stats = self.get_stats()
        print("\n" + "="*80)
        print("HIERARCHICAL LEARNING SUMMARY")
        print("="*80)

        # Print all levels dynamically
        for level in sorted(self.stats['patterns_by_level'].keys()):
            count = len(self.stats['patterns_by_level'][level])
            print(f"{level}: {count:,} patterns learned")

        print(f"Elapsed time: {stats['elapsed_formatted']}")
        print("="*80)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class HierarchicalNode:
    """
    Configuration for a single node in the hierarchy.

    This class encapsulates all configuration for one hierarchical level,
    allowing each node to have independent settings for chunk size, mode, etc.

    Example:
        # Create custom nodes with different chunk sizes
        nodes = [
            HierarchicalNode('node0', chunk_size=10, mode='chunking'),
            HierarchicalNode('node1', chunk_size=15, mode='chunking'),
            HierarchicalNode('node2', chunk_size=20, mode='chunking'),
            HierarchicalNode('node3', chunk_size=25, mode='chunking')
        ]

        learner = HierarchicalConceptLearner(nodes=nodes, tokenizer_name='gpt2')
    """

    def __init__(
        self,
        name: str,
        base_url: str = "http://kato:8000",
        mode: str = 'chunking',
        chunk_size: int = 15,
        min_sentence_tokens: int = 3,
        # KATO Configuration (Training Optimizations)
        process_predictions: bool = False,  # Disable for training performance
        max_pattern_length: int = 0,        # 0 = manual learning only
        stm_mode: str = 'CLEAR'             # Clear STM after each learn
    ):
        """
        Initialize a hierarchical node configuration.

        Args:
            name: Node identifier (e.g., 'node0', 'node1', 'node2', ...)
                  Must follow pattern 'nodeN' where N is the level index
            base_url: KATO server URL
            mode: Segmentation mode - 'chunking' or 'sentences'
                  Note: Only applies to node0 (raw token segmentation)
                  Higher nodes always receive pattern name sequences
            chunk_size: Number of items per chunk (default: 15)
                       - At node0: number of tokens per chunk
                       - At node1+: number of pattern names per observation
            min_sentence_tokens: Minimum tokens per sentence (default: 3)
                                Only used if mode='sentences' at node0
            process_predictions: Enable/disable prediction processing (default: False)
                                False = 2-3x faster training (predictions not computed)
                                True = Predictions available (for interactive exploration)
            max_pattern_length: Auto-learning threshold (default: 0)
                               0 = Manual learning only (recommended for training)
                               >0 = Auto-learn after N observations
            stm_mode: Short-term memory mode (default: 'CLEAR')
                     'CLEAR' = Clear STM after each learn (fresh context)
                     'ROLLING' = Keep rolling window (for sequential tasks)
        """
        if mode not in ['chunking', 'sentences']:
            raise ValueError(f"mode must be 'chunking' or 'sentences', got: {mode}")

        self.name = name
        self.base_url = base_url
        self.mode = mode
        self.chunk_size = chunk_size
        self.min_sentence_tokens = min_sentence_tokens
        # KATO configuration
        self.process_predictions = process_predictions
        self.max_pattern_length = max_pattern_length
        self.stm_mode = stm_mode
        self.kato_client = None  # Created by HierarchicalConceptLearner

    def __repr__(self):
        return f"HierarchicalNode(name={self.name}, mode={self.mode}, chunk_size={self.chunk_size})"


class HierarchicalConceptLearner:
    """
    Manages hierarchical learning with N nodes and delimiter-based streaming from datasets.

    This class implements delimiter-based hierarchical learning where:
    - N nodes are created (default 4, configurable to any depth)
    - Text is segmented by delimiter (sentence, word, bigram, etc.)
    - Each segment is tokenized (GPT-2, BERT, etc.)
    - Node0 learns each tokenized segment as a complete pattern
    - Pattern names can flow to higher nodes for multi-level hierarchy
    - Supports arbitrary depth hierarchies (5, 10, 20+ levels)

    Example:
        # Default 4 nodes
        learner = HierarchicalConceptLearner(tokenizer_name="gpt2")

        # Deep hierarchy with 10 nodes
        learner = HierarchicalConceptLearner(tokenizer_name="gpt2", num_nodes=10)

        # Process: delimiter='sentence' → "Hello world." → GPT-2 tokenize
        # → ["Hello", "Ġworld", "."] → node0.observe_sequence() → node0.learn()
        # → "PTRN|abc123..." → node1 (if hierarchical_levels > 1)
    """

    def __init__(self,
                 nodes: List['HierarchicalNode'] = None,
                 base_url: str = "http://kato:8000",
                 tokenizer_name: str = "gpt2",
                 num_nodes: int = 4,
                 segmentation_mode: str = 'chunking',
                 chunk_size: int = 15,
                 min_sentence_tokens: int = 3,
                 node0_batch_size: int = 1,
                 verbose_init: bool = True):
        """
        Initialize hierarchical learner with N KATO nodes.

        Two initialization modes:

        1. Custom nodes (NEW): Pass list of HierarchicalNode objects for per-node configuration
           Example:
               nodes = [
                   HierarchicalNode('node0', chunk_size=10),
                   HierarchicalNode('node1', chunk_size=15),
                   HierarchicalNode('node2', chunk_size=20)
               ]
               learner = HierarchicalConceptLearner(nodes=nodes, tokenizer_name='gpt2')

        2. Uniform nodes (BACKWARD COMPATIBLE): Pass global parameters
           Example:
               learner = HierarchicalConceptLearner(
                   num_nodes=4,
                   chunk_size=15,
                   tokenizer_name='gpt2'
               )

        Args:
            nodes: List of HierarchicalNode configs (NEW API). If provided, other params ignored.
            base_url: KATO server URL (used if nodes=None)
            tokenizer_name: Tokenizer for segment tokenization (gpt2, bert-base-uncased, etc.)
            num_nodes: Number of hierarchical nodes to create (used if nodes=None)
            segmentation_mode: 'chunking' or 'sentences' (used if nodes=None)
            chunk_size: Tokens per chunk (used if nodes=None)
            min_sentence_tokens: Min tokens per sentence (used if nodes=None)
            node0_batch_size: Number of chunks to accumulate before batching API call (1=no batching, 50=recommended)
            verbose_init: Whether to print initialization messages (default: True, set False for parallel workers)
        """
        if verbose_init:
            print("\n" + "="*80)
            print("INITIALIZING HIERARCHICAL CONCEPT LEARNER")
            print("="*80)

        self.tokenizer_name = tokenizer_name
        self.node0_batch_size = node0_batch_size

        # NEW API: Custom node configs
        if nodes is not None:
            if not isinstance(nodes, list) or len(nodes) == 0:
                raise ValueError("nodes must be a non-empty list of HierarchicalNode objects")

            self.node_configs = nodes
            self.num_nodes = len(nodes)

            # Validate node names match expected pattern: node0, node1, node2, ...
            for i, node_config in enumerate(nodes):
                expected_name = f'node{i}'
                if node_config.name != expected_name:
                    raise ValueError(
                        f"Node at index {i} must be named '{expected_name}', got '{node_config.name}'. "
                        f"Nodes must be ordered: [node0, node1, node2, ...]"
                    )

            if verbose_init:
                print(f"Using custom node configurations ({len(nodes)} nodes)")

        # OLD API: Uniform configuration (backward compatible)
        else:
            if num_nodes < 1:
                raise ValueError("num_nodes must be at least 1")

            self.node_configs = [
                HierarchicalNode(
                    name=f'node{i}',
                    base_url=base_url,
                    mode=segmentation_mode,
                    chunk_size=chunk_size,
                    min_sentence_tokens=min_sentence_tokens
                )
                for i in range(num_nodes)
            ]
            self.num_nodes = num_nodes

            if verbose_init:
                print(f"Using uniform configuration ({num_nodes} nodes)")

        # Backward compatibility: store global params (for old code that accesses them)
        self.segmentation_mode = self.node_configs[0].mode if nodes else segmentation_mode
        self.chunk_size = self.node_configs[0].chunk_size if nodes else chunk_size
        self.min_sentence_tokens = self.node_configs[0].min_sentence_tokens if nodes else min_sentence_tokens

        # Create KATO clients from node configs
        self.nodes = {}
        for node_config in self.node_configs:
            node_config.kato_client = KATOClient(
                node_id=node_config.name,  # node0, node1, node2, etc. (simple and clean)
                max_pattern_length=node_config.max_pattern_length,  # From node config
                stm_mode=node_config.stm_mode,                      # From node config
                process_predictions=node_config.process_predictions,  # From node config
                base_url=node_config.base_url
            )
            self.nodes[node_config.name] = node_config.kato_client

        # Initialize tokenizer
        self.token_processor = TokenProcessor(tokenizer_name)

        # Initialize progress tracker
        self.tracker = LearningTracker()

        # Print configuration summary
        if verbose_init:
            print(f"\n✓ {self.num_nodes} nodes initialized with:")
            # Show KATO configuration from first node (assuming all nodes use same KATO config)
            first_node_config = self.node_configs[0]
            print(f"  - max_pattern_length = {first_node_config.max_pattern_length} ({'manual learning' if first_node_config.max_pattern_length == 0 else 'auto-learning'})")
            print(f"  - stm_mode = {first_node_config.stm_mode} ({'STM clears after learn' if first_node_config.stm_mode == 'CLEAR' else 'rolling window'})")
            print(f"  - process_predictions = {first_node_config.process_predictions} ({'predictions disabled' if not first_node_config.process_predictions else 'predictions enabled'})")
            print(f"  - tokenizer = {tokenizer_name}")

            # Show per-node config if custom nodes, else show global config
            if nodes is not None:
                print("\nPer-node configuration:")
                for config in self.node_configs:
                    print(f"  {config.name}: mode={config.mode}, chunk_size={config.chunk_size}")
            else:
                print(f"  - segmentation_mode = {self.segmentation_mode}")
                if self.segmentation_mode == 'chunking':
                    print(f"  - chunk_size = {self.chunk_size} tokens")
                else:
                    print(f"  - min_sentence_tokens = {self.min_sentence_tokens}")

            print("="*80)

    def learn_segment_at_node0(self, tokens: List[str]) -> str:
        """
        Learn a tokenized segment at Node0 using observe_sequence + learn.

        Args:
            tokens: List of tokens from tokenizer

        Returns:
            Pattern name (e.g., "PTRN|abc123...")
        """
        if len(tokens) < 2:
            # KATO requires at least 2 strings to learn a pattern
            tokens.append("<EOS>")

        # Create observation sequence
        observations = [{'strings': [token]} for token in tokens]

        # Observe sequence and learn
        result = self.nodes['node0'].observe_sequence(
            observations=observations,
            learn_at_end=True
        )

        # Get pattern name
        pattern_name = result.get('final_learned_pattern', 'UNKNOWN')

        # Track progress
        self.tracker.record_pattern('node0', pattern_name)

        return pattern_name

    def learn_from_text(self,
                       text: str,
                       delimiter: str,
                       hierarchical_levels: int = 1,
                       max_tokens_per_segment: int = 512,
                       verbose: bool = True) -> Dict[str, Any]:
        """
        Learn from text using delimiter-based segmentation.

        Args:
            text: Input text to learn from
            delimiter: Segmentation delimiter ('sentence', 'word', 'bigram', 'trigram', '4-gram', '5-gram', 'paragraph')
            hierarchical_levels: Number of hierarchy levels to use (1 to num_nodes)
            max_tokens_per_segment: Max tokens per delimited segment
            verbose: Print progress

        Returns:
            Training statistics
        """
        if hierarchical_levels < 1 or hierarchical_levels > self.num_nodes:
            raise ValueError(f"hierarchical_levels must be between 1 and {self.num_nodes}")

        if verbose:
            print(f"\n{'='*80}")
            print(f"LEARNING FROM TEXT")
            print(f"{'='*80}")
            print(f"Delimiter: {delimiter}")
            print(f"Hierarchical levels: {hierarchical_levels}")
            print(f"Max tokens per segment: {max_tokens_per_segment}")

        # Segment and tokenize
        tokenized_segments = self.token_processor.process_with_delimiter(
            text, delimiter, max_tokens_per_segment
        )

        if verbose:
            print(f"Segments to learn: {len(tokenized_segments)}")

        # Learn at node0
        node0_patterns = []
        for i, tokens in enumerate(tokenized_segments, 1):
            pattern_name = self.learn_segment_at_node0(tokens)
            node0_patterns.append(pattern_name)

            if verbose and i % 100 == 0:
                print(f"  Processed {i}/{len(tokenized_segments)} segments...")

        if verbose:
            print(f"✓ node0 learned {len(node0_patterns)} patterns")

        # Hierarchical learning if requested - use loop instead of nested ifs
        if hierarchical_levels > 1:
            current_patterns = node0_patterns
            for level in range(1, hierarchical_levels):
                current_patterns = self._learn_hierarchy_level(
                    current_patterns, f'node{level}', verbose=verbose
                )

        stats = self.get_stats()
        if verbose:
            print(f"{'='*80}")
            print(f"Learning complete!")

        return stats

    def learn_from_stream(self,
                         dataset_key: str,
                         max_samples: int,
                         delimiter: str,
                         hierarchical_levels: int = 1,
                         max_tokens_per_segment: int = 512,
                         checkpoint_every: Optional[int] = None,
                         checkpoint_dir: str = "./checkpoints",
                         verbose: bool = True) -> Dict[str, Any]:
        """
        Learn from streaming dataset with delimiter-based segmentation.

        Args:
            dataset_key: Dataset identifier ('wikitext', 'c4', 'openwebtext', etc.)
            max_samples: Maximum text samples to process from stream
            delimiter: Segmentation delimiter ('sentence', 'word', 'bigram', 'trigram', '4-gram', '5-gram', 'paragraph')
            hierarchical_levels: Number of hierarchy levels (1 to num_nodes)
            max_tokens_per_segment: Max tokens per delimited segment
            checkpoint_every: Save checkpoint every N segments (None = no checkpointing)
            checkpoint_dir: Directory for checkpoints
            verbose: Print progress

        Returns:
            Training statistics

        Example:
            learner = HierarchicalConceptLearner()
            stats = learner.learn_from_stream(
                dataset_key='wikitext',
                max_samples=10000,
                delimiter='sentence',
                hierarchical_levels=1
            )
        """
        if hierarchical_levels < 1 or hierarchical_levels > self.num_nodes:
            raise ValueError(f"hierarchical_levels must be between 1 and {self.num_nodes}")

        if verbose:
            print(f"\n{'='*80}")
            print(f"STREAMING DATASET LEARNING")
            print(f"{'='*80}")
            print(f"Dataset: {dataset_key}")
            print(f"Max samples: {max_samples:,}")
            print(f"Delimiter: {delimiter}")
            print(f"Hierarchical levels: {hierarchical_levels}")
            print(f"Max tokens per segment: {max_tokens_per_segment}")
            if checkpoint_every:
                print(f"Checkpointing: every {checkpoint_every:,} segments")

        # Stream dataset
        if verbose:
            print(f"\nStarting data stream...")

        try:
            data_stream = StreamingDatasetLoader.load_streaming(dataset_key, max_samples)
        except NameError:
            raise ImportError("StreamingDatasetLoader not available. Please ensure it's imported.")

        # Process stream
        node0_patterns = []
        segments_processed = 0
        last_checkpoint = 0
        start_time = time.time()

        with tqdm(total=max_samples, desc="Streaming", unit="samples", disable=not verbose) as pbar:
            for text in data_stream:
                # Segment and tokenize
                tokenized_segments = self.token_processor.process_with_delimiter(
                    text, delimiter, max_tokens_per_segment
                )

                # Learn each segment at node0
                for tokens in tokenized_segments:
                    pattern_name = self.learn_segment_at_node0(tokens)
                    node0_patterns.append(pattern_name)
                    segments_processed += 1

                    # Checkpoint if needed
                    if checkpoint_every and (segments_processed - last_checkpoint) >= checkpoint_every:
                        TrainingCheckpoint.save_checkpoint(
                            self.nodes['node0'],
                            segments_processed,
                            dataset_key,
                            checkpoint_dir
                        )
                        last_checkpoint = segments_processed

                pbar.update(1)

                # Update stats periodically
                if pbar.n % 10 == 0:
                    stats = self.nodes['node0'].get_stats()
                    pbar.set_postfix({
                        'segments': segments_processed,
                        'patterns': stats.get('patterns_learned', 0)
                    })

        # Hierarchical learning if requested - use loop instead of nested ifs
        if hierarchical_levels > 1 and node0_patterns:
            if verbose:
                print(f"\nLearning hierarchical levels...")

            current_patterns = node0_patterns
            for level in range(1, hierarchical_levels):
                current_patterns = self._learn_hierarchy_level(
                    current_patterns, f'node{level}', verbose=verbose
                )
                if not current_patterns:
                    break  # Stop if no patterns were generated

        # Final stats
        total_time = time.time() - start_time
        stats = self.get_stats()

        if verbose:
            print(f"\n{'='*80}")
            print(f"STREAMING LEARNING COMPLETE")
            print(f"{'='*80}")
            print(f"Segments processed: {segments_processed:,}")
            print(f"Total time: {self.tracker._format_time(total_time)}")
            print(f"{'='*80}")

        return stats

    def _learn_hierarchy_level(self,
                               pattern_names: List[str],
                               node_key: str,
                               batch_size: int = 100,
                               verbose: bool = True) -> List[str]:
        """
        Learn patterns at a higher hierarchy level.

        Args:
            pattern_names: List of pattern names from previous level
            node_key: Node to learn at ('node1', 'node2', 'node3')
            batch_size: How many patterns to group into one higher-level pattern
            verbose: Print progress

        Returns:
            List of pattern names at this level
        """
        higher_patterns = []

        # Split into batches
        for i in range(0, len(pattern_names), batch_size):
            batch = pattern_names[i:i+batch_size]

            if len(batch) < 2:
                # KATO needs at least 2 strings
                continue

            # Observe sequence and learn
            observations = [{'strings': [pattern]} for pattern in batch]
            result = self.nodes[node_key].observe_sequence(
                observations=observations,
                learn_at_end=True
            )

            pattern_name = result.get('final_learned_pattern', 'UNKNOWN')
            higher_patterns.append(pattern_name)
            self.tracker.record_pattern(node_key, pattern_name)

        if verbose:
            print(f"✓ {node_key} learned {len(higher_patterns)} patterns from {len(pattern_names)} inputs")

        return higher_patterns

    def process_corpus(self, corpus: dict, verbose: bool = True):
        """
        Process entire corpus hierarchically.

        Args:
            corpus: Dictionary with 'books' list
            verbose: Print progress information
        """
        books = corpus.get('books', [])

        if verbose:
            print("\n" + "="*80)
            print("PROCESSING CORPUS")
            print("="*80)
            print(f"Total books: {len(books)}\n")

        for i, book in enumerate(books, 1):
            if verbose:
                print(f"\n[Book {i}/{len(books)}]")

            book_pattern = self.learn_book(book, verbose=verbose)

        if verbose:
            print("\n" + "="*80)
            print("CORPUS PROCESSING COMPLETE")
            print("="*80)
            self.tracker.print_summary()

    def get_stats(self) -> dict:
        """Get learning statistics."""
        return self.tracker.get_stats()

    def get_node_stats(self) -> dict:
        """Get statistics from all nodes."""
        return {
            level: node.get_stats()
            for level, node in self.nodes.items()
        }


# ============================================================================
# SINGLE-PASS HIERARCHICAL TRAINING
# ============================================================================

def batch_learn_node0(
    chunks: List[List[str]],
    learner: 'HierarchicalConceptLearner',
    metadata: Dict[str, Any] = None,
    metadata_injected: bool = False
) -> Tuple[List[str], bool]:
    """
    Learn multiple chunks at node0 using batched API calls.

    KATO API Limitation: observe_sequence doesn't support selective learning at
    specific positions. Current strategy: Process each chunk with learn_at_end=True.

    Performance Impact:
    - Reduces from N×chunk_size observe() calls to N observe_sequence() calls
    - Still one learn() per chunk (as designed)

    Args:
        chunks: List of token lists [[tok1, tok2, ...], [tok3, tok4, ...], ...]
        learner: HierarchicalConceptLearner instance
        metadata: Optional book metadata dict to inject in first observation
        metadata_injected: Whether metadata has already been injected (skip if True)

    Returns:
        Tuple of (learned_pattern_names, metadata_was_injected)
    """
    if not chunks:
        return [], metadata_injected

    node0 = learner.nodes['node0']
    learned_patterns = []

    # Process each chunk individually with observe_sequence + learn_at_end
    # This gives us one pattern per chunk (correct semantics) while still batching observations
    for chunk_idx, chunk in enumerate(chunks):
        if len(chunk) < 2:
            # Skip degenerate chunks (KATO requires >=2 tokens for learning)
            continue

        # Build observations for this chunk
        observations = []
        for token in chunk:
            observations.append({'strings': [token]})

        # Inject metadata into first observation of first chunk
        if metadata and not metadata_injected and chunk_idx == 0 and observations:
            observations[0]['metadata'] = metadata
            metadata_injected = True

        # Learn this chunk as a sequence
        result = node0.observe_sequence(
            observations=observations,
            learn_at_end=True  # Learn once from the accumulated sequence
        )

        # Extract the single learned pattern for this chunk
        pattern_name = result.get('final_learned_pattern') or result.get('pattern_name')
        if pattern_name:
            learned_patterns.append(pattern_name)

    return learned_patterns, metadata_injected


def train_hierarchical_single_pass(
    corpus: Dict,
    learner: 'HierarchicalConceptLearner',
    delimiter: str = 'sentence',
    num_levels: Optional[int] = None,
    verbose: bool = True,
    progress_mode: str = 'summary'
) -> Dict[str, Any]:
    """
    Single-pass hierarchical training with chunk-based learning at ALL levels.

    NEW BEHAVIOR: Each node learns when its buffer reaches the configured chunk_size,
    not at structural boundaries (paragraphs/chapters/books). This allows true
    multi-scale chunking with configurable granularity at each level.

    Process flow:
    1. node0: Every chunk_size[0] tokens → learn → add pattern to node1 buffer
    2. node1: Every chunk_size[1] node0 patterns → learn → add pattern to node2 buffer
    3. node2: Every chunk_size[2] node1 patterns → learn → add pattern to node3 buffer
    4. ... continues for all levels dynamically
    5. At end of corpus: flush all remaining patterns in buffers

    Hierarchical Composition Examples:
    - With uniform chunk_size=15 across all levels:
      * node0: 15 tokens → 1 pattern
      * node1: 15 node0 patterns (225 tokens) → 1 pattern
      * node2: 15 node1 patterns (3,375 tokens) → 1 pattern
      * node3: 15 node2 patterns (50,625 tokens) → 1 pattern

    - With custom per-node chunk sizes (5, 7, 9, 11):
      * node0: 5 tokens → 1 pattern
      * node1: 7 node0 patterns (35 tokens) → 1 pattern
      * node2: 9 node1 patterns (315 tokens) → 1 pattern
      * node3: 11 node2 patterns (3,465 tokens) → 1 pattern

    Key Features:
    - Supports arbitrary depth (2, 4, 6, 10, 20+ levels)
    - Per-node chunk_size configuration via HierarchicalNode
    - Pattern cascading: learning at level N may trigger learning at N+1, N+2, etc.
    - No hardcoded level limits

    Args:
        corpus: Hierarchical corpus from CorpusSegmenter.segment_book()
                Expected structure: {'books': [{'title': str, 'chapters': [{'paragraphs': [{'chunks': [[tokens]]}]}]}]}
        learner: HierarchicalConceptLearner with configured nodes
        delimiter: DEPRECATED - parameter exists for API compatibility only
        num_levels: Number of hierarchy levels to use. If None (default), automatically uses all
                   available nodes from learner (recommended). Set explicitly (2 to learner.num_nodes)
                   to train with a subset of nodes (useful for testing/debugging).
        verbose: Print start/end summary information
        progress_mode: Progress tracking mode:
            - 'silent': No progress output during training
            - 'summary': Print updates every 10 seconds with stats (recommended for large datasets)
            - 'bar': Single progress bar tracking chunks processed
            - 'detailed': Nested progress bars for books/chapters/paragraphs (not recommended for large datasets)

    Returns:
        Training statistics with patterns learned per level

    Example:
        # Custom per-node chunk sizes
        nodes = [
            HierarchicalNode('node0', chunk_size=5),
            HierarchicalNode('node1', chunk_size=7),
            HierarchicalNode('node2', chunk_size=9),
            HierarchicalNode('node3', chunk_size=11)
        ]
        learner = HierarchicalConceptLearner(nodes=nodes, tokenizer_name="gpt2")

        segmenter = CorpusSegmenter(chunk_size=5)  # Match node0
        book = segmenter.segment_book(text, book_metadata={'title': 'My Book'})
        corpus = {'books': [book]}

        # Auto-detect number of levels (uses all 4 nodes)
        stats = train_hierarchical_single_pass(corpus, learner)

        # Or explicitly specify (e.g., to test with only 3 nodes)
        stats = train_hierarchical_single_pass(corpus, learner, num_levels=3)
    """

    # Auto-detect num_levels if not specified
    if num_levels is None:
        num_levels = learner.num_nodes
        if verbose:
            print(f"✓ Auto-detected {num_levels} levels from learner configuration")
    else:
        # Validate provided num_levels
        if num_levels < 2 or num_levels > learner.num_nodes:
            raise ValueError(f"num_levels must be between 2 and {learner.num_nodes}")

        # Warn if user is not using all available nodes
        if num_levels < learner.num_nodes:
            unused_nodes = list(range(num_levels, learner.num_nodes))
            print(f"⚠️  WARNING: Using {num_levels}/{learner.num_nodes} available nodes. "
                  f"Node(s) {unused_nodes} will be ignored.")

    if 'books' not in corpus:
        raise ValueError("corpus must contain 'books' key from CorpusSegmenter")

    # Validate progress_mode
    valid_modes = ['silent', 'summary', 'bar', 'detailed']
    if progress_mode not in valid_modes:
        raise ValueError(f"progress_mode must be one of {valid_modes}, got '{progress_mode}'")

    # Initialize statistics and buffers
    stats = {f'node{i}_patterns': 0 for i in range(num_levels)}
    start_time = time.time()

    # Count total chunks for progress tracking
    total_chunks = sum(
        len(paragraph['chunks'])
        for book in corpus['books']
        for chapter in book['chapters']
        for paragraph in chapter['paragraphs']
    )
    chunks_processed = 0
    last_update_time = start_time

    # Initialize progress bar if needed
    pbar = None
    if progress_mode == 'bar':
        pbar = tqdm(total=total_chunks, desc="Training", unit="chunks", leave=True)

    # Pattern buffers for levels 1 through num_levels-1
    pattern_buffers = {f'node{i}': [] for i in range(1, num_levels)}

    # Track metadata injection for all levels
    metadata_injected_global = {f'node{i}': False for i in range(1, num_levels)}

    # Helper function: learn from buffer when chunk_size is reached
    def try_learn_at_level(level, book_metadata, force_learn=False):
        """
        Try to learn at the given level if buffer has enough patterns.
        Returns learned pattern name if successful, None otherwise.
        Adds learned pattern to next level's buffer but does NOT recursively learn
        (recursive learning was causing performance issues).
        """
        if level >= num_levels:
            return None

        node_key = f'node{level}'
        buffer = pattern_buffers[node_key]
        chunk_size = learner.node_configs[level].chunk_size

        # Check if we should learn
        patterns_to_learn = None
        if force_learn and len(buffer) >= 2:  # KATO minimum
            patterns_to_learn = buffer[:]
            buffer.clear()
        elif len(buffer) >= chunk_size:
            patterns_to_learn = buffer[:chunk_size]
            del buffer[:chunk_size]

        if patterns_to_learn is None:
            return None

        # Build observations with metadata in first observation
        node = learner.nodes[node_key]
        observations = []

        for i, pattern in enumerate(patterns_to_learn):
            if not metadata_injected_global[node_key] and i == 0:
                # Include metadata in first observation
                observations.append({'strings': [pattern], 'metadata': book_metadata})
                metadata_injected_global[node_key] = True
            else:
                observations.append({'strings': [pattern]})

        # Use observe_sequence for performance (batch operation)
        result = node.observe_sequence(
            observations=observations,
            learn_at_end=True
        )

        pattern_name = result.get('final_learned_pattern',
                                 result.get('pattern_name', 'UNKNOWN'))
        stats[f'{node_key}_patterns'] += 1

        # Add to next level's buffer (don't recurse - main loop will check)
        if level + 1 < num_levels:
            pattern_buffers[f'node{level + 1}'].append(pattern_name)

        return pattern_name

    # Print header
    if verbose:
        print(f"\n{'='*80}")
        print(f"SINGLE-PASS HIERARCHICAL TRAINING (CHUNK-BASED)")
        print(f"{'='*80}")
        print(f"Tokenizer: {learner.tokenizer_name}")
        print(f"Hierarchy levels: {num_levels}")
        print(f"Books: {len(corpus['books'])}")
        print(f"Node0 batch size: {learner.node0_batch_size} {'(batching enabled)' if learner.node0_batch_size > 1 else '(batching disabled)'}")
        print("\nPer-level chunk sizes:")
        for i in range(num_levels):
            print(f"  node{i}: chunk_size={learner.node_configs[i].chunk_size}")

    # Process each book with progress tracking
    books_iterator = corpus['books']
    if progress_mode == 'detailed':
        books_iterator = tqdm(books_iterator, desc="Processing books", unit="book")

    for book_idx, book in enumerate(books_iterator, 1):
        # Extract book metadata for injection into higher-level nodes
        book_metadata = {
            'title': book.get('title', 'Untitled'),
            'author': book.get('author', 'Unknown'),
            'book_idx': book_idx
        }

        # Node0 batch accumulator (only used if node0_batch_size > 1)
        node0_chunk_batch = []
        node0_metadata_injected = False

        # Helper: process accumulated node0 batch
        def flush_node0_batch():
            nonlocal node0_metadata_injected
            if not node0_chunk_batch:
                return

            # Learn all accumulated chunks in single API call
            learned_patterns, node0_metadata_injected = batch_learn_node0(
                chunks=node0_chunk_batch,
                learner=learner,
                metadata=book_metadata,
                metadata_injected=node0_metadata_injected
            )

            stats['node0_patterns'] += len(learned_patterns)

            # Add learned patterns to node1 buffer
            if num_levels > 1:
                for pattern in learned_patterns:
                    pattern_buffers['node1'].append(pattern)

                    # Check all levels that might be ready to learn
                    for level in range(1, num_levels):
                        node_key = f'node{level}'
                        buffer_size = len(pattern_buffers[node_key])
                        chunk_size_needed = learner.node_configs[level].chunk_size

                        if buffer_size >= chunk_size_needed:
                            try_learn_at_level(level, book_metadata, force_learn=False)

            # Clear batch
            node0_chunk_batch.clear()

        # Process chapters
        chapters_iterator = book['chapters']
        if progress_mode == 'detailed':
            book_title = book.get('title', 'Untitled')[:40]
            chapters_iterator = tqdm(
                chapters_iterator,
                desc=f"  Chapters ({book_title})",
                unit="ch",
                leave=False
            )

        for chapter_idx, chapter in enumerate(chapters_iterator, 1):
            # Process paragraphs
            paragraphs_iterator = chapter['paragraphs']
            if progress_mode == 'detailed':
                paragraphs_iterator = tqdm(
                    paragraphs_iterator,
                    desc=f"    Paragraphs (ch{chapter_idx})",
                    unit="para",
                    leave=False
                )

            for para_idx, paragraph in enumerate(paragraphs_iterator, 1):
                # LEVEL 0: Process each token chunk with batching
                for chunk_idx, chunk_tokens in enumerate(paragraph['chunks'], 1):
                    tokens = chunk_tokens

                    if len(tokens) < 2:
                        continue  # Skip degenerate chunks

                    # Accumulate chunk for batching
                    node0_chunk_batch.append(tokens)

                    # Process batch when full (or immediately if batching disabled)
                    if len(node0_chunk_batch) >= learner.node0_batch_size:
                        flush_node0_batch()

                    # Update progress tracking
                    chunks_processed += 1

                    # Update progress bar
                    if pbar is not None:
                        pbar.update(1)

                    # Print summary updates
                    elif progress_mode == 'summary':
                        current_time = time.time()
                        if current_time - last_update_time >= 10.0:  # Update every 10s
                            elapsed = current_time - start_time
                            rate = chunks_processed / elapsed if elapsed > 0 else 0
                            progress_pct = 100 * chunks_processed / total_chunks if total_chunks > 0 else 0
                            print(f"[{elapsed:.1f}s] Processed {chunks_processed:,} / {total_chunks:,} chunks "
                                  f"({progress_pct:.1f}%) | {rate:.0f} chunks/sec | "
                                  f"node0: {stats['node0_patterns']:,} patterns")
                            last_update_time = current_time

        # End of book: flush remaining node0 chunks and higher-level buffers
        flush_node0_batch()  # Process any remaining chunks in batch
        for level in range(1, num_levels):
            try_learn_at_level(level, book_metadata, force_learn=True)

    # Close progress bar if it was created
    if pbar is not None:
        pbar.close()

    total_time = time.time() - start_time
    num_documents = len(corpus['books'])

    if verbose:
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")

        # Pattern statistics per level
        for level in range(num_levels):
            patterns_learned = stats[f'node{level}_patterns']
            patterns_per_doc = patterns_learned / num_documents if num_documents > 0 else 0
            print(f"node{level}: {patterns_learned:,} patterns learned ({patterns_per_doc:.2f} patterns/doc)")

        print(f"Total time: {LearningTracker._format_time(total_time)}")

        # Validation metrics
        print(f"\n{'='*80}")
        print(f"HIERARCHY VALIDATION")
        print(f"{'='*80}")

        # Calculate receptive field coverage
        print(f"Receptive field coverage:")
        coverage = learner.node_configs[0].chunk_size
        for level in range(num_levels):
            chunk_size = learner.node_configs[level].chunk_size
            print(f"  node{level} (chunk_size={chunk_size}): {coverage:,} tokens")
            if level < num_levels - 1:
                coverage *= learner.node_configs[level + 1].chunk_size

        # Warn about underutilized levels
        print(f"\nUtilization warnings:")
        warnings_found = False
        for level in range(num_levels):
            patterns_learned = stats[f'node{level}_patterns']
            patterns_per_doc = patterns_learned / num_documents if num_documents > 0 else 0

            if patterns_per_doc < 0.5:
                warnings_found = True
                print(f"  ⚠️  node{level}: Low utilization ({patterns_per_doc:.2f} patterns/doc)")
                print(f"      Consider reducing hierarchy depth or adjusting chunk_size")

        if not warnings_found:
            print(f"  ✓ All levels well-utilized (≥0.5 patterns/doc)")

        print(f"{'='*80}")

    stats['total_time_seconds'] = total_time
    stats['num_documents'] = num_documents

    # Add patterns per document to stats
    for level in range(num_levels):
        patterns_learned = stats[f'node{level}_patterns']
        stats[f'node{level}_patterns_per_doc'] = patterns_learned / num_documents if num_documents > 0 else 0

    return stats


def cleanup_all_nodes(
    learner: 'HierarchicalConceptLearner',
    threshold: int = 2,
    mongo_uri: str = "mongodb://localhost:27017/",
    verbose: bool = True
) -> Dict[str, int]:
    """
    Clean up low-frequency patterns across all nodes.

    This function removes patterns with frequency below the threshold from all nodes
    in the hierarchical learner, helping to filter out noise and rare patterns.

    Args:
        learner: HierarchicalConceptLearner instance
        threshold: Delete patterns with frequency < threshold (patterns with freq >= threshold are kept)
        mongo_uri: MongoDB connection URI
        verbose: Print deletion statistics

    Returns:
        Dict mapping node_name → number of patterns deleted

    Example:
        learner = HierarchicalConceptLearner(num_nodes=4)
        # ... train ...
        deleted = cleanup_all_nodes(learner, threshold=3, verbose=True)
        # Output: node0: deleted 1,234 patterns with frequency < 3
    """
    results = {}

    for node_name, node in learner.nodes.items():
        try:
            analyzer = MongoDBAnalyzer(node, mongo_uri=mongo_uri)
            deleted = analyzer.delete_patterns_below_threshold(threshold)
            results[node_name] = deleted
            analyzer.close()

            if verbose and deleted > 0:
                print(f"{node_name}: deleted {deleted:,} patterns with frequency < {threshold}")
            elif verbose:
                print(f"{node_name}: no patterns with frequency < {threshold}")

        except Exception as e:
            if verbose:
                print(f"{node_name}: error during cleanup - {e}")
            results[node_name] = 0

    return results


def analyze_all_nodes(
    learner: 'HierarchicalConceptLearner',
    mongo_uri: str = "mongodb://localhost:27017/"
) -> Dict[str, Dict]:
    """
    Get comprehensive statistics for all nodes.

    Returns detailed statistics about patterns in each node's knowledge base,
    including total patterns, frequency distributions, and averages.

    Args:
        learner: HierarchicalConceptLearner instance
        mongo_uri: MongoDB connection URI

    Returns:
        Dict mapping node_name → stats_dict
        Stats dict contains: total_patterns, avg_frequency, max_frequency,
                           min_frequency, frequency_ranges

    Example:
        learner = HierarchicalConceptLearner(num_nodes=4)
        # ... train ...
        all_stats = analyze_all_nodes(learner)
        for node_name, stats in all_stats.items():
            print(f"{node_name}: {stats['total_patterns']} patterns, avg freq {stats['avg_frequency']}")
    """
    results = {}

    for node_name, node in learner.nodes.items():
        try:
            analyzer = MongoDBAnalyzer(node, mongo_uri=mongo_uri)
            results[node_name] = analyzer.get_stats()
            analyzer.close()
        except Exception as e:
            print(f"Error analyzing {node_name}: {e}")
            results[node_name] = {
                'total_patterns': 0,
                'error': str(e)
            }

    return results


# ============================================================================
# TRAINING CHECKPOINTS & RESUME
# ============================================================================

class TrainingCheckpoint:
    """Manage training checkpoints for long-running jobs."""

    @staticmethod
    def save_checkpoint(node: KATOClient,
                       samples_processed: int,
                       dataset_key: str,
                       checkpoint_dir: str = "./checkpoints",
                       checkpoint_name: str = None):
        """
        Save training checkpoint.

        Args:
            node: KATO client node
            samples_processed: Number of samples processed so far
            dataset_key: Dataset being used
            checkpoint_dir: Directory to save checkpoints
            checkpoint_name: Optional custom checkpoint name
        """
        # Create checkpoint directory if it doesn't exist
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Generate checkpoint filename
        if checkpoint_name is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            checkpoint_name = f"checkpoint_{node.node_id}_{timestamp}.pkl"

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        # Create checkpoint data
        checkpoint_data = {
            'node_id': node.node_id,
            'session_id': node.session_id,
            'samples_processed': samples_processed,
            'dataset_key': dataset_key,
            'stats': node.get_stats(),
            'patterns_learned': node.patterns_learned,
            'timestamp': time.time(),
            'max_pattern_length': node.max_pattern_length,
            'stm_mode': node.stm_mode
        }

        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        print(f"✓ Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"   Node ID: {checkpoint_data['node_id']}")
        print(f"   Samples processed: {checkpoint_data['samples_processed']:,}")
        print(f"   Patterns learned: {checkpoint_data['stats']['patterns_learned']}")

        return checkpoint_data

    @staticmethod
    def list_checkpoints(checkpoint_dir: str = "./checkpoints"):
        """List all available checkpoints."""
        if not os.path.exists(checkpoint_dir):
            print(f"No checkpoint directory found: {checkpoint_dir}")
            return []

        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]

        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}")
            return []

        print(f"\nAvailable Checkpoints in {checkpoint_dir}:")
        print("="*80)

        for cp_file in sorted(checkpoints):
            cp_path = os.path.join(checkpoint_dir, cp_file)
            try:
                cp_data = TrainingCheckpoint.load_checkpoint(cp_path)
                print()
            except Exception as e:
                print(f"Error loading {cp_file}: {e}")

        return checkpoints


# ============================================================================
# HIERARCHICAL CONNECTIONS
# ============================================================================

# Modeling Functions for Prediction Transfer
# These functions can be used with transfer_predictions() or in Jupyter notebooks

def transfer_all_names(predictions: List[Dict], field: str = 'name') -> List[str]:
    """
    Transfer all pattern names without filtering.
    Default modeling function that passes through all predictions.

    Args:
        predictions: List of prediction dicts from node
        field: Field to extract (default: 'name')

    Returns:
        List of pattern names or field values
    """
    return [p[field] for p in predictions]


def transfer_threshold(predictions: List[Dict],
                      field: str = 'name',
                      metric: str = 'potential',
                      threshold: float = 0.3) -> List[str]:
    """
    Filter predictions by metric threshold.

    Args:
        predictions: List of prediction dicts
        field: Field to extract ('name', 'future', 'matches', etc.)
        metric: Metric to filter by ('potential', 'confidence', 'similarity',
                'predictive_information', 'snr', etc.)
        threshold: Minimum metric value to include

    Returns:
        List of symbols/pattern names from filtered predictions

    Example:
        # Filter by potential (similarity × predictive_information) >= 0.4
        filtered = transfer_threshold(predictions, metric='potential', threshold=0.4)

        # Filter by confidence >= 0.6
        filtered = transfer_threshold(predictions, metric='confidence', threshold=0.6)
    """
    filtered = [p[field] for p in predictions if p.get(metric, 0) >= threshold]
    # Always return at least one prediction (the best one)
    return filtered if filtered else [predictions[0][field]] if predictions else []


def transfer_top_n(predictions: List[Dict],
                   field: str = 'name',
                   n: int = 3,
                   sort_by: str = 'potential') -> List[str]:
    """
    Return top N predictions sorted by metric.

    Args:
        predictions: List of prediction dicts
        field: Field to extract ('name', 'future', 'matches', etc.)
        n: Number of top predictions to return
        sort_by: Metric to sort by ('potential', 'confidence', 'similarity', etc.)

    Returns:
        List of top N symbols/pattern names

    Example:
        # Get top 5 predictions by potential
        top = transfer_top_n(predictions, n=5, sort_by='potential')

        # Get top 3 predictions by confidence
        top = transfer_top_n(predictions, n=3, sort_by='confidence')
    """
    if not predictions:
        return []

    sorted_preds = sorted(predictions, key=lambda p: p.get(sort_by, 0), reverse=True)
    return [p[field] for p in sorted_preds[:n]]


def transfer_weighted(predictions: List[Dict],
                     field: str = 'name',
                     weight_by: str = 'confidence',
                     min_weight: float = 0.3,
                     max_repeats: int = 5) -> List[str]:
    """
    Return pattern names with repetition weighted by metric.
    Higher metric values result in more repetitions of that pattern.

    Args:
        predictions: List of prediction dicts
        field: Field to extract ('name', 'future', 'matches', etc.)
        weight_by: Metric to weight by ('confidence', 'potential', 'similarity', etc.)
        min_weight: Minimum weight to include (0.0-1.0)
        max_repeats: Maximum repetitions per pattern

    Returns:
        List of symbols/pattern names with weighted repetition

    Example:
        # Weight by confidence, 1-5 repeats based on confidence value
        weighted = transfer_weighted(predictions, weight_by='confidence', max_repeats=5)

        # Weight by potential, minimum 0.4, up to 3 repeats
        weighted = transfer_weighted(predictions, weight_by='potential',
                                    min_weight=0.4, max_repeats=3)
    """
    if not predictions:
        return []

    weighted = []
    for pred in predictions:
        weight = pred.get(weight_by, 0)
        if weight >= min_weight:
            # Scale weight to repetition count (1 to max_repeats)
            repeat_count = max(1, min(max_repeats, int(weight * max_repeats)))
            weighted.extend([pred[field]] * repeat_count)

    # If nothing passed threshold, return best prediction
    return weighted if weighted else [predictions[0][field]]


def transfer_predictions(node_source: KATOClient,
                        node_target: KATOClient,
                        field: str,
                        modeling_function: Optional[callable] = None,
                        num_predictions: int = 5) -> Dict[str, Any]:
    """
    General-purpose function to transfer prediction data from one node to another.

    This function retrieves predictions from a source node, extracts a specific field
    from the prediction ensemble, optionally applies a modeling function to transform
    the data, and observes the results in the target node.

    Args:
        node_source: Source KATOClient node to get predictions from
        node_target: Target KATOClient node to observe data into
        field: Which field to extract from predictions. Valid values:
               'past', 'present', 'future', 'missing', 'matches', 'extras', 'name'
        modeling_function: Optional callable to transform prediction ensemble.
                          Signature: func(predictions: List[Dict], field: str) -> List[str]
                          The function receives the full prediction ensemble and field name,
                          and should return a list of strings to observe in the target node.
                          Has access to all prediction metrics: potential, normalized_entropy,
                          confidence, evidence, similarity, frequency, etc.
        num_predictions: Maximum number of predictions to retrieve from source

    Returns:
        Dictionary with transfer statistics:
        - 'predictions_retrieved': Number of predictions from source
        - 'items_transferred': Number of items observed in target
        - 'field': Field that was transferred
        - 'modeling_applied': Whether modeling function was used

    Examples:
        # Example 1: Transfer 'name' field filtered by normalized_potential threshold
        def threshold_filter(predictions, field):
            return [p['name'] for p in predictions if p.get('potential', 0) > 0.5]

        transfer_predictions(node0, node1, 'name', modeling_function=threshold_filter)

        # Example 2: Transfer 'matches' weighted by potential
        def weighted_matches(predictions, field):
            weighted = []
            for pred in predictions:
                weight = pred.get('potential', 0)
                matches = pred.get('matches', [])
                # Repeat matches based on weight (simple weighting)
                repeat_count = int(weight * 10)  # Scale to reasonable repeat count
                weighted.extend(matches * repeat_count)
            return weighted

        transfer_predictions(node0, node1, 'matches', modeling_function=weighted_matches)

        # Example 3: Transfer most likely 'future' event using probabilities
        def select_best_future(predictions, field):
            if not predictions:
                return []
            # Sort by potential (similarity * predictive_information)
            best = max(predictions, key=lambda p: p.get('potential', 0))
            future = best.get('future', [])
            # Flatten future events and return
            return [symbol for event in future for symbol in event]

        transfer_predictions(node1, node2, 'future', modeling_function=select_best_future)

        # Example 4: Simple transfer without modeling (pass field as-is)
        transfer_predictions(node0, node1, 'present')
    """
    # Validate field parameter
    valid_fields = ['past', 'present', 'future', 'missing', 'matches', 'extras', 'name']
    if field not in valid_fields:
        raise ValueError(f"Invalid field '{field}'. Must be one of: {valid_fields}")

    print(f"\n[Transfer] {node_source.node_id} → {node_target.node_id}")
    print(f"  Field: {field}")
    print(f"  Modeling: {'Yes' if modeling_function else 'No'}")

    # Get predictions from source node
    predictions = node_source.get_predictions()

    if not predictions:
        print("  ✗ No predictions available from source node")
        return {
            'predictions_retrieved': 0,
            'items_transferred': 0,
            'field': field,
            'modeling_applied': modeling_function is not None
        }

    # Limit to top N predictions
    predictions = predictions[:num_predictions]
    print(f"  Retrieved {len(predictions)} predictions")

    # Apply modeling function if provided
    if modeling_function:
        try:
            items_to_transfer = modeling_function(predictions, field)
            print(f"  Modeling function produced {len(items_to_transfer)} items")
        except Exception as e:
            print(f"  ✗ Error in modeling function: {e}")
            return {
                'predictions_retrieved': len(predictions),
                'items_transferred': 0,
                'field': field,
                'modeling_applied': True,
                'error': str(e)
            }
    else:
        # Extract field directly from predictions
        items_to_transfer = []
        for pred in predictions:
            field_value = pred.get(field, None)

            if field_value is None:
                continue

            # Handle different field types
            if field == 'name':
                # name is a string
                items_to_transfer.append(field_value)
            elif field in ['matches', 'missing', 'extras']:
                # These are lists of strings
                items_to_transfer.extend(field_value)
            elif field in ['past', 'present', 'future']:
                # These are lists of events (lists of lists)
                # Flatten to list of strings
                for event in field_value:
                    items_to_transfer.extend(event)

        print(f"  Extracted {len(items_to_transfer)} items from '{field}' field")

    # Observe items in target node
    if not items_to_transfer:
        print("  ✗ No items to transfer")
        return {
            'predictions_retrieved': len(predictions),
            'items_transferred': 0,
            'field': field,
            'modeling_applied': modeling_function is not None
        }

    # Observe each item in the target node
    for item in items_to_transfer:
        node_target.observe(strings=[item])

    print(f"  ✓ Transferred {len(items_to_transfer)} items to target node")

    return {
        'predictions_retrieved': len(predictions),
        'items_transferred': len(items_to_transfer),
        'field': field,
        'modeling_applied': modeling_function is not None
    }


# ============================================================================
# EVALUATION & VISUALIZATION
# ============================================================================

def visualize_training_stats(nodes: Dict[str, KATOClient]):
    """
    Visualize training statistics across hierarchical nodes.
    """
    print("\n" + "="*80)
    print("TRAINING STATISTICS - ALL NODES")
    print("="*80)

    # Collect stats
    stats_data = {}
    for level, node in nodes.items():
        stats_data[level] = node.get_stats()

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('KATO Hierarchical Training Statistics', fontsize=16, fontweight='bold')

    # 1. Patterns Learned by Level
    levels = list(stats_data.keys())
    patterns_learned = [stats_data[level]['patterns_learned'] for level in levels]

    axes[0, 0].bar(levels, patterns_learned, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 0].set_title('Patterns Learned by Level', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Patterns')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(patterns_learned):
        axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom')

    # 2. Tokens Processed
    tokens_processed = [stats_data[level]['tokens_processed'] for level in levels]

    axes[0, 1].bar(levels, tokens_processed, color=['#96CEB4', '#FFEAA7', '#DFE6E9', '#74B9FF'])
    axes[0, 1].set_title('Tokens Processed by Level', fontweight='bold')
    axes[0, 1].set_ylabel('Number of Tokens')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(tokens_processed):
        axes[0, 1].text(i, v + max(tokens_processed)*0.02, str(v), ha='center', va='bottom')

    # 3. Observations Made
    observations = [stats_data[level]['observations'] for level in levels]

    axes[1, 0].bar(levels, observations, color=['#74B9FF', '#A29BFE', '#FD79A8', '#55EFC4'])
    axes[1, 0].set_title('Observations Made by Level', fontweight='bold')
    axes[1, 0].set_ylabel('Number of Observations')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(observations):
        axes[1, 0].text(i, v + 0.5, str(v), ha='center', va='bottom')

    # 4. Auto-Learns Triggered
    auto_learns = [stats_data[level]['auto_learns'] for level in levels]

    axes[1, 1].bar(levels, auto_learns, color=['#55EFC4', '#FDCB6E', '#E17055', '#FF6B6B'])
    axes[1, 1].set_title('Auto-Learns Triggered (Rolling STM)', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Auto-Learns')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(auto_learns):
        axes[1, 1].text(i, v + 0.5, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Print detailed stats
    print("\nDetailed Statistics:")
    print("-" * 80)
    for level, stats in stats_data.items():
        print(f"\n{level.upper()}:")
        print(f"  Observations: {stats['observations']}")
        print(f"  Tokens Processed: {stats['tokens_processed']}")
        print(f"  Patterns Learned: {stats['patterns_learned']}")
        print(f"  Auto-Learns: {stats['auto_learns']}")
        if stats['observations'] > 0:
            print(f"  Avg Tokens/Observation: {stats['tokens_processed']/stats['observations']:.2f}")


def test_predictions(node: KATOClient,
                    test_texts: List[str],
                    level_name: str = "Level 0",
                    token_processor: TokenProcessor = None):
    """
    Test predictions on sample texts.
    """
    if token_processor is None:
        token_processor = TokenProcessor()

    print("\n" + "="*80)
    print(f"TESTING PREDICTIONS - {level_name}")
    print("="*80)

    for i, test_text in enumerate(test_texts, 1):
        print(f"\n[Test {i}] Input: {test_text[:60]}...")

        # Tokenize and observe
        tokens = token_processor.tokenize_for_level0(test_text, max_tokens=10)

        if len(tokens) < 2:
            print("  ✗ Not enough tokens for prediction")
            continue

        # Clear STM and observe fresh
        node.clear_stm()
        # Observe each token individually to preserve sequence in STM
        for token in tokens:
            node.observe(token)

        # Get predictions
        predictions = node.get_predictions()

        if not predictions:
            print("  ✗ No predictions generated")
            continue

        # Show top 3 predictions
        print(f"  ✓ Generated {len(predictions)} predictions (showing top 3):")

        for j, pred in enumerate(predictions[:3], 1):
            pattern_name = pred.get('name', 'UNKNOWN')
            confidence = pred.get('confidence', 0.0)
            similarity = pred.get('similarity', 0.0)
            frequency = pred.get('frequency', 0)

            print(f"\n    Prediction {j}:")
            print(f"      Pattern: {pattern_name[:30]}...")
            print(f"      Confidence: {confidence:.3f}")
            print(f"      Similarity: {similarity:.3f}")
            print(f"      Frequency: {frequency}")

            # Show future predictions if available
            future = pred.get('future', [])
            if future:
                future_tokens = [token for event in future for token in event]
                print(f"      Future: {' '.join(future_tokens[:5])}")

    print("\n" + "="*80)


def show_node_learning(node: KATOClient):
    """Display comprehensive learning information for a KATO node."""

    print("\n" + "="*80)
    print(f"LEARNING SUMMARY FOR NODE: {node.node_id}")
    print("="*80)

    # 1. Statistics
    stats = node.get_stats()
    print("\n📊 Training Statistics:")
    print(f"   Observations made: {stats['observations']:,}")
    print(f"   Tokens processed: {stats['tokens_processed']:,}")
    print(f"   Patterns learned: {stats['patterns_learned']:,}")
    print(f"   Auto-learns triggered: {stats['auto_learns']:,}")

    # 2. Learned Patterns
    print(f"\n🧠 Learned Patterns ({len(node.patterns_learned)}):")
    if node.patterns_learned:
        print("   First 10 patterns:")
        for i, pattern in enumerate(node.patterns_learned[:10], 1):
            print(f"   {i}. {pattern}")
        if len(node.patterns_learned) > 10:
            print(f"   ... and {len(node.patterns_learned) - 10} more patterns")
    else:
        print("   No patterns learned yet")

    # 3. Current STM State
    stm = node.get_stm()
    print(f"\n💭 Short-Term Memory (STM): {len(stm)} events")
    if stm:
        print("   Recent STM events (last 5):")
        for i, event in enumerate(stm[-5:], 1):
            # Show first few tokens of each event
            tokens_str = ' '.join(event[:5])
            if len(event) > 5:
                tokens_str += f" ... ({len(event)} tokens total)"
            print(f"   {i}. {tokens_str}")
    else:
        print("   STM is empty")

    # 4. Current Predictions
    predictions = node.get_predictions()
    print(f"\n🔮 Current Predictions: {len(predictions)} available")
    if predictions:
        print("   Top 3 predictions:")
        for i, pred in enumerate(predictions[:3], 1):
            print(f"   {i}. Pattern: {pred.get('name', 'N/A')[:30]}...")
            print(f"      Confidence: {pred.get('confidence', 0):.3f}")
            print(f"      Frequency: {pred.get('frequency', 0)}")
    else:
        print("   No predictions available (STM may be empty)")

    print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# HIERARCHICAL CONCEPT LEARNING DEMONSTRATION
# ============================================================================

def demonstrate_hierarchical_learning():
    """
    Demonstrate hierarchical concept learning with a sample text.

    This example shows the complete workflow:
    1. Segment text into hierarchical structure
    2. Learn patterns at each level
    3. Track progress and display statistics
    """
    print("\n" + "="*80)
    print("HIERARCHICAL CONCEPT LEARNING DEMONSTRATION")
    print("="*80)

    # Sample text for demonstration
    sample_text = """
Chapter 1: Introduction to Artificial Intelligence

Artificial intelligence (AI) is transforming the world. Machine learning algorithms can now recognize patterns in vast amounts of data. Deep learning networks have achieved remarkable results in image recognition and natural language processing.

The field of AI has grown rapidly over the past decade. Researchers continue to push the boundaries of what's possible. New architectures and training methods emerge regularly.

Chapter 2: Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence. It focuses on algorithms that can learn from data without explicit programming. Supervised learning uses labeled examples to train models.

Unsupervised learning finds patterns in unlabeled data. Reinforcement learning trains agents through trial and error. Each approach has its own strengths and applications.
"""

    # Step 1: Segment the text
    print("\n[Step 1] Segmenting text into hierarchical structure...")
    segmenter = CorpusSegmenter()
    book = segmenter.segment_book(
        sample_text,
        book_metadata={'title': 'AI Primer', 'author': 'Demo'}
    )

    print(f"✓ Segmented into:")
    print(f"  - {len(book['chapters'])} chapters")
    total_paragraphs = sum(len(ch['paragraphs']) for ch in book['chapters'])
    total_sentences = sum(
        len(p['sentences'])
        for ch in book['chapters']
        for p in ch['paragraphs']
    )
    print(f"  - {total_paragraphs} paragraphs")
    print(f"  - {total_sentences} sentences")

    # Step 2: Initialize hierarchical learner
    print("\n[Step 2] Initializing hierarchical concept learner...")
    learner = HierarchicalConceptLearner(
        base_url="http://kato:8000",
        tokenizer_name="gpt2"
    )

    # Step 3: Learn the book hierarchically
    print("\n[Step 3] Learning book hierarchically...")
    corpus = {'books': [book]}
    learner.process_corpus(corpus, verbose=True)

    # Step 4: Display detailed statistics
    print("\n[Step 4] Detailed Statistics:")
    print("="*80)
    stats = learner.get_stats()
    print(f"\nLearning Progress:")
    print(f"  Sentences learned: {stats['sentences_learned']:,}")
    print(f"  Paragraphs learned: {stats['paragraphs_learned']:,}")
    print(f"  Chapters learned: {stats['chapters_learned']:,}")
    print(f"  Books learned: {stats['books_learned']:,}")
    print(f"  Elapsed time: {stats['elapsed_formatted']}")

    node_stats = learner.get_node_stats()
    print(f"\nNode-Level Statistics:")
    for level, nstats in node_stats.items():
        print(f"\n  {level.upper()}:")
        print(f"    Observations: {nstats['observations']:,}")
        print(f"    Patterns learned: {nstats['patterns_learned']:,}")
        print(f"    Tokens processed: {nstats['tokens_processed']:,}")

    # Step 5: Show pattern hierarchy
    print("\n[Step 5] Pattern Hierarchy Visualization:")
    print("="*80)
    print("\nPattern Names by Level:")
    for level in ['node0', 'node1', 'node2', 'node3']:
        patterns = stats['patterns_by_level'][level]
        print(f"\n  {level.upper()} ({len(patterns)} patterns):")
        # Show first 3 patterns
        for i, pattern in enumerate(patterns[:3], 1):
            print(f"    {i}. {pattern}")
        if len(patterns) > 3:
            print(f"    ... and {len(patterns) - 3} more")

    print("\n" + "="*80)
    print("✓ Demonstration complete!")
    print("="*80)

    return learner, book


def visualize_hierarchical_stats(learner: HierarchicalConceptLearner):
    """
    Visualize hierarchical learning statistics.

    Args:
        learner: HierarchicalConceptLearner instance with learned data
    """
    stats = learner.get_stats()
    node_stats = learner.get_node_stats()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hierarchical Concept Learning Statistics', fontsize=16, fontweight='bold')

    # 1. Concepts Learned by Level
    levels = ['Sentences', 'Paragraphs', 'Chapters', 'Books']
    counts = [
        stats['sentences_learned'],
        stats['paragraphs_learned'],
        stats['chapters_learned'],
        stats['books_learned']
    ]

    axes[0, 0].bar(levels, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 0].set_title('Concepts Learned by Level', fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(counts):
        axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom')

    # 2. Patterns Learned by Node
    nodes = ['Node0', 'Node1', 'Node2', 'Node3']
    patterns = [node_stats[f'node{i}']['patterns_learned'] for i in range(4)]

    axes[0, 1].bar(nodes, patterns, color=['#96CEB4', '#FFEAA7', '#DFE6E9', '#74B9FF'])
    axes[0, 1].set_title('Patterns Learned by Node', fontweight='bold')
    axes[0, 1].set_ylabel('Number of Patterns')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(patterns):
        axes[0, 1].text(i, v + 0.5, str(v), ha='center', va='bottom')

    # 3. Observations by Node
    observations = [node_stats[f'node{i}']['observations'] for i in range(4)]

    axes[1, 0].bar(nodes, observations, color=['#74B9FF', '#A29BFE', '#FD79A8', '#55EFC4'])
    axes[1, 0].set_title('Observations by Node', fontweight='bold')
    axes[1, 0].set_ylabel('Number of Observations')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(observations):
        axes[1, 0].text(i, v + 0.5, str(v), ha='center', va='bottom')

    # 4. Tokens Processed by Node
    tokens = [node_stats[f'node{i}']['tokens_processed'] for i in range(4)]

    axes[1, 1].bar(nodes, tokens, color=['#55EFC4', '#FDCB6E', '#E17055', '#FF6B6B'])
    axes[1, 1].set_title('Tokens Processed by Node', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Tokens')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(tokens):
        axes[1, 1].text(i, v + max(tokens)*0.02, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print("\n✓ Visualization displayed")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("KATO HIERARCHICAL LEARNING WITH DELIMITER-BASED STREAMING")
    print("="*80)
    print(f"Session started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script demonstrates delimiter-based hierarchical learning:")
    print("  - Stream datasets from HuggingFace Hub (no downloads)")
    print("  - Segment by delimiter (sentence, word, bigram, trigram, 4-gram, 5-gram, paragraph)")
    print("  - Tokenize each segment (GPT-2, BERT, RoBERTa, etc.)")
    print("  - Learn as complete patterns at node0")
    print("  - Optional hierarchical levels (pattern names → node1 → node2 → node3)")
    print("="*80)

    # ========================================================================
    # EXAMPLE 1: Stream WikiText with sentence delimiter
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 1: Stream WikiText with Sentence Delimiter")
    print("="*80)
    print("\nThis will:")
    print("  - Stream 100 samples from WikiText dataset")
    print("  - Segment each sample by sentence")
    print("  - Tokenize each sentence with GPT-2")
    print("  - Learn each tokenized sentence as a pattern at node0")
    print("\nUncomment to run:")
    print("-" * 80)

    # learner = HierarchicalConceptLearner(base_url="http://kato:8000", tokenizer_name="gpt2")
    # stats = learner.learn_from_stream(
    #     dataset_key='wikitext',
    #     max_samples=100,
    #     delimiter='sentence',
    #     hierarchical_levels=1,  # Just node0
    #     max_tokens_per_segment=512
    # )
    # print(f"\n✓ Learned {stats['sentences_learned']} sentences")

    # ========================================================================
    # EXAMPLE 2: Stream with bigram delimiter
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 2: Stream with Bigram Delimiter")
    print("="*80)
    print("\nThis will:")
    print("  - Stream samples from WikiText")
    print("  - Segment by bigrams (2-word sequences)")
    print("  - Tokenize each bigram with GPT-2")
    print("  - Learn each as a pattern at node0")
    print("\nUncomment to run:")
    print("-" * 80)

    # learner = HierarchicalConceptLearner(tokenizer_name="gpt2")
    # stats = learner.learn_from_stream(
    #     dataset_key='wikitext',
    #     max_samples=100,
    #     delimiter='bigram',
    #     hierarchical_levels=1
    # )

    # ========================================================================
    # EXAMPLE 3: Multi-level hierarchy (node0 → node1)
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 3: Multi-Level Hierarchy")
    print("="*80)
    print("\nThis will:")
    print("  - Learn sentences at node0")
    print("  - Pass pattern names to node1")
    print("  - Learn higher-level patterns (batches of 100 sentence patterns)")
    print("\nUncomment to run:")
    print("-" * 80)

    # learner = HierarchicalConceptLearner(tokenizer_name="gpt2")
    # stats = learner.learn_from_stream(
    #     dataset_key='wikitext',
    #     max_samples=1000,
    #     delimiter='sentence',
    #     hierarchical_levels=2,  # node0 + node1
    #     max_tokens_per_segment=256
    # )
    # print(f"\n✓ Node0 learned: {stats['sentences_learned']} patterns")
    # print(f"✓ Node1 learned: {stats['paragraphs_learned']} patterns")

    # ========================================================================
    # EXAMPLE 4: Learn from direct text (no streaming)
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 4: Learn from Direct Text")
    print("="*80)
    print("\nThis will:")
    print("  - Process text directly (no dataset streaming)")
    print("  - Segment by delimiter")
    print("  - Learn hierarchically")
    print("\nUncomment to run:")
    print("-" * 80)

    # sample_text = """
    # Artificial intelligence is transforming the world. Machine learning algorithms
    # can recognize patterns in vast amounts of data. Deep learning networks achieve
    # remarkable results in image recognition and natural language processing.
    # """
    #
    # learner = HierarchicalConceptLearner(tokenizer_name="gpt2")
    # stats = learner.learn_from_text(
    #     text=sample_text,
    #     delimiter='sentence',
    #     hierarchical_levels=1,
    #     max_tokens_per_segment=512
    # )
    # print(f"\n✓ Learned {stats['sentences_learned']} sentences from text")

    # ========================================================================
    # EXAMPLE 5: With checkpointing (long-running jobs)
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 5: Streaming with Checkpointing")
    print("="*80)
    print("\nThis will:")
    print("  - Stream large dataset (C4, RefinedWeb, etc.)")
    print("  - Save checkpoints every N segments")
    print("  - Resume from checkpoint if interrupted")
    print("\nUncomment to run:")
    print("-" * 80)

    # learner = HierarchicalConceptLearner(tokenizer_name="gpt2")
    # stats = learner.learn_from_stream(
    #     dataset_key='c4',
    #     max_samples=10000,
    #     delimiter='paragraph',
    #     hierarchical_levels=1,
    #     checkpoint_every=1000,  # Save every 1000 segments
    #     checkpoint_dir="./checkpoints"
    # )

    # ========================================================================
    # EXAMPLE 6: Model prediction ensembles and transfer to another node
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 6: Model Prediction Ensembles")
    print("="*80)
    print("\nThis demonstrates how to:")
    print("  - Get predictions from node0")
    print("  - Model the ensemble (filter, weight, select)")
    print("  - Transfer to node1 for further processing")
    print("\nUncomment to run:")
    print("-" * 80)

    # # First, learn some patterns
    # learner = HierarchicalConceptLearner(tokenizer_name="gpt2")
    # stats = learner.learn_from_stream(
    #     dataset_key='wikitext',
    #     max_samples=100,
    #     delimiter='sentence',
    #     hierarchical_levels=1
    # )
    #
    # # Define a modeling function
    # def select_high_potential_futures(predictions, field):
    #     """Select future events from high-potential predictions."""
    #     futures = []
    #     for pred in predictions:
    #         # Filter by potential (similarity × predictive_information)
    #         if pred.get('potential', 0) > 0.4:
    #             future_events = pred.get('future', [])
    #             for event in future_events:
    #                 futures.extend(event)
    #     return futures
    #
    # # Transfer modeled predictions
    # transfer_predictions(
    #     node_source=learner.nodes['node0'],
    #     node_target=learner.nodes['node1'],
    #     field='future',
    #     modeling_function=select_high_potential_futures,
    #     num_predictions=10
    # )

    # ========================================================================
    # EXAMPLE 7: Deep Hierarchy with More Than 4 Nodes
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 7: Deep Hierarchy (10 Nodes)")
    print("="*80)
    print("\nThis demonstrates:")
    print("  - Create 10 hierarchical nodes instead of default 4")
    print("  - Learn patterns across 10 levels of abstraction")
    print("  - Each level learns from pattern names of previous level")
    print("\nUncomment to run:")
    print("-" * 80)

    # # Create learner with 10 nodes
    # learner = HierarchicalConceptLearner(
    #     tokenizer_name="gpt2",
    #     num_nodes=10  # Deep hierarchy!
    # )
    #
    # # Stream and learn across 10 levels
    # stats = learner.learn_from_stream(
    #     dataset_key='wikitext',
    #     max_samples=1000,
    #     delimiter='sentence',
    #     hierarchical_levels=10,  # All 10 levels
    #     max_tokens_per_segment=256
    # )
    #
    # # Print stats for each level
    # for i in range(10):
    #     count = stats.get(f'node{i}_patterns', 0)
    #     print(f"node{i}: {count} patterns learned")

    # ========================================================================
    # ADDITIONAL EXAMPLES
    # ========================================================================
    print("\n" + "="*80)
    print("ADDITIONAL EXAMPLES")
    print("="*80)
    print("\nDifferent delimiters:")
    print("  delimiter='word'       → Each word is a segment")
    print("  delimiter='bigram'     → 2-word sequences")
    print("  delimiter='trigram'    → 3-word sequences")
    print("  delimiter='4-gram'     → 4-word sequences")
    print("  delimiter='5-gram'     → 5-word sequences")
    print("  delimiter='sentence'   → Full sentences")
    print("  delimiter='paragraph'  → Full paragraphs")
    print("\nDifferent tokenizers:")
    print("  tokenizer_name='gpt2'                → GPT-2 BPE")
    print("  tokenizer_name='bert-base-uncased'   → BERT WordPiece")
    print("  tokenizer_name='roberta-base'        → RoBERTa BPE")
    print("  tokenizer_name='t5-small'            → T5")
    print("  tokenizer_name='meta-llama/Llama-2-7b-hf' → LLaMA 2 (requires auth)")
    print("\nAvailable datasets:")
    print("  'wikitext', 'openwebtext', 'c4', 'refinedweb', 'bookcorpus', 'pile'")

    # ========================================================================
    # LEGACY: Manual segmentation approach (from demonstrate_hierarchical_learning)
    # ========================================================================
    print("\n" + "="*80)
    print("LEGACY: Manual Segmentation Approach")
    print("="*80)
    print("\nFor manually pre-segmented text, use:")
    print("  1. CorpusSegmenter to segment book/article structure")
    print("  2. learner.process_corpus() for traditional hierarchical learning")
    print("\nUncomment to run:")
    print("-" * 80)

    # learner, book = demonstrate_hierarchical_learning()
    # visualize_hierarchical_stats(learner)

    print("\n" + "="*80)
    print("QUICK START")
    print("="*80)
    print("\nTo get started:")
    print("  1. Uncomment one of the examples above")
    print("  2. Ensure KATO server is running at http://kato:8000")
    print("  3. Run this script")
    print("\nFor more control, create your own learner:")
    print("  # Default (4 nodes)")
    print("  learner = HierarchicalConceptLearner(tokenizer_name='gpt2')")
    print("")
    print("  # Deep hierarchy (custom number of nodes)")
    print("  learner = HierarchicalConceptLearner(tokenizer_name='gpt2', num_nodes=10)")
    print("")
    print("  # Then stream and learn")
    print("  stats = learner.learn_from_stream(")
    print("      dataset_key='wikitext',")
    print("      max_samples=1000,")
    print("      delimiter='sentence',")
    print("      hierarchical_levels=5  # Up to num_nodes")
    print("  )")
    print(f"\nSession timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
