"""
Prediction analysis for hierarchy metrics.

Computes prediction-related metric:
- Prediction Fan-Out (number of predictions per level)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import random
import time

from .storage import HierarchyGraphStorage
from .config import (
    PredictionMetrics,
    HealthStatus,
    ThresholdConfig,
)


class PredictionAnalyzer:
    """
    Analyzes prediction behavior of hierarchy.

    Tests prediction cascade to measure:
    - Fan-out at each level (# of predictions)
    - Confidence distributions
    - Constraint effectiveness during generation

    Usage:
        analyzer = PredictionAnalyzer(learner, 'hierarchy_graph.db')
        pred_metrics = analyzer.compute_prediction_metrics(test_inputs)
    """

    def __init__(
        self,
        learner: Any,  # HierarchicalConceptLearner
        graph_db_path: Optional[str] = None,
        thresholds: Optional[ThresholdConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize analyzer.

        Args:
            learner: HierarchicalConceptLearner instance
            graph_db_path: Optional path to graph database
            thresholds: Optional threshold configuration
            verbose: Print debug information
        """
        self.learner = learner
        self.storage = HierarchyGraphStorage(graph_db_path, verbose=verbose) if graph_db_path else None
        self.thresholds = thresholds or ThresholdConfig()
        self.verbose = verbose

        # Extract node levels
        self.num_nodes = learner.num_nodes
        self.node_levels = [f"node{i}" for i in range(self.num_nodes)]

        if self.verbose:
            print(f"✓ PredictionAnalyzer initialized")
            print(f"  Node levels: {self.node_levels}")

    # ========================================================================
    # PREDICTION FAN-OUT
    # ========================================================================

    def compute_fanout_for_input(
        self,
        input_tokens: List[str],
        max_predictions: int = 100
    ) -> Dict[str, int]:
        """
        Compute prediction fan-out at each level for a single input.

        Args:
            input_tokens: Input token sequence
            max_predictions: Maximum predictions to request

        Returns:
            Dict mapping level → fan-out count
        """
        fanout = {}

        # Convert tokens to observations
        observations = [{'strings': [token]} for token in input_tokens]

        # Get predictions at each level
        for i, node in enumerate(self.learner.nodes):
            level = self.node_levels[i]

            try:
                # Get predictions (top_k)
                predictions = node.predict(
                    observations=observations,
                    top_k=max_predictions
                )

                # Count unique predictions
                if predictions and 'predictions' in predictions:
                    fanout[level] = len(predictions['predictions'])
                else:
                    fanout[level] = 0

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Prediction failed for {level}: {e}")
                fanout[level] = 0

        return fanout

    def compute_fanout_statistics(
        self,
        test_inputs: List[List[str]],
        max_predictions: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute fan-out statistics across multiple test inputs.

        Args:
            test_inputs: List of token sequences
            max_predictions: Maximum predictions per query

        Returns:
            Dict mapping level → {mean, std, min, max, median}
        """
        fanout_by_level = defaultdict(list)

        # Compute fan-out for each input
        for input_tokens in test_inputs:
            fanout = self.compute_fanout_for_input(input_tokens, max_predictions)

            for level, count in fanout.items():
                fanout_by_level[level].append(count)

        # Compute statistics
        stats = {}
        for level, counts in fanout_by_level.items():
            if counts:
                stats[level] = {
                    'mean': float(np.mean(counts)),
                    'std': float(np.std(counts)),
                    'min': int(np.min(counts)),
                    'max': int(np.max(counts)),
                    'median': float(np.median(counts)),
                }
            else:
                stats[level] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0,
                    'max': 0,
                    'median': 0.0,
                }

            if self.verbose:
                print(f"\n✓ Fan-out statistics for {level}:")
                print(f"  Mean: {stats[level]['mean']:.1f}")
                print(f"  Std: {stats[level]['std']:.1f}")
                print(f"  Range: [{stats[level]['min']}, {stats[level]['max']}]")

        return stats

    def compute_confidence_distributions(
        self,
        test_inputs: List[List[str]],
        max_predictions: int = 100
    ) -> Dict[str, List[float]]:
        """
        Compute confidence score distributions at each level.

        Args:
            test_inputs: List of token sequences
            max_predictions: Maximum predictions per query

        Returns:
            Dict mapping level → list of confidence scores
        """
        confidence_by_level = defaultdict(list)

        # Collect confidences for each input
        for input_tokens in test_inputs:
            observations = [{'strings': [token]} for token in input_tokens]

            for i, node in enumerate(self.learner.nodes):
                level = self.node_levels[i]

                try:
                    predictions = node.predict(
                        observations=observations,
                        top_k=max_predictions
                    )

                    if predictions and 'predictions' in predictions:
                        for pred in predictions['predictions']:
                            # Extract confidence if available
                            if 'confidence' in pred:
                                confidence_by_level[level].append(pred['confidence'])
                            elif 'probability' in pred:
                                confidence_by_level[level].append(pred['probability'])
                            elif 'score' in pred:
                                confidence_by_level[level].append(pred['score'])

                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Confidence extraction failed for {level}: {e}")

        # Convert to dict
        return dict(confidence_by_level)

    # ========================================================================
    # TEST INPUT GENERATION
    # ========================================================================

    def generate_test_inputs_from_graph(
        self,
        num_samples: int = 100,
        min_length: int = 5,
        max_length: int = 20
    ) -> List[List[str]]:
        """
        Generate test inputs by sampling from the graph.

        Args:
            num_samples: Number of test inputs to generate
            min_length: Minimum token length
            max_length: Maximum token length

        Returns:
            List of token sequences
        """
        if not self.storage:
            raise ValueError("No graph database provided. Cannot generate test inputs from graph.")

        test_inputs = []

        # Get node0 patterns (token-level)
        node0_patterns = self.storage.get_patterns_by_level('node0')

        if not node0_patterns:
            raise ValueError("No node0 patterns found in graph")

        # Sample patterns and reconstruct token sequences
        for _ in range(num_samples):
            # Sample a random length
            length = random.randint(min_length, max_length)

            # Sample random patterns
            sampled_patterns = random.sample(node0_patterns, min(length, len(node0_patterns)))

            # For now, use pattern IDs as proxy tokens
            # (In real implementation, would need to unravel to actual tokens)
            tokens = [p['pattern_id'] for p in sampled_patterns]

            test_inputs.append(tokens)

        if self.verbose:
            print(f"✓ Generated {num_samples} test inputs from graph")

        return test_inputs

    def generate_test_inputs_random(
        self,
        num_samples: int = 100,
        vocab: Optional[List[str]] = None,
        min_length: int = 5,
        max_length: int = 20
    ) -> List[List[str]]:
        """
        Generate random test inputs from vocabulary.

        Args:
            num_samples: Number of test inputs
            vocab: Optional vocabulary (default: common English words)
            min_length: Minimum token length
            max_length: Maximum token length

        Returns:
            List of token sequences
        """
        if vocab is None:
            # Default vocabulary (common English words)
            vocab = [
                "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "can", "may", "might", "must", "shall", "to", "of", "in",
                "for", "on", "with", "at", "by", "from", "as", "about", "like",
                "through", "over", "before", "after", "between", "under", "above",
                "cat", "dog", "house", "car", "tree", "book", "water", "food",
                "person", "time", "day", "year", "way", "thing", "man", "woman",
                "child", "world", "life", "hand", "part", "place", "case", "work",
            ]

        test_inputs = []

        for _ in range(num_samples):
            length = random.randint(min_length, max_length)
            tokens = random.choices(vocab, k=length)
            test_inputs.append(tokens)

        if self.verbose:
            print(f"✓ Generated {num_samples} random test inputs")

        return test_inputs

    # ========================================================================
    # COMPREHENSIVE ANALYSIS
    # ========================================================================

    def compute_prediction_metrics(
        self,
        test_inputs: Optional[List[List[str]]] = None,
        num_samples: int = 100,
        max_predictions: int = 100,
        collect_confidences: bool = False
    ) -> PredictionMetrics:
        """
        Compute all prediction-related metrics.

        Args:
            test_inputs: Optional test inputs (generated if not provided)
            num_samples: Number of test samples if generating
            max_predictions: Maximum predictions per query
            collect_confidences: Whether to collect confidence distributions

        Returns:
            PredictionMetrics dataclass with health scoring
        """
        # Generate test inputs if not provided
        if test_inputs is None:
            if self.storage:
                try:
                    test_inputs = self.generate_test_inputs_from_graph(num_samples)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Graph-based generation failed: {e}")
                    test_inputs = self.generate_test_inputs_random(num_samples)
            else:
                test_inputs = self.generate_test_inputs_random(num_samples)

        # Compute fan-out statistics
        fanout_stats = self.compute_fanout_statistics(test_inputs, max_predictions)

        # Optionally collect confidence distributions
        confidence_dists = None
        if collect_confidences:
            confidence_dists = self.compute_confidence_distributions(test_inputs, max_predictions)

        # Health scoring
        health_status = {}

        for level, stats in fanout_stats.items():
            mean_fanout = stats['mean']

            # Check against thresholds
            low, high = self.thresholds.fanout_excellent
            if low <= mean_fanout <= high:
                health = HealthStatus.EXCELLENT
            else:
                low, high = self.thresholds.fanout_good
                if low <= mean_fanout <= high:
                    health = HealthStatus.GOOD
                else:
                    low, high = self.thresholds.fanout_warning
                    if low <= mean_fanout <= high:
                        health = HealthStatus.WARNING
                    else:
                        health = HealthStatus.POOR

            health_status[f"fanout_{level}"] = health

        return PredictionMetrics(
            fanout_by_level=fanout_stats,
            confidence_distributions=confidence_dists,
            health_status=health_status
        )

    def analyze_all(
        self,
        test_inputs: Optional[List[List[str]]] = None,
        num_samples: int = 100
    ) -> PredictionMetrics:
        """
        Run all prediction analyses.

        Args:
            test_inputs: Optional test inputs
            num_samples: Number of samples if generating

        Returns:
            PredictionMetrics dataclass
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("PREDICTION ANALYSIS")
            print(f"{'='*80}\n")

        metrics = self.compute_prediction_metrics(
            test_inputs=test_inputs,
            num_samples=num_samples,
            collect_confidences=True
        )

        if self.verbose:
            print(f"\n{'='*80}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*80}\n")

        return metrics

    def close(self):
        """Close storage connection"""
        if self.storage:
            self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
