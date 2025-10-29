"""
Graph topology analyzer for hierarchy metrics.

Computes 10 core metrics from the persisted graph structure:
1. Compression Ratio (per level)
2. Pattern Count Progression
3. Effective Compression Rate
4. Pattern Reusability (parent count distribution)
5. Coverage (bottom-up utilization)
6. Branching Factor Statistics
7. Entropy Progression (basic calculation)
8. Context Window Alignment
9. Pattern Diversity (edit distance sampling)
10. Co-Occurrence Validation
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter
import random
import math

from .storage import HierarchyGraphStorage
from .config import (
    CompressionMetrics,
    ConnectivityMetrics,
    ContextMetrics,
    HealthStatus,
    ThresholdConfig,
)


class GraphAnalyzer:
    """
    Analyzes persisted hierarchy graph structure.

    Computes graph topology metrics, connectivity patterns,
    and structural health indicators.

    Usage:
        analyzer = GraphAnalyzer('hierarchy_graph.db')
        compression = analyzer.compute_compression_metrics()
        connectivity = analyzer.compute_connectivity_metrics()
        context = analyzer.compute_context_metrics()
    """

    def __init__(
        self,
        graph_db_path: str,
        thresholds: Optional[ThresholdConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize analyzer.

        Args:
            graph_db_path: Path to SQLite database
            thresholds: Optional threshold configuration
            verbose: Print debug information
        """
        self.storage = HierarchyGraphStorage(graph_db_path, verbose=verbose)
        self.thresholds = thresholds or ThresholdConfig()
        self.verbose = verbose

        # Load metadata
        self.chunk_sizes = self.storage.get_metadata('chunk_sizes') or []
        self.node_levels = self.storage.get_metadata('node_levels') or []
        self.samples_processed = self.storage.get_metadata('samples_processed') or 0

        if self.verbose:
            print(f"✓ GraphAnalyzer initialized")
            print(f"  Node levels: {self.node_levels}")
            print(f"  Chunk sizes: {self.chunk_sizes}")

    # ========================================================================
    # COMPRESSION METRICS (Metrics 1-3)
    # ========================================================================

    def compute_compression_ratios(self) -> Dict[str, float]:
        """
        Compute compression ratio at each level transition.

        Compression ratio = num_patterns[i] / num_patterns[i+1]

        Returns:
            Dict mapping level_pair → ratio
            E.g., {'node0->node1': 5.2, 'node1->node2': 4.8}
        """
        pattern_counts = self.storage.get_pattern_count_by_level()

        ratios = {}
        for i in range(len(self.node_levels) - 1):
            lower_level = self.node_levels[i]
            upper_level = self.node_levels[i + 1]

            lower_count = pattern_counts.get(lower_level, 0)
            upper_count = pattern_counts.get(upper_level, 0)

            if upper_count > 0:
                ratio = lower_count / upper_count
                ratios[f"{lower_level}->{upper_level}"] = ratio
            else:
                ratios[f"{lower_level}->{upper_level}"] = 0.0

        if self.verbose:
            print(f"\n✓ Compression ratios:")
            for pair, ratio in ratios.items():
                print(f"  {pair}: {ratio:.2f}x")

        return ratios

    def compute_pattern_counts(self) -> Dict[str, int]:
        """
        Get pattern count at each level.

        Returns:
            Dict mapping level → count
            E.g., {'node0': 150000, 'node1': 30000, 'node2': 6000}
        """
        counts = self.storage.get_pattern_count_by_level()

        if self.verbose:
            print(f"\n✓ Pattern counts:")
            for level in self.node_levels:
                count = counts.get(level, 0)
                print(f"  {level}: {count:,}")

        return counts

    def compute_effective_compression_rates(self) -> Dict[str, float]:
        """
        Compute effective compression rate (observations / unique patterns).

        This measures how much deduplication is happening.
        Higher values = more pattern reuse.

        Returns:
            Dict mapping level → compression_rate
        """
        pattern_counts = self.storage.get_pattern_count_by_level()

        # For now, use samples_processed as proxy for total observations
        # In future, could track actual observation counts per level
        rates = {}

        # node0: all tokens seen
        if 'node0' in pattern_counts:
            # Estimate: samples_processed * chunk_size
            chunk_size = self.chunk_sizes[0] if self.chunk_sizes else 15
            total_tokens = self.samples_processed * chunk_size
            unique_patterns = pattern_counts['node0']

            if unique_patterns > 0:
                rates['node0'] = total_tokens / unique_patterns

        # Upper levels: harder to estimate without tracking
        # For now, just report ratio to next level down
        for i in range(1, len(self.node_levels)):
            level = self.node_levels[i]
            lower_level = self.node_levels[i - 1]

            level_count = pattern_counts.get(level, 0)
            lower_count = pattern_counts.get(lower_level, 0)

            if level_count > 0:
                rates[level] = lower_count / level_count

        if self.verbose:
            print(f"\n✓ Effective compression rates:")
            for level, rate in rates.items():
                print(f"  {level}: {rate:.2f}x")

        return rates

    def compute_compression_metrics(self) -> CompressionMetrics:
        """
        Compute all compression-related metrics (1-3).

        Returns:
            CompressionMetrics dataclass with health scoring
        """
        compression_ratios = self.compute_compression_ratios()
        pattern_counts = self.compute_pattern_counts()
        compression_rates = self.compute_effective_compression_rates()

        # Health scoring
        health_status = {}

        for pair, ratio in compression_ratios.items():
            # Extract level index to get expected chunk_size
            level_idx = int(pair.split('->')[0].replace('node', ''))
            chunk_size = self.chunk_sizes[level_idx] if level_idx < len(self.chunk_sizes) else None

            health = self.thresholds.evaluate_metric(
                'compression_ratio',
                ratio,
                chunk_size=chunk_size
            )
            health_status[f"compression_{pair}"] = health

        return CompressionMetrics(
            compression_ratios=compression_ratios,
            pattern_counts=pattern_counts,
            effective_compression_rates=compression_rates,
            health_status=health_status
        )

    # ========================================================================
    # CONNECTIVITY METRICS (Metrics 4-6)
    # ========================================================================

    def compute_reusability(self, sample_size: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute pattern reusability statistics (parent count distribution).

        For each level, compute:
        - Mean parent count (how many parents use each child)
        - Median parent count
        - Orphan rate (% with 0 parents)
        - 90th percentile parent count

        Args:
            sample_size: Optional sample size for large graphs

        Returns:
            Dict mapping level → stats dict
        """
        reusability = {}

        for level in self.node_levels:
            # Get parent count distribution
            parent_dist = self.storage.get_parent_count_distribution(level)

            if not parent_dist:
                reusability[level] = {
                    'mean_parents': 0,
                    'median_parents': 0,
                    'orphan_rate': 0,
                    'p90_parents': 0,
                    'total_patterns': 0,
                }
                continue

            # Expand distribution to list of counts
            parent_counts = []
            for parent_count, pattern_count in parent_dist.items():
                parent_counts.extend([parent_count] * pattern_count)

            total_patterns = len(parent_counts)
            orphan_count = parent_dist.get(0, 0)

            if parent_counts:
                mean_parents = np.mean(parent_counts)
                median_parents = np.median(parent_counts)
                p90_parents = np.percentile(parent_counts, 90)
            else:
                mean_parents = median_parents = p90_parents = 0

            orphan_rate = orphan_count / total_patterns if total_patterns > 0 else 0

            reusability[level] = {
                'mean_parents': float(mean_parents),
                'median_parents': float(median_parents),
                'orphan_rate': float(orphan_rate),
                'p90_parents': float(p90_parents),
                'total_patterns': total_patterns,
            }

            if self.verbose:
                print(f"\n✓ Reusability for {level}:")
                print(f"  Mean parents: {mean_parents:.2f}")
                print(f"  Median parents: {median_parents:.2f}")
                print(f"  Orphan rate: {orphan_rate:.1%}")
                print(f"  90th percentile: {p90_parents:.0f}")

        return reusability

    def compute_coverage(self) -> Dict[str, float]:
        """
        Compute coverage: % of lower-level patterns used in upper level.

        Coverage measures bottom-up utilization.
        High coverage = most lower patterns are actually used.

        Returns:
            Dict mapping level_pair → coverage_rate
            E.g., {'node0->node1': 0.85} means 85% of node0 patterns used in node1
        """
        coverage = {}

        for i in range(len(self.node_levels) - 1):
            child_level = self.node_levels[i]
            parent_level = self.node_levels[i + 1]

            coverage_rate = self.storage.get_coverage(child_level, parent_level)
            coverage[f"{child_level}->{parent_level}"] = coverage_rate

        if self.verbose:
            print(f"\n✓ Coverage:")
            for pair, rate in coverage.items():
                print(f"  {pair}: {rate:.1%}")

        return coverage

    def compute_branching_factors(self) -> Dict[str, Dict[str, float]]:
        """
        Compute branching factor statistics (children per parent).

        For each level, compute:
        - Mean children per parent
        - Std dev
        - Coefficient of variation (CV)
        - Min/max

        Low CV = consistent branching (good)
        High CV = highly variable branching (may indicate issues)

        Returns:
            Dict mapping level → branching_stats
        """
        branching = {}

        for i in range(len(self.node_levels) - 1):
            parent_level = self.node_levels[i + 1]

            # Get all patterns at parent level
            parent_patterns = self.storage.get_patterns_by_level(parent_level)

            if not parent_patterns:
                branching[parent_level] = {
                    'mean_children': 0,
                    'std_children': 0,
                    'cv': 0,
                    'min_children': 0,
                    'max_children': 0,
                }
                continue

            # Count children for each parent
            child_counts = []
            for parent in parent_patterns:
                children = self.storage.get_children(parent['pattern_id'])
                child_counts.append(len(children))

            if child_counts:
                mean_children = np.mean(child_counts)
                std_children = np.std(child_counts)
                cv = std_children / mean_children if mean_children > 0 else 0
                min_children = np.min(child_counts)
                max_children = np.max(child_counts)
            else:
                mean_children = std_children = cv = min_children = max_children = 0

            branching[parent_level] = {
                'mean_children': float(mean_children),
                'std_children': float(std_children),
                'cv': float(cv),
                'min_children': int(min_children),
                'max_children': int(max_children),
            }

            if self.verbose:
                print(f"\n✓ Branching factors for {parent_level}:")
                print(f"  Mean children: {mean_children:.2f}")
                print(f"  Std dev: {std_children:.2f}")
                print(f"  CV: {cv:.2f}")

        return branching

    def compute_connectivity_metrics(self) -> ConnectivityMetrics:
        """
        Compute all connectivity-related metrics (4-6).

        Returns:
            ConnectivityMetrics dataclass with health scoring
        """
        reusability = self.compute_reusability()
        coverage = self.compute_coverage()
        branching = self.compute_branching_factors()

        # Health scoring
        health_status = {}

        # Orphan rate health
        for level, stats in reusability.items():
            orphan_rate = stats['orphan_rate']
            health = self.thresholds.evaluate_metric('orphan_rate', orphan_rate)
            health_status[f"orphan_{level}"] = health

        # Coverage health
        for pair, cov_rate in coverage.items():
            health = self.thresholds.evaluate_metric('coverage', cov_rate)
            health_status[f"coverage_{pair}"] = health

        # Branching CV health (lower is better)
        for level, stats in branching.items():
            cv = stats['cv']
            if cv <= self.thresholds.branching_cv_excellent:
                health = HealthStatus.EXCELLENT
            elif cv <= self.thresholds.branching_cv_good:
                health = HealthStatus.GOOD
            elif cv <= self.thresholds.branching_cv_warning:
                health = HealthStatus.WARNING
            else:
                health = HealthStatus.POOR
            health_status[f"branching_{level}"] = health

        return ConnectivityMetrics(
            reusability=reusability,
            coverage=coverage,
            branching_factors=branching,
            health_status=health_status
        )

    # ========================================================================
    # CONTEXT & COHERENCE METRICS (Metrics 7-10)
    # ========================================================================

    def compute_entropy_progression(self) -> Dict[str, float]:
        """
        Compute basic entropy at each level (histogram-based).

        Entropy H = -Σ p(x) log₂ p(x)

        Higher entropy = more uniform distribution
        Lower entropy = more concentrated (Zipfian)

        Returns:
            Dict mapping level → entropy
        """
        entropy = {}

        for level in self.node_levels:
            patterns = self.storage.get_patterns_by_level(level)

            if not patterns:
                entropy[level] = 0.0
                continue

            # Get frequency distribution
            frequencies = [p['frequency'] for p in patterns]
            total_freq = sum(frequencies)

            if total_freq == 0:
                entropy[level] = 0.0
                continue

            # Compute probabilities
            probs = [f / total_freq for f in frequencies]

            # Compute entropy
            H = -sum(p * math.log2(p) for p in probs if p > 0)

            entropy[level] = float(H)

        if self.verbose:
            print(f"\n✓ Entropy progression:")
            for level, H in entropy.items():
                print(f"  {level}: {H:.2f} bits")

        return entropy

    def compute_context_window_alignment(self) -> Dict[str, float]:
        """
        Compute context window alignment score.

        Measures how well pattern lengths align with expected context windows.

        For each level, check if mean branching factor ≈ chunk_size.

        Returns:
            Dict mapping level → alignment_score (0-1)
        """
        alignment = {}

        branching = self.compute_branching_factors()

        for i, level in enumerate(self.node_levels[1:], start=1):
            expected_chunk_size = self.chunk_sizes[i - 1] if i - 1 < len(self.chunk_sizes) else 15

            if level in branching:
                mean_children = branching[level]['mean_children']

                # Alignment score: 1 - (abs(mean - expected) / expected)
                # Perfect alignment = 1.0
                # 50% off = 0.5
                if expected_chunk_size > 0:
                    deviation = abs(mean_children - expected_chunk_size) / expected_chunk_size
                    score = max(0.0, 1.0 - deviation)
                else:
                    score = 0.0

                alignment[level] = float(score)
            else:
                alignment[level] = 0.0

        if self.verbose:
            print(f"\n✓ Context window alignment:")
            for level, score in alignment.items():
                print(f"  {level}: {score:.2f}")

        return alignment

    def compute_pattern_diversity(
        self,
        sample_size: int = 1000,
        metric: str = 'jaccard'
    ) -> Dict[str, float]:
        """
        Compute pattern diversity via sampling.

        Sample pairs of patterns at each level and compute similarity.

        Args:
            sample_size: Number of pairs to sample
            metric: 'jaccard' or 'levenshtein'

        Returns:
            Dict mapping level → mean_similarity (0-1)
        """
        diversity = {}

        for level in self.node_levels:
            patterns = self.storage.get_patterns_by_level(level)

            if len(patterns) < 2:
                diversity[level] = 0.0
                continue

            # Sample pairs
            pairs_to_sample = min(sample_size, len(patterns) * (len(patterns) - 1) // 2)

            similarities = []
            for _ in range(pairs_to_sample):
                # Random sample 2 patterns
                p1, p2 = random.sample(patterns, 2)

                # Get their children (or use pattern_id as proxy)
                children1 = set(self.storage.get_children(p1['pattern_id']))
                children2 = set(self.storage.get_children(p2['pattern_id']))

                if not children1 or not children2:
                    # Use pattern_id as proxy
                    children1 = set([p1['pattern_id']])
                    children2 = set([p2['pattern_id']])

                # Jaccard similarity
                if metric == 'jaccard':
                    if children1 or children2:
                        similarity = len(children1 & children2) / len(children1 | children2)
                    else:
                        similarity = 0.0
                else:
                    # Levenshtein (simplified - treat as set overlap for now)
                    similarity = len(children1 & children2) / max(len(children1), len(children2)) if children1 or children2 else 0.0

                similarities.append(similarity)

            mean_similarity = np.mean(similarities) if similarities else 0.0
            diversity[level] = float(mean_similarity)

        if self.verbose:
            print(f"\n✓ Pattern diversity (mean similarity):")
            for level, sim in diversity.items():
                print(f"  {level}: {sim:.3f}")

        return diversity

    def compute_cooccurrence_validation(
        self,
        sample_size: int = 100,
        window: int = 5
    ) -> Dict[str, float]:
        """
        Compute co-occurrence validation (coherence rate).

        For sampled patterns, check if their constituent children
        actually co-occur in the corpus (within a window).

        This validates that parent patterns represent real linguistic chunks,
        not arbitrary groupings.

        Args:
            sample_size: Number of patterns to sample per level
            window: Token window for co-occurrence

        Returns:
            Dict mapping level → coherence_rate (0-1)
        """
        coherence = {}

        # For now, we can't validate without access to raw corpus
        # Placeholder: assume coherence based on frequency distribution

        for i in range(1, len(self.node_levels)):
            level = self.node_levels[i]
            patterns = self.storage.get_patterns_by_level(level)

            if not patterns:
                coherence[level] = 0.0
                continue

            # Sample patterns
            sample = random.sample(patterns, min(sample_size, len(patterns)))

            coherent_count = 0
            for pattern in sample:
                children = self.storage.get_children(pattern['pattern_id'])

                # Heuristic: if pattern has children and frequency > 1, assume coherent
                # (Real validation would require corpus access)
                if len(children) > 0 and pattern['frequency'] > 1:
                    coherent_count += 1

            coherence_rate = coherent_count / len(sample) if sample else 0.0
            coherence[level] = float(coherence_rate)

        if self.verbose:
            print(f"\n✓ Co-occurrence validation (heuristic):")
            for level, rate in coherence.items():
                print(f"  {level}: {rate:.1%}")

        return coherence

    def compute_context_metrics(
        self,
        diversity_sample_size: int = 1000,
        coherence_sample_size: int = 100
    ) -> ContextMetrics:
        """
        Compute all context and coherence metrics (7-10).

        Args:
            diversity_sample_size: Pairs to sample for diversity
            coherence_sample_size: Patterns to sample for coherence

        Returns:
            ContextMetrics dataclass with health scoring
        """
        alignment = self.compute_context_window_alignment()
        diversity = self.compute_pattern_diversity(sample_size=diversity_sample_size)
        coherence = self.compute_cooccurrence_validation(sample_size=coherence_sample_size)

        # Health scoring
        health_status = {}

        # Alignment health (higher is better)
        for level, score in alignment.items():
            if score >= 0.9:
                health = HealthStatus.EXCELLENT
            elif score >= 0.8:
                health = HealthStatus.GOOD
            elif score >= 0.6:
                health = HealthStatus.WARNING
            else:
                health = HealthStatus.POOR
            health_status[f"alignment_{level}"] = health

        # Coherence health
        for level, rate in coherence.items():
            health = self.thresholds.evaluate_metric('coherence', rate)
            health_status[f"coherence_{level}"] = health

        return ContextMetrics(
            context_window_alignment=alignment,
            pattern_diversity=diversity,
            cooccurrence_validation=coherence,
            health_status=health_status
        )

    # ========================================================================
    # COMPREHENSIVE ANALYSIS
    # ========================================================================

    def analyze_all(self) -> Tuple[CompressionMetrics, ConnectivityMetrics, ContextMetrics]:
        """
        Run all graph analyses.

        Returns:
            (compression_metrics, connectivity_metrics, context_metrics)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("GRAPH ANALYSIS")
            print(f"{'='*80}\n")

        compression = self.compute_compression_metrics()
        connectivity = self.compute_connectivity_metrics()
        context = self.compute_context_metrics()

        if self.verbose:
            print(f"\n{'='*80}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*80}\n")

        return compression, connectivity, context

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.

        Returns:
            Dict with all metric categories
        """
        compression, connectivity, context = self.analyze_all()

        return {
            'compression': compression.to_dict(),
            'connectivity': connectivity.to_dict(),
            'context': context.to_dict(),
        }

    def close(self):
        """Close storage connection"""
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
