"""
Information-theoretic analysis for hierarchy metrics.

Computes 3 information theory metrics:
1. Mutual Information between adjacent levels
2. Conditional Entropy H(X|Y)
3. Constraint Effectiveness (normalized MI)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter
import math

from .storage import HierarchyGraphStorage
from .config import (
    InformationMetrics,
    HealthStatus,
    ThresholdConfig,
)


class InformationTheoryAnalyzer:
    """
    Analyzes information-theoretic properties of hierarchy.

    Computes mutual information, conditional entropy, and constraint
    effectiveness between adjacent levels.

    Usage:
        analyzer = InformationTheoryAnalyzer('hierarchy_graph.db')
        info_metrics = analyzer.compute_information_metrics()
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
        self.node_levels = self.storage.get_metadata('node_levels') or []

        if self.verbose:
            print(f"✓ InformationTheoryAnalyzer initialized")
            print(f"  Node levels: {self.node_levels}")

    # ========================================================================
    # ENTROPY CALCULATIONS
    # ========================================================================

    def _compute_entropy(self, frequencies: List[int]) -> float:
        """
        Compute Shannon entropy from frequency distribution.

        H(X) = -Σ p(x) log₂ p(x)

        Args:
            frequencies: List of frequency counts

        Returns:
            Entropy in bits
        """
        if not frequencies:
            return 0.0

        total = sum(frequencies)
        if total == 0:
            return 0.0

        probs = [f / total for f in frequencies if f > 0]
        return -sum(p * math.log2(p) for p in probs)

    def compute_marginal_entropy(self, level: str) -> float:
        """
        Compute marginal entropy H(X) for a level.

        Args:
            level: Node level (e.g., 'node0', 'node1')

        Returns:
            Entropy in bits
        """
        patterns = self.storage.get_patterns_by_level(level)

        if not patterns:
            return 0.0

        frequencies = [p['frequency'] for p in patterns]
        return self._compute_entropy(frequencies)

    def compute_joint_distribution(
        self,
        lower_level: str,
        upper_level: str
    ) -> Dict[Tuple[str, str], int]:
        """
        Compute joint distribution P(X, Y) between adjacent levels.

        Args:
            lower_level: Child level (e.g., 'node0')
            upper_level: Parent level (e.g., 'node1')

        Returns:
            Dict mapping (lower_pattern, upper_pattern) → co-occurrence count
        """
        joint_dist = defaultdict(int)

        # Get all patterns at upper level
        upper_patterns = self.storage.get_patterns_by_level(upper_level)

        for upper_pattern in upper_patterns:
            upper_id = upper_pattern['pattern_id']
            upper_freq = upper_pattern['frequency']

            # Get children of this upper pattern
            children = self.storage.get_children(upper_id)

            # For each child, increment joint count
            for child_id in children:
                joint_dist[(child_id, upper_id)] += upper_freq

        return dict(joint_dist)

    def compute_joint_entropy(
        self,
        lower_level: str,
        upper_level: str
    ) -> float:
        """
        Compute joint entropy H(X, Y) between adjacent levels.

        H(X,Y) = -Σ p(x,y) log₂ p(x,y)

        Args:
            lower_level: Child level
            upper_level: Parent level

        Returns:
            Joint entropy in bits
        """
        joint_dist = self.compute_joint_distribution(lower_level, upper_level)

        if not joint_dist:
            return 0.0

        frequencies = list(joint_dist.values())
        return self._compute_entropy(frequencies)

    def compute_conditional_entropy(
        self,
        lower_level: str,
        upper_level: str
    ) -> float:
        """
        Compute conditional entropy H(X|Y).

        H(X|Y) = H(X,Y) - H(Y)

        This measures how much uncertainty remains about X
        after observing Y.

        Lower values = Y constrains X more (good for hierarchy)

        Args:
            lower_level: X (child level)
            upper_level: Y (parent level)

        Returns:
            Conditional entropy in bits
        """
        H_XY = self.compute_joint_entropy(lower_level, upper_level)
        H_Y = self.compute_marginal_entropy(upper_level)

        return H_XY - H_Y

    # ========================================================================
    # MUTUAL INFORMATION
    # ========================================================================

    def compute_mutual_information(
        self,
        lower_level: str,
        upper_level: str
    ) -> float:
        """
        Compute mutual information I(X;Y) between adjacent levels.

        I(X;Y) = H(X) - H(X|Y)
              = H(Y) - H(Y|X)
              = H(X) + H(Y) - H(X,Y)

        Mutual information measures how much knowing Y
        reduces uncertainty about X.

        Higher values = stronger dependency (good for hierarchy)

        Args:
            lower_level: X (child level)
            upper_level: Y (parent level)

        Returns:
            Mutual information in bits
        """
        H_X = self.compute_marginal_entropy(lower_level)
        H_Y = self.compute_marginal_entropy(upper_level)
        H_XY = self.compute_joint_entropy(lower_level, upper_level)

        MI = H_X + H_Y - H_XY

        # Clamp to [0, min(H_X, H_Y)] due to numerical errors
        MI = max(0.0, min(MI, H_X, H_Y))

        return MI

    def compute_normalized_mutual_information(
        self,
        lower_level: str,
        upper_level: str
    ) -> float:
        """
        Compute normalized mutual information (constraint effectiveness).

        NMI = I(X;Y) / H(X)

        This measures what fraction of X's entropy is explained by Y.

        Values:
        - 0.0: X and Y are independent (no constraint)
        - 1.0: Y completely determines X (perfect constraint)

        Args:
            lower_level: X (child level)
            upper_level: Y (parent level)

        Returns:
            Normalized MI (0-1)
        """
        MI = self.compute_mutual_information(lower_level, upper_level)
        H_X = self.compute_marginal_entropy(lower_level)

        if H_X == 0:
            return 0.0

        return MI / H_X

    # ========================================================================
    # ENTROPY PROGRESSION
    # ========================================================================

    def compute_entropy_progression(self) -> Dict[str, float]:
        """
        Compute marginal entropy at each level.

        Entropy should generally decrease up the hierarchy
        (more concentrated distributions at higher levels).

        Returns:
            Dict mapping level → entropy
        """
        entropy_by_level = {}

        for level in self.node_levels:
            H = self.compute_marginal_entropy(level)
            entropy_by_level[level] = H

        if self.verbose:
            print(f"\n✓ Entropy progression:")
            for level in self.node_levels:
                H = entropy_by_level.get(level, 0)
                print(f"  {level}: {H:.2f} bits")

        return entropy_by_level

    # ========================================================================
    # COMPREHENSIVE ANALYSIS
    # ========================================================================

    def compute_information_metrics(self) -> InformationMetrics:
        """
        Compute all information-theoretic metrics.

        Returns:
            InformationMetrics dataclass with health scoring
        """
        mutual_information = {}
        conditional_entropy = {}
        constraint_effectiveness = {}
        entropy_progression = {}

        # Compute entropy at each level
        for level in self.node_levels:
            H = self.compute_marginal_entropy(level)
            entropy_progression[level] = H

        # Compute MI, conditional entropy, and effectiveness between adjacent levels
        for i in range(len(self.node_levels) - 1):
            lower_level = self.node_levels[i]
            upper_level = self.node_levels[i + 1]

            pair_key = f"{lower_level}-{upper_level}"

            # Mutual information
            MI = self.compute_mutual_information(lower_level, upper_level)
            mutual_information[pair_key] = MI

            # Conditional entropy
            H_cond = self.compute_conditional_entropy(lower_level, upper_level)
            conditional_entropy[f"H({lower_level}|{upper_level})"] = H_cond

            # Constraint effectiveness (normalized MI)
            effectiveness = self.compute_normalized_mutual_information(lower_level, upper_level)
            constraint_effectiveness[pair_key] = effectiveness

            if self.verbose:
                print(f"\n✓ Information metrics for {pair_key}:")
                print(f"  Mutual information: {MI:.3f} bits")
                print(f"  Conditional entropy: {H_cond:.3f} bits")
                print(f"  Constraint effectiveness: {effectiveness:.1%}")

        # Health scoring
        health_status = {}

        # MI health (use thresholds)
        for pair, mi_value in mutual_information.items():
            # Normalize by entropy to get 0-1 scale
            lower_level = pair.split('-')[0]
            H_lower = entropy_progression.get(lower_level, 1.0)

            normalized_mi = mi_value / H_lower if H_lower > 0 else 0.0

            low, high = self.thresholds.mi_excellent
            if low <= normalized_mi <= high:
                health = HealthStatus.EXCELLENT
            else:
                low, high = self.thresholds.mi_good
                if low <= normalized_mi <= high:
                    health = HealthStatus.GOOD
                else:
                    low, high = self.thresholds.mi_warning
                    if low <= normalized_mi <= high:
                        health = HealthStatus.WARNING
                    else:
                        health = HealthStatus.POOR

            health_status[f"mi_{pair}"] = health

        # Constraint effectiveness health
        for pair, effectiveness in constraint_effectiveness.items():
            health = self.thresholds.evaluate_metric('constraint_effectiveness', effectiveness)
            health_status[f"effectiveness_{pair}"] = health

        return InformationMetrics(
            mutual_information=mutual_information,
            conditional_entropy=conditional_entropy,
            constraint_effectiveness=constraint_effectiveness,
            entropy_progression=entropy_progression,
            health_status=health_status
        )

    def analyze_all(self) -> InformationMetrics:
        """
        Run all information-theoretic analyses.

        Returns:
            InformationMetrics dataclass
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("INFORMATION-THEORETIC ANALYSIS")
            print(f"{'='*80}\n")

        metrics = self.compute_information_metrics()

        if self.verbose:
            print(f"\n{'='*80}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*80}\n")

        return metrics

    def close(self):
        """Close storage connection"""
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
