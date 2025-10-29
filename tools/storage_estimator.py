#!/usr/bin/env python3
"""
Storage Estimator - Zipfian-Based MongoDB Storage Prediction

This module estimates MongoDB storage requirements for hierarchical KATO training
using Zipfian distribution modeling. It accounts for:
- Pattern frequency distributions (Zipf's Law)
- Deduplication rates at each hierarchical level
- Pattern complexity and metadata overhead
- MongoDB document structure and indexing

Usage:
    estimator = StorageEstimator()
    estimate = estimator.estimate_storage(
        num_samples=1000000,
        config={'num_levels': 4, 'chunk_sizes': [15, 15, 15, 15]},
        dataset_stats={'avg_tokens_per_sample': 500}
    )
    print(f"Estimated storage: {estimate.total_storage_gb:.2f} GB")
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class LevelStorageEstimate:
    """Storage estimate for a single hierarchical level"""
    level: int
    level_name: str
    total_observations: int
    unique_patterns_estimated: int
    deduplication_rate: float  # What % of observations are deduplicated
    avg_pattern_size_bytes: int
    metadata_overhead_bytes: int
    total_storage_bytes: int
    total_storage_mb: float
    patterns_per_sample: float


@dataclass
class StorageEstimate:
    """Complete storage estimate across all levels"""
    configuration: Dict[str, Any]
    dataset_stats: Dict[str, Any]
    num_samples: int

    # Per-level estimates
    level_estimates: List[LevelStorageEstimate] = field(default_factory=list)

    # Totals
    total_patterns: int = 0
    total_storage_bytes: int = 0
    total_storage_mb: float = 0.0
    total_storage_gb: float = 0.0

    # MongoDB overhead
    mongodb_overhead_percent: float = 20.0  # Indexes, internal structures, padding
    estimated_storage_with_overhead_gb: float = 0.0

    # Zipfian parameters used
    zipf_alpha: float = 1.0


class StorageEstimator:
    """
    Estimate MongoDB storage requirements using Zipfian distribution modeling.

    Zipf's Law: In natural language, word/pattern frequencies follow a power law:
                frequency(rank) = C / rank^α
                where α ≈ 1.0 for text data

    This means:
    - A few patterns appear VERY frequently (high deduplication)
    - Most patterns appear rarely (low deduplication)
    - Higher levels have less deduplication than lower levels
    """

    def __init__(self, verbose: bool = False, auto_calibrate: bool = True, history_db: str = './training_history.db'):
        """
        Initialize storage estimator.

        Args:
            verbose: Print detailed estimation steps
            auto_calibrate: Automatically calibrate from training history
            history_db: Path to training_history.db for calibration
        """
        self.verbose = verbose
        self.history_db = history_db

        # MongoDB document structure sizes (bytes)
        self.PATTERN_NAME_SIZE = 40  # "PTRN|<40-char-sha1>"
        self.FREQUENCY_SIZE = 8      # Integer field
        self.METADATA_BASE_SIZE = 150  # Base metadata (timestamps, etc.)
        self.EVENT_OVERHEAD = 20      # Per-event in pattern_data
        self.SYMBOL_AVG_SIZE = 15     # Average symbol string length

        # Zipfian parameters
        self.DEFAULT_ZIPF_ALPHA = 1.0  # Power law exponent for natural language
        self.calibrated_zipf_alpha = None  # Will be set if calibration succeeds

        # Auto-calibrate if requested
        if auto_calibrate:
            self._auto_calibrate_from_history()

        if self.verbose:
            if self.calibrated_zipf_alpha:
                print(f"✓ StorageEstimator initialized (calibrated α={self.calibrated_zipf_alpha:.3f})")
            else:
                print("✓ StorageEstimator initialized (using default α=1.0)")

    def estimate_storage(
        self,
        num_samples: int,
        config: Dict[str, Any],
        dataset_stats: Dict[str, Any],
        zipf_alpha: Optional[float] = None
    ) -> StorageEstimate:
        """
        Estimate storage for training configuration.

        Args:
            num_samples: Number of training samples
            config: Training configuration with 'num_levels', 'chunk_sizes'
            dataset_stats: Dataset characteristics with 'avg_tokens_per_sample'
            zipf_alpha: Zipfian distribution parameter (default: 1.0)

        Returns:
            StorageEstimate with detailed per-level breakdown

        Example:
            config = {
                'num_levels': 4,
                'chunk_sizes': [15, 15, 15, 15],
                'tokenizer': 'gpt2'
            }
            dataset_stats = {
                'avg_tokens_per_sample': 500,
                'dataset_name': 'wikitext'
            }
            estimate = estimator.estimate_storage(1000000, config, dataset_stats)
        """
        if zipf_alpha is None:
            # Use calibrated alpha if available, otherwise default
            zipf_alpha = self.calibrated_zipf_alpha if self.calibrated_zipf_alpha else self.DEFAULT_ZIPF_ALPHA

        num_levels = config['num_levels']
        chunk_sizes = config['chunk_sizes']
        avg_tokens = dataset_stats['avg_tokens_per_sample']

        if len(chunk_sizes) != num_levels:
            raise ValueError(f"chunk_sizes must have {num_levels} elements")

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"STORAGE ESTIMATION")
            print(f"{'='*80}")
            print(f"Samples: {num_samples:,}")
            print(f"Avg tokens/sample: {avg_tokens:.1f}")
            print(f"Hierarchy levels: {num_levels}")
            print(f"Chunk sizes: {chunk_sizes}")
            print(f"Zipfian α: {zipf_alpha}")

        level_estimates = []
        current_observations = 0
        current_patterns = 0

        for level in range(num_levels):
            chunk_size = chunk_sizes[level]

            # Calculate total observations at this level
            if level == 0:
                # node0: divide tokens by chunk size
                observations_per_sample = avg_tokens / chunk_size
                total_observations = int(num_samples * observations_per_sample)
            else:
                # Higher levels: divide previous level's patterns by chunk size
                observations_per_sample = current_patterns / num_samples
                chunks_per_sample = observations_per_sample / chunk_size
                total_observations = int(num_samples * chunks_per_sample)

            # Estimate unique patterns using Zipfian distribution
            unique_patterns = self._estimate_unique_patterns(
                total_observations=total_observations,
                level=level,
                zipf_alpha=zipf_alpha
            )

            # Deduplication rate
            if total_observations > 0:
                deduplication_rate = 1.0 - (unique_patterns / total_observations)
            else:
                deduplication_rate = 0.0

            # Pattern size estimation
            avg_pattern_bytes = self._estimate_pattern_size(
                level=level,
                chunk_size=chunk_size
            )

            # Metadata overhead per pattern
            metadata_bytes = self.METADATA_BASE_SIZE

            # Total storage for this level
            level_storage_bytes = unique_patterns * (avg_pattern_bytes + metadata_bytes)
            level_storage_mb = level_storage_bytes / 1024 / 1024

            level_estimate = LevelStorageEstimate(
                level=level,
                level_name=f'node{level}',
                total_observations=total_observations,
                unique_patterns_estimated=unique_patterns,
                deduplication_rate=deduplication_rate,
                avg_pattern_size_bytes=avg_pattern_bytes,
                metadata_overhead_bytes=metadata_bytes,
                total_storage_bytes=level_storage_bytes,
                total_storage_mb=level_storage_mb,
                patterns_per_sample=unique_patterns / num_samples
            )

            level_estimates.append(level_estimate)

            if self.verbose:
                print(f"\nnode{level}:")
                print(f"  Observations: {total_observations:,}")
                print(f"  Unique patterns: {unique_patterns:,}")
                print(f"  Deduplication: {deduplication_rate*100:.1f}%")
                print(f"  Storage: {level_storage_mb:.2f} MB")

            # Update for next level
            current_observations = unique_patterns
            current_patterns = unique_patterns

        # Aggregate totals
        total_patterns = sum(est.unique_patterns_estimated for est in level_estimates)
        total_storage_bytes = sum(est.total_storage_bytes for est in level_estimates)
        total_storage_mb = total_storage_bytes / 1024 / 1024
        total_storage_gb = total_storage_mb / 1024

        # MongoDB overhead (indexes, internal structures, padding)
        mongodb_overhead_percent = 20.0
        storage_with_overhead_gb = total_storage_gb * (1 + mongodb_overhead_percent / 100)

        estimate = StorageEstimate(
            configuration=config,
            dataset_stats=dataset_stats,
            num_samples=num_samples,
            level_estimates=level_estimates,
            total_patterns=total_patterns,
            total_storage_bytes=total_storage_bytes,
            total_storage_mb=total_storage_mb,
            total_storage_gb=total_storage_gb,
            mongodb_overhead_percent=mongodb_overhead_percent,
            estimated_storage_with_overhead_gb=storage_with_overhead_gb,
            zipf_alpha=zipf_alpha
        )

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"TOTAL ESTIMATES")
            print(f"{'='*80}")
            print(f"Total patterns: {total_patterns:,}")
            print(f"Raw storage: {total_storage_gb:.3f} GB")
            print(f"With MongoDB overhead (+{mongodb_overhead_percent}%): {storage_with_overhead_gb:.3f} GB")
            print(f"{'='*80}")

        return estimate

    def _estimate_unique_patterns(
        self,
        total_observations: int,
        level: int,
        zipf_alpha: float
    ) -> int:
        """
        Estimate unique patterns using Zipfian distribution.

        Zipf's Law states that the frequency of the k-th most common item is
        proportional to 1/k^α. This creates a power-law distribution where:
        - Few items are very frequent (high deduplication)
        - Many items are rare (low deduplication)

        The number of unique items grows logarithmically with observations:
        unique ≈ observations / (1 + log(observations)^α)

        However, higher levels have LESS deduplication because:
        - node0: Token combinations repeat frequently (high reuse)
        - node1+: Pattern combinations are more varied (lower reuse)

        Args:
            total_observations: Total observations to learn
            level: Hierarchical level (0, 1, 2, ...)
            zipf_alpha: Zipfian power law exponent

        Returns:
            Estimated number of unique patterns
        """
        if total_observations <= 0:
            return 0

        # Base Zipfian estimation
        if total_observations < 10:
            # Very small datasets: assume mostly unique
            base_unique = total_observations
        else:
            # Zipfian formula: unique patterns grow logarithmically
            base_unique = total_observations / (1 + np.log(total_observations) ** zipf_alpha)

        # Level-dependent adjustment
        # Higher levels have LESS deduplication (pattern combinations more varied)
        level_multiplier = 1.0 + (level * 0.3)  # Each level +30% more unique patterns

        unique_patterns = int(base_unique * level_multiplier)

        # Cap at total observations (can't have more unique than total)
        unique_patterns = min(unique_patterns, total_observations)

        return unique_patterns

    def _estimate_pattern_size(self, level: int, chunk_size: int) -> int:
        """
        Estimate average pattern size in bytes.

        Pattern structure in MongoDB:
        {
            "name": "PTRN|<sha1>",              # 40 bytes
            "pattern_data": [[sym1, sym2], ...], # Variable
            "frequency": 1,                      # 8 bytes
            "metadata": {...},                   # ~150 bytes
            "length": 5                          # 8 bytes
        }

        Args:
            level: Hierarchical level
            chunk_size: Number of items per pattern

        Returns:
            Estimated bytes per pattern
        """
        # Pattern name (always 40 bytes)
        size = self.PATTERN_NAME_SIZE

        # Frequency field
        size += self.FREQUENCY_SIZE

        # Length field
        size += 8

        # Pattern data: depends on level
        if level == 0:
            # node0: stores token strings
            # Each event is a list with one token string
            size += chunk_size * (self.EVENT_OVERHEAD + self.SYMBOL_AVG_SIZE)
        else:
            # Higher levels: store pattern names (40 chars each)
            size += chunk_size * (self.EVENT_OVERHEAD + self.PATTERN_NAME_SIZE)

        return size

    def calibrate_from_actual(
        self,
        actual_patterns: Dict[str, int],
        actual_storage_bytes: Dict[str, int],
        num_samples: int,
        config: Dict[str, Any]
    ) -> float:
        """
        Calibrate Zipfian alpha parameter from actual training data.

        After running a training session, use actual pattern counts and storage
        to refine the Zipfian distribution parameter for better future estimates.

        Args:
            actual_patterns: {'node0': 50000, 'node1': 8000, ...}
            actual_storage_bytes: {'node0': 25000000, 'node1': 5000000, ...}
            num_samples: Number of samples that were trained
            config: Configuration used for training

        Returns:
            Calibrated zipf_alpha value

        Example:
            # After training
            actual_patterns = {
                'node0': 48532,
                'node1': 8234,
                'node2': 1204,
                'node3': 156
            }
            actual_storage = {
                'node0': 22000000,
                'node1': 6000000,
                'node2': 1500000,
                'node3': 500000
            }
            new_alpha = estimator.calibrate_from_actual(
                actual_patterns, actual_storage, 10000, config
            )
            # Use new_alpha for future estimates
        """
        # Try different alpha values and find best fit
        alphas_to_try = np.linspace(0.5, 2.0, 30)
        best_alpha = 1.0
        best_error = float('inf')

        dataset_stats = {
            'avg_tokens_per_sample': 500  # Use actual if available
        }

        for alpha in alphas_to_try:
            estimate = self.estimate_storage(
                num_samples=num_samples,
                config=config,
                dataset_stats=dataset_stats,
                zipf_alpha=alpha
            )

            # Calculate error between estimated and actual
            error = 0
            for level_est in estimate.level_estimates:
                level_name = level_est.level_name
                if level_name in actual_patterns:
                    pattern_error = abs(level_est.unique_patterns_estimated - actual_patterns[level_name])
                    error += pattern_error

            if error < best_error:
                best_error = error
                best_alpha = alpha

        if self.verbose:
            print(f"✓ Calibrated Zipfian α: {best_alpha:.3f}")

        return best_alpha

    def _auto_calibrate_from_history(self):
        """
        Auto-calibrate Zipfian alpha from training_history.db.

        Analyzes actual pattern counts from completed training runs to
        refine the Zipfian distribution parameter.
        """
        try:
            import sqlite3

            conn = sqlite3.connect(self.history_db)
            cursor = conn.cursor()

            # Get runs with pattern data and configuration
            cursor.execute("""
                SELECT config, samples_processed,
                       node0_patterns, node1_patterns, node2_patterns, node3_patterns
                FROM training_runs
                WHERE node0_patterns IS NOT NULL
                AND samples_processed >= 1000
                ORDER BY timestamp DESC
                LIMIT 10
            """)

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                if self.verbose:
                    print("  No historical pattern data for calibration")
                return

            # Try different alpha values and find best fit
            alphas_to_try = np.linspace(0.5, 2.0, 30)
            best_alpha = 1.0
            best_error = float('inf')

            for alpha in alphas_to_try:
                total_error = 0
                valid_comparisons = 0

                for row in rows:
                    config = json.loads(row[0])
                    num_samples = row[1]
                    actual_patterns = {
                        'node0': row[2],
                        'node1': row[3],
                        'node2': row[4],
                        'node3': row[5]
                    }

                    # Filter out None values
                    actual_patterns = {k: v for k, v in actual_patterns.items() if v is not None}

                    if not actual_patterns:
                        continue

                    # Estimate with this alpha
                    try:
                        dataset_stats = {'avg_tokens_per_sample': 500}
                        estimate = self.estimate_storage(
                            num_samples=num_samples,
                            config=config,
                            dataset_stats=dataset_stats,
                            zipf_alpha=alpha
                        )

                        # Compare estimates vs actuals
                        for level_est in estimate.level_estimates:
                            level_name = level_est.level_name
                            if level_name in actual_patterns:
                                estimated = level_est.unique_patterns_estimated
                                actual = actual_patterns[level_name]
                                if actual > 0:
                                    error = abs(estimated - actual) / actual
                                    total_error += error
                                    valid_comparisons += 1
                    except:
                        continue

                if valid_comparisons > 0:
                    avg_error = total_error / valid_comparisons
                    if avg_error < best_error:
                        best_error = avg_error
                        best_alpha = alpha

            # Only use calibrated alpha if it's better than default
            if best_error < 0.5:  # Less than 50% average error
                self.calibrated_zipf_alpha = best_alpha
                if self.verbose:
                    print(f"  ✓ Calibrated Zipfian α: {best_alpha:.3f} (avg error: {best_error*100:.1f}%)")
            else:
                if self.verbose:
                    print(f"  ⚠️  Calibration error too high ({best_error*100:.1f}%), using default α=1.0")

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Auto-calibration failed: {e}")

    def export_json(self, estimate: StorageEstimate, filepath: str):
        """Export storage estimate as JSON"""
        data = {
            'configuration': estimate.configuration,
            'dataset_stats': estimate.dataset_stats,
            'num_samples': estimate.num_samples,
            'total_patterns': estimate.total_patterns,
            'total_storage_gb': estimate.total_storage_gb,
            'estimated_storage_with_overhead_gb': estimate.estimated_storage_with_overhead_gb,
            'mongodb_overhead_percent': estimate.mongodb_overhead_percent,
            'zipf_alpha': estimate.zipf_alpha,
            'level_estimates': [
                {
                    'level': est.level,
                    'level_name': est.level_name,
                    'total_observations': est.total_observations,
                    'unique_patterns_estimated': est.unique_patterns_estimated,
                    'deduplication_rate': est.deduplication_rate,
                    'avg_pattern_size_bytes': est.avg_pattern_size_bytes,
                    'total_storage_mb': est.total_storage_mb,
                    'patterns_per_sample': est.patterns_per_sample
                }
                for est in estimate.level_estimates
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Storage estimate exported to {filepath}")


def main():
    """Example usage"""
    estimator = StorageEstimator(verbose=True)

    # Example 1: WikiText with 1M samples
    print("\n" + "="*80)
    print("EXAMPLE 1: WikiText, 1M samples, 4 levels, chunk_size=15")
    print("="*80)

    config = {
        'num_levels': 4,
        'chunk_sizes': [15, 15, 15, 15],
        'tokenizer': 'gpt2'
    }

    dataset_stats = {
        'avg_tokens_per_sample': 500,
        'dataset_name': 'wikitext'
    }

    estimate = estimator.estimate_storage(
        num_samples=1000000,
        config=config,
        dataset_stats=dataset_stats
    )

    print(f"\n📊 RESULT: {estimate.estimated_storage_with_overhead_gb:.2f} GB needed")

    # Example 2: C4 with 100M samples (realistic full-scale)
    print("\n" + "="*80)
    print("EXAMPLE 2: C4, 100M samples, 4 levels, chunk_size=8")
    print("="*80)

    config_c4 = {
        'num_levels': 4,
        'chunk_sizes': [8, 8, 8, 8],
        'tokenizer': 'gpt2'
    }

    dataset_stats_c4 = {
        'avg_tokens_per_sample': 800,
        'dataset_name': 'c4'
    }

    estimate_c4 = estimator.estimate_storage(
        num_samples=100000000,
        config=config_c4,
        dataset_stats=dataset_stats_c4
    )

    print(f"\n📊 RESULT: {estimate_c4.estimated_storage_with_overhead_gb:.2f} GB needed")

    # Export
    estimator.export_json(estimate_c4, 'storage_estimate_c4_100M.json')


if __name__ == '__main__':
    main()
