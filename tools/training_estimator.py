#!/usr/bin/env python3
"""
Training Estimator - Data-Driven Performance Prediction

Uses historical training data to provide accurate estimates of:
- Training time (samples/sec throughput)
- Storage requirements
- Memory usage

This estimator is calibrated from actual training runs stored in training_history.db
and improves over time as more data accumulates.

Usage:
    estimator = TrainingEstimator(history_db='./training_history.db')

    estimate = estimator.estimate_training(
        config={'num_levels': 5, 'chunk_sizes': [8,8,8,8,8], 'batch_size': 100, 'num_workers': 6},
        num_samples=10000,
        dataset_key='wikitext'
    )

    estimate.print_summary()
"""

import numpy as np
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class TrainingTimeEstimate:
    """Training time estimation with confidence intervals"""
    num_samples: int
    config: Dict[str, Any]

    # Throughput estimates
    estimated_samples_per_sec: float
    estimated_time_seconds: float
    estimated_time_minutes: float
    estimated_time_hours: float

    # Confidence intervals (based on historical variance)
    lower_bound_seconds: float
    upper_bound_seconds: float
    confidence_level: float = 0.80  # 80% confidence interval

    # Contributing factors (for transparency)
    base_rate: float = 0.0
    chunk_multiplier: float = 1.0
    batch_multiplier: float = 1.0
    scale_multiplier: float = 1.0
    worker_multiplier: float = 1.0
    hardware_multiplier: float = 1.0

    # Metadata
    historical_runs_used: int = 0
    estimation_confidence: str = 'medium'  # low, medium, high
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def print_summary(self):
        """Print human-readable estimate"""
        print("\n" + "="*80)
        print("TRAINING TIME ESTIMATE")
        print("="*80)

        print(f"\n📋 CONFIGURATION")
        print(f"  Samples: {self.num_samples:,}")
        print(f"  Chunk sizes: {self.config.get('chunk_sizes', 'N/A')}")
        print(f"  Batch size: {self.config.get('batch_size', 'N/A')}")
        print(f"  Workers: {self.config.get('num_workers', 'N/A')}")

        print(f"\n⚡ PERFORMANCE ESTIMATE")
        print(f"  Throughput: {self.estimated_samples_per_sec:.2f} samples/sec")
        print(f"  Estimated time: {self.estimated_time_minutes:.1f} minutes ({self.estimated_time_hours:.2f} hours)")
        print(f"  Confidence interval ({self.confidence_level*100:.0f}%): "
              f"{self.lower_bound_seconds/60:.1f} - {self.upper_bound_seconds/60:.1f} minutes")

        print(f"\n🔍 BREAKDOWN")
        print(f"  Base rate: {self.base_rate:.2f} samples/sec")
        print(f"  Chunk size multiplier: {self.chunk_multiplier:.2f}x")
        print(f"  Batch size multiplier: {self.batch_multiplier:.2f}x")
        print(f"  Scale multiplier: {self.scale_multiplier:.2f}x")
        print(f"  Worker multiplier: {self.worker_multiplier:.2f}x")
        print(f"  Hardware multiplier: {self.hardware_multiplier:.2f}x")

        print(f"\n📊 CONFIDENCE")
        print(f"  Estimation confidence: {self.estimation_confidence.upper()}")
        print(f"  Based on {self.historical_runs_used} historical runs")

        if self.warnings:
            print(f"\n⚠️  WARNINGS")
            for warning in self.warnings:
                print(f"  • {warning}")

        print("\n" + "="*80)


class TrainingEstimator:
    """
    Data-driven training time estimator.

    Uses actual training history to predict performance for new configurations.
    Automatically improves as more training data accumulates.

    Key insight: Performance is dominated by minimum chunk_size, with secondary
    effects from batch_size, scale, workers, and hardware tier.
    """

    def __init__(self, history_db: str = './training_history.db', verbose: bool = True):
        """
        Initialize training estimator.

        Args:
            history_db: Path to training_history.db
            verbose: Print calibration info
        """
        self.history_db = history_db
        self.verbose = verbose

        # Calibrate from historical data
        self._calibrate_from_history()

        if self.verbose:
            print(f"✓ TrainingEstimator initialized with {self.num_historical_runs} runs")

    def _calibrate_from_history(self):
        """
        Calibrate estimation parameters from historical training runs.

        Analyzes training_history.db to derive:
        - Base throughput rate
        - Chunk size power law exponent
        - Batch size scaling factor
        - Scale effect parameters
        - Variance for confidence intervals
        """
        try:
            conn = sqlite3.connect(self.history_db)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT config, samples_processed, actual_time_seconds,
                       samples_per_second, peak_memory_mb
                FROM training_runs
                WHERE samples_per_second > 0
                ORDER BY timestamp
            """)

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                # No historical data - use conservative defaults
                self._use_default_parameters()
                return

            # Parse historical data
            self.historical_data = []
            for row in rows:
                config = json.loads(row[0])
                chunk_sizes = config.get('chunk_sizes', [8])
                batch_size = config.get('batch_size', 50)
                num_workers = config.get('num_workers', 6)

                self.historical_data.append({
                    'min_chunk_size': min(chunk_sizes),
                    'mean_chunk_size': np.mean(chunk_sizes),
                    'batch_size': batch_size,
                    'num_workers': num_workers,
                    'samples_processed': row[1],
                    'actual_time': row[2],
                    'samples_per_sec': row[3],
                    'peak_memory_mb': row[4] if row[4] else 0
                })

            self.num_historical_runs = len(self.historical_data)

            # Calibrate parameters from data
            self._fit_chunk_size_effect()
            self._fit_batch_size_effect()
            self._fit_scale_effect()
            self._calculate_variance()

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not load historical data: {e}")
                print(f"   Using default parameters")
            self._use_default_parameters()

    def _use_default_parameters(self):
        """Use conservative default parameters when no historical data exists"""
        self.num_historical_runs = 0
        self.historical_data = []

        # Conservative defaults (based on initial analysis)
        self.base_throughput = 1.5  # samples/sec for chunk=3, batch=50, 6 workers
        self.chunk_exponent = 1.3   # Power law exponent
        self.batch_exponent = 0.8   # Sub-linear scaling
        self.scale_coefficient = 0.05  # Logarithmic slowdown rate
        self.worker_exponent = 0.8  # Sub-linear worker scaling

        # Variance (for confidence intervals)
        self.throughput_std = 0.3  # 30% standard deviation

    def _fit_chunk_size_effect(self):
        """
        Fit chunk size power law: rate = base * (chunk / reference)^exponent

        From analysis: larger chunk sizes give exponential speedup
        """
        # Group by chunk size
        chunk_groups = {}
        for d in self.historical_data:
            cs = d['min_chunk_size']
            if cs not in chunk_groups:
                chunk_groups[cs] = []
            chunk_groups[cs].append(d['samples_per_sec'])

        if len(chunk_groups) < 2:
            # Not enough data - use defaults
            self.base_throughput = 1.5
            self.chunk_exponent = 1.3
            return

        # Calculate mean rate per chunk size
        chunk_sizes = np.array(sorted(chunk_groups.keys()), dtype=float)
        mean_rates = np.array([np.mean(chunk_groups[cs]) for cs in chunk_sizes])

        # Fit power law: rate = a * chunk^b
        # Using log-log regression: log(rate) = log(a) + b * log(chunk)
        log_chunks = np.log(chunk_sizes)
        log_rates = np.log(mean_rates)

        # Simple linear regression in log space
        n = len(chunk_sizes)
        sum_x = np.sum(log_chunks)
        sum_y = np.sum(log_rates)
        sum_xy = np.sum(log_chunks * log_rates)
        sum_x2 = np.sum(log_chunks ** 2)

        # Avoid division by zero
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            # Fall back to defaults
            self.base_throughput = 1.5
            self.chunk_exponent = 1.3
            return

        self.chunk_exponent = (n * sum_xy - sum_x * sum_y) / denominator
        log_a = (sum_y - self.chunk_exponent * sum_x) / n

        # Convert to base rate at reference chunk_size=3
        # rate = a * chunk^exponent
        # at chunk=3: rate_3 = a * 3^exponent
        # so: a = rate_3 / (3^exponent)
        # and we have: log(a) = log(rate_3) - exponent * log(3)
        # Therefore: log(rate_3) = log(a) + exponent * log(3)
        reference_chunk = 3
        log_base_rate = log_a + self.chunk_exponent * np.log(reference_chunk)
        self.base_throughput = np.exp(log_base_rate)

        # Sanity check and clamp
        self.chunk_exponent = np.clip(self.chunk_exponent, 0.8, 2.0)
        self.base_throughput = np.clip(self.base_throughput, 0.5, 5.0)

        if self.verbose:
            print(f"  Calibrated chunk effect: rate ∝ chunk^{self.chunk_exponent:.2f}, base={self.base_throughput:.2f}")

    def _fit_batch_size_effect(self):
        """Fit batch size scaling effect"""
        # Group by batch size
        batch_groups = {}
        for d in self.historical_data:
            bs = d['batch_size']
            if bs not in batch_groups:
                batch_groups[bs] = []
            batch_groups[bs].append(d['samples_per_sec'])

        if len(batch_groups) < 2:
            # Not enough data - use default
            self.batch_exponent = 0.8
            return

        # Calculate mean per batch size
        batch_sizes = np.array(sorted(batch_groups.keys()), dtype=float)
        mean_rates = np.array([np.mean(batch_groups[bs]) for bs in batch_sizes])

        # Fit multiplicative effect: rate_batch / rate_50 = (batch / 50)^exponent
        # Take ratio relative to batch=50
        reference_batch = 50
        if reference_batch in batch_groups:
            ref_rate = np.mean(batch_groups[reference_batch])
            # Simple average of log ratios
            exponents = []
            for bs, rate in zip(batch_sizes, mean_rates):
                if bs != reference_batch and rate > 0 and ref_rate > 0:
                    # (rate / ref_rate) = (bs / ref)^exp
                    # exp = log(rate / ref_rate) / log(bs / ref)
                    exp = np.log(rate / ref_rate) / np.log(bs / reference_batch)
                    exponents.append(exp)

            if exponents:
                self.batch_exponent = np.median(exponents)
            else:
                self.batch_exponent = 0.8
        else:
            # Fallback: simple power law fit
            log_batches = np.log(batch_sizes)
            log_rates = np.log(mean_rates)

            n = len(batch_sizes)
            sum_x = np.sum(log_batches)
            sum_y = np.sum(log_rates)
            sum_xy = np.sum(log_batches * log_rates)
            sum_x2 = np.sum(log_batches ** 2)

            denominator = n * sum_x2 - sum_x ** 2
            if abs(denominator) > 1e-10:
                self.batch_exponent = (n * sum_xy - sum_x * sum_y) / denominator
            else:
                self.batch_exponent = 0.8

        # Clamp to reasonable range
        self.batch_exponent = np.clip(self.batch_exponent, 0.5, 1.0)

        if self.verbose:
            print(f"  Calibrated batch effect: rate ∝ batch^{self.batch_exponent:.2f}")

    def _fit_scale_effect(self):
        """
        Fit scale effect: performance vs num_samples.

        Observed: slight logarithmic slowdown at larger scales
        """
        # Group by scale
        scale_groups = {}
        for d in self.historical_data:
            samples = d['samples_processed']
            if samples not in scale_groups:
                scale_groups[samples] = []
            scale_groups[samples].append(d['samples_per_sec'])

        if len(scale_groups) < 2:
            # Not enough data - assume minimal scale effect
            self.scale_coefficient = 0.03
            return

        # Analyze trend: does performance degrade with scale?
        scales = sorted(scale_groups.keys())
        mean_rates = [np.mean(scale_groups[s]) for s in scales]

        # Simple linear fit on log scale
        log_scales = np.log10(scales)

        # Normalize rates to remove chunk/batch effects
        # (just look at trend)
        normalized_rates = np.array(mean_rates) / np.mean(mean_rates)

        # Fit: rate_normalized = 1 - coefficient * log10(samples / 1000)
        # Simplified: assume minimal slowdown
        if normalized_rates[-1] < normalized_rates[0]:
            # Degradation observed
            self.scale_coefficient = 0.05  # 5% slowdown per 10x scale
        else:
            # No clear degradation
            self.scale_coefficient = 0.02  # Minimal effect

        if self.verbose:
            print(f"  Calibrated scale effect: coefficient={self.scale_coefficient:.3f}")

    def _calculate_variance(self):
        """Calculate throughput variance for confidence intervals"""
        if len(self.historical_data) < 3:
            self.throughput_std = 0.3  # 30% default
            return

        rates = [d['samples_per_sec'] for d in self.historical_data]
        self.throughput_std = np.std(rates) / np.mean(rates)  # Coefficient of variation

        if self.verbose:
            print(f"  Throughput variance: CV={self.throughput_std:.2f}")

    def estimate_training(
        self,
        config: Dict[str, Any],
        num_samples: int,
        dataset_key: str = 'unknown',
        hardware_tier: str = 'medium'
    ) -> TrainingTimeEstimate:
        """
        Estimate training time for a given configuration.

        Args:
            config: Training configuration with 'chunk_sizes', 'batch_size', 'num_workers'
            num_samples: Number of samples to train
            dataset_key: Dataset identifier (for future dataset-specific tuning)
            hardware_tier: Hardware tier ('low', 'medium', 'high', 'server')

        Returns:
            TrainingTimeEstimate with detailed breakdown
        """
        chunk_sizes = config.get('chunk_sizes', [8])
        batch_size = config.get('batch_size', 50)
        num_workers = config.get('num_workers', 6)

        warnings = []

        # 1. Base rate (reference: chunk=3, batch=50, 6 workers, medium hardware)
        base_rate = self.base_throughput

        # 2. Chunk size multiplier (exponential scaling)
        min_chunk = min(chunk_sizes)
        reference_chunk = 3
        chunk_multiplier = (min_chunk / reference_chunk) ** self.chunk_exponent

        # 3. Batch size multiplier (sub-linear scaling)
        reference_batch = 50
        batch_multiplier = (batch_size / reference_batch) ** self.batch_exponent

        # 4. Scale multiplier (logarithmic slowdown)
        if num_samples < 1000:
            scale_multiplier = 1.0
        else:
            slowdown = self.scale_coefficient * np.log10(num_samples / 1000)
            scale_multiplier = max(0.5, 1.0 - slowdown)  # Cap at 50% slowdown

        # 5. Worker multiplier (sub-linear scaling)
        reference_workers = 6
        worker_exponent = 0.8  # Sub-linear: adding workers has diminishing returns
        worker_multiplier = (num_workers / reference_workers) ** worker_exponent

        # 6. Hardware tier multiplier
        hardware_multipliers = {
            'low': 0.45,
            'medium': 1.0,
            'high': 2.3,
            'server': 4.5
        }
        hardware_multiplier = hardware_multipliers.get(hardware_tier, 1.0)

        # Combined estimate
        samples_per_sec = (base_rate * chunk_multiplier * batch_multiplier *
                          scale_multiplier * worker_multiplier * hardware_multiplier)

        estimated_time_seconds = num_samples / samples_per_sec
        estimated_time_minutes = estimated_time_seconds / 60
        estimated_time_hours = estimated_time_minutes / 60

        # Confidence interval (based on historical variance)
        # Using normal approximation: 80% CI ≈ ±1.28 * std
        z_score = 1.28  # 80% confidence
        std_seconds = estimated_time_seconds * self.throughput_std * z_score

        lower_bound = max(0, estimated_time_seconds - std_seconds)
        upper_bound = estimated_time_seconds + std_seconds

        # Estimation confidence
        if self.num_historical_runs >= 20:
            confidence = 'high'
        elif self.num_historical_runs >= 10:
            confidence = 'medium'
        else:
            confidence = 'low'
            warnings.append(f"Only {self.num_historical_runs} historical runs - estimates may be less accurate")

        # Check for extrapolation warnings
        if self.num_historical_runs > 0:
            historical_chunks = [d['min_chunk_size'] for d in self.historical_data]
            if min_chunk < min(historical_chunks) or min_chunk > max(historical_chunks):
                warnings.append(f"chunk_size={min_chunk} outside historical range "
                              f"[{min(historical_chunks)}, {max(historical_chunks)}]")

            historical_samples = [d['samples_processed'] for d in self.historical_data]
            if num_samples > max(historical_samples) * 2:
                warnings.append(f"{num_samples:,} samples is 2x larger than largest historical run")

        estimate = TrainingTimeEstimate(
            num_samples=num_samples,
            config=config,
            estimated_samples_per_sec=samples_per_sec,
            estimated_time_seconds=estimated_time_seconds,
            estimated_time_minutes=estimated_time_minutes,
            estimated_time_hours=estimated_time_hours,
            lower_bound_seconds=lower_bound,
            upper_bound_seconds=upper_bound,
            confidence_level=0.80,
            base_rate=base_rate,
            chunk_multiplier=chunk_multiplier,
            batch_multiplier=batch_multiplier,
            scale_multiplier=scale_multiplier,
            worker_multiplier=worker_multiplier,
            hardware_multiplier=hardware_multiplier,
            historical_runs_used=self.num_historical_runs,
            estimation_confidence=confidence,
            warnings=warnings
        )

        return estimate

    def validate_against_history(self, verbose: bool = True) -> Dict[str, float]:
        """
        Validate estimator accuracy against historical data.

        Returns:
            Dict with accuracy metrics (MAE, MAPE, R²)
        """
        if self.num_historical_runs == 0:
            return {'error': 'No historical data available'}

        actual_rates = []
        predicted_rates = []
        errors = []

        for d in self.historical_data:
            actual_rate = d['samples_per_sec']

            # Predict using current model
            config = {
                'chunk_sizes': [d['min_chunk_size']] * 5,
                'batch_size': d['batch_size'],
                'num_workers': d['num_workers']
            }

            estimate = self.estimate_training(
                config=config,
                num_samples=d['samples_processed']
            )

            predicted_rate = estimate.estimated_samples_per_sec

            actual_rates.append(actual_rate)
            predicted_rates.append(predicted_rate)
            errors.append(abs(actual_rate - predicted_rate))

        # Calculate metrics
        actual_rates = np.array(actual_rates)
        predicted_rates = np.array(predicted_rates)
        errors = np.array(errors)

        mae = np.mean(errors)  # Mean Absolute Error
        mape = np.mean(errors / actual_rates) * 100  # Mean Absolute Percentage Error
        rmse = np.sqrt(np.mean((actual_rates - predicted_rates) ** 2))  # Root Mean Squared Error

        # R² score
        ss_res = np.sum((actual_rates - predicted_rates) ** 2)
        ss_tot = np.sum((actual_rates - np.mean(actual_rates)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics = {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'r2': r2,
            'num_runs': self.num_historical_runs
        }

        if verbose:
            print("\n" + "="*80)
            print("ESTIMATOR VALIDATION")
            print("="*80)
            print(f"Historical runs: {self.num_historical_runs}")
            print(f"MAE: {mae:.3f} samples/sec")
            print(f"MAPE: {mape:.1f}%")
            print(f"RMSE: {rmse:.3f} samples/sec")
            print(f"R²: {r2:.3f}")
            print("="*80)

        return metrics


def main():
    """Example usage"""
    print("\n" + "="*80)
    print("TRAINING ESTIMATOR - Example Usage")
    print("="*80)

    # Initialize estimator
    estimator = TrainingEstimator(verbose=True)

    # Example 1: Estimate for chunk_size=8 configuration
    print("\n" + "-"*80)
    print("Example 1: chunk_size=8, batch=100, 10K samples")
    print("-"*80)

    config1 = {
        'chunk_sizes': [8, 8, 8, 8, 8],
        'batch_size': 100,
        'num_workers': 6
    }

    estimate1 = estimator.estimate_training(
        config=config1,
        num_samples=10000,
        hardware_tier='medium'
    )

    estimate1.print_summary()

    # Example 2: Smaller chunk size
    print("\n" + "-"*80)
    print("Example 2: chunk_size=3, batch=50, 10K samples")
    print("-"*80)

    config2 = {
        'chunk_sizes': [3, 3, 3, 3, 3],
        'batch_size': 50,
        'num_workers': 6
    }

    estimate2 = estimator.estimate_training(
        config=config2,
        num_samples=10000,
        hardware_tier='medium'
    )

    estimate2.print_summary()

    # Validate estimator
    print("\n")
    metrics = estimator.validate_against_history(verbose=True)

    print(f"\n✓ Estimator ready to use")
    print(f"  Prediction accuracy: {100 - metrics['mape']:.1f}%")


if __name__ == '__main__':
    main()
