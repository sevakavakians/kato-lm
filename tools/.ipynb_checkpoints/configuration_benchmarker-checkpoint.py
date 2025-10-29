#!/usr/bin/env python3
"""
Configuration Benchmarker - Automated Configuration Testing

This module explores the configuration space by testing multiple KATO hierarchical
configurations with quick benchmark runs. It helps identify optimal configurations
for specific hardware and dataset characteristics.

Tests configurations across:
- Number of hierarchical levels (3-10)
- Chunk sizes (5-25)
- Batch sizes
- Worker counts

Usage:
    benchmarker = ConfigurationBenchmarker(
        dataset_key='wikitext',
        samples_per_config=1000
    )
    results = benchmarker.benchmark_all_configs()
    benchmarker.print_rankings()
    benchmarker.export_results('config_comparison.csv')
"""

import time
import json
import csv
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from itertools import product
import numpy as np

# Import profiling modules
from tools.profiling_engine import ProfilingEngine
from tools.storage_estimator import StorageEstimator


@dataclass
class ConfigBenchmarkResult:
    """Results from benchmarking a single configuration"""
    config_id: str
    config: Dict[str, Any]

    # Performance metrics
    total_time_seconds: float
    samples_per_second: float
    tokens_per_second: float

    # Resource metrics
    peak_memory_mb: float
    avg_memory_mb: float
    avg_cpu_percent: float

    # Storage metrics
    estimated_storage_gb: float
    patterns_per_level: Dict[str, int]

    # Quality metrics
    patterns_per_sample: float
    utilization_score: float  # How well all levels are utilized
    force_learn_ratio: float  # Ratio of patterns that needed forcing

    # Efficiency score (composite metric)
    efficiency_score: float = 0.0

    # Errors/warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ConfigSpace:
    """Configuration space definition"""
    num_levels_range: List[int] = field(default_factory=lambda: [3, 4, 5])
    chunk_sizes_range: List[int] = field(default_factory=lambda: [8, 10, 15, 20])
    batch_sizes_range: List[int] = field(default_factory=lambda: [1, 25, 50])
    num_workers_range: List[int] = field(default_factory=lambda: [1, 4])

    def generate_configs(self, max_configs: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate all configurations from the space.

        Args:
            max_configs: Limit number of configs (for testing)

        Returns:
            List of configuration dictionaries
        """
        configs = []

        for num_levels in self.num_levels_range:
            for chunk_size in self.chunk_sizes_range:
                for batch_size in self.batch_sizes_range:
                    for num_workers in self.num_workers_range:
                        config = {
                            'num_levels': num_levels,
                            'chunk_sizes': [chunk_size] * num_levels,  # Uniform
                            'batch_size': batch_size,
                            'num_workers': num_workers
                        }
                        configs.append(config)

                        if max_configs and len(configs) >= max_configs:
                            return configs

        return configs


class ConfigurationBenchmarker:
    """
    Automated configuration benchmarking system.

    Tests multiple configurations with quick training runs to identify
    optimal settings for specific hardware and dataset characteristics.
    """

    def __init__(
        self,
        dataset_key: str = 'wikitext',
        samples_per_config: int = 1000,
        avg_tokens_per_sample: int = 500,
        config_space: Optional[ConfigSpace] = None,
        verbose: bool = True
    ):
        """
        Initialize configuration benchmarker.

        Args:
            dataset_key: Dataset identifier
            samples_per_config: Number of samples per benchmark
            avg_tokens_per_sample: Average tokens per sample (for estimation)
            config_space: Configuration space to explore
            verbose: Print progress
        """
        self.dataset_key = dataset_key
        self.samples_per_config = samples_per_config
        self.avg_tokens_per_sample = avg_tokens_per_sample
        self.verbose = verbose

        if config_space is None:
            self.config_space = ConfigSpace()
        else:
            self.config_space = config_space

        self.results: List[ConfigBenchmarkResult] = []
        self.storage_estimator = StorageEstimator(verbose=False)

        if self.verbose:
            print("✓ ConfigurationBenchmarker initialized")
            print(f"  Dataset: {dataset_key}")
            print(f"  Samples per config: {samples_per_config:,}")

    def benchmark_config(self, config: Dict[str, Any]) -> ConfigBenchmarkResult:
        """
        Benchmark a single configuration.

        NOTE: This is a SIMULATION for demonstration. In actual use, you would
        call train_hierarchical_single_pass() with profiling enabled.

        Args:
            config: Configuration to test

        Returns:
            ConfigBenchmarkResult with metrics
        """
        config_id = self._generate_config_id(config)

        if self.verbose:
            print(f"\n⏳ Benchmarking: {config_id}")

        errors = []
        warnings = []

        # SIMULATION: In real implementation, call training here
        # For now, simulate based on configuration characteristics

        # Simulate time based on config complexity
        base_time = 60  # Base 60 seconds
        level_penalty = config['num_levels'] * 5  # More levels = more time
        chunk_penalty = 100 / config['chunk_sizes'][0]  # Smaller chunks = more patterns
        batch_bonus = config['batch_size'] / 10  # Batching speeds up
        worker_bonus = config['num_workers'] * 5  # More workers = faster

        simulated_time = base_time + level_penalty + chunk_penalty - batch_bonus - worker_bonus
        simulated_time = max(10, simulated_time)  # Minimum 10s

        samples_per_second = self.samples_per_config / simulated_time
        tokens_per_second = samples_per_second * self.avg_tokens_per_sample

        # Simulate memory usage
        base_memory = 500  # Base 500 MB
        level_memory = config['num_levels'] * 50
        batch_memory = config['batch_size'] * 5
        worker_memory = config['num_workers'] * 100

        peak_memory = base_memory + level_memory + batch_memory + worker_memory
        avg_memory = peak_memory * 0.8

        # Simulate CPU usage
        avg_cpu = min(95, 40 + config['num_workers'] * 15)

        # Estimate storage using StorageEstimator
        dataset_stats = {
            'avg_tokens_per_sample': self.avg_tokens_per_sample,
            'dataset_name': self.dataset_key
        }

        storage_estimate = self.storage_estimator.estimate_storage(
            num_samples=self.samples_per_config,
            config=config,
            dataset_stats=dataset_stats
        )

        estimated_storage_gb = storage_estimate.estimated_storage_with_overhead_gb

        # Extract patterns per level
        patterns_per_level = {
            f"node{est.level}": est.unique_patterns_estimated
            for est in storage_estimate.level_estimates
        }

        # Quality metrics
        patterns_per_sample = storage_estimate.total_patterns / self.samples_per_config

        # Utilization score: how well are all levels being used?
        patterns_per_doc_by_level = [
            est.patterns_per_sample for est in storage_estimate.level_estimates
        ]

        # Check for underutilized levels (< 0.5 patterns/doc)
        underutilized_count = sum(1 for ppd in patterns_per_doc_by_level if ppd < 0.5)
        utilization_score = 1.0 - (underutilized_count / config['num_levels'])

        if underutilized_count > 0:
            warnings.append(f"{underutilized_count} underutilized levels (< 0.5 patterns/doc)")

        # Force learn ratio (simulated)
        force_learn_ratio = 0.1 if config['chunk_sizes'][0] > 20 else 0.02

        # Calculate efficiency score (composite metric)
        # Higher is better: fast, low memory, good utilization
        time_score = 1.0 / (simulated_time / 60)  # Normalize to 1 for 60s
        memory_score = 1.0 / (peak_memory / 1000)  # Normalize to 1 for 1GB
        utilization_penalty = utilization_score

        efficiency_score = (time_score * 0.5 + memory_score * 0.3 + utilization_penalty * 0.2)

        result = ConfigBenchmarkResult(
            config_id=config_id,
            config=config,
            total_time_seconds=simulated_time,
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            avg_cpu_percent=avg_cpu,
            estimated_storage_gb=estimated_storage_gb,
            patterns_per_level=patterns_per_level,
            patterns_per_sample=patterns_per_sample,
            utilization_score=utilization_score,
            force_learn_ratio=force_learn_ratio,
            efficiency_score=efficiency_score,
            errors=errors,
            warnings=warnings
        )

        if self.verbose:
            print(f"  ✓ Time: {simulated_time:.1f}s, Memory: {peak_memory:.0f}MB, "
                  f"Efficiency: {efficiency_score:.3f}")

        return result

    def benchmark_all_configs(self, max_configs: Optional[int] = None) -> List[ConfigBenchmarkResult]:
        """
        Benchmark all configurations in the space.

        Args:
            max_configs: Limit number of configs to test

        Returns:
            List of benchmark results
        """
        configs = self.config_space.generate_configs(max_configs=max_configs)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"CONFIGURATION BENCHMARKING")
            print(f"{'='*80}")
            print(f"Total configurations to test: {len(configs)}")
            print(f"Samples per configuration: {self.samples_per_config:,}")

        self.results = []

        for i, config in enumerate(configs, 1):
            if self.verbose:
                print(f"\n[{i}/{len(configs)}]", end=' ')

            result = self.benchmark_config(config)
            self.results.append(result)

        if self.verbose:
            print(f"\n\n✓ Benchmarking complete: {len(self.results)} configurations tested")

        return self.results

    def get_rankings(self, sort_by: str = 'efficiency_score') -> List[ConfigBenchmarkResult]:
        """
        Get ranked results.

        Args:
            sort_by: Metric to sort by ('efficiency_score', 'samples_per_second',
                     'peak_memory_mb', 'estimated_storage_gb')

        Returns:
            Sorted list of results (best first)
        """
        if not self.results:
            raise RuntimeError("No results available. Run benchmark_all_configs() first.")

        reverse = True
        if sort_by in ['peak_memory_mb', 'estimated_storage_gb', 'total_time_seconds']:
            reverse = False  # Lower is better for these metrics

        return sorted(self.results, key=lambda r: getattr(r, sort_by), reverse=reverse)

    def print_rankings(self, top_n: int = 10, sort_by: str = 'efficiency_score'):
        """Print top N configurations"""
        rankings = self.get_rankings(sort_by=sort_by)[:top_n]

        print(f"\n{'='*80}")
        print(f"TOP {len(rankings)} CONFIGURATIONS (sorted by {sort_by})")
        print(f"{'='*80}")

        for i, result in enumerate(rankings, 1):
            print(f"\n#{i}: {result.config_id}")
            print(f"  Config: {result.config['num_levels']} levels, chunk_size={result.config['chunk_sizes'][0]}, "
                  f"batch={result.config['batch_size']}, workers={result.config['num_workers']}")
            print(f"  Time: {result.total_time_seconds:.1f}s ({result.samples_per_second:.1f} samples/s)")
            print(f"  Memory: {result.peak_memory_mb:.0f}MB peak, {result.avg_memory_mb:.0f}MB avg")
            print(f"  Storage: {result.estimated_storage_gb:.3f}GB")
            print(f"  Utilization: {result.utilization_score*100:.1f}%")
            print(f"  Efficiency: {result.efficiency_score:.3f}")

            if result.warnings:
                print(f"  ⚠️  Warnings: {', '.join(result.warnings)}")

        print(f"\n{'='*80}")

    def export_csv(self, filepath: str):
        """Export results as CSV"""
        if not self.results:
            raise RuntimeError("No results to export")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'config_id',
                'num_levels',
                'chunk_size',
                'batch_size',
                'num_workers',
                'time_seconds',
                'samples_per_sec',
                'peak_memory_mb',
                'avg_memory_mb',
                'storage_gb',
                'utilization_score',
                'efficiency_score',
                'warnings'
            ])

            # Data
            for result in self.results:
                writer.writerow([
                    result.config_id,
                    result.config['num_levels'],
                    result.config['chunk_sizes'][0],
                    result.config['batch_size'],
                    result.config['num_workers'],
                    f"{result.total_time_seconds:.2f}",
                    f"{result.samples_per_second:.2f}",
                    f"{result.peak_memory_mb:.0f}",
                    f"{result.avg_memory_mb:.0f}",
                    f"{result.estimated_storage_gb:.4f}",
                    f"{result.utilization_score:.3f}",
                    f"{result.efficiency_score:.3f}",
                    '; '.join(result.warnings) if result.warnings else ''
                ])

        print(f"✓ Results exported to {filepath}")

    def export_json(self, filepath: str):
        """Export results as JSON"""
        if not self.results:
            raise RuntimeError("No results to export")

        data = {
            'dataset_key': self.dataset_key,
            'samples_per_config': self.samples_per_config,
            'avg_tokens_per_sample': self.avg_tokens_per_sample,
            'total_configs_tested': len(self.results),
            'results': [
                {
                    'config_id': r.config_id,
                    'config': r.config,
                    'time_seconds': r.total_time_seconds,
                    'samples_per_second': r.samples_per_second,
                    'tokens_per_second': r.tokens_per_second,
                    'peak_memory_mb': r.peak_memory_mb,
                    'avg_memory_mb': r.avg_memory_mb,
                    'avg_cpu_percent': r.avg_cpu_percent,
                    'estimated_storage_gb': r.estimated_storage_gb,
                    'patterns_per_level': r.patterns_per_level,
                    'patterns_per_sample': r.patterns_per_sample,
                    'utilization_score': r.utilization_score,
                    'efficiency_score': r.efficiency_score,
                    'warnings': r.warnings,
                    'errors': r.errors
                }
                for r in self.results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Results exported to {filepath}")

    def get_best_config(self, sort_by: str = 'efficiency_score') -> ConfigBenchmarkResult:
        """Get single best configuration"""
        rankings = self.get_rankings(sort_by=sort_by)
        return rankings[0]

    def _generate_config_id(self, config: Dict[str, Any]) -> str:
        """Generate human-readable config ID"""
        return (f"L{config['num_levels']}_C{config['chunk_sizes'][0]}_"
                f"B{config['batch_size']}_W{config['num_workers']}")


def main():
    """Example usage"""
    # Define custom configuration space
    config_space = ConfigSpace(
        num_levels_range=[3, 4, 5],
        chunk_sizes_range=[8, 15, 20],
        batch_sizes_range=[1, 50],
        num_workers_range=[1, 4]
    )

    # Create benchmarker
    benchmarker = ConfigurationBenchmarker(
        dataset_key='wikitext',
        samples_per_config=1000,
        avg_tokens_per_sample=500,
        config_space=config_space,
        verbose=True
    )

    # Run benchmarks
    results = benchmarker.benchmark_all_configs()

    # Show top configurations
    benchmarker.print_rankings(top_n=5, sort_by='efficiency_score')

    # Export results
    benchmarker.export_csv('config_benchmarks.csv')
    benchmarker.export_json('config_benchmarks.json')

    # Get best config
    best = benchmarker.get_best_config()
    print(f"\n🏆 Best configuration: {best.config_id}")
    print(f"   Efficiency score: {best.efficiency_score:.3f}")
    print(f"   Config: {best.config}")


if __name__ == '__main__':
    main()
