#!/usr/bin/env python3
"""
Training History - Track and Learn from Training Runs

This module maintains a persistent database of training runs to:
- Compare estimated vs actual metrics over time
- Improve estimation accuracy with each run
- Track performance trends across configurations
- Enable historical analysis and debugging
- Capture complete knowledge base snapshots for comparison

Usage:
    history = TrainingHistory()

    # Record a training run
    run_id = history.record_run(
        config={'num_levels': 4, 'chunk_sizes': [15, 15, 15, 15]},
        estimated_time=120.0,
        actual_time=135.2,
        estimated_storage_gb=2.5,
        actual_storage_gb=2.8,
        profiling_report=profiler.generate_report()
    )

    # Capture snapshot AFTER training
    snapshot = history.capture_snapshot(learner, run_id)

    # Analyze accuracy
    accuracy = history.get_estimation_accuracy()
    print(f"Time estimation accuracy: {accuracy['time_error_percent']:.1f}%")

    # Compare runs
    comparison_df = history.compare_runs(['run_1', 'run_2', 'run_3'])
"""

import json
import time
import sqlite3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class TrainingRun:
    """Single training run record"""
    # Required fields (no defaults)
    run_id: str
    timestamp: float
    config: Dict[str, Any]

    # Estimates (predicted before training) - required
    estimated_time_seconds: float
    estimated_storage_gb: float

    # Actuals (measured during training) - required
    actual_time_seconds: float
    actual_storage_gb: float

    # Optional fields (with defaults) - must come after required fields
    estimated_memory_mb: Optional[float] = None
    actual_memory_mb: Optional[float] = None

    # Training metrics
    samples_processed: int = 0
    samples_per_second: float = 0.0
    patterns_learned: Dict[str, int] = field(default_factory=dict)

    # Resource usage
    peak_memory_mb: Optional[float] = None
    avg_cpu_percent: Optional[float] = None
    disk_write_speed_mbps: Optional[float] = None

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    dataset_key: str = 'unknown'
    hardware_tier: str = 'unknown'
    notes: str = ''

    def get_time_error_percent(self) -> float:
        """Calculate time estimation error"""
        if self.estimated_time_seconds is not None and self.actual_time_seconds > 0:
            return abs(self.estimated_time_seconds - self.actual_time_seconds) / self.actual_time_seconds * 100
        return 0.0

    def get_storage_error_percent(self) -> float:
        """Calculate storage estimation error"""
        if self.estimated_storage_gb is not None and self.actual_storage_gb > 0:
            return abs(self.estimated_storage_gb - self.actual_storage_gb) / self.actual_storage_gb * 100
        return 0.0

    def get_memory_error_percent(self) -> Optional[float]:
        """Calculate memory estimation error"""
        if self.estimated_memory_mb and self.actual_memory_mb and self.actual_memory_mb > 0:
            return abs(self.estimated_memory_mb - self.actual_memory_mb) / self.actual_memory_mb * 100
        return None


@dataclass
class EstimationAccuracy:
    """Estimation accuracy metrics"""
    total_runs: int
    avg_time_error_percent: float
    avg_storage_error_percent: float
    avg_memory_error_percent: Optional[float]
    time_improvement_trend: str  # 'improving', 'stable', 'degrading'
    storage_improvement_trend: str


class TrainingHistory:
    """
    Persistent storage and analysis of training runs.

    Uses SQLite for efficient storage and querying of historical data.
    """

    def __init__(self, db_path: str = './training_history.db', verbose: bool = True):
        """
        Initialize training history database.

        Args:
            db_path: Path to SQLite database file
            verbose: Print operations
        """
        self.db_path = db_path
        self.verbose = verbose

        # Create database and tables
        self._init_database()

        if self.verbose:
            count = self.get_run_count()
            print(f"✓ TrainingHistory initialized ({count} runs in database)")

    def _init_database(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                timestamp REAL,
                config TEXT,
                estimated_time_seconds REAL,
                estimated_storage_gb REAL,
                estimated_memory_mb REAL,
                actual_time_seconds REAL,
                actual_storage_gb REAL,
                actual_memory_mb REAL,
                samples_processed INTEGER,
                samples_per_second REAL,
                patterns_learned TEXT,
                peak_memory_mb REAL,
                avg_cpu_percent REAL,
                disk_write_speed_mbps REAL,
                errors TEXT,
                warnings TEXT,
                dataset_key TEXT,
                hardware_tier TEXT,
                notes TEXT
            )
        ''')

        # Migrate: Add snapshot columns if they don't exist
        self._migrate_add_snapshot_columns(cursor)

        conn.commit()
        conn.close()

    def _migrate_add_snapshot_columns(self, cursor):
        """Add snapshot-related columns to existing database (migration)"""
        # Get existing columns
        cursor.execute("PRAGMA table_info(training_runs)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Snapshot metadata columns
        new_columns = {
            'snapshot_path': 'TEXT',
            'total_patterns': 'INTEGER',
            'total_storage_mb': 'REAL',
            'total_observations': 'INTEGER',
        }

        # Add missing columns
        for col_name, col_type in new_columns.items():
            if col_name not in existing_columns:
                cursor.execute(f'ALTER TABLE training_runs ADD COLUMN {col_name} {col_type}')

        # Dynamic per-node columns (support up to 10 nodes)
        for i in range(10):
            node_name = f'node{i}'
            node_columns = {
                f'{node_name}_patterns': 'INTEGER',
                f'{node_name}_storage_mb': 'REAL',
                f'{node_name}_mean_freq': 'REAL',
                f'{node_name}_zipf_alpha': 'REAL',
                f'{node_name}_mean_shannon': 'REAL',
            }

            for col_name, col_type in node_columns.items():
                if col_name not in existing_columns:
                    cursor.execute(f'ALTER TABLE training_runs ADD COLUMN {col_name} {col_type}')

    def _generate_descriptive_run_id(
        self,
        config: Dict[str, Any],
        samples_processed: int
    ) -> str:
        """
        Generate descriptive run ID from configuration.

        Format:
        - Uniform chunks: run_c{size}x{levels}_w{workers}_s{samples}
        - Non-uniform chunks: run_c{size1}-{size2}-..._w{workers}_s{samples}

        Args:
            config: Training configuration
            samples_processed: Number of samples

        Returns:
            Descriptive run ID string
        """
        chunk_sizes = config.get('chunk_sizes', [])
        num_workers = config.get('num_workers', 0)

        # Format chunk size part
        if chunk_sizes:
            if len(set(chunk_sizes)) == 1:
                # Uniform: c8x5 (chunk_size=8, 5 levels)
                chunk_part = f"c{chunk_sizes[0]}x{len(chunk_sizes)}"
            else:
                # Non-uniform: c4-5-10-15
                chunk_part = f"c{'-'.join(map(str, chunk_sizes))}"
        else:
            chunk_part = "c0x0"

        # Format: run_c8x5_w6_s100
        run_id = f"run_{chunk_part}_w{num_workers}_s{samples_processed}"

        return run_id

    def _ensure_unique_run_id(self, run_id: str) -> str:
        """
        Ensure run ID is unique by appending timestamp if duplicate exists.

        Args:
            run_id: Proposed run ID

        Returns:
            Unique run ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT run_id FROM training_runs WHERE run_id = ?', (run_id,))
        existing = cursor.fetchone()
        conn.close()

        if existing:
            # Append timestamp to make unique
            run_id = f"{run_id}_{int(time.time())}"

        return run_id

    def record_run(
        self,
        config: Dict[str, Any],
        estimated_time: float,
        actual_time: float,
        estimated_storage_gb: float,
        actual_storage_gb: float,
        samples_processed: int = 0,
        patterns_learned: Optional[Dict[str, int]] = None,
        profiling_report: Optional[Any] = None,
        dataset_key: str = 'unknown',
        hardware_tier: str = 'unknown',
        notes: str = ''
    ) -> str:
        """
        Record a training run.

        Args:
            config: Training configuration
            estimated_time: Estimated time in seconds
            actual_time: Actual time in seconds
            estimated_storage_gb: Estimated storage
            actual_storage_gb: Actual storage
            samples_processed: Number of samples trained
            patterns_learned: Dict of patterns per node
            profiling_report: Optional ProfilingReport object
            dataset_key: Dataset identifier
            hardware_tier: Hardware tier used
            notes: Optional notes

        Returns:
            run_id of recorded run
        """
        # Generate descriptive run ID
        run_id = self._generate_descriptive_run_id(config, samples_processed)
        run_id = self._ensure_unique_run_id(run_id)

        # Extract metrics from profiling report
        estimated_memory_mb = None
        actual_memory_mb = None
        peak_memory_mb = None
        avg_cpu_percent = None
        disk_write_speed_mbps = None
        samples_per_second = 0.0

        if profiling_report:
            actual_memory_mb = profiling_report.avg_memory_mb
            peak_memory_mb = profiling_report.peak_memory_mb
            avg_cpu_percent = profiling_report.avg_cpu_percent
            disk_write_speed_mbps = profiling_report.avg_write_speed_mbps
            samples_per_second = profiling_report.samples_per_second

        # Create run record
        run = TrainingRun(
            run_id=run_id,
            timestamp=time.time(),
            config=config,
            estimated_time_seconds=estimated_time,
            estimated_storage_gb=estimated_storage_gb,
            estimated_memory_mb=estimated_memory_mb,
            actual_time_seconds=actual_time,
            actual_storage_gb=actual_storage_gb,
            actual_memory_mb=actual_memory_mb,
            samples_processed=samples_processed,
            samples_per_second=samples_per_second,
            patterns_learned=patterns_learned or {},
            peak_memory_mb=peak_memory_mb,
            avg_cpu_percent=avg_cpu_percent,
            disk_write_speed_mbps=disk_write_speed_mbps,
            errors=[],
            warnings=[],
            dataset_key=dataset_key,
            hardware_tier=hardware_tier,
            notes=notes
        )

        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert only the base columns (snapshot columns will be NULL initially)
        cursor.execute('''
            INSERT INTO training_runs (
                run_id, timestamp, config,
                estimated_time_seconds, estimated_storage_gb, estimated_memory_mb,
                actual_time_seconds, actual_storage_gb, actual_memory_mb,
                samples_processed, samples_per_second, patterns_learned,
                peak_memory_mb, avg_cpu_percent, disk_write_speed_mbps,
                errors, warnings, dataset_key, hardware_tier, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run.run_id,
            run.timestamp,
            json.dumps(run.config),
            run.estimated_time_seconds,
            run.estimated_storage_gb,
            run.estimated_memory_mb,
            run.actual_time_seconds,
            run.actual_storage_gb,
            run.actual_memory_mb,
            run.samples_processed,
            run.samples_per_second,
            json.dumps(run.patterns_learned),
            run.peak_memory_mb,
            run.avg_cpu_percent,
            run.disk_write_speed_mbps,
            json.dumps(run.errors),
            json.dumps(run.warnings),
            run.dataset_key,
            run.hardware_tier,
            run.notes
        ))

        conn.commit()
        conn.close()

        if self.verbose:
            time_error = run.get_time_error_percent()
            storage_error = run.get_storage_error_percent()
            print(f"✓ Recorded run {run_id}")
            print(f"  Time error: {time_error:.1f}%")
            print(f"  Storage error: {storage_error:.1f}%")

        return run_id

    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        """Get a specific run by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM training_runs WHERE run_id = ?', (run_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_run(row)

    def get_all_runs(self) -> List[TrainingRun]:
        """Get all training runs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM training_runs ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def get_recent_runs(self, n: int = 10) -> List[TrainingRun]:
        """Get N most recent runs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM training_runs ORDER BY timestamp DESC LIMIT ?', (n,))
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def find_similar_runs(
        self,
        config: Dict[str, Any],
        max_results: int = 10
    ) -> List[TrainingRun]:
        """
        Find training runs with similar configurations.

        Args:
            config: Configuration to match
            max_results: Maximum number of results

        Returns:
            List of similar TrainingRun objects
        """
        all_runs = self.get_all_runs()

        # Score runs by configuration similarity
        scored_runs = []
        for run in all_runs:
            similarity_score = self._calculate_config_similarity(config, run.config)
            scored_runs.append((similarity_score, run))

        # Sort by similarity (descending)
        scored_runs.sort(key=lambda x: x[0], reverse=True)

        return [run for _, run in scored_runs[:max_results]]

    def get_estimation_accuracy(self) -> EstimationAccuracy:
        """
        Calculate overall estimation accuracy.

        Returns:
            EstimationAccuracy with error metrics
        """
        runs = self.get_all_runs()

        if not runs:
            return EstimationAccuracy(
                total_runs=0,
                avg_time_error_percent=0.0,
                avg_storage_error_percent=0.0,
                avg_memory_error_percent=None,
                time_improvement_trend='unknown',
                storage_improvement_trend='unknown'
            )

        # Calculate average errors
        time_errors = [run.get_time_error_percent() for run in runs]
        storage_errors = [run.get_storage_error_percent() for run in runs]
        memory_errors = [run.get_memory_error_percent() for run in runs if run.get_memory_error_percent() is not None]

        avg_time_error = np.mean(time_errors)
        avg_storage_error = np.mean(storage_errors)
        avg_memory_error = np.mean(memory_errors) if memory_errors else None

        # Analyze improvement trends (first half vs second half)
        time_trend = self._analyze_trend(time_errors)
        storage_trend = self._analyze_trend(storage_errors)

        return EstimationAccuracy(
            total_runs=len(runs),
            avg_time_error_percent=avg_time_error,
            avg_storage_error_percent=avg_storage_error,
            avg_memory_error_percent=avg_memory_error,
            time_improvement_trend=time_trend,
            storage_improvement_trend=storage_trend
        )

    def print_summary(self):
        """Print summary of training history"""
        runs = self.get_all_runs()
        accuracy = self.get_estimation_accuracy()

        print("\n" + "="*80)
        print("TRAINING HISTORY SUMMARY")
        print("="*80)

        print(f"\n📊 DATABASE STATISTICS")
        print(f"  Total runs: {len(runs)}")

        if runs:
            print(f"  Date range: {time.strftime('%Y-%m-%d', time.localtime(runs[-1].timestamp))} → "
                  f"{time.strftime('%Y-%m-%d', time.localtime(runs[0].timestamp))}")

        print(f"\n🎯 ESTIMATION ACCURACY")
        print(f"  Time error: {accuracy.avg_time_error_percent:.1f}% (trend: {accuracy.time_improvement_trend})")
        print(f"  Storage error: {accuracy.avg_storage_error_percent:.1f}% (trend: {accuracy.storage_improvement_trend})")
        if accuracy.avg_memory_error_percent:
            print(f"  Memory error: {accuracy.avg_memory_error_percent:.1f}%")

        if runs:
            print(f"\n🏆 RECENT RUNS")
            for run in runs[:5]:
                print(f"\n  {run.run_id} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(run.timestamp))})")
                print(f"    Config: {run.config.get('num_levels', '?')} levels, "
                      f"chunk_size={run.config.get('chunk_sizes', ['?'])[0]}")

                # Time stats
                if run.estimated_time_seconds is not None:
                    print(f"    Time: {run.actual_time_seconds:.1f}s (estimated: {run.estimated_time_seconds:.1f}s, "
                          f"error: {run.get_time_error_percent():.1f}%)")
                else:
                    print(f"    Time: {run.actual_time_seconds:.1f}s (no estimate)")

                # Storage stats
                if run.estimated_storage_gb is not None:
                    print(f"    Storage: {run.actual_storage_gb:.2f}GB (estimated: {run.estimated_storage_gb:.2f}GB, "
                          f"error: {run.get_storage_error_percent():.1f}%)")
                else:
                    print(f"    Storage: {run.actual_storage_gb:.2f}GB (no estimate)")

        print("\n" + "="*80)

    def export_csv(self, filepath: str):
        """Export all runs as CSV"""
        import csv

        runs = self.get_all_runs()

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'run_id', 'timestamp', 'num_levels', 'chunk_size',
                'estimated_time_s', 'actual_time_s', 'time_error_pct',
                'estimated_storage_gb', 'actual_storage_gb', 'storage_error_pct',
                'samples_processed', 'samples_per_sec',
                'peak_memory_mb', 'avg_cpu_pct'
            ])

            # Data
            for run in runs:
                writer.writerow([
                    run.run_id,
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run.timestamp)),
                    run.config.get('num_levels', ''),
                    run.config.get('chunk_sizes', [''])[0],
                    run.estimated_time_seconds,
                    run.actual_time_seconds,
                    run.get_time_error_percent(),
                    run.estimated_storage_gb,
                    run.actual_storage_gb,
                    run.get_storage_error_percent(),
                    run.samples_processed,
                    run.samples_per_second,
                    run.peak_memory_mb or '',
                    run.avg_cpu_percent or ''
                ])

        print(f"✓ Exported {len(runs)} runs to {filepath}")

    def get_run_count(self) -> int:
        """Get total number of runs in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM training_runs')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def clear_all_runs(self):
        """Delete all runs from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM training_runs')
        conn.commit()
        conn.close()

        if self.verbose:
            print("✓ All runs cleared from database")

    def _row_to_run(self, row) -> TrainingRun:
        """Convert database row to TrainingRun object"""
        return TrainingRun(
            run_id=row[0],
            timestamp=row[1],
            config=json.loads(row[2]),
            estimated_time_seconds=row[3],
            estimated_storage_gb=row[4],
            estimated_memory_mb=row[5],
            actual_time_seconds=row[6],
            actual_storage_gb=row[7],
            actual_memory_mb=row[8],
            samples_processed=row[9],
            samples_per_second=row[10],
            patterns_learned=json.loads(row[11]) if row[11] else {},
            peak_memory_mb=row[12],
            avg_cpu_percent=row[13],
            disk_write_speed_mbps=row[14],
            errors=json.loads(row[15]) if row[15] else [],
            warnings=json.loads(row[16]) if row[16] else [],
            dataset_key=row[17],
            hardware_tier=row[18],
            notes=row[19]
        )

    def _calculate_config_similarity(self, config1: Dict, config2: Dict) -> float:
        """Calculate similarity score between two configurations (0-1)"""
        score = 0.0
        total_weight = 0.0

        # Compare num_levels (weight: 0.3)
        if 'num_levels' in config1 and 'num_levels' in config2:
            if config1['num_levels'] == config2['num_levels']:
                score += 0.3
            total_weight += 0.3

        # Compare chunk_sizes (weight: 0.4)
        if 'chunk_sizes' in config1 and 'chunk_sizes' in config2:
            cs1 = config1['chunk_sizes']
            cs2 = config2['chunk_sizes']
            if cs1 == cs2:
                score += 0.4
            elif len(cs1) > 0 and len(cs2) > 0 and cs1[0] == cs2[0]:
                score += 0.2  # Partial credit for first chunk size match
            total_weight += 0.4

        # Compare batch_size (weight: 0.15)
        if 'batch_size' in config1 and 'batch_size' in config2:
            if config1['batch_size'] == config2['batch_size']:
                score += 0.15
            total_weight += 0.15

        # Compare num_workers (weight: 0.15)
        if 'num_workers' in config1 and 'num_workers' in config2:
            if config1['num_workers'] == config2['num_workers']:
                score += 0.15
            total_weight += 0.15

        return score / total_weight if total_weight > 0 else 0.0

    def _analyze_trend(self, errors: List[float]) -> str:
        """Analyze if errors are improving, stable, or degrading"""
        if len(errors) < 4:
            return 'insufficient_data'

        # Compare first half to second half
        mid = len(errors) // 2
        first_half_avg = np.mean(errors[:mid])
        second_half_avg = np.mean(errors[mid:])

        if first_half_avg == 0:
            return 'insufficient_data'

        improvement = (first_half_avg - second_half_avg) / first_half_avg * 100

        if improvement > 10:
            return 'improving'
        elif improvement < -10:
            return 'degrading'
        else:
            return 'stable'

    def capture_snapshot(
        self,
        learner: Any,
        run_id: str,
        mongo_uri: str = 'mongodb://localhost:27017/',
        snapshots_dir: str = './snapshots',
        verbose: bool = True
    ):
        """
        Capture complete knowledge base snapshot after training.

        IMPORTANT: Call this BEFORE clearing nodes for the next training run!

        Args:
            learner: HierarchicalConceptLearner instance
            run_id: Run identifier to associate snapshot with
            mongo_uri: MongoDB connection string
            snapshots_dir: Directory to save snapshot JSON files
            verbose: Print progress

        Returns:
            TrainingRunSnapshot object
        """
        from .training_snapshot import TrainingRunSnapshot

        # Create snapshot
        snapshot = TrainingRunSnapshot.create_from_learner(
            learner=learner,
            run_id=run_id,
            mongo_uri=mongo_uri,
            verbose=verbose
        )

        # Save to file
        snapshot_path = f"{snapshots_dir}/{run_id}_snapshot.json"
        snapshot.save(snapshot_path)

        # Update database with snapshot metadata
        summary = snapshot.get_summary()
        self._update_run_with_snapshot(run_id, summary)

        if verbose:
            print(f"✓ Snapshot saved: {snapshot_path}")

        return snapshot

    def _update_run_with_snapshot(self, run_id: str, snapshot_summary: Dict[str, Any]):
        """Update training run record with snapshot data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build UPDATE query dynamically
        set_clauses = []
        values = []

        for key, value in snapshot_summary.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)

        values.append(run_id)  # For WHERE clause

        query = f"UPDATE training_runs SET {', '.join(set_clauses)} WHERE run_id = ?"
        cursor.execute(query, values)

        conn.commit()
        conn.close()

    def load_snapshot(self, run_id: str) -> Optional[Any]:
        """
        Load snapshot for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            TrainingRunSnapshot or None if not found
        """
        from .training_snapshot import TrainingRunSnapshot

        # Get snapshot path from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT snapshot_path FROM training_runs WHERE run_id = ?', (run_id,))
        row = cursor.fetchone()
        conn.close()

        if not row or not row[0]:
            return None

        snapshot_path = row[0]

        try:
            return TrainingRunSnapshot.load(snapshot_path)
        except Exception as e:
            print(f"⚠️  Failed to load snapshot: {e}")
            return None

    def compare_runs(
        self,
        run_ids: Optional[List[str]] = None,
        n_recent: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compare multiple training runs in a DataFrame.

        Args:
            run_ids: Specific run IDs to compare (if None, use n_recent)
            n_recent: Number of recent runs to compare (default: all)

        Returns:
            pandas DataFrame with comparison data
        """
        if run_ids:
            runs = [self.get_run(rid) for rid in run_ids]
            runs = [r for r in runs if r is not None]
        elif n_recent:
            runs = self.get_recent_runs(n=n_recent)
        else:
            runs = self.get_all_runs()

        if not runs:
            return pd.DataFrame()

        # Build comparison data
        comparison_data = []

        for run in runs:
            # Extract configuration
            chunk_sizes = run.config.get('chunk_sizes', [])

            # Compute chunk size metadata
            if chunk_sizes:
                # Check if uniform
                chunk_uniform = len(set(chunk_sizes)) == 1
                chunk_min = min(chunk_sizes)
                chunk_max = max(chunk_sizes)
                chunk_mean = np.mean(chunk_sizes)

                # Detect pattern type
                if chunk_uniform:
                    chunk_pattern = 'uniform'
                    chunk_size_str = str(chunk_sizes[0])
                elif all(chunk_sizes[i] <= chunk_sizes[i+1] for i in range(len(chunk_sizes)-1)):
                    chunk_pattern = 'increasing'
                    chunk_size_str = ','.join(map(str, chunk_sizes))
                elif all(chunk_sizes[i] >= chunk_sizes[i+1] for i in range(len(chunk_sizes)-1)):
                    chunk_pattern = 'decreasing'
                    chunk_size_str = ','.join(map(str, chunk_sizes))
                else:
                    chunk_pattern = 'mixed'
                    chunk_size_str = ','.join(map(str, chunk_sizes))
            else:
                chunk_uniform = True
                chunk_min = chunk_max = chunk_mean = None
                chunk_pattern = 'unknown'
                chunk_size_str = 'N/A'

            # Load snapshot data if available
            snapshot = self.load_snapshot(run.run_id)
            total_patterns = snapshot.total_patterns if snapshot else run.patterns_learned.get('total', 0)
            total_storage_mb = snapshot.total_storage_mb if snapshot else run.actual_storage_gb * 1024

            # Get node-specific data from snapshot
            zipf_alpha_node0 = None
            if snapshot and 'node0' in snapshot.nodes:
                zipf_alpha_node0 = snapshot.nodes['node0'].zipf_alpha

            # Calculate rate from actual data (profiling engine may not have tracked samples)
            rate_samples_per_sec = run.samples_processed / run.actual_time_seconds if run.actual_time_seconds > 0 else 0

            row = {
                'Run ID': run.run_id[:20],
                'Timestamp': time.strftime('%Y-%m-%d %H:%M', time.localtime(run.timestamp)),
                'Dataset': run.dataset_key,
                'Samples': run.samples_processed,
                'Chunk Size': chunk_size_str,
                'Chunk Pattern': chunk_pattern,
                'Chunk Min': chunk_min,
                'Chunk Max': chunk_max,
                'Chunk Mean': round(chunk_mean, 1) if chunk_mean else None,
                'Chunk Uniform': chunk_uniform,
                'Levels': run.config.get('num_levels', 'N/A'),
                'Batch Size': run.config.get('batch_size', 'N/A'),
                'Workers': run.config.get('num_workers', 'N/A'),
                'Time (s)': round(run.actual_time_seconds, 2),
                'Rate (samples/s)': round(rate_samples_per_sec, 2),
                'Storage (MB)': round(total_storage_mb, 2),
                'Total Patterns': total_patterns,
                'Zipf Alpha': round(zipf_alpha_node0, 3) if zipf_alpha_node0 else None,
                'Hardware': run.hardware_tier,
                'Peak Mem (MB)': round(run.peak_memory_mb, 1) if run.peak_memory_mb else None,
                'Avg CPU (%)': round(run.avg_cpu_percent, 1) if run.avg_cpu_percent else None,
            }

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by timestamp (newest first)
        if 'Timestamp' in df.columns:
            df = df.sort_values('Timestamp', ascending=False)

        return df


def main():
    """Example usage"""
    history = TrainingHistory(db_path='./test_training_history.db', verbose=True)

    # Simulate recording several runs
    configs = [
        {'num_levels': 3, 'chunk_sizes': [15, 15, 15], 'batch_size': 1},
        {'num_levels': 4, 'chunk_sizes': [15, 15, 15, 15], 'batch_size': 50},
        {'num_levels': 4, 'chunk_sizes': [10, 10, 10, 10], 'batch_size': 50},
    ]

    for i, config in enumerate(configs):
        # Simulate improving estimates
        est_time = 100 + i * 20
        act_time = est_time * (1.2 - i * 0.05)  # Estimates get better

        est_storage = 2.0 + i * 0.5
        act_storage = est_storage * (1.15 - i * 0.03)  # Estimates get better

        history.record_run(
            config=config,
            estimated_time=est_time,
            actual_time=act_time,
            estimated_storage_gb=est_storage,
            actual_storage_gb=act_storage,
            samples_processed=1000 * (i + 1),
            patterns_learned={'node0': 500 + i * 100, 'node1': 80 + i * 20},
            dataset_key='wikitext',
            hardware_tier='medium',
            notes=f'Test run {i+1}'
        )

        # Small delay to ensure different timestamps
        time.sleep(0.1)

    # Analyze
    history.print_summary()

    # Export
    history.export_csv('training_history.csv')


if __name__ == '__main__':
    main()
