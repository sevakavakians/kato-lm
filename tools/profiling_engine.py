#!/usr/bin/env python3
"""
Profiling Engine - Real-time Resource Monitoring for KATO Training

This module provides comprehensive profiling capabilities to track:
- Memory usage (RAM: peak, average, per-sample)
- CPU utilization (per-core, bottleneck detection)
- Disk I/O (MongoDB write/read speeds)
- Network latency (KATO API calls)
- Per-node learning rates

Usage:
    profiler = ProfilingEngine()
    profiler.start()

    # ... training code ...
    profiler.record_event('node0_learn', duration_ms=45.2)
    profiler.record_sample_processed()

    profiler.stop()
    report = profiler.generate_report()
"""

import time
import psutil
import threading
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import json


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement"""
    timestamp: float
    memory_mb: float
    memory_percent: float
    cpu_percent: float
    cpu_per_core: List[float]
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float


@dataclass
class ProfilingReport:
    """Comprehensive profiling results"""
    # Time metrics
    total_duration_seconds: float
    training_duration_seconds: float

    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    memory_per_sample_mb: float
    memory_trend: str  # 'stable', 'growing', 'declining'

    # CPU metrics
    avg_cpu_percent: float
    peak_cpu_percent: float
    cpu_bottleneck_detected: bool
    avg_cpu_per_core: List[float]

    # Disk I/O metrics
    total_disk_read_mb: float
    total_disk_write_mb: float
    avg_write_speed_mbps: float
    disk_io_bottleneck_detected: bool

    # Network metrics
    total_network_sent_mb: float
    total_network_recv_mb: float
    avg_network_latency_ms: float

    # Training metrics
    samples_processed: int
    samples_per_second: float

    # Bottleneck analysis
    primary_bottleneck: str  # 'cpu', 'memory', 'disk', 'network', 'none'
    bottleneck_confidence: float

    # Optional fields with defaults (must come after required fields)
    tokens_per_second: Optional[float] = None

    # Per-node metrics
    node_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Event metrics
    event_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


class ProfilingEngine:
    """
    Real-time profiling engine for KATO hierarchical training.

    Monitors system resources and training events to provide comprehensive
    performance analysis and bottleneck identification.
    """

    def __init__(self, sampling_interval_seconds: float = 1.0, verbose: bool = False):
        """
        Initialize profiling engine.

        Args:
            sampling_interval_seconds: How often to sample system resources
            verbose: Print profiling events in real-time
        """
        self.sampling_interval = sampling_interval_seconds
        self.verbose = verbose

        # State
        self.is_running = False
        self.start_time = None
        self.stop_time = None

        # Resource snapshots
        self.snapshots: List[ResourceSnapshot] = []

        # Event tracking
        self.events = defaultdict(list)  # event_type -> [durations]
        self.node_events = defaultdict(lambda: defaultdict(list))  # node -> event_type -> [durations]

        # Training metrics
        self.samples_processed = 0
        self.tokens_processed = 0
        self.sample_timestamps = []

        # Network latency tracking
        self.network_latencies = []

        # Threading
        self.monitor_thread = None
        self._stop_event = threading.Event()

        # Initial measurements
        self.process = psutil.Process()
        self.initial_io = psutil.disk_io_counters()
        self.initial_net = psutil.net_io_counters()

        if self.verbose:
            print("✓ ProfilingEngine initialized")

    def start(self):
        """Start profiling and resource monitoring"""
        if self.is_running:
            raise RuntimeError("Profiling already running")

        self.is_running = True
        self.start_time = time.time()
        self._stop_event.clear()

        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        if self.verbose:
            print(f"✓ Profiling started at {time.strftime('%H:%M:%S')}")

    def stop(self):
        """Stop profiling"""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_time = time.time()
        self._stop_event.set()

        # Wait for monitor thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        if self.verbose:
            duration = self.stop_time - self.start_time
            print(f"✓ Profiling stopped after {duration:.2f}s")

    def _monitor_loop(self):
        """Background thread that samples system resources"""
        while not self._stop_event.is_set():
            try:
                snapshot = self._capture_snapshot()
                self.snapshots.append(snapshot)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to capture snapshot: {e}")

            # Sleep for sampling interval
            self._stop_event.wait(self.sampling_interval)

    def _capture_snapshot(self) -> ResourceSnapshot:
        """Capture current system resource state"""
        # Memory
        mem = self.process.memory_info()
        mem_mb = mem.rss / 1024 / 1024
        mem_percent = self.process.memory_percent()

        # CPU
        cpu_percent = self.process.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

        # Disk I/O
        try:
            io = psutil.disk_io_counters()
            disk_read_mb = (io.read_bytes - self.initial_io.read_bytes) / 1024 / 1024
            disk_write_mb = (io.write_bytes - self.initial_io.write_bytes) / 1024 / 1024
        except:
            disk_read_mb = disk_write_mb = 0

        # Network I/O
        try:
            net = psutil.net_io_counters()
            net_sent_mb = (net.bytes_sent - self.initial_net.bytes_sent) / 1024 / 1024
            net_recv_mb = (net.bytes_recv - self.initial_net.bytes_recv) / 1024 / 1024
        except:
            net_sent_mb = net_recv_mb = 0

        return ResourceSnapshot(
            timestamp=time.time(),
            memory_mb=mem_mb,
            memory_percent=mem_percent,
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb
        )

    def record_event(self, event_type: str, duration_ms: float, node: Optional[str] = None):
        """
        Record a training event with its duration.

        Args:
            event_type: Type of event (e.g., 'node0_learn', 'tokenization', 'mongodb_write')
            duration_ms: Duration in milliseconds
            node: Optional node identifier (e.g., 'node0', 'node1')
        """
        self.events[event_type].append(duration_ms)

        if node:
            self.node_events[node][event_type].append(duration_ms)

        if self.verbose and len(self.events[event_type]) % 100 == 0:
            avg_ms = np.mean(self.events[event_type])
            print(f"  {event_type}: {len(self.events[event_type])} events, avg {avg_ms:.2f}ms")

    def record_sample_processed(self, num_tokens: Optional[int] = None):
        """
        Record that a sample was processed.

        Args:
            num_tokens: Optional token count for this sample
        """
        self.samples_processed += 1
        self.sample_timestamps.append(time.time())

        if num_tokens:
            self.tokens_processed += num_tokens

    def record_network_latency(self, latency_ms: float):
        """Record KATO API network latency"""
        self.network_latencies.append(latency_ms)

    def generate_report(self) -> ProfilingReport:
        """
        Generate comprehensive profiling report.

        Returns:
            ProfilingReport with all metrics and analysis
        """
        if self.is_running:
            self.stop()

        if not self.snapshots:
            raise RuntimeError("No profiling data collected")

        # Time metrics
        total_duration = self.stop_time - self.start_time

        # Memory metrics
        memory_values = [s.memory_mb for s in self.snapshots]
        peak_memory = max(memory_values)
        avg_memory = np.mean(memory_values)
        memory_per_sample = avg_memory / self.samples_processed if self.samples_processed > 0 else 0

        # Detect memory trend
        if len(memory_values) > 10:
            first_half_avg = np.mean(memory_values[:len(memory_values)//2])
            second_half_avg = np.mean(memory_values[len(memory_values)//2:])
            growth_rate = (second_half_avg - first_half_avg) / first_half_avg

            if abs(growth_rate) < 0.1:
                memory_trend = 'stable'
            elif growth_rate > 0:
                memory_trend = 'growing'
            else:
                memory_trend = 'declining'
        else:
            memory_trend = 'unknown'

        # CPU metrics
        cpu_values = [s.cpu_percent for s in self.snapshots]
        avg_cpu = np.mean(cpu_values)
        peak_cpu = max(cpu_values)
        cpu_bottleneck = avg_cpu > 80  # Threshold for bottleneck detection

        # Average CPU per core
        num_cores = len(self.snapshots[0].cpu_per_core) if self.snapshots else 0
        avg_cpu_per_core = []
        if num_cores > 0:
            for core_idx in range(num_cores):
                core_values = [s.cpu_per_core[core_idx] for s in self.snapshots if len(s.cpu_per_core) > core_idx]
                avg_cpu_per_core.append(np.mean(core_values) if core_values else 0)

        # Disk I/O metrics
        final_snapshot = self.snapshots[-1]
        total_disk_read = final_snapshot.disk_io_read_mb
        total_disk_write = final_snapshot.disk_io_write_mb
        avg_write_speed = total_disk_write / total_duration if total_duration > 0 else 0
        disk_io_bottleneck = avg_write_speed < 10  # Less than 10 MB/s indicates bottleneck

        # Network metrics
        total_net_sent = final_snapshot.network_sent_mb
        total_net_recv = final_snapshot.network_recv_mb
        avg_net_latency = np.mean(self.network_latencies) if self.network_latencies else 0

        # Training metrics
        samples_per_second = self.samples_processed / total_duration if total_duration > 0 else 0
        tokens_per_second = self.tokens_processed / total_duration if total_duration > 0 and self.tokens_processed > 0 else None

        # Per-node statistics
        node_stats = {}
        for node, events_dict in self.node_events.items():
            node_stats[node] = {}
            for event_type, durations in events_dict.items():
                node_stats[node][event_type] = {
                    'count': len(durations),
                    'avg_ms': np.mean(durations),
                    'std_ms': np.std(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                    'total_seconds': sum(durations) / 1000
                }

        # Event statistics
        event_stats = {}
        for event_type, durations in self.events.items():
            event_stats[event_type] = {
                'count': len(durations),
                'avg_ms': np.mean(durations),
                'std_ms': np.std(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'total_seconds': sum(durations) / 1000
            }

        # Bottleneck analysis
        bottleneck, confidence = self._identify_bottleneck(
            avg_cpu=avg_cpu,
            avg_memory=avg_memory,
            avg_write_speed=avg_write_speed,
            avg_net_latency=avg_net_latency
        )

        return ProfilingReport(
            total_duration_seconds=total_duration,
            training_duration_seconds=total_duration,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            memory_per_sample_mb=memory_per_sample,
            memory_trend=memory_trend,
            avg_cpu_percent=avg_cpu,
            peak_cpu_percent=peak_cpu,
            cpu_bottleneck_detected=cpu_bottleneck,
            avg_cpu_per_core=avg_cpu_per_core,
            total_disk_read_mb=total_disk_read,
            total_disk_write_mb=total_disk_write,
            avg_write_speed_mbps=avg_write_speed,
            disk_io_bottleneck_detected=disk_io_bottleneck,
            total_network_sent_mb=total_net_sent,
            total_network_recv_mb=total_net_recv,
            avg_network_latency_ms=avg_net_latency,
            samples_processed=self.samples_processed,
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
            node_stats=node_stats,
            event_stats=event_stats,
            primary_bottleneck=bottleneck,
            bottleneck_confidence=confidence
        )

    def _identify_bottleneck(self, avg_cpu: float, avg_memory: float,
                            avg_write_speed: float, avg_net_latency: float) -> tuple:
        """
        Identify primary performance bottleneck.

        Returns:
            (bottleneck_type, confidence) tuple
        """
        # Scoring system
        scores = {
            'cpu': 0.0,
            'memory': 0.0,
            'disk': 0.0,
            'network': 0.0
        }

        # CPU bottleneck indicators
        if avg_cpu > 90:
            scores['cpu'] = 1.0
        elif avg_cpu > 75:
            scores['cpu'] = 0.7
        elif avg_cpu > 60:
            scores['cpu'] = 0.4

        # Memory bottleneck indicators (>80% memory usage)
        if avg_memory > 80:
            scores['memory'] = 1.0
        elif avg_memory > 70:
            scores['memory'] = 0.6

        # Disk bottleneck indicators (slow write speed)
        if avg_write_speed < 5:
            scores['disk'] = 1.0
        elif avg_write_speed < 15:
            scores['disk'] = 0.6
        elif avg_write_speed < 30:
            scores['disk'] = 0.3

        # Network bottleneck indicators
        if avg_net_latency > 100:
            scores['network'] = 1.0
        elif avg_net_latency > 50:
            scores['network'] = 0.6
        elif avg_net_latency > 20:
            scores['network'] = 0.3

        # Find primary bottleneck
        if max(scores.values()) < 0.3:
            return 'none', 0.0

        primary = max(scores, key=scores.get)
        confidence = scores[primary]

        return primary, confidence

    def print_summary(self):
        """Print a human-readable summary of profiling results"""
        report = self.generate_report()

        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)

        print(f"\n⏱️  TIMING")
        print(f"  Total duration: {report.total_duration_seconds:.2f}s")
        print(f"  Samples processed: {report.samples_processed:,}")
        print(f"  Throughput: {report.samples_per_second:.2f} samples/sec")
        if report.tokens_per_second:
            print(f"  Token throughput: {report.tokens_per_second:.1f} tokens/sec")

        print(f"\n💾 MEMORY")
        print(f"  Peak: {report.peak_memory_mb:.1f} MB")
        print(f"  Average: {report.avg_memory_mb:.1f} MB")
        print(f"  Per sample: {report.memory_per_sample_mb:.3f} MB/sample")
        print(f"  Trend: {report.memory_trend}")

        print(f"\n🔧 CPU")
        print(f"  Average utilization: {report.avg_cpu_percent:.1f}%")
        print(f"  Peak utilization: {report.peak_cpu_percent:.1f}%")
        if report.cpu_bottleneck_detected:
            print(f"  ⚠️  CPU BOTTLENECK DETECTED")

        print(f"\n💿 DISK I/O")
        print(f"  Total write: {report.total_disk_write_mb:.2f} MB")
        print(f"  Write speed: {report.avg_write_speed_mbps:.2f} MB/s")
        if report.disk_io_bottleneck_detected:
            print(f"  ⚠️  DISK I/O BOTTLENECK DETECTED")

        print(f"\n🌐 NETWORK")
        print(f"  Sent: {report.total_network_sent_mb:.2f} MB")
        print(f"  Received: {report.total_network_recv_mb:.2f} MB")
        if report.avg_network_latency_ms > 0:
            print(f"  Avg latency: {report.avg_network_latency_ms:.2f}ms")

        print(f"\n🔍 BOTTLENECK ANALYSIS")
        print(f"  Primary bottleneck: {report.primary_bottleneck.upper()}")
        print(f"  Confidence: {report.bottleneck_confidence*100:.1f}%")

        if report.node_stats:
            print(f"\n📊 PER-NODE STATISTICS")
            for node, stats in sorted(report.node_stats.items()):
                print(f"\n  {node}:")
                for event_type, metrics in stats.items():
                    print(f"    {event_type}: {metrics['count']:,} events, avg {metrics['avg_ms']:.2f}ms")

        print("\n" + "="*80)

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # For unknown types, try to convert to string
            return str(obj)

    def export_json(self, filepath: str):
        """Export profiling report as JSON"""
        report = self.generate_report()

        # Convert dataclass to dict
        data = {
            'total_duration_seconds': report.total_duration_seconds,
            'training_duration_seconds': report.training_duration_seconds,
            'peak_memory_mb': report.peak_memory_mb,
            'avg_memory_mb': report.avg_memory_mb,
            'memory_per_sample_mb': report.memory_per_sample_mb,
            'memory_trend': report.memory_trend,
            'avg_cpu_percent': report.avg_cpu_percent,
            'peak_cpu_percent': report.peak_cpu_percent,
            'cpu_bottleneck_detected': report.cpu_bottleneck_detected,
            'avg_cpu_per_core': report.avg_cpu_per_core,
            'total_disk_read_mb': report.total_disk_read_mb,
            'total_disk_write_mb': report.total_disk_write_mb,
            'avg_write_speed_mbps': report.avg_write_speed_mbps,
            'disk_io_bottleneck_detected': report.disk_io_bottleneck_detected,
            'total_network_sent_mb': report.total_network_sent_mb,
            'total_network_recv_mb': report.total_network_recv_mb,
            'avg_network_latency_ms': report.avg_network_latency_ms,
            'samples_processed': report.samples_processed,
            'samples_per_second': report.samples_per_second,
            'tokens_per_second': report.tokens_per_second,
            'node_stats': report.node_stats,
            'event_stats': report.event_stats,
            'primary_bottleneck': report.primary_bottleneck,
            'bottleneck_confidence': report.bottleneck_confidence
        }

        # Convert numpy types to native Python types
        data = self._convert_to_json_serializable(data)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Profiling report exported to {filepath}")


def main():
    """Example usage"""
    import random

    profiler = ProfilingEngine(sampling_interval_seconds=0.5, verbose=True)
    profiler.start()

    # Simulate training
    for i in range(100):
        time.sleep(random.uniform(0.01, 0.05))
        profiler.record_event('node0_learn', duration_ms=random.uniform(30, 60))
        profiler.record_sample_processed(num_tokens=random.randint(50, 150))

        if i % 10 == 0:
            profiler.record_event('node1_learn', duration_ms=random.uniform(40, 80), node='node1')

    profiler.stop()
    profiler.print_summary()
    profiler.export_json('profiling_report.json')


if __name__ == '__main__':
    main()
