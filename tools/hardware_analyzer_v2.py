#!/usr/bin/env python3
"""
Hardware Analyzer V2 - Enhanced Hardware Detection and Benchmarking

Extends the original hardware_analyzer.py with:
- MongoDB performance benchmarking
- Disk I/O speed measurement
- Network latency testing
- GPU detection (for future use)
- Detailed performance profiling

Usage:
    analyzer = HardwareAnalyzerV2()
    report = analyzer.analyze_system()
    report.print_summary()
    report.export_json('hardware_report.json')
"""

import platform
import subprocess
import time
import requests
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import tempfile
import os

# Import base analyzer
from tools.hardware_analyzer import HardwareAnalyzer


@dataclass
class ClickHouseBenchmark:
    """ClickHouse performance benchmark results (TODO: Implement)"""
    write_speed_rows_per_sec: float = 0.0
    read_speed_rows_per_sec: float = 0.0
    avg_write_latency_ms: float = 0.0
    avg_read_latency_ms: float = 0.0
    connection_latency_ms: float = 0.0
    available: bool = False
    error_message: Optional[str] = "ClickHouse benchmarking not implemented"


@dataclass
class DiskIOBenchmark:
    """Disk I/O performance benchmark results"""
    sequential_write_mbps: float
    sequential_read_mbps: float
    random_write_mbps: float
    random_read_mbps: float
    test_file_size_mb: int = 100


@dataclass
class NetworkBenchmark:
    """Network latency benchmarks"""
    kato_api_latency_ms: float
    kato_api_available: bool
    localhost_latency_ms: float


@dataclass
class GPUInfo:
    """GPU information (if available)"""
    gpu_available: bool
    gpu_count: int = 0
    gpu_names: List[str] = None
    gpu_memory_mb: List[int] = None

    def __post_init__(self):
        if self.gpu_names is None:
            self.gpu_names = []
        if self.gpu_memory_mb is None:
            self.gpu_memory_mb = []


@dataclass
class HardwareReportV2:
    """Comprehensive hardware analysis report"""
    # Basic info (from base analyzer)
    cpu_model: str
    cpu_cores: int
    memory_gb: float
    platform: str
    tier: str
    baseline_rate: float

    # Enhanced benchmarks
    clickhouse_benchmark: Optional[ClickHouseBenchmark] = None
    disk_io_benchmark: Optional[DiskIOBenchmark] = None
    network_benchmark: Optional[NetworkBenchmark] = None
    gpu_info: Optional[GPUInfo] = None

    # Performance estimates
    estimated_samples_per_sec: float
    estimated_tokens_per_sec: float
    bottleneck_prediction: str  # 'cpu', 'memory', 'disk', 'network', 'none'

    def print_summary(self):
        """Print human-readable hardware report"""
        print("\n" + "="*80)
        print("HARDWARE ANALYSIS REPORT V2")
        print("="*80)

        print(f"\n💻 SYSTEM INFO")
        print(f"  CPU: {self.cpu_model}")
        print(f"  Cores: {self.cpu_cores}")
        print(f"  Memory: {self.memory_gb:.1f} GB")
        print(f"  Platform: {self.platform}")
        print(f"  Performance Tier: {self.tier.upper()}")

        print(f"\n💾 CLICKHOUSE PERFORMANCE")
        if self.clickhouse_benchmark and self.clickhouse_benchmark.available:
            print(f"  Write speed: {self.clickhouse_benchmark.write_speed_rows_per_sec:.0f} rows/sec")
            print(f"  Read speed: {self.clickhouse_benchmark.read_speed_rows_per_sec:.0f} rows/sec")
            print(f"  Write latency: {self.clickhouse_benchmark.avg_write_latency_ms:.2f}ms")
            print(f"  Read latency: {self.clickhouse_benchmark.avg_read_latency_ms:.2f}ms")
            print(f"  Connection latency: {self.clickhouse_benchmark.connection_latency_ms:.2f}ms")
        elif self.clickhouse_benchmark:
            print(f"  ⚠️  ClickHouse not available: {self.clickhouse_benchmark.error_message}")
        else:
            print(f"  ⚠️  ClickHouse benchmark not run")

        print(f"\n💿 DISK I/O PERFORMANCE")
        print(f"  Sequential write: {self.disk_io_benchmark.sequential_write_mbps:.1f} MB/s")
        print(f"  Sequential read: {self.disk_io_benchmark.sequential_read_mbps:.1f} MB/s")
        print(f"  Random write: {self.disk_io_benchmark.random_write_mbps:.1f} MB/s")
        print(f"  Random read: {self.disk_io_benchmark.random_read_mbps:.1f} MB/s")

        print(f"\n🌐 NETWORK PERFORMANCE")
        if self.network_benchmark.kato_api_available:
            print(f"  KATO API latency: {self.network_benchmark.kato_api_latency_ms:.2f}ms")
        else:
            print(f"  ⚠️  KATO API not available")
        print(f"  Localhost latency: {self.network_benchmark.localhost_latency_ms:.2f}ms")

        if self.gpu_info.gpu_available:
            print(f"\n🎮 GPU INFO")
            print(f"  GPUs detected: {self.gpu_info.gpu_count}")
            for i, (name, mem) in enumerate(zip(self.gpu_info.gpu_names, self.gpu_info.gpu_memory_mb)):
                print(f"    GPU {i}: {name} ({mem} MB)")
        else:
            print(f"\n🎮 GPU INFO")
            print(f"  No GPU detected")

        print(f"\n⚡ PERFORMANCE ESTIMATES")
        print(f"  Estimated throughput: {self.estimated_samples_per_sec:.1f} samples/sec")
        print(f"  Estimated tokens: {self.estimated_tokens_per_sec:.0f} tokens/sec")

        print(f"\n🔍 BOTTLENECK PREDICTION")
        print(f"  Primary bottleneck: {self.bottleneck_prediction.upper()}")

        print("\n" + "="*80)

    def export_json(self, filepath: str):
        """Export report as JSON"""
        data = {
            'cpu_model': self.cpu_model,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'platform': self.platform,
            'tier': self.tier,
            'baseline_rate': self.baseline_rate,
            'clickhouse_benchmark': {
                'available': self.clickhouse_benchmark.available if self.clickhouse_benchmark else False,
                'write_speed_rows_per_sec': self.clickhouse_benchmark.write_speed_rows_per_sec if self.clickhouse_benchmark else 0,
                'read_speed_rows_per_sec': self.clickhouse_benchmark.read_speed_rows_per_sec if self.clickhouse_benchmark else 0,
                'avg_write_latency_ms': self.clickhouse_benchmark.avg_write_latency_ms if self.clickhouse_benchmark else 0,
                'avg_read_latency_ms': self.clickhouse_benchmark.avg_read_latency_ms if self.clickhouse_benchmark else 0,
                'connection_latency_ms': self.clickhouse_benchmark.connection_latency_ms if self.clickhouse_benchmark else 0,
                'error_message': self.clickhouse_benchmark.error_message if self.clickhouse_benchmark else "Not run"
            },
            'disk_io_benchmark': {
                'sequential_write_mbps': self.disk_io_benchmark.sequential_write_mbps,
                'sequential_read_mbps': self.disk_io_benchmark.sequential_read_mbps,
                'random_write_mbps': self.disk_io_benchmark.random_write_mbps,
                'random_read_mbps': self.disk_io_benchmark.random_read_mbps,
                'test_file_size_mb': self.disk_io_benchmark.test_file_size_mb
            },
            'network_benchmark': {
                'kato_api_latency_ms': self.network_benchmark.kato_api_latency_ms,
                'kato_api_available': self.network_benchmark.kato_api_available,
                'localhost_latency_ms': self.network_benchmark.localhost_latency_ms
            },
            'gpu_info': {
                'gpu_available': self.gpu_info.gpu_available,
                'gpu_count': self.gpu_info.gpu_count,
                'gpu_names': self.gpu_info.gpu_names,
                'gpu_memory_mb': self.gpu_info.gpu_memory_mb
            },
            'estimated_samples_per_sec': self.estimated_samples_per_sec,
            'estimated_tokens_per_sec': self.estimated_tokens_per_sec,
            'bottleneck_prediction': self.bottleneck_prediction
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Hardware report exported to {filepath}")


class HardwareAnalyzerV2:
    """
    Enhanced hardware analyzer with comprehensive benchmarking.

    Extends the base HardwareAnalyzer with:
    - MongoDB performance testing
    - Disk I/O benchmarking
    - Network latency measurement
    - GPU detection
    - Bottleneck prediction
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize enhanced hardware analyzer.

        Args:
            verbose: Print progress during analysis
        """
        self.verbose = verbose
        self.base_analyzer = HardwareAnalyzer()

        if self.verbose:
            print("✓ HardwareAnalyzerV2 initialized")

    def analyze_system(
        self,
        clickhouse_uri: str = "clickhouse://localhost:9000/",
        kato_url: str = "http://localhost:8000",
        training_config: Optional[Dict[str, Any]] = None,
        num_samples: int = 10000
    ) -> HardwareReportV2:
        """
        Perform comprehensive system analysis.

        Args:
            clickhouse_uri: ClickHouse connection URI for benchmarking (not yet implemented)
            kato_url: KATO API URL for latency testing
            training_config: Optional training config for accurate throughput estimates
            num_samples: Sample count for throughput estimation (default 10000)

        Returns:
            HardwareReportV2 with all benchmark results
        """
        if self.verbose:
            print("\n" + "="*80)
            print("RUNNING HARDWARE ANALYSIS V2")
            print("="*80)

        # Basic info from base analyzer
        cpu_model = self.base_analyzer.get_cpu_model()
        cpu_cores = self.base_analyzer.get_cpu_count()
        memory_gb = self.base_analyzer.get_memory_gb()
        platform_str = f"{platform.system()} {platform.release()}"
        tier = self.base_analyzer.classify_hardware()
        baseline_rate = self.base_analyzer.get_current_rate()

        if self.verbose:
            print(f"\n✓ System: {cpu_cores} cores, {memory_gb:.1f}GB RAM, tier={tier}")

        # ClickHouse benchmark (TODO: Implement)
        if self.verbose:
            print(f"\n⏳ ClickHouse benchmark not yet implemented, skipping...")
        clickhouse_bench = None  # TODO: Implement _benchmark_clickhouse(clickhouse_uri)

        # Disk I/O benchmark
        if self.verbose:
            print(f"\n⏳ Benchmarking disk I/O...")
        disk_bench = self._benchmark_disk_io()

        # Network benchmark
        if self.verbose:
            print(f"\n⏳ Testing network latency...")
        network_bench = self._benchmark_network(kato_url)

        # GPU detection
        if self.verbose:
            print(f"\n⏳ Detecting GPUs...")
        gpu_info = self._detect_gpus()

        # Performance estimates
        # Use TrainingEstimator if config provided, otherwise fallback to baseline
        if training_config:
            try:
                from tools.training_estimator import TrainingEstimator
                estimator = TrainingEstimator(verbose=False)
                time_estimate = estimator.estimate_training(
                    config=training_config,
                    num_samples=num_samples,
                    hardware_tier=tier
                )
                estimated_samples_per_sec = time_estimate.estimated_samples_per_sec
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not use TrainingEstimator: {e}")
                    print(f"   Falling back to baseline rate")
                estimated_samples_per_sec = baseline_rate
        else:
            estimated_samples_per_sec = baseline_rate

        estimated_tokens_per_sec = estimated_samples_per_sec * 500  # Assume avg 500 tokens/sample

        # Bottleneck prediction
        clickhouse_write_speed = clickhouse_bench.write_speed_rows_per_sec if clickhouse_bench else 0
        bottleneck = self._predict_bottleneck(
            tier=tier,
            storage_write_speed=clickhouse_write_speed,
            disk_write_speed=disk_bench.sequential_write_mbps,
            network_latency=network_bench.kato_api_latency_ms
        )

        report = HardwareReportV2(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            platform=platform_str,
            tier=tier,
            baseline_rate=baseline_rate,
            clickhouse_benchmark=clickhouse_bench,
            disk_io_benchmark=disk_bench,
            network_benchmark=network_bench,
            gpu_info=gpu_info,
            estimated_samples_per_sec=estimated_samples_per_sec,
            estimated_tokens_per_sec=estimated_tokens_per_sec,
            bottleneck_prediction=bottleneck
        )

        if self.verbose:
            print(f"\n✓ Hardware analysis complete")

        return report

    def _benchmark_clickhouse(self, clickhouse_uri: str) -> ClickHouseBenchmark:
        """
        Benchmark ClickHouse read/write performance.

        TODO: Implement ClickHouse benchmarking using clickhouse-connect library.
        Should test:
        - Connection latency test
        - Bulk insert test (1000 rows)
        - Read test (1000 rows)
        - Cleanup test table

        Returns:
            ClickHouseBenchmark with performance metrics (stub for now)
        """
        return ClickHouseBenchmark()  # Returns default "not implemented" benchmark

    def _benchmark_disk_io(self, test_size_mb: int = 100) -> DiskIOBenchmark:
        """
        Benchmark disk I/O performance.

        Tests:
        - Sequential write
        - Sequential read
        - Random write (simulated)
        - Random read (simulated)

        Args:
            test_size_mb: Size of test file in MB

        Returns:
            DiskIOBenchmark with speeds in MB/s
        """
        test_file = tempfile.NamedTemporaryFile(delete=False)
        test_path = test_file.name
        test_file.close()

        try:
            # Sequential write
            data = os.urandom(test_size_mb * 1024 * 1024)
            write_start = time.time()
            with open(test_path, 'wb') as f:
                f.write(data)
            write_duration = time.time() - write_start
            seq_write_mbps = test_size_mb / write_duration

            # Sequential read
            read_start = time.time()
            with open(test_path, 'rb') as f:
                _ = f.read()
            read_duration = time.time() - read_start
            seq_read_mbps = test_size_mb / read_duration

            # Random write (simulated with small chunks)
            chunk_size = 4 * 1024  # 4KB chunks
            num_chunks = 1000
            rand_write_start = time.time()
            with open(test_path, 'r+b') as f:
                for _ in range(num_chunks):
                    offset = np.random.randint(0, test_size_mb * 1024 * 1024 - chunk_size)
                    f.seek(offset)
                    f.write(os.urandom(chunk_size))
            rand_write_duration = time.time() - rand_write_start
            rand_write_mbps = (num_chunks * chunk_size / 1024 / 1024) / rand_write_duration

            # Random read (simulated)
            rand_read_start = time.time()
            with open(test_path, 'rb') as f:
                for _ in range(num_chunks):
                    offset = np.random.randint(0, test_size_mb * 1024 * 1024 - chunk_size)
                    f.seek(offset)
                    _ = f.read(chunk_size)
            rand_read_duration = time.time() - rand_read_start
            rand_read_mbps = (num_chunks * chunk_size / 1024 / 1024) / rand_read_duration

            return DiskIOBenchmark(
                sequential_write_mbps=seq_write_mbps,
                sequential_read_mbps=seq_read_mbps,
                random_write_mbps=rand_write_mbps,
                random_read_mbps=rand_read_mbps,
                test_file_size_mb=test_size_mb
            )

        finally:
            # Cleanup
            try:
                os.unlink(test_path)
            except:
                pass

    def _benchmark_network(self, kato_url: str) -> NetworkBenchmark:
        """
        Benchmark network latency to KATO API and localhost.

        Args:
            kato_url: KATO API URL

        Returns:
            NetworkBenchmark with latency measurements
        """
        # Test KATO API
        kato_available = False
        kato_latency = 0.0

        try:
            latencies = []
            for _ in range(10):
                start = time.time()
                response = requests.get(f"{kato_url}/health", timeout=2)
                latency = (time.time() - start) * 1000
                latencies.append(latency)

            kato_latency = np.mean(latencies)
            kato_available = True
        except:
            pass

        # Test localhost latency (TCP handshake simulation)
        import socket
        localhost_latencies = []
        for _ in range(10):
            start = time.time()
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect(('localhost', 80))  # Try common port
                sock.close()
            except:
                pass
            latency = (time.time() - start) * 1000
            localhost_latencies.append(latency)

        localhost_latency = np.mean(localhost_latencies) if localhost_latencies else 0.5

        return NetworkBenchmark(
            kato_api_latency_ms=kato_latency,
            kato_api_available=kato_available,
            localhost_latency_ms=localhost_latency
        )

    def _detect_gpus(self) -> GPUInfo:
        """
        Detect available GPUs.

        Checks for:
        - NVIDIA GPUs (via nvidia-smi)
        - AMD GPUs (via rocm-smi)
        - Apple Silicon GPU (via system_profiler)

        Returns:
            GPUInfo with detected GPUs
        """
        gpu_names = []
        gpu_memory = []

        # Try NVIDIA
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        name = parts[0].strip()
                        mem_str = parts[1].strip().split()[0]
                        mem_mb = int(mem_str)
                        gpu_names.append(name)
                        gpu_memory.append(mem_mb)
        except:
            pass

        # Try Apple Silicon
        if platform.system() == 'Darwin' and not gpu_names:
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPDisplaysDataType'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if 'Apple' in result.stdout or 'M1' in result.stdout or 'M2' in result.stdout or 'M3' in result.stdout:
                    gpu_names.append('Apple Silicon GPU')
                    # Memory shared with system on Apple Silicon
                    gpu_memory.append(0)
            except:
                pass

        gpu_available = len(gpu_names) > 0

        return GPUInfo(
            gpu_available=gpu_available,
            gpu_count=len(gpu_names),
            gpu_names=gpu_names,
            gpu_memory_mb=gpu_memory
        )

    def _predict_bottleneck(
        self,
        tier: str,
        storage_write_speed: float,
        disk_write_speed: float,
        network_latency: float
    ) -> str:
        """
        Predict primary performance bottleneck.

        Args:
            tier: Hardware tier ('low', 'medium', 'high', 'server')
            storage_write_speed: Rows/sec for storage system (ClickHouse)
            disk_write_speed: MB/s
            network_latency: ms

        Returns:
            Bottleneck type: 'cpu', 'memory', 'disk', 'network', 'none'
        """
        # Scoring
        scores = {
            'cpu': 0.0,
            'memory': 0.0,
            'disk': 0.0,
            'network': 0.0
        }

        # CPU: low-tier systems are CPU-bound
        if tier == 'low':
            scores['cpu'] = 0.8
        elif tier == 'medium':
            scores['cpu'] = 0.4

        # Storage/Disk: slow writes indicate bottleneck
        if storage_write_speed > 0 and storage_write_speed < 500:
            scores['disk'] = 0.7
        elif disk_write_speed < 50:
            scores['disk'] = 0.6

        # Network: high latency indicates bottleneck
        if network_latency > 50:
            scores['network'] = 0.7
        elif network_latency > 20:
            scores['network'] = 0.4

        # Find primary
        if max(scores.values()) < 0.3:
            return 'none'

        return max(scores, key=scores.get)


def main():
    """Example usage"""
    analyzer = HardwareAnalyzerV2(verbose=True)
    report = analyzer.analyze_system()
    report.print_summary()
    report.export_json('hardware_report_v2.json')


if __name__ == '__main__':
    main()
