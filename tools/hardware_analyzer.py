#!/usr/bin/env python3
"""
Hardware Analyzer - Standalone script for analyzing system hardware.
Detects CPU, memory, and classifies performance tier.
"""
import platform
import subprocess

class HardwareAnalyzer:
    """Analyze system hardware for performance estimation."""
    
    @staticmethod
    def get_cpu_count() -> int:
        """Get number of CPU cores."""
        try:
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'hw.ncpu'],
                                      capture_output=True, text=True)
                return int(result.stdout.strip())
            elif platform.system() == 'Linux':
                result = subprocess.run(['nproc'], capture_output=True, text=True)
                return int(result.stdout.strip())
            else:
                import multiprocessing
                return multiprocessing.cpu_count()
        except Exception:
            return 4  # Default fallback
    
    @staticmethod
    def get_memory_gb() -> float:
        """Get total system memory in GB."""
        try:
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                      capture_output=True, text=True)
                bytes_mem = int(result.stdout.strip())
                return bytes_mem / (1024**3)
            elif platform.system() == 'Linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            kb = int(line.split()[1])
                            return kb / (1024**2)
            else:
                # Windows or other platforms
                import psutil
                return psutil.virtual_memory().total / (1024**3)
        except Exception:
            return 16.0  # Default fallback
    
    @staticmethod
    def get_cpu_model() -> str:
        """Get CPU model name."""
        try:
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                      capture_output=True, text=True)
                return result.stdout.strip()
            elif platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':')[1].strip()
            else:
                # Windows or other platforms
                return platform.processor()
        except Exception:
            return "Unknown CPU"
    
    @staticmethod
    def classify_hardware() -> str:
        """Classify hardware tier based on specs."""
        cores = HardwareAnalyzer.get_cpu_count()
        memory = HardwareAnalyzer.get_memory_gb()
        
        if cores >= 32 and memory >= 128:
            return 'server'
        elif cores >= 16 and memory >= 64:
            return 'high'
        elif cores >= 8 and memory >= 16:
            return 'medium'
        else:
            return 'low'
    
    @staticmethod
    def get_performance_multiplier(tier: str = None) -> float:
        """Get performance multiplier for hardware tier."""
        if tier is None:
            tier = HardwareAnalyzer.classify_hardware()
        
        multipliers = {
            'low': 0.45,
            'medium': 1.0,   # Baseline (like current i9-8950HK)
            'high': 2.3,
            'server': 4.5
        }
        return multipliers.get(tier, 1.0)
    
    @staticmethod
    def get_baseline_rate() -> float:
        """
        Get baseline processing rate (samples/second).
        Based on observed performance: Intel i9-8950HK @ 44 samples/sec
        """
        return 44.0  # samples/second for medium-tier hardware
    
    @staticmethod
    def get_current_rate() -> float:
        """Get estimated processing rate for current hardware."""
        tier = HardwareAnalyzer.classify_hardware()
        baseline = HardwareAnalyzer.get_baseline_rate()
        multiplier = HardwareAnalyzer.get_performance_multiplier(tier)
        return baseline * multiplier
    
    @staticmethod
    def display_hardware_info():
        """Display hardware information and performance tier."""
        cpu = HardwareAnalyzer.get_cpu_model()
        cores = HardwareAnalyzer.get_cpu_count()
        memory = HardwareAnalyzer.get_memory_gb()
        tier = HardwareAnalyzer.classify_hardware()
        rate = HardwareAnalyzer.get_current_rate()
        
        print("\n" + "="*80)
        print("HARDWARE ANALYSIS")
        print("="*80)
        print(f"CPU Model:    {cpu}")
        print(f"CPU Cores:    {cores}")
        print(f"Memory:       {memory:.1f} GB")
        print(f"Platform:     {platform.system()} {platform.release()}")
        print(f"\nPerformance Tier: {tier.upper()}")
        print(f"Estimated Rate:   {rate:.1f} samples/second")
        print("="*80)
        
        return {
            'cpu': cpu,
            'cores': cores,
            'memory_gb': memory,
            'tier': tier,
            'rate': rate
        }


def main():
    """Main entry point for the script."""
    hw_info = HardwareAnalyzer.display_hardware_info()
    print(hw_info)
    return hw_info


if __name__ == '__main__':
    main()
