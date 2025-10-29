#!/usr/bin/env python3
"""
Hardware Recommender - Generate Hardware Specifications for KATO Training

This module generates hardware purchase recommendations based on:
- Target dataset size
- Desired training time
- Budget constraints
- Deployment type (cloud vs on-premise)

Usage:
    recommender = HardwareRecommender()

    # Generate recommendation
    recommendation = recommender.recommend(
        target_samples=100_000_000,  # 100M samples
        desired_time_hours=24,       # Complete in 24 hours
        budget_usd=10000,
        deployment='on_premise'
    )

    recommendation.print_report()
    recommendation.export_markdown('hardware_recommendation.md')
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class CPUSpec:
    """CPU specification"""
    model: str
    cores: int
    threads: int
    base_clock_ghz: float
    turbo_clock_ghz: float
    tdp_watts: int
    price_usd: float


@dataclass
class MemorySpec:
    """RAM specification"""
    capacity_gb: int
    type: str  # DDR4, DDR5
    speed_mhz: int
    channels: int
    price_usd: float


@dataclass
class StorageSpec:
    """Storage specification"""
    capacity_gb: int
    type: str  # NVMe SSD, SATA SSD, HDD
    read_speed_mbps: int
    write_speed_mbps: int
    price_usd: float


@dataclass
class CloudInstance:
    """Cloud compute instance specification"""
    provider: str  # AWS, GCP, Azure
    instance_type: str
    vcpus: int
    memory_gb: int
    storage_gb: int
    cost_per_hour_usd: float
    estimated_total_cost_usd: float


@dataclass
class HardwareRecommendation:
    """Complete hardware recommendation"""
    # Input parameters
    target_samples: int
    desired_time_hours: float
    budget_usd: float
    deployment_type: str  # 'cloud', 'on_premise'

    # Calculated requirements
    required_cpu_cores: int
    required_memory_gb: int
    required_storage_gb: int
    required_throughput_samples_per_sec: float

    # On-premise recommendations
    cpu_spec: Optional[CPUSpec] = None
    memory_spec: Optional[MemorySpec] = None
    storage_spec: Optional[StorageSpec] = None
    total_on_premise_cost_usd: float = 0.0

    # Cloud recommendations
    cloud_instances: List[CloudInstance] = field(default_factory=list)
    recommended_cloud_instance: Optional[CloudInstance] = None

    # Analysis
    meets_budget: bool = True
    meets_timeline: bool = True
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Performance estimates
    estimated_training_time_hours: float = 0.0
    estimated_samples_per_second: float = 0.0

    def print_report(self):
        """Print human-readable hardware recommendation"""
        print("\n" + "="*80)
        print("HARDWARE RECOMMENDATION REPORT")
        print("="*80)

        print(f"\n🎯 TARGET REQUIREMENTS")
        print(f"  Dataset size: {self.target_samples:,} samples")
        print(f"  Desired training time: {self.desired_time_hours:.1f} hours")
        print(f"  Budget: ${self.budget_usd:,}")
        print(f"  Deployment: {self.deployment_type}")

        print(f"\n📊 CALCULATED REQUIREMENTS")
        print(f"  CPU cores: {self.required_cpu_cores}")
        print(f"  Memory: {self.required_memory_gb} GB")
        print(f"  Storage: {self.required_storage_gb} GB")
        print(f"  Required throughput: {self.required_throughput_samples_per_sec:.1f} samples/sec")

        if self.deployment_type == 'on_premise' and self.cpu_spec:
            print(f"\n💻 ON-PREMISE RECOMMENDATION")
            print(f"\n  CPU:")
            print(f"    Model: {self.cpu_spec.model}")
            print(f"    Cores: {self.cpu_spec.cores} cores / {self.cpu_spec.threads} threads")
            print(f"    Clock: {self.cpu_spec.base_clock_ghz:.2f} GHz (Turbo: {self.cpu_spec.turbo_clock_ghz:.2f} GHz)")
            print(f"    Price: ${self.cpu_spec.price_usd:,.0f}")

            print(f"\n  Memory:")
            print(f"    Capacity: {self.memory_spec.capacity_gb} GB {self.memory_spec.type}")
            print(f"    Speed: {self.memory_spec.speed_mhz} MHz")
            print(f"    Channels: {self.memory_spec.channels}")
            print(f"    Price: ${self.memory_spec.price_usd:,.0f}")

            print(f"\n  Storage:")
            print(f"    Capacity: {self.storage_spec.capacity_gb} GB {self.storage_spec.type}")
            print(f"    Read: {self.storage_spec.read_speed_mbps:,} MB/s")
            print(f"    Write: {self.storage_spec.write_speed_mbps:,} MB/s")
            print(f"    Price: ${self.storage_spec.price_usd:,.0f}")

            print(f"\n  💰 TOTAL ON-PREMISE COST: ${self.total_on_premise_cost_usd:,.0f}")

        if self.deployment_type == 'cloud' and self.recommended_cloud_instance:
            print(f"\n☁️  CLOUD RECOMMENDATION")
            inst = self.recommended_cloud_instance
            print(f"\n  Provider: {inst.provider}")
            print(f"  Instance type: {inst.instance_type}")
            print(f"  vCPUs: {inst.vcpus}")
            print(f"  Memory: {inst.memory_gb} GB")
            print(f"  Storage: {inst.storage_gb} GB")
            print(f"  Cost: ${inst.cost_per_hour_usd:.3f}/hour")
            print(f"\n  💰 ESTIMATED TOTAL COST: ${inst.estimated_total_cost_usd:,.2f}")
            print(f"     (for {self.estimated_training_time_hours:.1f} hours of training)")

            if len(self.cloud_instances) > 1:
                print(f"\n  Alternative options:")
                for alt in self.cloud_instances[:3]:
                    if alt.instance_type != inst.instance_type:
                        print(f"    - {alt.instance_type} ({alt.provider}): ${alt.estimated_total_cost_usd:,.2f}")

        print(f"\n⏱️  PERFORMANCE ESTIMATES")
        print(f"  Estimated training time: {self.estimated_training_time_hours:.1f} hours")
        print(f"  Estimated throughput: {self.estimated_samples_per_second:.1f} samples/sec")

        if not self.meets_timeline:
            print(f"  ⚠️  WARNING: May exceed desired timeline of {self.desired_time_hours} hours")

        if self.warnings:
            print(f"\n⚠️  WARNINGS")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            for rec in self.recommendations:
                print(f"  - {rec}")

        print("\n" + "="*80)

    def export_markdown(self, filepath: str):
        """Export recommendation as Markdown report"""
        md = f"""# Hardware Recommendation Report

## Target Requirements
- **Dataset size**: {self.target_samples:,} samples
- **Desired training time**: {self.desired_time_hours:.1f} hours
- **Budget**: ${self.budget_usd:,}
- **Deployment**: {self.deployment_type}

## Calculated Requirements
- **CPU cores**: {self.required_cpu_cores}
- **Memory**: {self.required_memory_gb} GB
- **Storage**: {self.required_storage_gb} GB
- **Required throughput**: {self.required_throughput_samples_per_sec:.1f} samples/sec

"""

        if self.deployment_type == 'on_premise' and self.cpu_spec:
            md += f"""## On-Premise Recommendation

### CPU
- **Model**: {self.cpu_spec.model}
- **Cores/Threads**: {self.cpu_spec.cores} cores / {self.cpu_spec.threads} threads
- **Clock Speed**: {self.cpu_spec.base_clock_ghz:.2f} GHz (Turbo: {self.cpu_spec.turbo_clock_ghz:.2f} GHz)
- **TDP**: {self.cpu_spec.tdp_watts}W
- **Price**: ${self.cpu_spec.price_usd:,.0f}

### Memory
- **Capacity**: {self.memory_spec.capacity_gb} GB {self.memory_spec.type}
- **Speed**: {self.memory_spec.speed_mhz} MHz
- **Channels**: {self.memory_spec.channels}
- **Price**: ${self.memory_spec.price_usd:,.0f}

### Storage
- **Capacity**: {self.storage_spec.capacity_gb} GB
- **Type**: {self.storage_spec.type}
- **Read Speed**: {self.storage_spec.read_speed_mbps:,} MB/s
- **Write Speed**: {self.storage_spec.write_speed_mbps:,} MB/s
- **Price**: ${self.storage_spec.price_usd:,.0f}

### Total Cost
**${self.total_on_premise_cost_usd:,.0f}**

"""

        if self.deployment_type == 'cloud' and self.recommended_cloud_instance:
            inst = self.recommended_cloud_instance
            md += f"""## Cloud Recommendation

### Recommended Instance
- **Provider**: {inst.provider}
- **Instance Type**: {inst.instance_type}
- **vCPUs**: {inst.vcpus}
- **Memory**: {inst.memory_gb} GB
- **Storage**: {inst.storage_gb} GB
- **Hourly Cost**: ${inst.cost_per_hour_usd:.3f}/hour

### Total Estimated Cost
**${inst.estimated_total_cost_usd:,.2f}** (for {self.estimated_training_time_hours:.1f} hours)

"""

        md += f"""## Performance Estimates
- **Estimated training time**: {self.estimated_training_time_hours:.1f} hours
- **Estimated throughput**: {self.estimated_samples_per_second:.1f} samples/sec

"""

        if self.warnings:
            md += "## ⚠️ Warnings\n"
            for warning in self.warnings:
                md += f"- {warning}\n"
            md += "\n"

        if self.recommendations:
            md += "## 💡 Recommendations\n"
            for rec in self.recommendations:
                md += f"- {rec}\n"

        with open(filepath, 'w') as f:
            f.write(md)

        print(f"✓ Recommendation exported to {filepath}")

    def export_json(self, filepath: str):
        """Export recommendation as JSON"""
        data = {
            'target_samples': self.target_samples,
            'desired_time_hours': self.desired_time_hours,
            'budget_usd': self.budget_usd,
            'deployment_type': self.deployment_type,
            'requirements': {
                'cpu_cores': self.required_cpu_cores,
                'memory_gb': self.required_memory_gb,
                'storage_gb': self.required_storage_gb,
                'throughput_samples_per_sec': self.required_throughput_samples_per_sec
            },
            'estimated_performance': {
                'training_time_hours': self.estimated_training_time_hours,
                'samples_per_second': self.estimated_samples_per_second
            },
            'meets_budget': self.meets_budget,
            'meets_timeline': self.meets_timeline,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }

        if self.deployment_type == 'on_premise' and self.cpu_spec:
            data['on_premise'] = {
                'cpu': {
                    'model': self.cpu_spec.model,
                    'cores': self.cpu_spec.cores,
                    'price_usd': self.cpu_spec.price_usd
                },
                'memory': {
                    'capacity_gb': self.memory_spec.capacity_gb,
                    'type': self.memory_spec.type,
                    'price_usd': self.memory_spec.price_usd
                },
                'storage': {
                    'capacity_gb': self.storage_spec.capacity_gb,
                    'type': self.storage_spec.type,
                    'price_usd': self.storage_spec.price_usd
                },
                'total_cost_usd': self.total_on_premise_cost_usd
            }

        if self.deployment_type == 'cloud' and self.recommended_cloud_instance:
            data['cloud'] = {
                'provider': self.recommended_cloud_instance.provider,
                'instance_type': self.recommended_cloud_instance.instance_type,
                'vcpus': self.recommended_cloud_instance.vcpus,
                'memory_gb': self.recommended_cloud_instance.memory_gb,
                'cost_per_hour_usd': self.recommended_cloud_instance.cost_per_hour_usd,
                'total_cost_usd': self.recommended_cloud_instance.estimated_total_cost_usd
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Recommendation exported to {filepath}")


class HardwareRecommender:
    """
    Generate hardware purchase recommendations for KATO training.

    Uses scaling analysis and hardware benchmarks to recommend
    specific CPU, RAM, and storage configurations.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize hardware recommender.

        Args:
            verbose: Print recommendation process
        """
        self.verbose = verbose

        # CPU database (prices as of 2025, approximate)
        self.cpu_database = [
            CPUSpec("AMD Ryzen 9 7950X", 16, 32, 4.5, 5.7, 170, 550),
            CPUSpec("AMD Ryzen 9 7900X", 12, 24, 4.7, 5.4, 170, 400),
            CPUSpec("AMD Ryzen Threadripper PRO 5975WX", 32, 64, 3.6, 4.5, 280, 3200),
            CPUSpec("Intel Core i9-14900K", 24, 32, 3.2, 6.0, 253, 600),
            CPUSpec("Intel Xeon W-3375", 38, 76, 2.5, 4.0, 270, 4500),
            CPUSpec("AMD EPYC 7763", 64, 128, 2.45, 3.5, 280, 7200),
        ]

        # Cloud instance database
        self.cloud_instances = {
            'AWS': [
                ('c7i.4xlarge', 16, 32, 500, 0.68),
                ('c7i.8xlarge', 32, 64, 1000, 1.36),
                ('c7i.16xlarge', 64, 128, 2000, 2.72),
                ('m7i.4xlarge', 16, 64, 500, 0.77),
                ('m7i.8xlarge', 32, 128, 1000, 1.54),
            ],
            'GCP': [
                ('c3-standard-4', 4, 16, 500, 0.19),
                ('c3-standard-8', 8, 32, 500, 0.38),
                ('c3-standard-22', 22, 88, 1000, 1.04),
                ('n2-standard-16', 16, 64, 500, 0.78),
                ('n2-standard-32', 32, 128, 1000, 1.55),
            ],
            'Azure': [
                ('F16s_v2', 16, 32, 500, 0.68),
                ('F32s_v2', 32, 64, 1000, 1.35),
                ('D16s_v5', 16, 64, 500, 0.77),
                ('D32s_v5', 32, 128, 1000, 1.54),
            ]
        }

        if self.verbose:
            print("✓ HardwareRecommender initialized")

    def recommend(
        self,
        target_samples: int,
        desired_time_hours: float,
        budget_usd: float = 10000,
        deployment: str = 'on_premise',
        baseline_throughput: float = 44.0,  # samples/sec on baseline hardware
        estimated_storage_gb_per_million_samples: float = 10.0
    ) -> HardwareRecommendation:
        """
        Generate hardware recommendation.

        Args:
            target_samples: Number of samples to train
            desired_time_hours: Desired training time in hours
            budget_usd: Budget constraint
            deployment: 'on_premise' or 'cloud'
            baseline_throughput: Baseline samples/sec (from hardware_analyzer)
            estimated_storage_gb_per_million_samples: Storage estimate

        Returns:
            HardwareRecommendation with specs
        """
        if self.verbose:
            print(f"\n⏳ Generating recommendation for {target_samples:,} samples...")

        warnings = []
        recommendations = []

        # Calculate required throughput
        desired_time_seconds = desired_time_hours * 3600
        required_throughput = target_samples / desired_time_seconds

        if self.verbose:
            print(f"  Required throughput: {required_throughput:.1f} samples/sec")

        # Calculate CPU cores needed
        # Assume linear scaling with cores up to 32 cores, then diminishing returns
        throughput_multiplier = required_throughput / baseline_throughput
        if throughput_multiplier <= 1:
            required_cores = 4
        elif throughput_multiplier <= 4:
            required_cores = int(throughput_multiplier * 4)
        elif throughput_multiplier <= 8:
            required_cores = 16 + int((throughput_multiplier - 4) * 4)
        else:
            required_cores = 32 + int((throughput_multiplier - 8) * 2)
            warnings.append(f"Very high throughput required ({required_throughput:.0f} samples/sec). Consider distributed training.")

        # Calculate memory needed
        # Base: 2GB + 100MB per core + 1MB per 1000 samples in memory
        required_memory = 2 + (required_cores * 0.1) + (target_samples / 1000 * 0.001)
        required_memory = max(16, int(required_memory))  # Minimum 16GB

        # Round to common memory sizes
        memory_sizes = [16, 32, 64, 128, 256, 512]
        required_memory = min(s for s in memory_sizes if s >= required_memory)

        # Calculate storage needed
        required_storage = int((target_samples / 1_000_000) * estimated_storage_gb_per_million_samples)
        required_storage = max(500, required_storage)  # Minimum 500GB

        # Round to common storage sizes
        storage_sizes = [500, 1000, 2000, 4000, 8000]
        required_storage = min((s for s in storage_sizes if s >= required_storage), default=8000)

        if self.verbose:
            print(f"  Required CPU cores: {required_cores}")
            print(f"  Required memory: {required_memory} GB")
            print(f"  Required storage: {required_storage} GB")

        # Generate recommendation based on deployment type
        if deployment == 'on_premise':
            recommendation = self._recommend_on_premise(
                required_cores, required_memory, required_storage,
                budget_usd, warnings, recommendations
            )
        else:
            recommendation = self._recommend_cloud(
                required_cores, required_memory, required_storage,
                budget_usd, desired_time_hours, warnings, recommendations
            )

        # Populate common fields
        recommendation.target_samples = target_samples
        recommendation.desired_time_hours = desired_time_hours
        recommendation.budget_usd = budget_usd
        recommendation.deployment_type = deployment
        recommendation.required_cpu_cores = required_cores
        recommendation.required_memory_gb = required_memory
        recommendation.required_storage_gb = required_storage
        recommendation.required_throughput_samples_per_sec = required_throughput
        recommendation.warnings = warnings
        recommendation.recommendations = recommendations

        # Estimate actual performance
        if deployment == 'on_premise' and recommendation.cpu_spec:
            estimated_multiplier = recommendation.cpu_spec.cores / 4  # Relative to baseline
            recommendation.estimated_samples_per_second = baseline_throughput * estimated_multiplier
        elif deployment == 'cloud' and recommendation.recommended_cloud_instance:
            estimated_multiplier = recommendation.recommended_cloud_instance.vcpus / 4
            recommendation.estimated_samples_per_second = baseline_throughput * estimated_multiplier

        recommendation.estimated_training_time_hours = (
            target_samples / recommendation.estimated_samples_per_second / 3600
        )

        # Check if meets timeline
        recommendation.meets_timeline = recommendation.estimated_training_time_hours <= desired_time_hours * 1.1

        if not recommendation.meets_timeline:
            warnings.append(
                f"Estimated time ({recommendation.estimated_training_time_hours:.1f}h) "
                f"exceeds desired time ({desired_time_hours:.1f}h)"
            )

        if self.verbose:
            print(f"✓ Recommendation generated")

        return recommendation

    def _recommend_on_premise(
        self,
        required_cores: int,
        required_memory: int,
        required_storage: int,
        budget: float,
        warnings: List[str],
        recommendations: List[str]
    ) -> HardwareRecommendation:
        """Generate on-premise hardware recommendation"""

        # Select CPU
        suitable_cpus = [cpu for cpu in self.cpu_database if cpu.cores >= required_cores]
        if not suitable_cpus:
            suitable_cpus = [self.cpu_database[-1]]  # Use most powerful
            warnings.append(f"No CPU found with {required_cores} cores. Using best available.")

        cpu = min(suitable_cpus, key=lambda c: c.price_usd)  # Cheapest suitable CPU

        # Memory spec
        memory = MemorySpec(
            capacity_gb=required_memory,
            type='DDR5',
            speed_mhz=5600,
            channels=2,
            price_usd=required_memory * 5  # ~$5/GB
        )

        # Storage spec
        storage = StorageSpec(
            capacity_gb=required_storage,
            type='NVMe SSD',
            read_speed_mbps=7000,
            write_speed_mbps=5000,
            price_usd=required_storage * 0.1  # ~$0.10/GB for NVMe
        )

        total_cost = cpu.price_usd + memory.price_usd + storage.price_usd + 500  # +$500 for motherboard/PSU/case

        meets_budget = total_cost <= budget

        if not meets_budget:
            warnings.append(f"Total cost (${total_cost:,.0f}) exceeds budget (${budget:,.0f})")
            recommendations.append("Consider cloud deployment or increase budget")

        return HardwareRecommendation(
            target_samples=0,
            desired_time_hours=0,
            budget_usd=budget,
            deployment_type='on_premise',
            required_cpu_cores=required_cores,
            required_memory_gb=required_memory,
            required_storage_gb=required_storage,
            required_throughput_samples_per_sec=0,
            cpu_spec=cpu,
            memory_spec=memory,
            storage_spec=storage,
            total_on_premise_cost_usd=total_cost,
            meets_budget=meets_budget
        )

    def _recommend_cloud(
        self,
        required_cores: int,
        required_memory: int,
        required_storage: int,
        budget: float,
        training_hours: float,
        warnings: List[str],
        recommendations: List[str]
    ) -> HardwareRecommendation:
        """Generate cloud deployment recommendation"""

        cloud_options = []

        for provider, instances in self.cloud_instances.items():
            for inst_type, vcpus, memory_gb, storage_gb, cost_per_hour in instances:
                if vcpus >= required_cores and memory_gb >= required_memory:
                    total_cost = cost_per_hour * training_hours

                    cloud_options.append(CloudInstance(
                        provider=provider,
                        instance_type=inst_type,
                        vcpus=vcpus,
                        memory_gb=memory_gb,
                        storage_gb=storage_gb,
                        cost_per_hour_usd=cost_per_hour,
                        estimated_total_cost_usd=total_cost
                    ))

        if not cloud_options:
            warnings.append("No suitable cloud instances found. Consider on-premise.")
            return HardwareRecommendation(
                target_samples=0,
                desired_time_hours=0,
                budget_usd=budget,
                deployment_type='cloud',
                required_cpu_cores=required_cores,
                required_memory_gb=required_memory,
                required_storage_gb=required_storage,
                required_throughput_samples_per_sec=0,
                meets_budget=False
            )

        # Sort by total cost
        cloud_options.sort(key=lambda c: c.estimated_total_cost_usd)

        # Select cheapest option that meets budget
        recommended = cloud_options[0]
        meets_budget = recommended.estimated_total_cost_usd <= budget

        if not meets_budget:
            warnings.append(
                f"Cheapest cloud option (${recommended.estimated_total_cost_usd:,.2f}) "
                f"exceeds budget (${budget:,.0f})"
            )

        return HardwareRecommendation(
            target_samples=0,
            desired_time_hours=0,
            budget_usd=budget,
            deployment_type='cloud',
            required_cpu_cores=required_cores,
            required_memory_gb=required_memory,
            required_storage_gb=required_storage,
            required_throughput_samples_per_sec=0,
            cloud_instances=cloud_options[:5],  # Top 5 options
            recommended_cloud_instance=recommended,
            meets_budget=meets_budget
        )


def main():
    """Example usage"""
    recommender = HardwareRecommender(verbose=True)

    # Example 1: On-premise for 100M samples
    print("\n" + "="*80)
    print("EXAMPLE 1: On-premise deployment for 100M samples")
    print("="*80)

    rec1 = recommender.recommend(
        target_samples=100_000_000,
        desired_time_hours=48,
        budget_usd=10000,
        deployment='on_premise'
    )

    rec1.print_report()
    rec1.export_markdown('hardware_rec_on_premise.md')

    # Example 2: Cloud for 10M samples
    print("\n" + "="*80)
    print("EXAMPLE 2: Cloud deployment for 10M samples")
    print("="*80)

    rec2 = recommender.recommend(
        target_samples=10_000_000,
        desired_time_hours=12,
        budget_usd=500,
        deployment='cloud'
    )

    rec2.print_report()
    rec2.export_json('hardware_rec_cloud.json')


if __name__ == '__main__':
    main()
