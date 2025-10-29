#!/usr/bin/env python3
"""
Scaling Analyzer - Training Complexity Analysis and Extrapolation

This module analyzes how training time and storage scale with dataset size
by fitting complexity curves (O(n), O(n log n), O(n²)) to actual training data.

It enables extrapolation to predict requirements for much larger datasets.

Usage:
    analyzer = ScalingAnalyzer()

    # Add training measurements
    analyzer.add_measurement(samples=100, time_seconds=5.2, storage_mb=12)
    analyzer.add_measurement(samples=500, time_seconds=28.1, storage_mb=58)
    analyzer.add_measurement(samples=1000, time_seconds=58.3, storage_mb=115)

    # Fit complexity curves
    analyzer.fit_curves()

    # Extrapolate
    prediction = analyzer.predict(samples=1000000)
    print(f"1M samples: {prediction.time_hours:.1f} hours, {prediction.storage_gb:.2f} GB")
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy import optimize
import json
import matplotlib.pyplot as plt


@dataclass
class ScalingMeasurement:
    """Single scaling measurement"""
    num_samples: int
    time_seconds: float
    storage_mb: float
    memory_mb: Optional[float] = None
    samples_per_second: Optional[float] = None

    def __post_init__(self):
        if self.samples_per_second is None and self.time_seconds > 0:
            self.samples_per_second = self.num_samples / self.time_seconds


@dataclass
class ComplexityCurve:
    """Fitted complexity curve"""
    complexity_type: str  # 'linear', 'n_log_n', 'quadratic', 'power'
    coefficients: List[float]
    r_squared: float  # Goodness of fit
    equation: str  # Human-readable equation


@dataclass
class ScalingPrediction:
    """Prediction for a specific dataset size"""
    num_samples: int

    # Time predictions
    time_seconds: float
    time_minutes: float
    time_hours: float
    time_days: float

    # Storage predictions
    storage_mb: float
    storage_gb: float

    # Memory predictions (if available)
    memory_mb: Optional[float] = None
    memory_gb: Optional[float] = None

    # Throughput predictions
    samples_per_second: Optional[float] = None

    # Confidence
    prediction_confidence: float = 0.0  # 0-1, based on R²


@dataclass
class ScalingAnalysisReport:
    """Complete scaling analysis results"""
    measurements: List[ScalingMeasurement]

    # Fitted curves
    time_curve: ComplexityCurve
    storage_curve: ComplexityCurve
    memory_curve: Optional[ComplexityCurve] = None

    # Extrapolations
    predictions: Dict[int, ScalingPrediction] = field(default_factory=dict)

    # Analysis
    time_complexity_class: str = 'unknown'  # 'linear', 'n_log_n', 'quadratic'
    storage_complexity_class: str = 'unknown'
    scaling_efficiency: str = 'unknown'  # 'excellent', 'good', 'fair', 'poor'

    def print_summary(self):
        """Print human-readable analysis summary"""
        print("\n" + "="*80)
        print("SCALING ANALYSIS REPORT")
        print("="*80)

        print(f"\n📊 MEASUREMENTS")
        print(f"  Data points: {len(self.measurements)}")
        print(f"  Sample range: {min(m.num_samples for m in self.measurements):,} - "
              f"{max(m.num_samples for m in self.measurements):,}")

        print(f"\n⏱️  TIME COMPLEXITY")
        print(f"  Best fit: {self.time_curve.complexity_type}")
        print(f"  Equation: {self.time_curve.equation}")
        print(f"  R²: {self.time_curve.r_squared:.4f}")
        print(f"  Classification: {self.time_complexity_class}")

        print(f"\n💾 STORAGE COMPLEXITY")
        print(f"  Best fit: {self.storage_curve.complexity_type}")
        print(f"  Equation: {self.storage_curve.equation}")
        print(f"  R²: {self.storage_curve.r_squared:.4f}")
        print(f"  Classification: {self.storage_complexity_class}")

        if self.memory_curve:
            print(f"\n🧠 MEMORY COMPLEXITY")
            print(f"  Best fit: {self.memory_curve.complexity_type}")
            print(f"  Equation: {self.memory_curve.equation}")
            print(f"  R²: {self.memory_curve.r_squared:.4f}")

        print(f"\n🎯 OVERALL SCALING EFFICIENCY")
        print(f"  Rating: {self.scaling_efficiency}")

        if self.predictions:
            print(f"\n🔮 EXTRAPOLATIONS")
            for samples, pred in sorted(self.predictions.items()):
                print(f"\n  {samples:,} samples:")
                if pred.time_hours < 1:
                    print(f"    Time: {pred.time_minutes:.1f} minutes")
                elif pred.time_days < 1:
                    print(f"    Time: {pred.time_hours:.1f} hours")
                else:
                    print(f"    Time: {pred.time_days:.1f} days")
                print(f"    Storage: {pred.storage_gb:.2f} GB")
                if pred.memory_gb:
                    print(f"    Memory: {pred.memory_gb:.1f} GB")
                print(f"    Confidence: {pred.prediction_confidence*100:.1f}%")

        print("\n" + "="*80)


class ScalingAnalyzer:
    """
    Analyze training scaling behavior through complexity curve fitting.

    Fits O(n), O(n log n), O(n²) and power-law curves to training data
    to enable accurate extrapolation to larger dataset sizes.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize scaling analyzer.

        Args:
            verbose: Print analysis steps
        """
        self.verbose = verbose
        self.measurements: List[ScalingMeasurement] = []
        self.time_curve: Optional[ComplexityCurve] = None
        self.storage_curve: Optional[ComplexityCurve] = None
        self.memory_curve: Optional[ComplexityCurve] = None

        if self.verbose:
            print("✓ ScalingAnalyzer initialized")

    def add_measurement(
        self,
        samples: int,
        time_seconds: float,
        storage_mb: float,
        memory_mb: Optional[float] = None
    ):
        """
        Add a training measurement.

        Args:
            samples: Number of samples trained
            time_seconds: Training duration
            storage_mb: MongoDB storage used
            memory_mb: Peak memory usage
        """
        measurement = ScalingMeasurement(
            num_samples=samples,
            time_seconds=time_seconds,
            storage_mb=storage_mb,
            memory_mb=memory_mb
        )

        self.measurements.append(measurement)

        if self.verbose:
            print(f"✓ Added measurement: {samples:,} samples, {time_seconds:.1f}s, {storage_mb:.1f}MB")

    def fit_curves(self) -> ScalingAnalysisReport:
        """
        Fit complexity curves to all measurements.

        Returns:
            ScalingAnalysisReport with fitted curves
        """
        if len(self.measurements) < 3:
            raise ValueError("Need at least 3 measurements to fit curves")

        if self.verbose:
            print(f"\n⏳ Fitting complexity curves to {len(self.measurements)} measurements...")

        # Extract data
        n_values = np.array([m.num_samples for m in self.measurements])
        time_values = np.array([m.time_seconds for m in self.measurements])
        storage_values = np.array([m.storage_mb for m in self.measurements])

        # Fit time complexity
        self.time_curve = self._fit_best_curve(n_values, time_values, "time")

        # Fit storage complexity
        self.storage_curve = self._fit_best_curve(n_values, storage_values, "storage")

        # Fit memory complexity (if data available)
        memory_values = np.array([m.memory_mb for m in self.measurements if m.memory_mb])
        if len(memory_values) >= 3:
            n_memory = n_values[:len(memory_values)]
            self.memory_curve = self._fit_best_curve(n_memory, memory_values, "memory")

        # Classify complexity
        time_class = self._classify_complexity(self.time_curve)
        storage_class = self._classify_complexity(self.storage_curve)

        # Assess overall scaling efficiency
        scaling_efficiency = self._assess_scaling_efficiency(time_class, storage_class)

        report = ScalingAnalysisReport(
            measurements=self.measurements,
            time_curve=self.time_curve,
            storage_curve=self.storage_curve,
            memory_curve=self.memory_curve,
            time_complexity_class=time_class,
            storage_complexity_class=storage_class,
            scaling_efficiency=scaling_efficiency
        )

        if self.verbose:
            print(f"✓ Time complexity: {self.time_curve.complexity_type} (R²={self.time_curve.r_squared:.4f})")
            print(f"✓ Storage complexity: {self.storage_curve.complexity_type} (R²={self.storage_curve.r_squared:.4f})")

        return report

    def predict(self, samples: int) -> ScalingPrediction:
        """
        Predict resource requirements for a specific number of samples.

        Args:
            samples: Number of samples to predict for

        Returns:
            ScalingPrediction with time and storage estimates
        """
        if not self.time_curve or not self.storage_curve:
            raise RuntimeError("Must call fit_curves() before prediction")

        # Predict time
        time_seconds = self._evaluate_curve(self.time_curve, samples)
        time_minutes = time_seconds / 60
        time_hours = time_minutes / 60
        time_days = time_hours / 24

        # Predict storage
        storage_mb = self._evaluate_curve(self.storage_curve, samples)
        storage_gb = storage_mb / 1024

        # Predict memory (if curve available)
        memory_mb = None
        memory_gb = None
        if self.memory_curve:
            memory_mb = self._evaluate_curve(self.memory_curve, samples)
            memory_gb = memory_mb / 1024

        # Predict throughput
        samples_per_second = samples / time_seconds if time_seconds > 0 else None

        # Confidence based on R² values
        confidence = (self.time_curve.r_squared + self.storage_curve.r_squared) / 2

        return ScalingPrediction(
            num_samples=samples,
            time_seconds=time_seconds,
            time_minutes=time_minutes,
            time_hours=time_hours,
            time_days=time_days,
            storage_mb=storage_mb,
            storage_gb=storage_gb,
            memory_mb=memory_mb,
            memory_gb=memory_gb,
            samples_per_second=samples_per_second,
            prediction_confidence=confidence
        )

    def generate_report(
        self,
        extrapolate_to: List[int] = None
    ) -> ScalingAnalysisReport:
        """
        Generate complete scaling analysis report with extrapolations.

        Args:
            extrapolate_to: List of sample counts to extrapolate to
                           Default: [10K, 100K, 1M, 10M, 100M]

        Returns:
            ScalingAnalysisReport with all predictions
        """
        if extrapolate_to is None:
            extrapolate_to = [10000, 100000, 1000000, 10000000, 100000000]

        report = self.fit_curves()

        # Generate predictions
        for samples in extrapolate_to:
            prediction = self.predict(samples)
            report.predictions[samples] = prediction

        return report

    def _fit_best_curve(
        self,
        n_values: np.ndarray,
        y_values: np.ndarray,
        metric_name: str
    ) -> ComplexityCurve:
        """
        Fit multiple complexity curves and return best fit.

        Tests: linear, n log n, quadratic, power law

        Args:
            n_values: Sample counts
            y_values: Measured values (time or storage)
            metric_name: Name for logging

        Returns:
            Best-fitting ComplexityCurve
        """
        curves = []

        # 1. Linear: y = a*n + b
        try:
            def linear(n, a, b):
                return a * n + b

            popt, _ = optimize.curve_fit(linear, n_values, y_values)
            y_pred = linear(n_values, *popt)
            r2 = self._calculate_r_squared(y_values, y_pred)

            curves.append(ComplexityCurve(
                complexity_type='linear',
                coefficients=list(popt),
                r_squared=r2,
                equation=f"y = {popt[0]:.2e}*n + {popt[1]:.2e}"
            ))
        except:
            pass

        # 2. n log n: y = a*n*log(n) + b
        try:
            def n_log_n(n, a, b):
                return a * n * np.log(n) + b

            popt, _ = optimize.curve_fit(n_log_n, n_values, y_values)
            y_pred = n_log_n(n_values, *popt)
            r2 = self._calculate_r_squared(y_values, y_pred)

            curves.append(ComplexityCurve(
                complexity_type='n_log_n',
                coefficients=list(popt),
                r_squared=r2,
                equation=f"y = {popt[0]:.2e}*n*log(n) + {popt[1]:.2e}"
            ))
        except:
            pass

        # 3. Quadratic: y = a*n² + b*n + c
        try:
            def quadratic(n, a, b, c):
                return a * n**2 + b * n + c

            popt, _ = optimize.curve_fit(quadratic, n_values, y_values)
            y_pred = quadratic(n_values, *popt)
            r2 = self._calculate_r_squared(y_values, y_pred)

            curves.append(ComplexityCurve(
                complexity_type='quadratic',
                coefficients=list(popt),
                r_squared=r2,
                equation=f"y = {popt[0]:.2e}*n² + {popt[1]:.2e}*n + {popt[2]:.2e}"
            ))
        except:
            pass

        # 4. Power law: y = a*n^b
        try:
            def power_law(n, a, b):
                return a * np.power(n, b)

            popt, _ = optimize.curve_fit(power_law, n_values, y_values, p0=[1, 1])
            y_pred = power_law(n_values, *popt)
            r2 = self._calculate_r_squared(y_values, y_pred)

            curves.append(ComplexityCurve(
                complexity_type='power',
                coefficients=list(popt),
                r_squared=r2,
                equation=f"y = {popt[0]:.2e}*n^{popt[1]:.3f}"
            ))
        except:
            pass

        if not curves:
            raise RuntimeError(f"Failed to fit any curves to {metric_name} data")

        # Return curve with best R²
        best_curve = max(curves, key=lambda c: c.r_squared)

        if self.verbose:
            print(f"  {metric_name}: best fit = {best_curve.complexity_type} (R²={best_curve.r_squared:.4f})")

        return best_curve

    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² goodness of fit"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _evaluate_curve(self, curve: ComplexityCurve, n: int) -> float:
        """Evaluate curve at specific n value"""
        if curve.complexity_type == 'linear':
            a, b = curve.coefficients
            return a * n + b
        elif curve.complexity_type == 'n_log_n':
            a, b = curve.coefficients
            return a * n * np.log(n) + b
        elif curve.complexity_type == 'quadratic':
            a, b, c = curve.coefficients
            return a * n**2 + b * n + c
        elif curve.complexity_type == 'power':
            a, b = curve.coefficients
            return a * np.power(n, b)
        else:
            raise ValueError(f"Unknown complexity type: {curve.complexity_type}")

    def _classify_complexity(self, curve: ComplexityCurve) -> str:
        """Classify complexity into standard classes"""
        if curve.complexity_type == 'linear':
            return 'O(n) - Linear'
        elif curve.complexity_type == 'n_log_n':
            return 'O(n log n) - Linearithmic'
        elif curve.complexity_type == 'quadratic':
            return 'O(n²) - Quadratic'
        elif curve.complexity_type == 'power':
            exponent = curve.coefficients[1]
            if exponent < 1.2:
                return 'O(n) - Near Linear'
            elif exponent < 1.8:
                return 'O(n^1.5) - Superlinear'
            elif exponent < 2.5:
                return 'O(n²) - Near Quadratic'
            else:
                return f'O(n^{exponent:.1f}) - Polynomial'
        else:
            return 'Unknown'

    def _assess_scaling_efficiency(self, time_class: str, storage_class: str) -> str:
        """Assess overall scaling efficiency"""
        # Excellent: both linear or n log n
        if ('Linear' in time_class and 'Linear' in storage_class):
            return 'EXCELLENT'

        # Good: one linear, one linearithmic
        if ('Linear' in time_class or 'Linearithmic' in time_class) and \
           ('Linear' in storage_class or 'Linearithmic' in storage_class):
            return 'GOOD'

        # Fair: no quadratic
        if 'Quadratic' not in time_class and 'Quadratic' not in storage_class:
            return 'FAIR'

        # Poor: quadratic or worse
        return 'POOR'

    def plot_curves(self, save_path: Optional[str] = None):
        """Plot fitted curves with measurements"""
        if not self.time_curve or not self.storage_curve:
            raise RuntimeError("Must call fit_curves() before plotting")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Extract data
        n_values = np.array([m.num_samples for m in self.measurements])
        time_values = np.array([m.time_seconds for m in self.measurements])
        storage_values = np.array([m.storage_mb for m in self.measurements])

        # Generate curve points
        n_curve = np.linspace(min(n_values), max(n_values) * 2, 100)
        time_curve = np.array([self._evaluate_curve(self.time_curve, n) for n in n_curve])
        storage_curve = np.array([self._evaluate_curve(self.storage_curve, n) for n in n_curve])

        # Plot time
        ax1.scatter(n_values, time_values, color='blue', label='Measurements', s=100, alpha=0.7)
        ax1.plot(n_curve, time_curve, color='red', label=f'Fit: {self.time_curve.complexity_type}', linewidth=2)
        ax1.set_xlabel('Number of Samples', fontsize=12)
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.set_title(f'Time Complexity: {self.time_curve.complexity_type}\nR² = {self.time_curve.r_squared:.4f}', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot storage
        ax2.scatter(n_values, storage_values / 1024, color='green', label='Measurements', s=100, alpha=0.7)
        ax2.plot(n_curve, storage_curve / 1024, color='orange', label=f'Fit: {self.storage_curve.complexity_type}', linewidth=2)
        ax2.set_xlabel('Number of Samples', fontsize=12)
        ax2.set_ylabel('Storage (GB)', fontsize=12)
        ax2.set_title(f'Storage Complexity: {self.storage_curve.complexity_type}\nR² = {self.storage_curve.r_squared:.4f}', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()

    def export_json(self, filepath: str):
        """Export analysis as JSON"""
        report = self.generate_report()

        data = {
            'measurements': [
                {
                    'num_samples': m.num_samples,
                    'time_seconds': m.time_seconds,
                    'storage_mb': m.storage_mb,
                    'memory_mb': m.memory_mb,
                    'samples_per_second': m.samples_per_second
                }
                for m in report.measurements
            ],
            'time_curve': {
                'type': report.time_curve.complexity_type,
                'equation': report.time_curve.equation,
                'r_squared': report.time_curve.r_squared
            },
            'storage_curve': {
                'type': report.storage_curve.complexity_type,
                'equation': report.storage_curve.equation,
                'r_squared': report.storage_curve.r_squared
            },
            'classifications': {
                'time_complexity': report.time_complexity_class,
                'storage_complexity': report.storage_complexity_class,
                'scaling_efficiency': report.scaling_efficiency
            },
            'predictions': {
                str(samples): {
                    'time_hours': pred.time_hours,
                    'storage_gb': pred.storage_gb,
                    'confidence': pred.prediction_confidence
                }
                for samples, pred in report.predictions.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Analysis exported to {filepath}")


def main():
    """Example usage"""
    analyzer = ScalingAnalyzer(verbose=True)

    # Add progressive measurements (simulated)
    measurements = [
        (100, 5.2, 12),
        (500, 28.1, 58),
        (1000, 58.3, 115),
        (5000, 312.5, 580),
        (10000, 645.8, 1150)
    ]

    for samples, time_s, storage_mb in measurements:
        analyzer.add_measurement(samples, time_s, storage_mb)

    # Generate report with extrapolations
    report = analyzer.generate_report(
        extrapolate_to=[100000, 1000000, 10000000, 100000000]
    )

    report.print_summary()

    # Plot
    analyzer.plot_curves(save_path='scaling_curves.png')

    # Export
    analyzer.export_json('scaling_analysis.json')


if __name__ == '__main__':
    main()
