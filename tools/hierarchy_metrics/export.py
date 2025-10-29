"""
Export utilities for web dashboards and data analysis.

Provides functions to export metrics in web-ready formats (JSON, Plotly, Parquet).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .report import MetricsReport
from .config import HealthStatus


class DashboardExporter:
    """
    Export metrics for web dashboards.

    Generates JSON configurations compatible with:
    - Plotly.js
    - Chart.js
    - D3.js
    - Generic web frameworks
    """

    def __init__(self, report: MetricsReport):
        """
        Initialize exporter.

        Args:
            report: MetricsReport instance
        """
        self.report = report

    # ========================================================================
    # PLOTLY EXPORT
    # ========================================================================

    def export_plotly_compression_ratios(self) -> Dict[str, Any]:
        """
        Export compression ratios in Plotly format.

        Returns:
            Plotly figure dict
        """
        ratios = self.report.compression.compression_ratios

        pairs = list(ratios.keys())
        values = list(ratios.values())

        # Health status colors
        colors = []
        for pair in pairs:
            health = self.report.compression.health_status.get(f"compression_{pair}", HealthStatus.UNKNOWN)
            if health == HealthStatus.EXCELLENT:
                colors.append('#2ecc71')
            elif health == HealthStatus.GOOD:
                colors.append('#27ae60')
            elif health == HealthStatus.WARNING:
                colors.append('#f39c12')
            else:
                colors.append('#e74c3c')

        return {
            'data': [{
                'x': pairs,
                'y': values,
                'type': 'bar',
                'marker': {'color': colors},
                'name': 'Compression Ratio'
            }],
            'layout': {
                'title': 'Compression Ratios Across Hierarchy',
                'xaxis': {'title': 'Level Pair'},
                'yaxis': {'title': 'Compression Ratio'},
                'showlegend': False
            }
        }

    def export_plotly_pattern_counts(self, log_scale: bool = True) -> Dict[str, Any]:
        """
        Export pattern counts in Plotly format.

        Args:
            log_scale: Use log scale for y-axis

        Returns:
            Plotly figure dict
        """
        counts = self.report.compression.pattern_counts

        levels = sorted(counts.keys())
        values = [counts[level] for level in levels]

        yaxis_config = {'title': 'Pattern Count'}
        if log_scale:
            yaxis_config['type'] = 'log'
            yaxis_config['title'] = 'Pattern Count (log scale)'

        return {
            'data': [{
                'x': levels,
                'y': values,
                'type': 'bar',
                'marker': {'color': 'steelblue'},
                'name': 'Pattern Count',
                'text': [f'{v:,}' for v in values],
                'textposition': 'auto'
            }],
            'layout': {
                'title': 'Pattern Count Progression',
                'xaxis': {'title': 'Hierarchy Level'},
                'yaxis': yaxis_config,
                'showlegend': False
            }
        }

    def export_plotly_reusability(self, level: str) -> Dict[str, Any]:
        """
        Export reusability stats in Plotly format.

        Args:
            level: Level to export (e.g., 'node0')

        Returns:
            Plotly figure dict
        """
        reusability = self.report.connectivity.reusability.get(level)

        if not reusability:
            return {}

        stats = {
            'Mean Parents': reusability['mean_parents'],
            'Median Parents': reusability['median_parents'],
            '90th Percentile': reusability['p90_parents'],
        }

        return {
            'data': [{
                'x': list(stats.keys()),
                'y': list(stats.values()),
                'type': 'bar',
                'marker': {'color': 'coral'},
                'name': 'Parent Count',
                'text': [f'{v:.2f}' for v in stats.values()],
                'textposition': 'auto'
            }],
            'layout': {
                'title': f'Reusability Statistics - {level}',
                'xaxis': {'title': ''},
                'yaxis': {'title': 'Parent Count'},
                'annotations': [{
                    'text': f"Orphan Rate: {reusability['orphan_rate']:.1%}",
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.95,
                    'y': 0.95,
                    'showarrow': False,
                    'bgcolor': '#ffe6e6',
                    'bordercolor': '#e74c3c',
                    'borderwidth': 2
                }]
            }
        }

    def export_plotly_coverage(self) -> Dict[str, Any]:
        """
        Export coverage heatmap in Plotly format.

        Returns:
            Plotly figure dict
        """
        coverage = self.report.connectivity.coverage

        pairs = list(coverage.keys())
        values = list(coverage.values())

        # Health colors
        colors = []
        for pair in pairs:
            health = self.report.connectivity.health_status.get(f"coverage_{pair}", HealthStatus.UNKNOWN)
            if health == HealthStatus.EXCELLENT:
                colors.append('#2ecc71')
            elif health == HealthStatus.GOOD:
                colors.append('#27ae60')
            elif health == HealthStatus.WARNING:
                colors.append('#f39c12')
            else:
                colors.append('#e74c3c')

        return {
            'data': [{
                'y': pairs,
                'x': values,
                'type': 'bar',
                'orientation': 'h',
                'marker': {'color': colors},
                'name': 'Coverage',
                'text': [f'{v:.1%}' for v in values],
                'textposition': 'auto'
            }],
            'layout': {
                'title': 'Coverage: % of Lower Patterns Used in Upper Level',
                'xaxis': {'title': 'Coverage Rate', 'range': [0, 1]},
                'yaxis': {'title': ''},
                'showlegend': False
            }
        }

    def export_plotly_training_dynamics(self) -> Dict[str, Any]:
        """
        Export training dynamics in Plotly format.

        Returns:
            Plotly figure dict with subplots
        """
        if not self.report.training_dynamics:
            return {}

        dynamics = self.report.training_dynamics
        checkpoints = dynamics.checkpoints

        samples = [cp['samples_processed'] for cp in checkpoints]
        total_patterns = [sum(cp['pattern_counts'].values()) for cp in checkpoints]

        return {
            'data': [
                {
                    'x': samples,
                    'y': total_patterns,
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Actual',
                    'marker': {'size': 8, 'color': 'blue'}
                },
                {
                    'x': samples,
                    'y': total_patterns,  # Would need fitted line calculation
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': f'Power-law fit (exponent={dynamics.growth_exponent:.3f})',
                    'line': {'dash': 'dash', 'color': 'red'}
                }
            ],
            'layout': {
                'title': f'Pattern Growth (R²={dynamics.growth_r_squared:.3f})',
                'xaxis': {'title': 'Samples Processed', 'type': 'log'},
                'yaxis': {'title': 'Total Patterns', 'type': 'log'},
                'showlegend': True
            }
        }

    def export_all_plotly(self, output_dir: str):
        """
        Export all charts in Plotly format.

        Args:
            output_dir: Directory to save JSON files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compression ratios
        with open(output_dir / 'compression_ratios.json', 'w') as f:
            json.dump(self.export_plotly_compression_ratios(), f, indent=2)

        # Pattern counts
        with open(output_dir / 'pattern_counts.json', 'w') as f:
            json.dump(self.export_plotly_pattern_counts(), f, indent=2)

        # Reusability (all levels)
        for level in self.report.compression.pattern_counts.keys():
            if level in self.report.connectivity.reusability:
                with open(output_dir / f'reusability_{level}.json', 'w') as f:
                    json.dump(self.export_plotly_reusability(level), f, indent=2)

        # Coverage
        with open(output_dir / 'coverage.json', 'w') as f:
            json.dump(self.export_plotly_coverage(), f, indent=2)

        # Training dynamics
        if self.report.training_dynamics:
            with open(output_dir / 'training_dynamics.json', 'w') as f:
                json.dump(self.export_plotly_training_dynamics(), f, indent=2)

        print(f"✓ Plotly charts exported to {output_dir}")

    # ========================================================================
    # GENERIC JSON EXPORT
    # ========================================================================

    def export_dashboard_config(self) -> Dict[str, Any]:
        """
        Export comprehensive dashboard configuration.

        Returns:
            Dashboard config dict with all data
        """
        config = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0.0',
            },
            'summary': self.report.metrics_summary.to_dict(),
            'metrics': {
                'compression': self.report.compression.to_dict(),
                'connectivity': self.report.connectivity.to_dict(),
                'information': self.report.information.to_dict(),
            },
            'charts': {
                'compression_ratios': self.export_plotly_compression_ratios(),
                'pattern_counts': self.export_plotly_pattern_counts(),
                'coverage': self.export_plotly_coverage(),
            }
        }

        # Add optional metrics
        if self.report.prediction:
            config['metrics']['prediction'] = self.report.prediction.to_dict()

        if self.report.training_dynamics:
            config['metrics']['training_dynamics'] = self.report.training_dynamics.to_dict()
            config['charts']['training_dynamics'] = self.export_plotly_training_dynamics()

        return config

    def export_dashboard_json(self, output_path: str):
        """
        Export full dashboard configuration as JSON.

        Args:
            output_path: Path to output JSON file
        """
        config = self.export_dashboard_config()

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Dashboard config exported to {output_path}")

    # ========================================================================
    # TIME SERIES EXPORT
    # ========================================================================

    def export_time_series(self) -> Dict[str, Any]:
        """
        Export training dynamics as time series data.

        Returns:
            Time series dict
        """
        if not self.report.training_dynamics:
            return {}

        checkpoints = self.report.training_dynamics.checkpoints

        time_series = {
            'timestamps': [],
            'samples_processed': [],
            'total_patterns': [],
            'pattern_counts_by_level': {},
            'reusability_by_level': {},
        }

        for cp in checkpoints:
            time_series['timestamps'].append(cp['timestamp'])
            time_series['samples_processed'].append(cp['samples_processed'])
            time_series['total_patterns'].append(sum(cp['pattern_counts'].values()))

            # Pattern counts by level
            for level, count in cp['pattern_counts'].items():
                if level not in time_series['pattern_counts_by_level']:
                    time_series['pattern_counts_by_level'][level] = []
                time_series['pattern_counts_by_level'][level].append(count)

            # Reusability by level
            if 'reusability' in cp['metrics_snapshot']:
                for level, stats in cp['metrics_snapshot']['reusability'].items():
                    if level not in time_series['reusability_by_level']:
                        time_series['reusability_by_level'][level] = []
                    time_series['reusability_by_level'][level].append(stats['mean_parents'])

        return time_series

    def export_time_series_json(self, output_path: str):
        """
        Export time series data as JSON.

        Args:
            output_path: Path to output JSON file
        """
        time_series = self.export_time_series()

        with open(output_path, 'w') as f:
            json.dump(time_series, f, indent=2)

        print(f"✓ Time series data exported to {output_path}")

    # ========================================================================
    # PARQUET EXPORT (for data analysis)
    # ========================================================================

    def export_parquet(self, output_dir: str):
        """
        Export metrics as Parquet files for data analysis.

        Requires pandas and pyarrow.

        Args:
            output_dir: Directory to save Parquet files
        """
        try:
            import pandas as pd
        except ImportError:
            print("Warning: pandas not installed. Skipping Parquet export.")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compression metrics
        compression_data = []
        for pair, ratio in self.report.compression.compression_ratios.items():
            lower, upper = pair.split('->')
            compression_data.append({
                'lower_level': lower,
                'upper_level': upper,
                'compression_ratio': ratio,
                'health': self.report.compression.health_status.get(f"compression_{pair}", HealthStatus.UNKNOWN).value
            })

        if compression_data:
            df_compression = pd.DataFrame(compression_data)
            df_compression.to_parquet(output_dir / 'compression.parquet', index=False)

        # Pattern counts
        pattern_data = []
        for level, count in self.report.compression.pattern_counts.items():
            pattern_data.append({
                'level': level,
                'count': count
            })

        if pattern_data:
            df_patterns = pd.DataFrame(pattern_data)
            df_patterns.to_parquet(output_dir / 'pattern_counts.parquet', index=False)

        # Reusability
        reusability_data = []
        for level, stats in self.report.connectivity.reusability.items():
            reusability_data.append({
                'level': level,
                'mean_parents': stats['mean_parents'],
                'median_parents': stats['median_parents'],
                'orphan_rate': stats['orphan_rate'],
                'p90_parents': stats['p90_parents'],
                'total_patterns': stats['total_patterns']
            })

        if reusability_data:
            df_reusability = pd.DataFrame(reusability_data)
            df_reusability.to_parquet(output_dir / 'reusability.parquet', index=False)

        # Time series (if available)
        if self.report.training_dynamics:
            time_series_data = []
            for cp in self.report.training_dynamics.checkpoints:
                time_series_data.append({
                    'checkpoint_id': cp['checkpoint_id'],
                    'samples_processed': cp['samples_processed'],
                    'timestamp': cp['timestamp'],
                    'total_patterns': sum(cp['pattern_counts'].values()),
                    **{f'patterns_{k}': v for k, v in cp['pattern_counts'].items()}
                })

            if time_series_data:
                df_time_series = pd.DataFrame(time_series_data)
                df_time_series.to_parquet(output_dir / 'time_series.parquet', index=False)

        print(f"✓ Parquet files exported to {output_dir}")

    # ========================================================================
    # COMPREHENSIVE EXPORT
    # ========================================================================

    def export_all(self, output_dir: str, formats: List[str] = None):
        """
        Export all data in multiple formats.

        Args:
            output_dir: Base output directory
            formats: List of formats to export ('plotly', 'json', 'parquet', 'csv')
                    Default: all formats
        """
        if formats is None:
            formats = ['plotly', 'json', 'parquet', 'csv']

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print("EXPORTING METRICS FOR DASHBOARD")
        print(f"{'='*80}\n")

        # Plotly charts
        if 'plotly' in formats:
            plotly_dir = output_dir / 'plotly'
            self.export_all_plotly(str(plotly_dir))

        # Dashboard JSON
        if 'json' in formats:
            self.export_dashboard_json(str(output_dir / 'dashboard_config.json'))
            self.export_time_series_json(str(output_dir / 'time_series.json'))

        # Parquet
        if 'parquet' in formats:
            parquet_dir = output_dir / 'parquet'
            self.export_parquet(str(parquet_dir))

        # CSV (via report)
        if 'csv' in formats:
            csv_dir = output_dir / 'csv'
            self.report.export_csv(str(csv_dir))

        print(f"\n{'='*80}")
        print("EXPORT COMPLETE")
        print(f"{'='*80}\n")


def export_for_web(
    report: MetricsReport,
    output_dir: str,
    formats: List[str] = None
):
    """
    Convenience function to export metrics for web dashboards.

    Args:
        report: MetricsReport instance
        output_dir: Output directory
        formats: List of formats to export
    """
    exporter = DashboardExporter(report)
    exporter.export_all(output_dir, formats)
