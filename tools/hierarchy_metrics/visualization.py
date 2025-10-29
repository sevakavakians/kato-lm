"""
Visualization utilities for hierarchy metrics.

Provides plotting functions for Jupyter notebooks using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from .report import MetricsReport
from .config import HealthStatus


# Health status colors
HEALTH_COLORS = {
    HealthStatus.EXCELLENT: '#2ecc71',  # Green
    HealthStatus.GOOD: '#27ae60',       # Dark green
    HealthStatus.WARNING: '#f39c12',    # Orange
    HealthStatus.POOR: '#e67e22',       # Dark orange
    HealthStatus.CRITICAL: '#e74c3c',   # Red
    HealthStatus.UNKNOWN: '#95a5a6',    # Gray
}


def plot_compression_ratios(
    report: MetricsReport,
    figsize: Tuple[int, int] = (10, 6),
    show_thresholds: bool = True
):
    """
    Plot compression ratios across hierarchy levels.

    Args:
        report: MetricsReport instance
        figsize: Figure size
        show_thresholds: Show threshold bands
    """
    ratios = report.compression.compression_ratios

    if not ratios:
        print("No compression ratios to plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    pairs = list(ratios.keys())
    values = list(ratios.values())

    # Get health statuses
    colors = [
        HEALTH_COLORS[report.compression.health_status.get(f"compression_{pair}", HealthStatus.UNKNOWN)]
        for pair in pairs
    ]

    # Bar plot
    bars = ax.bar(pairs, values, color=colors, alpha=0.7, edgecolor='black')

    # Show thresholds if requested
    if show_thresholds:
        # Assume chunk_size=15 for threshold bands
        chunk_size = 15
        ax.axhline(y=chunk_size * 0.8, color='green', linestyle='--', alpha=0.3, label='Excellent range')
        ax.axhline(y=chunk_size * 1.2, color='green', linestyle='--', alpha=0.3)
        ax.fill_between(range(len(pairs)), chunk_size * 0.8, chunk_size * 1.2, alpha=0.1, color='green')

    ax.set_xlabel('Level Pair', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Compression Ratios Across Hierarchy', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Legend for health status
    legend_elements = [
        mpatches.Patch(color=HEALTH_COLORS[HealthStatus.EXCELLENT], label='Excellent'),
        mpatches.Patch(color=HEALTH_COLORS[HealthStatus.GOOD], label='Good'),
        mpatches.Patch(color=HEALTH_COLORS[HealthStatus.WARNING], label='Warning'),
        mpatches.Patch(color=HEALTH_COLORS[HealthStatus.POOR], label='Poor'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_pattern_counts(
    report: MetricsReport,
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = True
):
    """
    Plot pattern counts at each hierarchy level.

    Args:
        report: MetricsReport instance
        figsize: Figure size
        log_scale: Use logarithmic y-axis
    """
    counts = report.compression.pattern_counts

    if not counts:
        print("No pattern counts to plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    levels = sorted(counts.keys())
    values = [counts[level] for level in levels]

    # Bar plot
    bars = ax.bar(levels, values, color='steelblue', alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Hierarchy Level', fontsize=12)
    ax.set_ylabel('Pattern Count', fontsize=12)
    ax.set_title('Pattern Count Progression', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Pattern Count (log scale)', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_reusability_distribution(
    report: MetricsReport,
    level: str,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot reusability statistics for a specific level.

    Args:
        report: MetricsReport instance
        level: Level to plot (e.g., 'node0')
        figsize: Figure size
    """
    reusability = report.connectivity.reusability.get(level)

    if not reusability:
        print(f"No reusability data for {level}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Stats to display
    stats = {
        'Mean Parents': reusability['mean_parents'],
        'Median Parents': reusability['median_parents'],
        '90th Percentile': reusability['p90_parents'],
    }

    labels = list(stats.keys())
    values = list(stats.values())

    # Bar plot
    bars = ax.bar(labels, values, color='coral', alpha=0.7, edgecolor='black')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Parent Count', fontsize=12)
    ax.set_title(f'Reusability Statistics - {level}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add orphan rate as text annotation
    orphan_rate = reusability['orphan_rate']
    health = report.connectivity.health_status.get(f"orphan_{level}", HealthStatus.UNKNOWN)
    color = HEALTH_COLORS[health]

    ax.text(0.98, 0.95, f"Orphan Rate: {orphan_rate:.1%}",
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()
    plt.show()


def plot_coverage_heatmap(
    report: MetricsReport,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot coverage as a heatmap.

    Args:
        report: MetricsReport instance
        figsize: Figure size
    """
    coverage = report.connectivity.coverage

    if not coverage:
        print("No coverage data to plot")
        return

    # Parse level pairs
    pairs = []
    values = []

    for pair, rate in sorted(coverage.items()):
        pairs.append(pair)
        values.append(rate)

    fig, ax = plt.subplots(figsize=figsize)

    # Horizontal bar chart
    y_pos = np.arange(len(pairs))

    # Color bars by health
    colors = [
        HEALTH_COLORS[report.connectivity.health_status.get(f"coverage_{pair}", HealthStatus.UNKNOWN)]
        for pair in pairs
    ]

    bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')

    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.02, i, f'{val:.1%}',
                ha='left', va='center', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs)
    ax.set_xlabel('Coverage Rate', fontsize=12)
    ax.set_title('Coverage: % of Lower Patterns Used in Upper Level', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.0])
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_mutual_information(
    report: MetricsReport,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot mutual information between levels.

    Args:
        report: MetricsReport instance
        figsize: Figure size
    """
    mi = report.information.mutual_information
    effectiveness = report.information.constraint_effectiveness

    if not mi:
        print("No mutual information data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    pairs = list(mi.keys())

    # Plot 1: Mutual Information (bits)
    mi_values = [mi[pair] for pair in pairs]
    bars1 = ax1.bar(pairs, mi_values, color='teal', alpha=0.7, edgecolor='black')

    ax1.set_xlabel('Level Pair', fontsize=12)
    ax1.set_ylabel('Mutual Information (bits)', fontsize=12)
    ax1.set_title('Mutual Information', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Constraint Effectiveness (normalized)
    eff_values = [effectiveness[pair] for pair in pairs]

    # Color by health
    colors = [
        HEALTH_COLORS[report.information.health_status.get(f"effectiveness_{pair}", HealthStatus.UNKNOWN)]
        for pair in pairs
    ]

    bars2 = ax2.bar(pairs, eff_values, color=colors, alpha=0.7, edgecolor='black')

    ax2.set_xlabel('Level Pair', fontsize=12)
    ax2.set_ylabel('Constraint Effectiveness', fontsize=12)
    ax2.set_title('Constraint Effectiveness (NMI)', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add percentage labels to effectiveness
    for bar, val in zip(bars2, eff_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_entropy_progression(
    report: MetricsReport,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot entropy progression across hierarchy.

    Args:
        report: MetricsReport instance
        figsize: Figure size
    """
    entropy = report.information.entropy_progression

    if not entropy:
        print("No entropy data to plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    levels = sorted(entropy.keys())
    values = [entropy[level] for level in levels]

    # Line plot with markers
    ax.plot(levels, values, marker='o', markersize=10, linewidth=2,
            color='purple', label='Entropy')

    # Fill area under curve
    ax.fill_between(range(len(levels)), values, alpha=0.2, color='purple')

    ax.set_xlabel('Hierarchy Level', fontsize=12)
    ax.set_ylabel('Entropy (bits)', fontsize=12)
    ax.set_title('Entropy Progression Across Hierarchy', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_training_dynamics(
    report: MetricsReport,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot training dynamics (growth curves and reusability trends).

    Args:
        report: MetricsReport instance
        figsize: Figure size
    """
    if not report.training_dynamics:
        print("No training dynamics data to plot")
        return

    dynamics = report.training_dynamics
    checkpoints = dynamics.checkpoints

    if not checkpoints:
        print("No checkpoints available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract data
    samples = [cp['samples_processed'] for cp in checkpoints]
    total_patterns = [sum(cp['pattern_counts'].values()) for cp in checkpoints]
    reusability_trend = dynamics.reusability_trend

    # Plot 1: Pattern Growth (log-log scale)
    ax1.scatter(samples, total_patterns, s=50, alpha=0.6, color='blue', label='Actual')

    # Plot fitted power-law
    if len(samples) >= 2:
        log_samples = np.log(samples)
        log_patterns = np.log(total_patterns)
        coeffs = np.polyfit(log_samples, log_patterns, deg=1)

        # Generate fitted line
        samples_fit = np.linspace(min(samples), max(samples), 100)
        patterns_fit = np.exp(coeffs[1]) * (samples_fit ** coeffs[0])

        ax1.plot(samples_fit, patterns_fit, 'r--', linewidth=2,
                label=f'Power-law fit: N ∝ S^{dynamics.growth_exponent:.3f}')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Samples Processed', fontsize=12)
    ax1.set_ylabel('Total Patterns', fontsize=12)
    ax1.set_title(f'Pattern Growth (R²={dynamics.growth_r_squared:.3f})', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Plot 2: Reusability Trend
    if reusability_trend:
        checkpoint_nums = list(range(len(reusability_trend)))

        ax2.plot(checkpoint_nums, reusability_trend, marker='o', markersize=8,
                linewidth=2, color='green', label='Mean Parents (node0)')

        # Plot linear trend
        if len(reusability_trend) >= 2:
            coeffs = np.polyfit(checkpoint_nums, reusability_trend, deg=1)
            trend_line = np.poly1d(coeffs)
            ax2.plot(checkpoint_nums, trend_line(checkpoint_nums), 'r--', linewidth=2,
                    label=f'Trend: slope={dynamics.reusability_trend_slope:.4f}')

        ax2.set_xlabel('Checkpoint', fontsize=12)
        ax2.set_ylabel('Mean Parents', fontsize=12)
        ax2.set_title('Reusability Trend Over Training', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_prediction_fanout(
    report: MetricsReport,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot prediction fan-out at each level.

    Args:
        report: MetricsReport instance
        figsize: Figure size
    """
    if not report.prediction:
        print("No prediction data to plot")
        return

    fanout = report.prediction.fanout_by_level

    if not fanout:
        print("No fan-out data available")
        return

    fig, ax = plt.subplots(figsize=figsize)

    levels = sorted(fanout.keys())
    means = [fanout[level]['mean'] for level in levels]
    stds = [fanout[level]['std'] for level in levels]

    # Color by health
    colors = [
        HEALTH_COLORS[report.prediction.health_status.get(f"fanout_{level}", HealthStatus.UNKNOWN)]
        for level in levels
    ]

    # Bar plot with error bars
    bars = ax.bar(levels, means, yerr=stds, color=colors, alpha=0.7,
                  edgecolor='black', capsize=5, error_kw={'linewidth': 2})

    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Hierarchy Level', fontsize=12)
    ax.set_ylabel('Mean Prediction Fan-Out', fontsize=12)
    ax.set_title('Prediction Fan-Out at Each Level', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_health_summary(
    report: MetricsReport,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot overall health summary dashboard.

    Args:
        report: MetricsReport instance
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    summary = report.metrics_summary

    # Overall health (large text)
    ax_overall = fig.add_subplot(gs[0, :])
    ax_overall.axis('off')

    overall_color = HEALTH_COLORS[summary.overall_health]
    ax_overall.text(0.5, 0.7, 'OVERALL HEALTH', ha='center', va='center',
                   fontsize=18, fontweight='bold')
    ax_overall.text(0.5, 0.3, summary.overall_health.value.upper(), ha='center', va='center',
                   fontsize=32, fontweight='bold', color=overall_color)

    # Category health (horizontal bars)
    ax_categories = fig.add_subplot(gs[1, :])

    categories = ['Compression', 'Connectivity', 'Information', 'Prediction', 'Training Dynamics']
    statuses = [
        summary.compression,
        summary.connectivity,
        summary.information,
        summary.prediction,
        summary.training_dynamics
    ]

    y_pos = np.arange(len(categories))
    colors = [HEALTH_COLORS[status] for status in statuses]

    # Encode health as numeric values for bar length
    health_values = {
        HealthStatus.EXCELLENT: 5,
        HealthStatus.GOOD: 4,
        HealthStatus.WARNING: 3,
        HealthStatus.POOR: 2,
        HealthStatus.CRITICAL: 1,
        HealthStatus.UNKNOWN: 0,
    }
    values = [health_values[status] for status in statuses]

    bars = ax_categories.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')

    # Add status labels
    for i, (bar, status) in enumerate(zip(bars, statuses)):
        ax_categories.text(bar.get_width() + 0.1, i, status.value,
                          ha='left', va='center', fontsize=11, fontweight='bold')

    ax_categories.set_yticks(y_pos)
    ax_categories.set_yticklabels(categories)
    ax_categories.set_xlabel('Health Score', fontsize=12)
    ax_categories.set_title('Category Health Scores', fontsize=13, fontweight='bold')
    ax_categories.set_xlim([0, 6])
    ax_categories.grid(axis='x', alpha=0.3)

    # Critical issues (bottom left)
    ax_issues = fig.add_subplot(gs[2, 0])
    ax_issues.axis('off')

    issue_text = "CRITICAL ISSUES\n\n"
    if summary.critical_issues:
        for issue in summary.critical_issues[:5]:  # Show top 5
            issue_text += f"⚠️  {issue}\n"
    else:
        issue_text += "None"

    ax_issues.text(0.1, 0.9, issue_text, ha='left', va='top', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.5))

    # Recommendations (bottom right)
    ax_recs = fig.add_subplot(gs[2, 1])
    ax_recs.axis('off')

    rec_text = "RECOMMENDATIONS\n\n"
    if summary.recommendations:
        for rec in summary.recommendations[:5]:  # Show top 5
            rec_text += f"→  {rec}\n"
    else:
        rec_text += "None"

    ax_recs.text(0.1, 0.9, rec_text, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#e6ffe6', alpha=0.5))

    plt.suptitle('Hierarchy Metrics Health Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_full_dashboard(report: MetricsReport):
    """
    Plot comprehensive dashboard with all key metrics.

    Args:
        report: MetricsReport instance
    """
    # Health summary
    plot_health_summary(report, figsize=(12, 8))

    # Compression and connectivity
    plot_compression_ratios(report)
    plot_pattern_counts(report)

    # Reusability for each level
    for level in sorted(report.compression.pattern_counts.keys()):
        if level in report.connectivity.reusability:
            plot_reusability_distribution(report, level)

    # Coverage
    plot_coverage_heatmap(report)

    # Information theory
    plot_mutual_information(report)
    plot_entropy_progression(report)

    # Training dynamics
    if report.training_dynamics:
        plot_training_dynamics(report)

    # Prediction
    if report.prediction:
        plot_prediction_fanout(report)
