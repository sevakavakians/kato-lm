#!/usr/bin/env python3
"""
Training Run Comparison - Visualization and Analysis Utilities

Comprehensive tools for comparing training runs with different configurations.
Provides visualization functions to identify optimal hyperparameters and
understand performance trade-offs.

Key Features:
- Configuration comparison tables
- Performance scatter plots (config params vs metrics)
- Frequency distribution overlays (Zipfian analysis)
- Shannon entropy comparisons
- Hierarchy utilization visualization
- Storage breakdown charts
- Scaling analysis curves
- Efficiency heatmaps
- Configuration recommendations

Usage:
    from tools import TrainingHistory, plot_frequency_comparison

    history = TrainingHistory()
    comparison_df = history.compare_runs(n_recent=10)

    # Display styled comparison table
    styled_table = plot_configuration_table(comparison_df)
    display(styled_table)

    # Visualize frequency distributions
    plot_frequency_comparison(history, ['run_1', 'run_2', 'run_3'])

    # Find optimal configuration
    optimal = find_optimal_configuration(history, goal='throughput')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Any, Optional, Tuple
from matplotlib.ticker import MaxNLocator
import re


def _extract_numeric_sort_key(value: Any) -> float:
    """
    Extract numeric sort key from various column types.

    Handles:
    - Numeric values: returned as-is
    - Strings with numbers: "8" → 8, "4,5,10,15" → 4 (first number)
    - None/NaN: returns infinity (sorts last)

    Args:
        value: Value to extract sort key from

    Returns:
        Numeric sort key
    """
    if pd.isna(value):
        return float('inf')
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # For chunk sizes like "8" or "4,5,10,15", extract first number
        match = re.search(r'\d+', value)
        if match:
            return float(match.group())
    return 0.0


def _sort_dataframe_for_plotting(
    df: pd.DataFrame,
    sort_col: str,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Sort DataFrame for plotting with numeric ordering.

    Automatically handles numeric and categorical columns,
    ensuring values appear in numeric order on charts.

    Args:
        df: DataFrame to sort
        sort_col: Column name to sort by
        ascending: Sort order

    Returns:
        Sorted DataFrame copy
    """
    if sort_col not in df.columns:
        return df

    df = df.copy()
    df['_sort_key'] = df[sort_col].apply(_extract_numeric_sort_key)
    df = df.sort_values('_sort_key', ascending=ascending)
    df = df.drop(columns=['_sort_key'])
    return df


def plot_configuration_table(
    comparison_df: pd.DataFrame,
    highlight_best: bool = True
) -> pd.DataFrame:
    """
    Create styled configuration comparison table.

    Args:
        comparison_df: DataFrame from TrainingHistory.compare_runs()
        highlight_best: Highlight best values in each metric column

    Returns:
        Styled pandas DataFrame
    """
    if comparison_df.empty:
        print("⚠️  No data to display")
        return comparison_df

    # Define columns to highlight (higher is better)
    higher_better = ['Rate (samples/s)', 'Total Patterns']
    # Lower is better
    lower_better = ['Time (s)', 'Storage (MB)']

    def highlight_max(s):
        """Highlight maximum value in green"""
        is_max = s == s.max()
        return ['background-color: #90EE90' if v else '' for v in is_max]

    def highlight_min(s):
        """Highlight minimum value in green"""
        is_min = s == s.min()
        return ['background-color: #90EE90' if v else '' for v in is_min]

    styled = comparison_df.style

    if highlight_best:
        for col in higher_better:
            if col in comparison_df.columns:
                styled = styled.apply(highlight_max, subset=[col])

        for col in lower_better:
            if col in comparison_df.columns:
                styled = styled.apply(highlight_min, subset=[col])

    # Format numeric columns
    format_dict = {
        'Time (s)': '{:.2f}',
        'Rate (samples/s)': '{:.2f}',
        'Storage (MB)': '{:.2f}',
        'Total Patterns': '{:,.0f}',
        'Zipf Alpha': '{:.3f}',
        'Peak Mem (MB)': '{:.1f}',
        'Avg CPU (%)': '{:.1f}',
        'Chunk Min': '{:.0f}',
        'Chunk Max': '{:.0f}',
        'Chunk Mean': '{:.1f}',
    }

    for col, fmt in format_dict.items():
        if col in comparison_df.columns:
            styled = styled.format({col: fmt}, na_rep='-')

    return styled


def plot_frequency_comparison(
    history: Any,  # TrainingHistory
    run_ids: List[str],
    max_rank: int = 1000,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Overlay frequency distributions for multiple runs (Zipfian analysis).

    Args:
        history: TrainingHistory instance
        run_ids: List of run IDs to compare
        max_rank: Maximum rank to plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))

    for idx, run_id in enumerate(run_ids):
        snapshot = history.load_snapshot(run_id)

        if not snapshot or 'node0' not in snapshot.nodes:
            print(f"⚠️  No snapshot for {run_id}")
            continue

        node0 = snapshot.nodes['node0']
        freq_hist = node0.frequency_histogram

        if not freq_hist:
            continue

        # Convert to rank-frequency
        ranks = []
        frequencies = []
        for freq, count in sorted(freq_hist.items(), reverse=True):
            start_rank = len(ranks) + 1
            for i in range(count):
                if start_rank + i > max_rank:
                    break
                ranks.append(start_rank + i)
                frequencies.append(freq)
            if len(ranks) >= max_rank:
                break

        # Plot
        label = f"{run_id[:12]}... (α={node0.zipf_alpha:.2f})" if node0.zipf_alpha else run_id[:12]
        plt.loglog(ranks, frequencies, 'o-', alpha=0.6, color=colors[idx],
                   markersize=3, label=label, linewidth=1.5)

    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Frequency Distribution Comparison (node0)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()


def plot_hierarchy_utilization(
    history: Any,
    run_ids: List[str],
    figsize: Tuple[int, int] = (12, 7)
):
    """
    Stacked bar chart of pattern counts per node across runs.

    Args:
        history: TrainingHistory instance
        run_ids: List of run IDs to compare
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    data_by_run = {}

    for run_id in run_ids:
        snapshot = history.load_snapshot(run_id)

        if not snapshot:
            continue

        patterns_by_node = {}
        for node_name in sorted(snapshot.nodes.keys()):
            patterns_by_node[node_name] = snapshot.nodes[node_name].total_patterns

        data_by_run[run_id[:12]] = patterns_by_node

    if not data_by_run:
        print("⚠️  No data to visualize")
        return

    # Sort run IDs numerically (extracts numbers from run ID format)
    run_ids_with_keys = [(run_id, _extract_numeric_sort_key(run_id)) for run_id in data_by_run.keys()]
    run_ids_sorted = [x[0] for x in sorted(run_ids_with_keys, key=lambda x: x[1])]

    # Prepare data for stacked bars
    run_labels = run_ids_sorted
    node_names = sorted(next(iter(data_by_run.values())).keys())

    # Create stacked bars
    bottom = np.zeros(len(run_labels))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(node_names)))

    for idx, node_name in enumerate(node_names):
        values = [data_by_run[run][node_name] for run in run_labels]
        plt.bar(run_labels, values, bottom=bottom, label=node_name, color=colors[idx])
        bottom += values

    plt.xlabel('Training Run', fontsize=12)
    plt.ylabel('Total Patterns', fontsize=12)
    plt.title('Hierarchy Utilization (Patterns per Node)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_storage_breakdown(
    history: Any,
    run_ids: List[str],
    figsize: Tuple[int, int] = (12, 7)
):
    """
    Stacked bar chart of storage per node across runs.

    Args:
        history: TrainingHistory instance
        run_ids: List of run IDs to compare
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    data_by_run = {}

    for run_id in run_ids:
        snapshot = history.load_snapshot(run_id)

        if not snapshot:
            continue

        storage_by_node = {}
        for node_name in sorted(snapshot.nodes.keys()):
            storage_by_node[node_name] = snapshot.nodes[node_name].db_size_mb

        data_by_run[run_id[:12]] = storage_by_node

    if not data_by_run:
        print("⚠️  No data to visualize")
        return

    # Sort run IDs numerically (extracts numbers from run ID format)
    run_ids_with_keys = [(run_id, _extract_numeric_sort_key(run_id)) for run_id in data_by_run.keys()]
    run_ids_sorted = [x[0] for x in sorted(run_ids_with_keys, key=lambda x: x[1])]

    # Prepare data
    run_labels = run_ids_sorted
    node_names = sorted(next(iter(data_by_run.values())).keys())

    # Create stacked bars
    bottom = np.zeros(len(run_labels))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(node_names)))

    for idx, node_name in enumerate(node_names):
        values = [data_by_run[run][node_name] for run in run_labels]
        plt.bar(run_labels, values, bottom=bottom, label=node_name, color=colors[idx])
        bottom += values

    plt.xlabel('Training Run', fontsize=12)
    plt.ylabel('Storage (MB)', fontsize=12)
    plt.title('Storage Breakdown per Node', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_performance_scatter(
    comparison_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_by: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7)
):
    """
    Scatter plot of configuration parameters vs metrics.

    Args:
        comparison_df: DataFrame from compare_runs()
        x_col: Column name for X-axis
        y_col: Column name for Y-axis
        color_by: Optional column to color points by
        figsize: Figure size
    """
    if comparison_df.empty:
        print("⚠️  No data to plot")
        return

    # Handle non-uniform chunk sizes
    if x_col == 'Chunk Size' and 'Chunk Uniform' in comparison_df.columns:
        has_non_uniform = (~comparison_df['Chunk Uniform']).any()
        if has_non_uniform:
            print(f"⚠️  Non-uniform chunk sizes detected. Using 'Chunk Mean' for X-axis.")
            x_col = 'Chunk Mean'

    # Sort data by x_col for proper numeric ordering
    comparison_df = _sort_dataframe_for_plotting(comparison_df, x_col, ascending=True)

    plt.figure(figsize=figsize)

    if color_by and color_by in comparison_df.columns:
        # Color by categorical or numeric column
        unique_values = comparison_df[color_by].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
        color_map = dict(zip(unique_values, colors))

        for value in unique_values:
            mask = comparison_df[color_by] == value
            plt.scatter(
                comparison_df.loc[mask, x_col],
                comparison_df.loc[mask, y_col],
                label=f"{color_by}={value}",
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )

        plt.legend(loc='best')
    else:
        plt.scatter(
            comparison_df[x_col],
            comparison_df[y_col],
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_scaling_analysis(
    comparison_df: pd.DataFrame,
    group_by: str = 'Chunk Size',
    figsize: Tuple[int, int] = (14, 7)
):
    """
    Plot how patterns and storage scale with sample count.

    Args:
        comparison_df: DataFrame from compare_runs()
        group_by: Column to group configurations by
        figsize: Figure size
    """
    if comparison_df.empty:
        print("⚠️  No data to plot")
        return

    # Handle non-uniform chunk sizes - suggest grouping by pattern
    if group_by == 'Chunk Size' and 'Chunk Pattern' in comparison_df.columns:
        has_non_uniform = 'Chunk Uniform' in comparison_df.columns and (~comparison_df['Chunk Uniform']).any()
        if has_non_uniform:
            print(f"⚠️  Non-uniform chunk sizes detected. Consider grouping by 'Chunk Pattern' instead.")
            print(f"   Grouping by '{group_by}' will separate configs like [4,5,10] and [4,5,15].")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Group by configuration
    groups = comparison_df.groupby(group_by)

    # Sort groups by numeric value of group key for consistent ordering
    sorted_groups = sorted(groups, key=lambda x: _extract_numeric_sort_key(x[0]))

    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_groups)))

    for idx, (config_value, group_df) in enumerate(sorted_groups):
        # Sort by samples
        group_df = group_df.sort_values('Samples')

        # Plot patterns vs samples
        ax1.plot(
            group_df['Samples'],
            group_df['Total Patterns'],
            'o-',
            label=f"{group_by}={config_value}",
            color=colors[idx],
            linewidth=2,
            markersize=6
        )

        # Plot storage vs samples
        ax2.plot(
            group_df['Samples'],
            group_df['Storage (MB)'],
            'o-',
            label=f"{group_by}={config_value}",
            color=colors[idx],
            linewidth=2,
            markersize=6
        )

    ax1.set_xlabel('Samples Processed', fontsize=11)
    ax1.set_ylabel('Total Patterns', fontsize=11)
    ax1.set_title('Pattern Growth', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Samples Processed', fontsize=11)
    ax2.set_ylabel('Storage (MB)', fontsize=11)
    ax2.set_title('Storage Growth', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_efficiency_metrics(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6),
    sort_by: str = 'Chunk Size'
):
    """
    Plot efficiency metrics: patterns/second and patterns/MB.

    Args:
        comparison_df: DataFrame from compare_runs()
        figsize: Figure size
        sort_by: Column to sort bars by (default: 'Chunk Size')
    """
    if comparison_df.empty:
        print("⚠️  No data to plot")
        return

    # Compute efficiency metrics
    df = comparison_df.copy()
    df['Patterns/Sec'] = df['Total Patterns'] / df['Time (s)']
    df['Patterns/MB'] = df['Total Patterns'] / df['Storage (MB)']

    # Sort by specified column for consistent ordering
    if sort_by in df.columns:
        df = _sort_dataframe_for_plotting(df, sort_by, ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Patterns per second
    ax1.barh(range(len(df)), df['Patterns/Sec'], color='skyblue', edgecolor='black')
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df['Run ID'], fontsize=9)
    ax1.set_xlabel('Patterns/Second', fontsize=11)
    ax1.set_title('Learning Efficiency (Patterns/Sec)', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Patterns per MB
    ax2.barh(range(len(df)), df['Patterns/MB'], color='lightcoral', edgecolor='black')
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels(df['Run ID'], fontsize=9)
    ax2.set_xlabel('Patterns/MB', fontsize=11)
    ax2.set_title('Storage Efficiency (Patterns/MB)', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_configuration_heatmap(
    comparison_df: pd.DataFrame,
    row_param: str,
    col_param: str,
    value_metric: str,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Heatmap showing metric values across 2D parameter space.

    Args:
        comparison_df: DataFrame from compare_runs()
        row_param: Parameter for rows
        col_param: Parameter for columns
        value_metric: Metric to display as color
        figsize: Figure size
    """
    if comparison_df.empty:
        print("⚠️  No data to plot")
        return

    # Pivot data into heatmap format
    try:
        pivot_data = comparison_df.pivot_table(
            index=row_param,
            columns=col_param,
            values=value_metric,
            aggfunc='mean'  # Average if multiple runs with same config
        )
    except Exception as e:
        print(f"⚠️  Cannot create heatmap: {e}")
        return

    # Sort rows and columns numerically
    row_sort_keys = [_extract_numeric_sort_key(x) for x in pivot_data.index]
    col_sort_keys = [_extract_numeric_sort_key(x) for x in pivot_data.columns]

    row_order = sorted(range(len(row_sort_keys)), key=lambda i: row_sort_keys[i])
    col_order = sorted(range(len(col_sort_keys)), key=lambda i: col_sort_keys[i])

    pivot_data = pivot_data.iloc[row_order, col_order]

    plt.figure(figsize=figsize)
    im = plt.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')

    # Set ticks and labels
    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
    plt.yticks(range(len(pivot_data.index)), pivot_data.index)

    plt.xlabel(col_param, fontsize=12)
    plt.ylabel(row_param, fontsize=12)
    plt.title(f'{value_metric} Heatmap', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(value_metric, rotation=270, labelpad=20)

    # Annotate cells with values
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.values[i, j]
            if not np.isnan(value):
                plt.text(j, i, f'{value:.1f}', ha='center', va='center',
                        color='black' if value > pivot_data.values.mean() else 'white',
                        fontsize=10)

    plt.tight_layout()
    plt.show()


def find_optimal_configuration(
    history: Any,
    goal: str = 'throughput',
    min_samples: int = 100
) -> Dict[str, Any]:
    """
    Find optimal configuration based on goal.

    Args:
        history: TrainingHistory instance
        goal: 'throughput', 'quality', 'storage', or 'balanced'
        min_samples: Minimum samples processed to consider

    Returns:
        Dictionary with optimal run info
    """
    df = history.compare_runs()

    if df.empty:
        return {'error': 'No runs available'}

    # Filter by minimum samples
    df = df[df['Samples'] >= min_samples]

    if df.empty:
        return {'error': f'No runs with >={min_samples} samples'}

    # Add tie-breaking preference for uniform chunk sizes
    has_chunk_uniform = 'Chunk Uniform' in df.columns
    if has_chunk_uniform:
        df['_uniform_bonus'] = df['Chunk Uniform'].fillna(False).astype(float) * 0.001

    if goal == 'throughput':
        # Maximize samples/second (with uniform preference)
        df['_score'] = df['Rate (samples/s)']
        if has_chunk_uniform:
            df['_score'] = df['_score'] + df['_uniform_bonus'] * df['_score'].max()
        best_idx = df['_score'].idxmax()
        metric = 'Rate (samples/s)'

    elif goal == 'quality':
        # Maximize Zipf alpha (with uniform preference)
        df['_score'] = df['Zipf Alpha']
        if has_chunk_uniform:
            df['_score'] = df['_score'] + df['_uniform_bonus'] * df['_score'].max()
        best_idx = df['_score'].idxmax()
        metric = 'Zipf Alpha'

    elif goal == 'storage':
        # Minimize storage per pattern (with uniform preference)
        df['Storage/Pattern'] = df['Storage (MB)'] / df['Total Patterns']
        df['_score'] = -df['Storage/Pattern']  # Negate for maximization
        if has_chunk_uniform:
            df['_score'] = df['_score'] + df['_uniform_bonus'] * abs(df['_score']).max()
        best_idx = df['_score'].idxmax()
        metric = 'Storage/Pattern'

    elif goal == 'balanced':
        # Balance throughput and quality (with uniform preference)
        df['_score'] = df['Rate (samples/s)'].rank(pct=True) + df['Zipf Alpha'].rank(pct=True)
        if has_chunk_uniform:
            df['_score'] = df['_score'] + df['_uniform_bonus'] * df['_score'].max()
        best_idx = df['_score'].idxmax()
        metric = 'Balanced Score'

    else:
        return {'error': f'Unknown goal: {goal}'}

    best_run = df.loc[best_idx]

    return {
        'goal': goal,
        'metric': metric,
        'run_id': best_run['Run ID'],
        'chunk_size': best_run['Chunk Size'],
        'levels': best_run['Levels'],
        'batch_size': best_run['Batch Size'],
        'workers': best_run['Workers'],
        'rate': best_run['Rate (samples/s)'],
        'patterns': best_run['Total Patterns'],
        'storage_mb': best_run['Storage (MB)'],
        'zipf_alpha': best_run.get('Zipf Alpha', None)
    }


def plot_hierarchy_alpha_comparison(
    history: Any,
    run_ids: List[str],
    figsize: Tuple[int, int] = (12, 7)
):
    """
    Plot how Zipf alpha changes across hierarchy levels for multiple runs.

    Shows whether compression improves or degrades at higher abstraction levels.
    Ideal: alpha increases with level (more reuse at higher abstractions)
    Actual: may decrease if documents are structurally unique

    Args:
        history: TrainingHistory instance
        run_ids: List of run IDs to compare
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))

    for idx, run_id in enumerate(run_ids):
        snapshot = history.load_snapshot(run_id)

        if not snapshot:
            print(f"⚠️  No snapshot for {run_id}")
            continue

        # Extract alpha values for each node
        node_levels = []
        alpha_values = []

        for node_name in sorted(snapshot.nodes.keys()):
            node = snapshot.nodes[node_name]
            level = int(node_name.replace('node', ''))

            # Only plot if alpha exists (not None)
            if node.zipf_alpha is not None:
                node_levels.append(level)
                alpha_values.append(node.zipf_alpha)

        if not alpha_values:
            print(f"⚠️  No alpha values for {run_id}")
            continue

        # Plot with markers for each data point
        label = run_id if len(run_id) <= 20 else run_id[:20] + '...'
        plt.plot(
            node_levels,
            alpha_values,
            'o-',
            color=colors[idx],
            label=label,
            linewidth=2,
            markersize=8,
            alpha=0.7
        )

    # Add reference zone for ideal Zipfian range
    plt.axhspan(1.0, 1.5, alpha=0.1, color='green', label='Ideal range (α=1.0-1.5)')

    plt.xlabel('Hierarchy Level (node)', fontsize=12)
    plt.ylabel('Zipf Alpha (α)', fontsize=12)
    plt.title('Zipfian Alpha Across Hierarchy Levels', fontsize=14, fontweight='bold')
    plt.yscale('log')  # Log scale to show small values
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(range(5), ['node0', 'node1', 'node2', 'node3', 'node4'])

    # Add horizontal line at α=1.0
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='α=1.0')

    plt.tight_layout()
    plt.show()


def plot_alpha_vs_patterns_scatter(
    history: Any,
    run_ids: List[str],
    figsize: Tuple[int, int] = (10, 7)
):
    """
    Scatter plot of pattern counts vs alpha values, color-coded by node level.

    Helps identify if low alpha is due to insufficient pattern counts.

    Args:
        history: TrainingHistory instance
        run_ids: List of run IDs to compare
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Collect data points
    data_points = []

    for run_id in run_ids:
        snapshot = history.load_snapshot(run_id)

        if not snapshot:
            continue

        for node_name in sorted(snapshot.nodes.keys()):
            node = snapshot.nodes[node_name]
            level = int(node_name.replace('node', ''))

            if node.zipf_alpha is not None:
                data_points.append({
                    'level': level,
                    'node_name': node_name,
                    'patterns': node.total_patterns,
                    'alpha': node.zipf_alpha,
                    'r_squared': node.zipf_r_squared or 0.0
                })

    if not data_points:
        print("⚠️  No data points to plot")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(data_points)

    # Create color map by level
    levels = df['level'].unique()
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(levels)))
    color_map = dict(zip(sorted(levels), colors))

    # Plot each level separately for legend
    for level in sorted(levels):
        level_df = df[df['level'] == level]
        plt.scatter(
            level_df['patterns'],
            level_df['alpha'],
            s=100,
            alpha=0.7,
            c=[color_map[level]] * len(level_df),
            label=f'node{level}',
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel('Total Patterns', fontsize=12)
    plt.ylabel('Zipf Alpha (α)', fontsize=12)
    plt.title('Pattern Count vs Zipfian Alpha', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3, which='both')

    # Add reference line at α=1.0
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='α=1.0 (ideal)')

    plt.tight_layout()
    plt.show()


def print_comparison_summary(comparison_df: pd.DataFrame):
    """Print statistical summary of comparison data"""
    if comparison_df.empty:
        print("⚠️  No data to summarize")
        return

    print(f"\n{'='*80}")
    print("TRAINING RUN COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    print(f"Total runs compared: {len(comparison_df)}\n")

    # Numeric columns to summarize
    numeric_cols = [
        'Samples', 'Time (s)', 'Rate (samples/s)',
        'Storage (MB)', 'Total Patterns', 'Zipf Alpha'
    ]

    summary_data = []

    for col in numeric_cols:
        if col in comparison_df.columns:
            values = comparison_df[col].dropna()
            if len(values) > 0:
                summary_data.append({
                    'Metric': col,
                    'Min': values.min(),
                    'Max': values.max(),
                    'Mean': values.mean(),
                    'Std': values.std()
                })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print(f"\n{'='*80}\n")


def main():
    """Example usage"""
    print("Training Comparison Utilities loaded")
    print("\nKey functions:")
    print("  - plot_frequency_comparison()")
    print("  - plot_hierarchy_utilization()")
    print("  - plot_performance_scatter()")
    print("  - plot_scaling_analysis()")
    print("  - find_optimal_configuration()")


if __name__ == '__main__':
    main()
