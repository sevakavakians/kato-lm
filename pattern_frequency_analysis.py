#!/usr/bin/env python3
"""
Pattern Frequency Analysis for KATO Hierarchical Learning

Analyzes pattern repetition by querying frequency values from Redis
and generates histograms for each node.

Usage:
    python pattern_frequency_analysis.py
    python pattern_frequency_analysis.py --output-dir ./my_analysis
    python pattern_frequency_analysis.py --nodes node0_kato node1_kato
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

# Project imports
from tools.kato_storage.connection_manager import get_redis_client, get_clickhouse_client
from tools.kato_storage.redis_writer import RedisWriter


def discover_kato_nodes(clickhouse_client):
    """
    Discover all KATO knowledge bases by querying ClickHouse.

    Returns:
        list: List of kb_id strings
    """
    try:
        result = clickhouse_client.query("""
            SELECT DISTINCT kb_id
            FROM kato.patterns_data
            WHERE kb_id LIKE '%_kato'
            ORDER BY kb_id
        """)

        kb_ids = [row[0] for row in result.result_rows]
        return kb_ids
    except Exception as e:
        print(f"Error discovering nodes: {e}")
        return []


def discover_nodes_and_patterns(clickhouse_client):
    """
    Discover all nodes and get pattern names for each.

    Returns:
        dict: {kb_id: [pattern_name1, pattern_name2, ...]}
    """
    # 1. Discover nodes
    nodes = discover_kato_nodes(clickhouse_client)

    # 2. For each node, query all pattern names
    node_patterns = {}
    for kb_id in nodes:
        result = clickhouse_client.query(f"""
            SELECT name FROM kato.patterns_data
            WHERE kb_id = '{kb_id}'
        """)

        pattern_names = [row[0] for row in result.result_rows]
        node_patterns[kb_id] = pattern_names

    return node_patterns


def get_frequencies_for_node(kb_id, pattern_names, redis_client):
    """
    Query Redis for frequencies of all patterns in a node.

    Args:
        kb_id: Node identifier (e.g., 'node0_kato')
        pattern_names: List of pattern name hashes
        redis_client: Redis client connection

    Returns:
        list: Frequency values (same order as pattern_names)
    """
    writer = RedisWriter(kb_id=kb_id, redis_client=redis_client)

    frequencies = []
    for pattern_name in tqdm(pattern_names, desc=f"Fetching {kb_id} frequencies"):
        freq = writer.get_frequency(pattern_name)
        frequencies.append(freq)

    return frequencies


def analyze_frequency_distribution(frequencies, kb_id):
    """
    Calculate statistical properties of frequency distribution.

    Returns:
        dict: Statistics including mean, median, percentiles, etc.
    """
    freq_array = np.array(frequencies)

    # Filter out zeros (patterns never seen)
    nonzero_freq = freq_array[freq_array > 0]

    stats = {
        'kb_id': kb_id,
        'total_patterns': len(frequencies),
        'patterns_with_frequency': len(nonzero_freq),
        'patterns_zero_frequency': len(frequencies) - len(nonzero_freq),
        'mean_frequency': np.mean(nonzero_freq) if len(nonzero_freq) > 0 else 0,
        'median_frequency': np.median(nonzero_freq) if len(nonzero_freq) > 0 else 0,
        'min_frequency': np.min(nonzero_freq) if len(nonzero_freq) > 0 else 0,
        'max_frequency': np.max(nonzero_freq) if len(nonzero_freq) > 0 else 0,
        'std_frequency': np.std(nonzero_freq) if len(nonzero_freq) > 0 else 0,
        'percentiles': {
            '25th': np.percentile(nonzero_freq, 25) if len(nonzero_freq) > 0 else 0,
            '50th': np.percentile(nonzero_freq, 50) if len(nonzero_freq) > 0 else 0,
            '75th': np.percentile(nonzero_freq, 75) if len(nonzero_freq) > 0 else 0,
            '90th': np.percentile(nonzero_freq, 90) if len(nonzero_freq) > 0 else 0,
            '95th': np.percentile(nonzero_freq, 95) if len(nonzero_freq) > 0 else 0,
            '99th': np.percentile(nonzero_freq, 99) if len(nonzero_freq) > 0 else 0,
        },
        'top_10_frequencies': sorted(nonzero_freq, reverse=True)[:10] if len(nonzero_freq) > 0 else []
    }

    return stats


def generate_histogram(frequencies, kb_id, stats, output_dir='./frequency_analysis'):
    """
    Generate histogram visualization for pattern frequencies.

    Creates two plots:
    1. Linear scale histogram (for overall distribution)
    2. Log scale histogram (for Zipfian distribution analysis)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freq_array = np.array(frequencies)
    nonzero_freq = freq_array[freq_array > 0]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Log scale Y-axis
    ax1.hist(nonzero_freq, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Frequency', fontsize=12)
    ax1.set_ylabel('Number of Patterns (log scale)', fontsize=12)
    ax1.set_title(f'{kb_id} - Pattern Frequency Distribution (Log Scale)', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')

    # Format axes to show integers
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}' if x >= 1 else ''))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{int(y):,}'))

    # Add statistics text
    stats_text = (
        f"Total Patterns: {stats['total_patterns']:,}\n"
        f"Mean Frequency: {stats['mean_frequency']:.2f}\n"
        f"Median Frequency: {stats['median_frequency']:.2f}\n"
        f"Max Frequency: {stats['max_frequency']:,}"
    )
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

    # Subplot 2: Log-log scale (Zipfian analysis)
    # Create rank-frequency plot
    sorted_freq = sorted(nonzero_freq, reverse=True)
    ranks = np.arange(1, len(sorted_freq) + 1)

    ax2.loglog(ranks, sorted_freq, color='darkred', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Rank (log scale)', fontsize=12)
    ax2.set_ylabel('Frequency (log scale)', fontsize=12)
    ax2.set_title(f'{kb_id} - Zipfian Distribution (Rank vs Frequency)', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')

    # Format axes to show integers
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}' if x >= 1 else ''))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{int(y):,}' if y >= 1 else ''))

    # Add Zipf's law reference line (y = k/x)
    # Estimate k from data
    k = sorted_freq[0]
    zipf_line = k / ranks
    ax2.plot(ranks, zipf_line, '--', color='gray', linewidth=1,
             label="Zipf's Law (1/x)", alpha=0.6)
    ax2.legend()

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / f'{kb_id}_frequency_histogram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved histogram: {output_path}")

    return str(output_path)


def generate_summary_report(all_stats, output_dir='./frequency_analysis'):
    """
    Generate text summary report of all nodes.
    """
    output_path = Path(output_dir) / 'frequency_analysis_summary.txt'

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("KATO Pattern Frequency Analysis Summary\n")
        f.write("="*80 + "\n\n")

        for stats in all_stats:
            f.write(f"\n{stats['kb_id'].upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Patterns: {stats['total_patterns']:,}\n")
            f.write(f"Patterns with frequency > 0: {stats['patterns_with_frequency']:,}\n")
            f.write(f"Patterns with frequency = 0: {stats['patterns_zero_frequency']:,}\n")
            f.write(f"\nFrequency Statistics:\n")
            f.write(f"  Mean: {stats['mean_frequency']:.2f}\n")
            f.write(f"  Median: {stats['median_frequency']:.2f}\n")
            f.write(f"  Std Dev: {stats['std_frequency']:.2f}\n")
            f.write(f"  Min: {stats['min_frequency']:.0f}\n")
            f.write(f"  Max: {stats['max_frequency']:,.0f}\n")
            f.write(f"\nPercentiles:\n")
            for pct, val in stats['percentiles'].items():
                f.write(f"  {pct}: {val:.2f}\n")
            f.write(f"\nTop 10 Frequencies:\n")
            for i, freq in enumerate(stats['top_10_frequencies'], 1):
                f.write(f"  #{i}: {freq:.0f}\n")
            f.write("\n")

        f.write("="*80 + "\n")

    print(f"\n✓ Summary report saved: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze pattern frequency distribution and generate histograms'
    )
    parser.add_argument(
        '--output-dir',
        default='./frequency_analysis',
        help='Output directory for histograms and reports'
    )
    parser.add_argument(
        '--nodes',
        nargs='+',
        help='Specific nodes to analyze (default: all discovered nodes)'
    )
    parser.add_argument(
        '--max-patterns',
        type=int,
        help='Limit analysis to first N patterns per node (for testing)'
    )

    args = parser.parse_args()

    print("="*80)
    print("KATO Pattern Frequency Analysis")
    print("="*80)

    try:
        # Phase 1: Connect to databases
        print("\n[1/5] Connecting to databases...")
        clickhouse_client = get_clickhouse_client()
        redis_client = get_redis_client()

        if not clickhouse_client or not redis_client:
            raise RuntimeError("Failed to connect to ClickHouse or Redis")

        print("  ✓ Connected to ClickHouse and Redis")

        # Phase 2: Discover nodes and patterns
        print("\n[2/5] Discovering nodes and collecting pattern names...")
        node_patterns = discover_nodes_and_patterns(clickhouse_client)

        if args.nodes:
            # Filter to specific nodes
            node_patterns = {k: v for k, v in node_patterns.items() if k in args.nodes}

        if args.max_patterns:
            # Limit patterns for testing
            node_patterns = {k: v[:args.max_patterns] for k, v in node_patterns.items()}

        for kb_id, patterns in node_patterns.items():
            print(f"  ✓ {kb_id}: {len(patterns):,} patterns")

        # Phase 3: Query frequencies
        print("\n[3/5] Querying pattern frequencies from Redis...")
        node_frequencies = {}
        for kb_id, pattern_names in node_patterns.items():
            frequencies = get_frequencies_for_node(kb_id, pattern_names, redis_client)
            node_frequencies[kb_id] = frequencies
            print(f"  ✓ {kb_id}: Retrieved {len(frequencies):,} frequency values")

        # Phase 4: Analyze and generate histograms
        print("\n[4/5] Generating histograms and statistics...")
        all_stats = []
        for kb_id, frequencies in node_frequencies.items():
            stats = analyze_frequency_distribution(frequencies, kb_id)
            all_stats.append(stats)

            generate_histogram(frequencies, kb_id, stats, args.output_dir)

        # Phase 5: Generate summary report
        print("\n[5/5] Generating summary report...")
        generate_summary_report(all_stats, args.output_dir)

        print("\n" + "="*80)
        print("✓ Analysis complete!")
        print(f"Results saved to: {args.output_dir}")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
