#!/usr/bin/env python3
"""
Storage Comparison Analysis for KATO Hierarchical Learning

Compares memory storage requirements between:
1. Raw Wikitext data (actual samples from checkpoint)
2. KATO hierarchical storage (dynamically discovered nodes)

Features:
- Dynamically discovers number of nodes (not hardcoded)
- Reads actual sample count from checkpoint file
- Queries ClickHouse for per-node storage statistics
- Calculates raw data storage (UTF-8, Int16, Int32 representations)
- Generates tabular comparison output
- Calculates compression ratios and deduplication metrics

**Usage**

# Quick test with estimated raw data (fast)

source .venv/bin/activate && python storage_comparison.py --skip-raw

# Full analysis with actual raw data calculation (slower, ~several minutes)

source .venv/bin/activate && python storage_comparison.py

# Custom checkpoint

source .venv/bin/activate && python storage_comparison.py --checkpoint ./path/to/checkpoint.json
"""

import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2Tokenizer

# Project imports
from tools.streaming_dataset_loader import StreamingDatasetLoader
from tools.kato_storage.connection_manager import get_clickhouse_client


def load_checkpoint_data(checkpoint_path='./checkpoints/wikitext_v2_checkpoint.json'):
    """Load training checkpoint to get actual sample count."""
    try:
        with open(checkpoint_path) as f:
            data = json.load(f)

        return {
            'samples_completed': data['samples_completed'],
            'chunk_sizes': data['model_config']['chunk_sizes'],
            'tokenizer': data['model_config']['tokenizer'],
            'num_layers': data['model_config']['num_layers'],
            'dataset_key': data['dataset_key']
        }
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            f"Please ensure training has been run and checkpoint exists."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def discover_kato_nodes(clickhouse_client):
    """
    Dynamically discover KATO nodes by querying ClickHouse.

    Returns list of kb_ids that match node pattern (node0_kato, node1_kato, etc.).
    """
    try:
        # Query all distinct kb_ids
        result = clickhouse_client.query(
            "SELECT DISTINCT kb_id FROM kato.patterns_data ORDER BY kb_id"
        )

        kb_ids = [row[0] for row in result.result_rows]

        # Filter for node pattern: node<digit>_kato or just node<digit>
        # Accept either format for flexibility
        import re
        node_pattern = re.compile(r'^node\d+(_kato)?$')
        node_ids = [kb for kb in kb_ids if node_pattern.match(kb)]

        # Extract node numbers and sort
        def get_node_number(kb_id):
            # Extract the number from 'node0_kato' or 'node0'
            match = re.search(r'node(\d+)', kb_id)
            return int(match.group(1)) if match else 0

        nodes = sorted(node_ids, key=get_node_number)

        return nodes
    except Exception as e:
        raise RuntimeError(f"Failed to discover KATO nodes: {e}")


def calculate_raw_storage(dataset_key, num_samples, tokenizer_name):
    """
    Calculate raw storage for first N Wikitext samples.

    Returns dict with token counts, byte sizes, char counts.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

    stream = StreamingDatasetLoader.load_streaming(
        dataset_key=dataset_key,
        max_samples=num_samples,
        skip=0
    )

    total_tokens = 0
    total_chars = 0
    total_bytes_utf8 = 0
    sample_count = 0

    for sample in tqdm(stream, total=num_samples, desc="Analyzing raw data"):
        text = sample.get('text', '')
        tokens = tokenizer.encode(text)

        total_tokens += len(tokens)
        total_chars += len(text)
        total_bytes_utf8 += len(text.encode('utf-8'))
        sample_count += 1

    return {
        'samples': sample_count,
        'tokens': total_tokens,
        'chars': total_chars,
        'bytes_utf8': total_bytes_utf8,
        'bytes_int16': total_tokens * 2,
        'bytes_int32': total_tokens * 4,
        'avg_tokens_per_sample': total_tokens / sample_count if sample_count > 0 else 0
    }


def estimate_raw_storage(num_samples):
    """
    Estimate raw storage based on typical Wikitext statistics.

    Used when --skip-raw is enabled. Based on empirical averages.
    """
    # Typical Wikitext stats (from analysis of 10k samples)
    AVG_TOKENS_PER_SAMPLE = 500
    AVG_CHARS_PER_TOKEN = 4.0

    total_tokens = num_samples * AVG_TOKENS_PER_SAMPLE
    total_chars = int(total_tokens * AVG_CHARS_PER_TOKEN)
    total_bytes_utf8 = int(total_chars * 1.1)  # UTF-8 overhead

    return {
        'samples': num_samples,
        'tokens': total_tokens,
        'chars': total_chars,
        'bytes_utf8': total_bytes_utf8,
        'bytes_int16': total_tokens * 2,
        'bytes_int32': total_tokens * 4,
        'avg_tokens_per_sample': AVG_TOKENS_PER_SAMPLE,
        'estimated': True
    }


def query_node_storage(clickhouse_client, kb_id):
    """
    Query ClickHouse for storage statistics for a single node.

    Returns dict with pattern count, data bytes, metadata bytes.
    """
    try:
        result = clickhouse_client.query(f"""
            SELECT
                COUNT(*) as pattern_count,
                SUM(length(toString(pattern_data))) as pattern_data_bytes,
                SUM(token_count) as total_tokens_in_patterns,
                AVG(length) as avg_pattern_length,
                SUM(length(toString(minhash_sig))) as minhash_bytes,
                SUM(length(toString(lsh_bands))) as lsh_bytes,
                SUM(length(toString(token_set))) as token_set_bytes
            FROM kato.patterns_data
            WHERE kb_id = '{kb_id}'
        """)

        if not result.result_rows:
            return None

        row = result.result_rows[0]

        # Calculate total ClickHouse storage (data + indices + overhead)
        pattern_data_bytes = row[1] or 0
        minhash_bytes = row[4] or 0
        lsh_bytes = row[5] or 0
        token_set_bytes = row[6] or 0

        # Add ClickHouse column overhead (~100 bytes per pattern for metadata columns)
        overhead_bytes = row[0] * 100

        total_clickhouse = (pattern_data_bytes + minhash_bytes + lsh_bytes +
                           token_set_bytes + overhead_bytes)

        return {
            'kb_id': kb_id,
            'pattern_count': row[0],
            'pattern_data_bytes': pattern_data_bytes,
            'total_tokens': row[2] or 0,
            'avg_pattern_length': row[3] or 0,
            'minhash_bytes': minhash_bytes,
            'lsh_bytes': lsh_bytes,
            'token_set_bytes': token_set_bytes,
            'overhead_bytes': overhead_bytes,
            'total_clickhouse_bytes': total_clickhouse
        }
    except Exception as e:
        raise RuntimeError(f"Failed to query node storage for {kb_id}: {e}")


def estimate_redis_storage(pattern_count):
    """
    Estimate Redis storage for pattern metadata.

    Redis stores:
    - Frequency counter: 8 bytes
    - Emotives: ~50 bytes JSON
    - Metadata: ~40 bytes JSON
    - Key overhead: ~40 bytes

    Total: ~138 bytes per pattern
    """
    REDIS_BYTES_PER_PATTERN = 138
    return pattern_count * REDIS_BYTES_PER_PATTERN


def calculate_kato_storage(clickhouse_client, nodes):
    """
    Calculate total KATO storage across all nodes.

    Returns dict with per-node stats and total.
    """
    node_stats = {}
    total_stats = {
        'patterns': 0,
        'clickhouse_bytes': 0,
        'redis_bytes': 0,
        'total_bytes': 0
    }

    for kb_id in nodes:
        stats = query_node_storage(clickhouse_client, kb_id)

        if stats:
            redis_bytes = estimate_redis_storage(stats['pattern_count'])
            stats['redis_bytes'] = redis_bytes
            stats['total_bytes'] = stats['total_clickhouse_bytes'] + redis_bytes

            node_stats[kb_id] = stats

            # Aggregate
            total_stats['patterns'] += stats['pattern_count']
            total_stats['clickhouse_bytes'] += stats['total_clickhouse_bytes']
            total_stats['redis_bytes'] += redis_bytes
            total_stats['total_bytes'] += stats['total_bytes']

    return node_stats, total_stats


def calculate_comparison_metrics(raw_stats, kato_stats):
    """
    Calculate compression ratios, space saved, deduplication rates.
    """
    raw_bytes = raw_stats['bytes_utf8']
    kato_bytes = kato_stats['total_bytes']

    compression_ratio = kato_bytes / raw_bytes if raw_bytes > 0 else 0
    space_saved_pct = (1 - compression_ratio) * 100

    return {
        'compression_ratio': compression_ratio,
        'space_saved_pct': space_saved_pct,
        'raw_mb': raw_bytes / 1024 / 1024,
        'kato_mb': kato_bytes / 1024 / 1024,
        'saved_mb': (raw_bytes - kato_bytes) / 1024 / 1024
    }


def calculate_deduplication_rates(raw_stats, node_stats, chunk_sizes):
    """
    Calculate deduplication rate per node.

    Deduplication rate = 1 - (unique_patterns / total_observations)
    """
    dedup_rates = {}

    # Convert node_stats keys to a list to access by index
    node_list = list(node_stats.items())

    for i, (kb_id, stats) in enumerate(node_list):
        # Estimate observations at this level
        if i == 0:  # node0
            observations = raw_stats['tokens'] / chunk_sizes[0]
        else:
            # Higher levels: observations = previous_level_patterns / chunk_size
            prev_kb_id = node_list[i-1][0]  # Get the actual kb_id from previous node
            observations = node_stats[prev_kb_id]['pattern_count'] / chunk_sizes[i]

        unique_patterns = stats['pattern_count']

        if observations > 0:
            dedup_rate = 1 - (unique_patterns / observations)
            reuse_factor = observations / unique_patterns
        else:
            dedup_rate = 0
            reuse_factor = 0

        dedup_rates[kb_id] = {
            'observations': observations,
            'unique_patterns': unique_patterns,
            'dedup_rate': dedup_rate,
            'reuse_factor': reuse_factor
        }

    return dedup_rates


def print_comparison_table(raw_stats, node_stats, total_kato, metrics):
    """
    Print main storage comparison table.
    """
    print("\n" + "="*80)
    print("STORAGE COMPARISON: Raw Wikitext vs KATO Hierarchical")
    print("="*80)

    # Calculate values
    raw_utf8_mb = raw_stats['bytes_utf8'] / 1024 / 1024
    raw_int16_mb = raw_stats['bytes_int16'] / 1024 / 1024
    raw_int32_mb = raw_stats['bytes_int32'] / 1024 / 1024

    # Try to use tabulate if available
    try:
        from tabulate import tabulate

        headers = ["Storage Type", "Size (MB)", "Size (Bytes)", "% of Raw", "Space Saved"]

        rows = [
            ["Raw Text (UTF-8)", f"{raw_utf8_mb:.2f}", f"{raw_stats['bytes_utf8']:,}", "100.0%", "0.0%"],
            ["Raw Tokens (Int16)", f"{raw_int16_mb:.2f}", f"{raw_stats['bytes_int16']:,}",
             f"{raw_int16_mb/raw_utf8_mb*100:.1f}%", f"{(1-raw_int16_mb/raw_utf8_mb)*100:.1f}%"],
            ["Raw Tokens (Int32)", f"{raw_int32_mb:.2f}", f"{raw_stats['bytes_int32']:,}",
             f"{raw_int32_mb/raw_utf8_mb*100:.1f}%", f"{(1-raw_int32_mb/raw_utf8_mb)*100:.1f}%"],
            ["", "", "", "", ""],  # Separator
        ]

        # Add per-node rows
        for kb_id, stats in node_stats.items():
            node_mb = stats['total_bytes'] / 1024 / 1024
            pct_raw = stats['total_bytes'] / raw_stats['bytes_utf8'] * 100
            space_saved = (1 - stats['total_bytes'] / raw_stats['bytes_utf8']) * 100

            rows.append([
                f"KATO {kb_id}",
                f"{node_mb:.2f}",
                f"{stats['total_bytes']:,}",
                f"{pct_raw:.1f}%",
                f"{space_saved:.1f}%"
            ])

        # Add total KATO row
        kato_mb = total_kato['total_bytes'] / 1024 / 1024
        rows.append([
            "KATO Total (All Nodes)",
            f"{kato_mb:.2f}",
            f"{total_kato['total_bytes']:,}",
            f"{metrics['compression_ratio']*100:.1f}%",
            f"{metrics['space_saved_pct']:.1f}%"
        ])

        print(tabulate(rows, headers=headers, tablefmt="grid"))

    except ImportError:
        # Manual formatting fallback
        print(f"{'Storage Type':<25} {'Size (MB)':>12} {'Size (Bytes)':>15} {'% of Raw':>10} {'Space Saved':>12}")
        print("-" * 80)

        # Raw data rows
        print(f"{'Raw Text (UTF-8)':<25} {raw_utf8_mb:>12.2f} {raw_stats['bytes_utf8']:>15,} {'100.0%':>10} {'0.0%':>12}")
        print(f"{'Raw Tokens (Int16)':<25} {raw_int16_mb:>12.2f} {raw_stats['bytes_int16']:>15,} "
              f"{raw_int16_mb/raw_utf8_mb*100:>9.1f}% {(1-raw_int16_mb/raw_utf8_mb)*100:>11.1f}%")
        print(f"{'Raw Tokens (Int32)':<25} {raw_int32_mb:>12.2f} {raw_stats['bytes_int32']:>15,} "
              f"{raw_int32_mb/raw_utf8_mb*100:>9.1f}% {(1-raw_int32_mb/raw_utf8_mb)*100:>11.1f}%")
        print()

        # KATO rows
        for kb_id, stats in node_stats.items():
            node_mb = stats['total_bytes'] / 1024 / 1024
            pct_raw = stats['total_bytes'] / raw_stats['bytes_utf8'] * 100
            space_saved = (1 - stats['total_bytes'] / raw_stats['bytes_utf8']) * 100

            print(f"{f'KATO {kb_id}':<25} {node_mb:>12.2f} {stats['total_bytes']:>15,} "
                  f"{pct_raw:>9.1f}% {space_saved:>11.1f}%")

        # Total
        kato_mb = total_kato['total_bytes'] / 1024 / 1024
        print(f"{'KATO Total (All Nodes)':<25} {kato_mb:>12.2f} {total_kato['total_bytes']:>15,} "
              f"{metrics['compression_ratio']*100:>9.1f}% {metrics['space_saved_pct']:>11.1f}%")


def print_node_details_table(node_stats, dedup_rates):
    """
    Print detailed per-node statistics.
    """
    print("\n" + "="*80)
    print("PER-NODE STORAGE DETAILS")
    print("="*80)

    try:
        from tabulate import tabulate

        headers = ["Node", "Patterns", "ClickHouse", "Redis", "Total (MB)",
                   "Dedup Rate", "Reuse Factor"]

        rows = []
        for kb_id, stats in node_stats.items():
            dedup = dedup_rates[kb_id]

            rows.append([
                kb_id,
                f"{stats['pattern_count']:,}",
                f"{stats['total_clickhouse_bytes']/1024/1024:.2f} MB",
                f"{stats['redis_bytes']/1024/1024:.2f} MB",
                f"{stats['total_bytes']/1024/1024:.2f}",
                f"{dedup['dedup_rate']*100:.1f}%",
                f"{dedup['reuse_factor']:.1f}x"
            ])

        print(tabulate(rows, headers=headers, tablefmt="grid"))

    except ImportError:
        # Manual formatting
        print(f"{'Node':<10} {'Patterns':>12} {'ClickHouse':>15} {'Redis':>12} "
              f"{'Total (MB)':>12} {'Dedup Rate':>12} {'Reuse Factor':>14}")
        print("-" * 90)

        for kb_id, stats in node_stats.items():
            dedup = dedup_rates[kb_id]

            print(f"{kb_id:<10} {stats['pattern_count']:>12,} "
                  f"{stats['total_clickhouse_bytes']/1024/1024:>14.2f} MB "
                  f"{stats['redis_bytes']/1024/1024:>11.2f} MB "
                  f"{stats['total_bytes']/1024/1024:>12.2f} "
                  f"{dedup['dedup_rate']*100:>11.1f}% "
                  f"{dedup['reuse_factor']:>13.1f}x")


def print_summary(checkpoint_data, raw_stats, metrics):
    """
    Print summary statistics.
    """
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Dataset: {checkpoint_data['dataset_key']}")
    print(f"Samples analyzed: {checkpoint_data['samples_completed']:,}")
    print(f"Total tokens: {raw_stats['tokens']:,}")
    print(f"Avg tokens/sample: {raw_stats['avg_tokens_per_sample']:.1f}")
    if raw_stats.get('estimated'):
        print("⚠️  Raw storage values are ESTIMATED (use without --skip-raw for exact values)")
    print()
    print(f"Raw storage (UTF-8): {metrics['raw_mb']:.2f} MB")
    print(f"KATO storage (Total): {metrics['kato_mb']:.2f} MB")
    print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
    print(f"Space saved: {metrics['space_saved_pct']:.1f}% ({metrics['saved_mb']:.2f} MB)")
    print("="*80)


def main():
    """
    Main entry point for storage comparison script.
    """
    parser = argparse.ArgumentParser(
        description='Compare raw Wikitext storage vs KATO hierarchical storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-discover nodes, read checkpoint)
  python storage_comparison.py

  # Specify checkpoint path
  python storage_comparison.py --checkpoint ./checkpoints/wikitext_v2_checkpoint.json

  # Skip raw data calculation (use estimates - faster)
  python storage_comparison.py --skip-raw

  # Use custom KATO URL
  python storage_comparison.py --kato-url http://192.168.1.100:8000
        """
    )
    parser.add_argument(
        '--checkpoint',
        default='./checkpoints/wikitext_v2_checkpoint.json',
        help='Path to training checkpoint file'
    )
    parser.add_argument(
        '--skip-raw',
        action='store_true',
        help='Skip raw data calculation (use cached/estimated values)'
    )

    args = parser.parse_args()

    print("="*80)
    print("KATO Storage Comparison Analysis")
    print("="*80)

    try:
        # Phase 1: Load checkpoint
        print("\n[1/6] Loading checkpoint data...")
        checkpoint_data = load_checkpoint_data(args.checkpoint)
        print(f"  ✓ Samples completed: {checkpoint_data['samples_completed']:,}")
        print(f"  ✓ Layers: {checkpoint_data['num_layers']}")

        # Phase 2: Discover nodes
        print("\n[2/6] Discovering KATO nodes...")
        clickhouse_client = get_clickhouse_client()
        if not clickhouse_client:
            raise RuntimeError("ClickHouse client unavailable. Is KATO server running?")

        nodes = discover_kato_nodes(clickhouse_client)
        print(f"  ✓ Found {len(nodes)} nodes: {nodes}")

        # Validate against checkpoint
        if len(nodes) != checkpoint_data['num_layers']:
            print(f"  ⚠️  Warning: Found {len(nodes)} nodes, "
                  f"expected {checkpoint_data['num_layers']} from checkpoint")

        # Phase 3: Calculate raw storage
        if not args.skip_raw:
            print("\n[3/6] Calculating raw data storage...")
            print("  ⏳ This may take several minutes for 65k samples...")
            raw_stats = calculate_raw_storage(
                checkpoint_data['dataset_key'],
                checkpoint_data['samples_completed'],
                checkpoint_data['tokenizer']
            )
            print(f"  ✓ Processed {raw_stats['samples']:,} samples")
            print(f"  ✓ Total tokens: {raw_stats['tokens']:,}")
        else:
            print("\n[3/6] Skipping raw data calculation (using estimates)...")
            raw_stats = estimate_raw_storage(checkpoint_data['samples_completed'])
            print(f"  ✓ Estimated {raw_stats['samples']:,} samples")
            print(f"  ✓ Estimated tokens: {raw_stats['tokens']:,}")

        # Phase 4: Query KATO storage
        print("\n[4/6] Querying KATO storage...")
        node_stats, total_kato = calculate_kato_storage(clickhouse_client, nodes)
        print(f"  ✓ Total patterns: {total_kato['patterns']:,}")
        print(f"  ✓ Total storage: {total_kato['total_bytes']/1024/1024:.2f} MB")

        # Phase 5: Calculate metrics
        print("\n[5/6] Calculating comparison metrics...")
        metrics = calculate_comparison_metrics(raw_stats, total_kato)
        dedup_rates = calculate_deduplication_rates(
            raw_stats,
            node_stats,
            checkpoint_data['chunk_sizes']
        )
        print(f"  ✓ Compression ratio: {metrics['compression_ratio']:.2f}x")
        print(f"  ✓ Space saved: {metrics['space_saved_pct']:.1f}%")

        # Phase 6: Print results
        print("\n[6/6] Generating output...")
        print_comparison_table(raw_stats, node_stats, total_kato, metrics)
        print_node_details_table(node_stats, dedup_rates)
        print_summary(checkpoint_data, raw_stats, metrics)

        print("\n✓ Analysis complete!")

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
