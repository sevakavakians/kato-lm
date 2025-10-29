from typing import Dict, Any, Optional, Iterator
from datasets import load_dataset
from tqdm import tqdm
import time
from tools import HardwareAnalyzer

class StreamingDatasetLoader:
    """Loader for streaming LLM training datasets."""

    # Dataset configurations (updated to use script-free datasets)
    DATASETS = {
        'c4': {
            'name': 'allenai/c4',
            'config': 'en',
            'text_field': 'text',
            'description': 'Common Crawl (C4) - Cleaned web text',
            'est_samples': 364868892  # ~365M samples
        },
        'redpajama_sample': {
            'name': 'togethercomputer/RedPajama-Data-1T-Sample',
            'config': None,
            'text_field': 'text',
            'description': 'RedPajama Sample - Open LLaMA reproduction dataset (sample)',
            'est_samples': 1000000  # ~1M samples
        },
        'pile': {
            'name': 'monology/pile-uncopyrighted',
            'config': None,
            'text_field': 'text',
            'description': 'The Pile (Uncopyrighted) - 825 GiB from 22 sources',
            'est_samples': 210000000  # ~210M samples
        },
        'refinedweb': {
            'name': 'tiiuae/falcon-refinedweb',
            'config': None,
            'text_field': 'content',
            'description': 'RefinedWeb - Falcon high-quality web corpus',
            'est_samples': 968000015  # ~968M samples
        },
        'dolma': {
            'name': 'allenai/dolma',
            'config': None,
            'text_field': 'text',
            'description': 'Dolma - AI2 3T token open corpus',
            'est_samples': 3000000000  # ~3B samples
        },
        'wikitext': {
            'name': 'Salesforce/wikitext',
            'config': 'wikitext-103-raw-v1',
            'text_field': 'text',
            'description': 'WikiText-103 - Wikipedia articles (script-free)',
            'est_samples': 1801350  # ~1.8M samples
        },
        'openwebtext': {
            'name': 'Skylion007/openwebtext',
            'config': None,
            'text_field': 'text',
            'description': 'OpenWebText - Reddit-curated web content',
            'est_samples': 8013769  # ~8M samples
        },
        'bookcorpus': {
            'name': 'bookcorpusopen',
            'config': None,
            'text_field': 'text',
            'description': 'BookCorpus - Books dataset',
            'est_samples': 74000000  # ~74M samples
        },
        'simple_wiki': {
            'name': 'legacy-datasets/wikipedia',
            'config': '20220301.simple',
            'text_field': 'text',
            'description': 'Simple Wikipedia - Simplified English articles',
            'est_samples': 205328  # ~205K samples
        }
    }
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"
    
    @staticmethod
    def estimate_time(dataset_key: str, num_samples: int, 
                     hardware_tier: str = None) -> Dict[str, Any]:
        """
        Estimate processing time for a dataset.
        
        Args:
            dataset_key: Dataset to estimate
            num_samples: Number of samples to process
            hardware_tier: Hardware tier ('low', 'medium', 'high', 'server')
        
        Returns:
            Dictionary with time estimates
        """
        if hardware_tier is None:
            hardware_tier = HardwareAnalyzer.classify_hardware()
        
        baseline_rate = HardwareAnalyzer.get_baseline_rate()
        multiplier = HardwareAnalyzer.get_performance_multiplier(hardware_tier)
        rate = baseline_rate * multiplier
        
        time_seconds = num_samples / rate
        
        return {
            'dataset': dataset_key,
            'num_samples': num_samples,
            'hardware_tier': hardware_tier,
            'rate': rate,
            'time_seconds': time_seconds,
            'time_formatted': StreamingDatasetLoader.format_time(time_seconds)
        }
    
    @staticmethod
    def show_time_estimates():
        """Show time estimates for all datasets on current hardware."""
        hw_tier = HardwareAnalyzer.classify_hardware()
        rate = HardwareAnalyzer.get_current_rate()
        
        print(f"\nDataset Processing Time Estimates ({hw_tier.upper()} Hardware)")
        print(f"Estimated Rate: {rate:.1f} samples/second")
        print("="*110)
        print(f"{'Dataset':<20} {'Samples':<15} {'1K':<10} {'10K':<10} {'100K':<10} {'1M':<10} {'Full':<15}")
        print("-"*110)
        
        sample_counts = [1000, 10000, 100000, 1000000]
        
        for key, config in StreamingDatasetLoader.DATASETS.items():
            est_samples = config.get('est_samples', 0)
            
            # Calculate times
            times = []
            for count in sample_counts:
                est = StreamingDatasetLoader.estimate_time(key, count, hw_tier)
                times.append(est['time_formatted'])
            
            # Full dataset time
            full_est = StreamingDatasetLoader.estimate_time(key, est_samples, hw_tier)
            full_time = full_est['time_formatted']
            
            # Warning for large datasets
            warning = " ⚠️" if full_est['time_seconds'] > 86400 else ""  # > 1 day
            
            samples_str = f"{est_samples:,}" if est_samples > 0 else "Unknown"
            
            print(f"{key:<20} {samples_str:<15} {times[0]:<10} {times[1]:<10} "
                  f"{times[2]:<10} {times[3]:<10} {full_time:<15}{warning}")
        
        print("="*110)
        print("\n⚠️  = Full dataset will take >1 day (use max_samples to limit)")
        print(f"\nYour Hardware: {hw_tier.upper()}")
        print("For faster processing: Upgrade to higher-tier hardware or use smaller max_samples")
    
    @staticmethod
    def get_dataset_info(dataset_key: str) -> Dict[str, Any]:
        """Get information about a dataset including size."""
        if dataset_key not in StreamingDatasetLoader.DATASETS:
            return {"error": f"Unknown dataset: {dataset_key}"}
        
        config = StreamingDatasetLoader.DATASETS[dataset_key]
        info = {
            'key': dataset_key,
            'name': config['name'],
            'description': config['description'],
            'config': config['config'],
            'text_field': config['text_field'],
            'est_samples': config.get('est_samples', 0)
        }
        
        return info
    
    @staticmethod
    def load_streaming(dataset_key: str, max_samples: int = None) -> Iterator[str]:
        """
        Load dataset in streaming mode.
        
        Args:
            dataset_key: Key from DATASETS dict
            max_samples: Maximum samples to yield (None = unlimited)
            
        Yields:
            Text samples from the dataset
        """
        if dataset_key not in StreamingDatasetLoader.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        config = StreamingDatasetLoader.DATASETS[dataset_key]
        
        print(f"\n📡 Streaming: {config['description']}")
        print(f"   Dataset: {config['name']}")
        if max_samples:
            est = StreamingDatasetLoader.estimate_time(dataset_key, max_samples)
            print(f"   Samples: {max_samples:,}")
            print(f"   Est. Time: {est['time_formatted']}")
        
        try:
            # Load dataset in streaming mode (NO trust_remote_code parameter)
            if config['config']:
                dataset = load_dataset(
                    config['name'],
                    config['config'],
                    split='train',
                    streaming=True
                )
            else:
                dataset = load_dataset(
                    config['name'],
                    split='train',
                    streaming=True
                )
            
            # Yield samples
            count = 0
            for sample in dataset:
                text = sample.get(config['text_field'], '')

                # Filter: English only, non-empty
                if text and len(text.strip()) > 10:
                    yield sample  # Return full sample dict with all metadata
                    count += 1

                    if max_samples and count >= max_samples:
                        break
            
            print(f"   ✓ Streamed {count} samples")
            
        except Exception as e:
            print(f"   ✗ Error streaming dataset: {e}")
            print(f"   Tip: Try a different dataset from the list above")
    
    @staticmethod
    def list_datasets():
        """List all available datasets."""
        print("\nAvailable Streaming Datasets:")
        print("="*80)
        for key, config in StreamingDatasetLoader.DATASETS.items():
            est_samples = config.get('est_samples', 0)
            samples_str = f"~{est_samples:,} samples" if est_samples > 0 else "Unknown size"
            print(f"\n{key.upper()}:")
            print(f"  {config['description']}")
            print(f"  Source: {config['name']}")
            print(f"  Size: {samples_str}")

    @staticmethod
    def build_corpus_from_stream(
        dataset_key: str,
        max_samples: int,
        segmenter: Optional['CorpusSegmenter'] = None,
        segment_method: str = 'simple',
        tokenizer_name: str = 'gpt2',
        verbose: bool = True
    ) -> dict:
        """
        Build a corpus structure from a streaming dataset with synthetic metadata.

        ⚠️  WARNING: This function loads ALL samples into memory. Use only for small datasets!
        For large-scale training (>10K samples), use train_from_streaming_dataset() instead.

        Args:
            dataset_key: Dataset to load ('c4', 'wikitext', 'refinedweb', etc.)
            max_samples: Number of samples to load from stream
            segmenter: Optional CorpusSegmenter instance (creates new one if None)
            segment_method: 'simple' (default), 'article', or 'book' - determines segmentation approach
            tokenizer_name: Tokenizer to use for sentence segmentation (default: 'gpt2')
            verbose: Show progress bar and info

        Returns:
            Dictionary with corpus structure: {'books': [book1, book2, ...]}
            Each book includes synthetic metadata:
                - title: "{dataset_name}_sample_{idx}"
                - author: dataset_key
                - dataset_key: Original dataset key
                - sample_idx: Sample number in stream
                - url: If present in dataset
                - timestamp: If present in dataset
        """
        # Import here to avoid circular dependency
        from tools.hierarchical_learning import CorpusSegmenter

        # Memory safety check
        SAFE_LIMIT = 10000
        if max_samples > SAFE_LIMIT:
            est_memory_mb = max_samples * 2 / 1000  # Rough estimate: 2 KB per sample
            print(f"\n{'='*80}")
            print(f"⚠️  MEMORY WARNING")
            print(f"{'='*80}")
            print(f"You requested {max_samples:,} samples, which will load ALL data into memory.")
            print(f"Estimated memory usage: ~{est_memory_mb:.0f} MB (minimum)")
            print(f"Recommended limit: {SAFE_LIMIT:,} samples for this function")
            print(f"\nFor large-scale training, use train_from_streaming_dataset() instead:")
            print(f"  from tools import train_from_streaming_dataset")
            print(f"  stats = train_from_streaming_dataset(")
            print(f"      dataset_key='{dataset_key}',")
            print(f"      max_samples={max_samples},")
            print(f"      learner=learner,")
            print(f"      num_levels=4")
            print(f"  )")
            print(f"\nThis will use constant memory (~1-10 MB per sample) and avoid OOM crashes.")
            print(f"{'='*80}\n")

            response = input("Continue anyway? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Aborted. Please use train_from_streaming_dataset() for large datasets.")
                return {'books': []}

        if segmenter is None:
            segmenter = CorpusSegmenter(tokenizer_name=tokenizer_name)

        # Validate segment method
        valid_methods = ['simple', 'article', 'book']
        if segment_method not in valid_methods:
            raise ValueError(f"segment_method must be one of {valid_methods}, got: {segment_method}")

        # Get dataset info
        dataset_info = StreamingDatasetLoader.get_dataset_info(dataset_key)
        if 'error' in dataset_info:
            raise ValueError(dataset_info['error'])

        dataset_config = StreamingDatasetLoader.DATASETS[dataset_key]
        text_field = dataset_config['text_field']
        dataset_name = dataset_info['description']

        if verbose:
            print(f"\n{'='*80}")
            print(f"BUILDING CORPUS FROM STREAMING DATASET")
            print(f"{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"Source: {dataset_info['name']}")
            print(f"Samples: {max_samples:,}")
            print(f"Segmentation: {segment_method}")

            # Estimate time
            est = StreamingDatasetLoader.estimate_time(dataset_key, max_samples)
            print(f"Est. Time: {est['time_formatted']}")
            print(f"{'='*80}\n")

        books = []
        errors = 0

        # Stream samples with progress bar
        stream_iterator = StreamingDatasetLoader.load_streaming(dataset_key, max_samples)

        if verbose:
            stream_iterator = tqdm(stream_iterator, total=max_samples, desc="Processing samples")

        for idx, sample in enumerate(stream_iterator):
            try:
                # Extract text from sample
                text = sample.get(text_field, '')

                if not text or len(text.strip()) < 10:
                    continue

                # Create synthetic metadata
                book_metadata = {
                    'title': f'{dataset_key}_sample_{idx}',
                    'author': dataset_key,
                    'dataset_key': dataset_key,
                    'sample_idx': idx,
                }

                # Add optional metadata fields if present
                if 'url' in sample:
                    book_metadata['url'] = sample['url']
                if 'timestamp' in sample:
                    book_metadata['timestamp'] = sample['timestamp']

                # Segment text based on method
                if segment_method == 'simple':
                    book = segmenter.segment_simple_text(text, metadata=book_metadata)
                elif segment_method == 'article':
                    book = segmenter.segment_article(text, article_metadata=book_metadata)
                elif segment_method == 'book':
                    book = segmenter.segment_book(text, book_metadata=book_metadata)

                # Only add if segmentation produced valid structure
                if book and 'chapters' in book and len(book['chapters']) > 0:
                    books.append(book)
                else:
                    errors += 1

            except Exception as e:
                errors += 1
                if verbose and errors <= 5:  # Show first 5 errors
                    print(f"\n⚠️  Error processing sample {idx}: {str(e)[:100]}")
                continue

        if verbose:
            print(f"\n{'='*80}")
            print(f"CORPUS BUILD COMPLETE")
            print(f"{'='*80}")
            print(f"✓ Successfully processed: {len(books):,} samples")
            if errors > 0:
                print(f"⚠️  Errors/skipped: {errors:,} samples")
            print(f"{'='*80}\n")

        return {'books': books}

    @staticmethod
    def train_from_streaming_dataset(
        dataset_key: str,
        max_samples: int,
        learner: 'HierarchicalConceptLearner',
        num_levels: int = 4,
        segment_method: str = 'simple',
        tokenizer_name: str = 'gpt2',
        chunk_size: int = 15,
        checkpoint_interval: int = 10000,
        checkpoint_dir: str = './checkpoints',
        resume_from_checkpoint: bool = False,
        verbose: bool = True
    ) -> dict:
        """
        Train hierarchical learner from streaming dataset with constant memory usage.

        This function streams samples one at a time, segments them, trains the learner,
        and immediately discards the sample. Memory usage is constant (~1-10 MB per sample)
        regardless of dataset size.

        Args:
            dataset_key: Dataset to load ('c4', 'wikitext', 'refinedweb', etc.')
            max_samples: Number of samples to process from stream
            learner: HierarchicalConceptLearner instance to train
            num_levels: Number of hierarchical levels (default: 4)
            segment_method: 'simple' (default), 'article', or 'book'
            tokenizer_name: Tokenizer to use (default: 'gpt2')
            chunk_size: Number of tokens per chunk (default: 15, recommended: 10-25)
            checkpoint_interval: Save checkpoint every N samples (default: 10000)
            checkpoint_dir: Directory to save checkpoints (default: './checkpoints')
            resume_from_checkpoint: Resume from last checkpoint if available (default: False)
            verbose: Show progress bar and info

        Returns:
            Dictionary with training statistics:
                - samples_processed: Number of samples trained on
                - node0_patterns: Patterns learned at level 0
                - node1_patterns: Patterns learned at level 1
                - ... (for each level)
                - total_time_seconds: Total training time
                - checkpoints_saved: Number of checkpoints saved
        """
        # Import here to avoid circular dependency
        from tools.hierarchical_learning import CorpusSegmenter, train_hierarchical_single_pass
        import os
        import json
        import pickle
        from pathlib import Path

        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(checkpoint_dir) / f"{dataset_key}_streaming_checkpoint.json"

        # Initialize tracking
        start_idx = 0
        samples_processed = 0
        checkpoints_saved = 0
        start_time = time.time()

        # Resume from checkpoint if requested
        if resume_from_checkpoint and checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                start_idx = checkpoint_data.get('samples_processed', 0)
                samples_processed = start_idx
                print(f"\n📂 Resuming from checkpoint at sample {start_idx:,}")
            except Exception as e:
                print(f"\n⚠️  Failed to load checkpoint: {e}")
                print("Starting from beginning...")

        # Create segmenter with chunk_size
        segmenter = CorpusSegmenter(tokenizer_name=tokenizer_name, chunk_size=chunk_size)

        # Get dataset info
        dataset_info = StreamingDatasetLoader.get_dataset_info(dataset_key)
        if 'error' in dataset_info:
            raise ValueError(dataset_info['error'])

        dataset_config = StreamingDatasetLoader.DATASETS[dataset_key]
        text_field = dataset_config['text_field']
        dataset_name = dataset_info['description']

        if verbose:
            print(f"\n{'='*80}")
            print(f"STREAMING HIERARCHICAL TRAINING")
            print(f"{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"Source: {dataset_info['name']}")
            print(f"Samples: {max_samples:,}")
            print(f"Segmentation: {segment_method}")
            print(f"Checkpoint Interval: {checkpoint_interval:,} samples")
            if resume_from_checkpoint and start_idx > 0:
                print(f"Resuming from: sample {start_idx:,}")

            # Estimate time
            est = StreamingDatasetLoader.estimate_time(dataset_key, max_samples)
            print(f"Est. Time: {est['time_formatted']}")
            print(f"{'='*80}\n")

        # Stream and train
        stream_iterator = StreamingDatasetLoader.load_streaming(dataset_key, max_samples + start_idx)

        # Create manual progress bar for accurate tracking
        pbar = None
        if verbose:
            pbar = tqdm(total=max_samples, desc="Training samples", unit="sample")
            if start_idx > 0:
                pbar.update(samples_processed)  # Resume from checkpoint

        errors = 0
        sample_idx = 0

        for sample in stream_iterator:
            # Skip samples until we reach start_idx (for checkpoint resume)
            if sample_idx < start_idx:
                sample_idx += 1
                continue

            try:
                # Extract text from sample
                text = sample.get(text_field, '')

                if not text or len(text.strip()) < 10:
                    sample_idx += 1
                    continue

                # Create synthetic metadata
                book_metadata = {
                    'title': f'{dataset_key}_sample_{sample_idx}',
                    'author': dataset_key,
                    'dataset_key': dataset_key,
                    'sample_idx': sample_idx,
                }

                # Add optional metadata fields if present
                if 'url' in sample:
                    book_metadata['url'] = sample['url']
                if 'timestamp' in sample:
                    book_metadata['timestamp'] = sample['timestamp']

                # Segment text
                if segment_method == 'simple':
                    book = segmenter.segment_simple_text(text, metadata=book_metadata)
                elif segment_method == 'article':
                    book = segmenter.segment_article(text, article_metadata=book_metadata)
                elif segment_method == 'book':
                    book = segmenter.segment_book(text, book_metadata=book_metadata)

                # Only train if segmentation produced valid structure
                if book and 'chapters' in book and len(book['chapters']) > 0:
                    # Create mini corpus with single book
                    mini_corpus = {'books': [book]}

                    # Train on this sample (immediately, then discard)
                    train_hierarchical_single_pass(
                        corpus=mini_corpus,
                        learner=learner,
                        delimiter='sentence',
                        num_levels=num_levels,
                        verbose=False  # Don't spam output for each sample
                    )

                    samples_processed += 1

                    # Update progress bar
                    if pbar:
                        pbar.update(1)
                        # Update description with rate every 100 samples
                        if samples_processed % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = samples_processed / elapsed if elapsed > 0 else 0
                            pbar.set_description(f"Training ({rate:.1f} samples/sec, {errors} errors)")

                    # Save checkpoint periodically
                    if samples_processed % checkpoint_interval == 0:
                        checkpoint_data = {
                            'dataset_key': dataset_key,
                            'samples_processed': samples_processed,
                            'timestamp': time.time(),
                            'elapsed_seconds': time.time() - start_time
                        }

                        with open(checkpoint_path, 'w') as f:
                            json.dump(checkpoint_data, f, indent=2)

                        checkpoints_saved += 1

                        if verbose:
                            elapsed = time.time() - start_time
                            rate = samples_processed / elapsed if elapsed > 0 else 0
                            print(f"\n💾 Checkpoint saved: {samples_processed:,} samples ({rate:.1f} samples/sec)")

                else:
                    errors += 1

            except Exception as e:
                errors += 1
                if verbose and errors <= 5:  # Show first 5 errors
                    print(f"\n⚠️  Error processing sample {sample_idx}: {str(e)[:100]}")

            sample_idx += 1

            # Stop if we've processed max_samples
            if samples_processed >= max_samples:
                break

        # Close progress bar
        if pbar:
            pbar.close()

        # Final statistics
        elapsed_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*80}")
            print(f"STREAMING TRAINING COMPLETE")
            print(f"{'='*80}")
            print(f"✓ Samples processed: {samples_processed:,}")
            if errors > 0:
                print(f"⚠️  Errors/skipped: {errors:,}")
            print(f"⏱️  Total time: {StreamingDatasetLoader.format_time(elapsed_time)}")
            print(f"📊 Rate: {samples_processed / elapsed_time:.1f} samples/second")
            print(f"💾 Checkpoints saved: {checkpoints_saved}")
            print(f"{'='*80}\n")

        # Get pattern counts from learner
        stats = {
            'samples_processed': samples_processed,
            'total_time_seconds': elapsed_time,
            'checkpoints_saved': checkpoints_saved,
            'errors': errors
        }

        # Add pattern counts for each node
        for i in range(num_levels):
            node_name = f'node{i}'
            if node_name in learner.nodes:
                # Get pattern count from MongoDB
                from tools.hierarchical_learning import MongoDBAnalyzer
                analyzer = MongoDBAnalyzer(learner.nodes[node_name])
                node_stats = analyzer.get_stats()
                analyzer.close()
                stats[f'{node_name}_patterns'] = node_stats['total_patterns']

        return stats

# Module-level wrapper for easy import
def train_from_streaming_dataset(
    dataset_key: str,
    max_samples: int,
    learner: 'HierarchicalConceptLearner',
    num_levels: int = 4,
    segment_method: str = 'simple',
    tokenizer_name: str = 'gpt2',
    chunk_size: int = 15,
    checkpoint_interval: int = 10000,
    checkpoint_dir: str = './checkpoints',
    resume_from_checkpoint: bool = False,
    verbose: bool = True
) -> dict:
    """
    Module-level wrapper for StreamingDatasetLoader.train_from_streaming_dataset().

    Train hierarchical learner from streaming dataset with constant memory usage.
    See StreamingDatasetLoader.train_from_streaming_dataset() for full documentation.
    """
    return StreamingDatasetLoader.train_from_streaming_dataset(
        dataset_key=dataset_key,
        max_samples=max_samples,
        learner=learner,
        num_levels=num_levels,
        segment_method=segment_method,
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        verbose=verbose
    )

def recommend_dataset_configuration():
    """Recommend dataset and sample count based on available time."""

    rate = HardwareAnalyzer.get_current_rate()
    hw_tier = HardwareAnalyzer.classify_hardware()
    
    print("\n" + "="*80)
    print("DATASET CONFIGURATION RECOMMENDATIONS")
    print("="*80)
    
    print(f"\nYour Hardware: {hw_tier.upper()}")
    print(f"Processing Rate: ~{rate:.1f} samples/second")
    
    print("\n" + "-"*80)
    print("Recommended Configurations by Available Time:")
    print("-"*80)
    
    recommendations = [
        {
            'time': '< 5 minutes',
            'use_case': 'Quick Test',
            'dataset': 'wikitext',
            'samples': 1000,
            'description': 'Test the pipeline and verify everything works'
        },
        {
            'time': '5-30 minutes',
            'use_case': 'Development',
            'dataset': 'wikitext',
            'samples': 10000,
            'description': 'Develop and debug training code'
        },
        {
            'time': '30 min - 2 hours',
            'use_case': 'Small Training',
            'dataset': 'wikitext',
            'samples': 100000,
            'description': 'Initial model training and evaluation'
        },
        {
            'time': '2-6 hours',
            'use_case': 'Medium Training',
            'dataset': 'openwebtext',
            'samples': 500000,
            'description': 'Solid training run with diverse data'
        },
        {
            'time': '6-24 hours',
            'use_case': 'Large Training',
            'dataset': 'openwebtext',
            'samples': 2000000,
            'description': 'Substantial training for production models'
        },
        {
            'time': '1-7 days',
            'use_case': 'Production Training',
            'dataset': 'c4',
            'samples': 10000000,
            'description': 'Large-scale training on web data'
        },
        {
            'time': '1+ weeks',
            'use_case': 'Full Dataset Training',
            'dataset': 'refinedweb',
            'samples': 100000000,
            'description': 'Maximum scale training (requires checkpointing!)'
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        est = StreamingDatasetLoader.estimate_time(rec['dataset'], rec['samples'], hw_tier)
        
        print(f"\n{i}. {rec['use_case']} ({rec['time']})")
        print(f"   Dataset: {rec['dataset']}")
        print(f"   Samples: {rec['samples']:,}")
        print(f"   Est. Time: {est['time_formatted']}")
        print(f"   Use Case: {rec['description']}")
    
    print("="*80)

