# Training Run Comparison Guide

## Overview

The **"Compare Multiple Training Runs"** feature allows you to analyze and compare different training experiments side-by-side. However, this requires proper setup to avoid database conflicts.

## The Problem

**By default, using the same node IDs across training runs causes database conflicts:**

```python
# Training Run 1
nodes = [
    HierarchicalNode('node0_level0', ...),  # → Database: node0_level0_kato
    HierarchicalNode('node1_level1', ...),  # → Database: node1_level1_kato
]
# Train... creates patterns in databases

# Training Run 2 (SAME node IDs)
nodes = [
    HierarchicalNode('node0_level0', ...),  # → SAME database: node0_level0_kato ❌
    HierarchicalNode('node1_level1', ...),  # → SAME database: node1_level1_kato ❌
]
# Train... OVERWRITES or APPENDS to same databases!
```

**Result:** Both training runs share the same databases → Cannot compare them!

## The Solution: Unique Node IDs Per Run

### Option 1: Use `create_training_run_nodes()` (Recommended)

```python
from tools import create_training_run_nodes

# Training Run 1: WikiText 100 samples
nodes_100 = create_training_run_nodes(run_id='wikitext_100samples')
# Creates databases:
#   node0_wikitext_100samples_kato
#   node1_wikitext_100samples_kato
#   node2_wikitext_100samples_kato
#   node3_wikitext_100samples_kato

learner_100 = HierarchicalConceptLearner(nodes=nodes_100, tokenizer_name='gpt2')
# ... train ...

# Training Run 2: WikiText 500 samples
nodes_500 = create_training_run_nodes(run_id='wikitext_500samples')
# Creates SEPARATE databases:
#   node0_wikitext_500samples_kato
#   node1_wikitext_500samples_kato
#   node2_wikitext_500samples_kato
#   node3_wikitext_500samples_kato

learner_500 = HierarchicalConceptLearner(nodes=nodes_500, tokenizer_name='gpt2')
# ... train ...

# Now you can compare both runs!
```

### Option 2: Manual Unique IDs

```python
# Run 1: Baseline
nodes_baseline = [
    HierarchicalNode('node0', node_id='node0_baseline', ...),
    HierarchicalNode('node1', node_id='node1_baseline', ...),
    HierarchicalNode('node2', node_id='node2_baseline', ...),
    HierarchicalNode('node3', node_id='node3_baseline', ...),
]

# Run 2: Experiment
nodes_experiment = [
    HierarchicalNode('node0', node_id='node0_experiment', ...),
    HierarchicalNode('node1', node_id='node1_experiment', ...),
    HierarchicalNode('node2', node_id='node2_experiment', ...),
    HierarchicalNode('node3', node_id='node3_experiment', ...),
]
```

## Comparing Training Runs in Analysis Notebook

Once you have multiple training runs with unique IDs:

```python
from tools import list_all_training_runs, StandaloneMongoDBAnalyzer

# List all available runs
runs = list_all_training_runs()
print(f"Available runs: {list(runs.keys())}")

# Output:
# Available runs: ['wikitext_100samples', 'wikitext_500samples', 'baseline', 'experiment']

# Compare two runs
for run_id in ['wikitext_100samples', 'wikitext_500samples']:
    print(f"\n{run_id}:")
    for db_name in runs[run_id]:
        analyzer = StandaloneMongoDBAnalyzer(db_name)
        stats = analyzer.get_stats()
        print(f"  {db_name}: {stats['total_patterns']:,} patterns")
        analyzer.close()
```

## Managing Training Runs

### List All Runs

```python
from tools import list_all_training_runs

runs = list_all_training_runs()
# Returns: {'run_id': ['db1', 'db2', ...], ...}
```

### Delete Old Runs

```python
from tools import delete_training_run

# Delete with confirmation prompt
delete_training_run('old_experiment')

# Delete without confirmation (use carefully!)
delete_training_run('old_experiment', confirm=False)
```

### Discover All Databases

```python
from tools import discover_training_databases

databases = discover_training_databases()
# Returns: ['node0_level0_kato', 'node0_baseline_kato', ...]
```

## Best Practices

### 1. **Use Descriptive Run IDs**

```python
# Good
create_training_run_nodes(run_id='wikitext_100k_chunk8')
create_training_run_nodes(run_id='wikitext_100k_chunk15')
create_training_run_nodes(run_id='c4_500k_baseline')

# Less helpful
create_training_run_nodes(run_id='test1')
create_training_run_nodes(run_id='run2')
```

### 2. **Document Training Parameters**

Save training configuration in manifest metadata:

```python
manifest = TrainingManifest.create_from_learner(
    learner,
    dataset='wikitext',
    samples_trained=100000
)
manifest.metadata = {
    'chunk_size': 8,
    'tokenizer': 'gpt2',
    'purpose': 'baseline comparison'
}
manifest.save(f'manifests/{manifest.training_id}.json')
```

### 3. **Clean Up Old Runs**

```python
# List all runs
runs = list_all_training_runs()

# Delete experiments you no longer need
for run_id in runs:
    if run_id.startswith('test_'):
        delete_training_run(run_id)
```

### 4. **Auto-Generate IDs for Exploratory Work**

```python
# Auto-generates timestamp-based ID
nodes = create_training_run_nodes()  # → run_20250118_143022
```

## Common Scenarios

### Scenario 1: Compare Different Sample Sizes

```python
# Training with increasing sample counts
for num_samples in [100, 500, 1000, 5000]:
    nodes = create_training_run_nodes(run_id=f'wikitext_{num_samples}samples')
    learner = HierarchicalConceptLearner(nodes=nodes, tokenizer_name='gpt2')
    train_from_streaming_dataset_parallel(
        dataset_key='wikitext',
        max_samples=num_samples,
        learner=learner,
        num_workers=4
    )
```

### Scenario 2: Compare Different Chunk Sizes

```python
# Compare different chunking strategies
for chunk_size in [8, 15, 25]:
    nodes = create_training_run_nodes(
        run_id=f'chunk{chunk_size}',
        chunk_size=chunk_size
    )
    learner = HierarchicalConceptLearner(nodes=nodes, tokenizer_name='gpt2')
    # ... train ...
```

### Scenario 3: A/B Testing Datasets

```python
# Compare WikiText vs C4
for dataset in ['wikitext', 'c4']:
    nodes = create_training_run_nodes(run_id=f'{dataset}_10k')
    learner = HierarchicalConceptLearner(nodes=nodes, tokenizer_name='gpt2')
    train_from_streaming_dataset_parallel(
        dataset_key=dataset,
        max_samples=10000,
        learner=learner,
        num_workers=4
    )
```

## Troubleshooting

### "I ran training twice and the comparison shows the same data!"

**Cause:** You used the same node IDs, so both runs share the same databases.

**Solution:** Use `create_training_run_nodes()` with unique `run_id` for each training run.

### "How do I know which databases belong to which run?"

```python
runs = list_all_training_runs()
for run_id, databases in runs.items():
    print(f"{run_id}: {databases}")
```

### "Can I rename a training run?"

No direct rename, but you can:
1. Create new nodes with desired ID
2. Copy MongoDB databases manually
3. Delete old run

### "What happens if I don't specify run_id?"

`create_training_run_nodes()` auto-generates a timestamp-based ID like `run_20250118_143022`.

## Summary

✅ **DO:** Use unique run IDs for each training experiment
✅ **DO:** Use `create_training_run_nodes()` for easy setup
✅ **DO:** Document your experiments with descriptive run IDs
✅ **DO:** Clean up old experiments to save disk space

❌ **DON'T:** Reuse the same node IDs across training runs
❌ **DON'T:** Expect comparison to work without unique IDs
❌ **DON'T:** Forget to save manifests (they're auto-saved now!)

---

**Related Documentation:**
- `analysis_only_template.ipynb` - Session-independent analysis
- `hierarchical_parallel_training.ipynb` - Training examples
- `PROJECT_OVERVIEW.md` - Project architecture
