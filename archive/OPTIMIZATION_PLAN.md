# Performance Optimization Plan: Hierarchical Training

**Date**: 2025-10-17
**Project**: KATO Hierarchical Concept Learning
**Current Performance**: 12-14 seconds/sample (~0.07 samples/sec)
**Target Performance**: 0.5-1 second/sample (~1-2 samples/sec)
**Goal**: 15-28x speedup through 3-phase optimization

---

## Executive Summary

Current bottleneck is **client-side API call overhead**, not KATO server performance (KATO already optimized 3.57x). The training loop makes ~115 API calls per sample:
- **node0**: 100 calls (500 tokens ÷ 5 tokens/chunk)
- **node1+**: ~15 calls (higher-level learning)

This plan targets client-side optimizations to dramatically reduce API call count and enable concurrent processing.

**Phase Impact Summary**:
- **Phase 1**: Batch node0 calls → 4-7x speedup (highest impact)
- **Phase 2**: KATO config tuning → 1.15x speedup (quick win)
- **Phase 3**: Parallel processing → 2-3x speedup (advanced)
- **Combined**: 15-28x total speedup

---

## Current Performance Analysis

### Baseline Metrics (As of 2025-10-17)

**Training Configuration**:
```python
nodes = [
    HierarchicalNode('node0', chunk_size=5),
    HierarchicalNode('node1', chunk_size=7),
    HierarchicalNode('node2', chunk_size=9),
    HierarchicalNode('node3', chunk_size=11),
    HierarchicalNode('node4', chunk_size=13),
    HierarchicalNode('node5', chunk_size=17)
]
```

**Measured Performance**:
- Time per sample: 12-14 seconds
- Throughput: ~0.07 samples/second
- For 1000 samples: ~3.5-4 hours
- For 100K samples: ~14-16 days

**API Call Breakdown (per sample)**:
```
node0: 100 calls (learn individual chunks)
  ↓
node1: ~14 calls (every 7 node0 patterns)
  ↓
node2: ~2 calls (every 9 node1 patterns)
  ↓
node3+: ~1 call (higher levels)
---
Total: ~115-120 API calls/sample
```

**Bottleneck Identification**:
- ❌ **NOT KATO server** (already optimized: 3.57x improvement complete)
- ❌ **NOT MongoDB** (pattern lookups are fast)
- ✅ **YES: Client-side API call overhead**
  - HTTP request/response overhead
  - JSON serialization/deserialization
  - Network latency (even localhost has ~1-2ms)
  - Python function call overhead

---

## Phase 1: Batch Node0 API Calls

**Priority**: CRITICAL (highest impact)
**Estimated Speedup**: 4-7x
**Estimated Time to Implement**: 2-4 hours
**Risk**: Medium (requires careful pattern extraction)

### Current Behavior (node0)

```python
# Current: One API call per chunk (100 calls/sample)
for chunk in chunks:  # 100 iterations
    observations = [{'strings': [token]} for token in chunk]
    result = node0.observe_sequence(observations, learn_at_end=True)
    pattern_name = result['pattern_name']
    pattern_buffers['node1'].append(pattern_name)
```

**Problem**: 100 API calls @ ~100-120ms each = 10-12 seconds per sample

### Proposed Optimization

```python
# Proposed: Accumulate chunks, then batch learn (2-5 API calls/sample)
batch_size = 50  # Learn 50 chunks at once
chunk_batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]

for batch in chunk_batches:  # 2 iterations (100÷50)
    observations = []
    for chunk_idx, chunk in enumerate(batch):
        for token in chunk:
            obs = {'strings': [token]}
            # Mark boundaries for learn_after_each
            if is_chunk_end(token, chunk):
                obs['learn'] = True  # Signal to learn at this boundary
            observations.append(obs)

    # One API call learns all 50 chunks
    result = node0.observe_sequence(
        observations=observations,
        learn_after_each=True  # Learn at each 'learn': True marker
    )

    # Extract all learned patterns from result
    for pattern_name in result['auto_learned_patterns']:
        pattern_buffers['node1'].append(pattern_name)
```

### Implementation Details

**File to Modify**: `/Users/sevakavakians/PROGRAMMING/kato-notebooks/tools/hierarchical_learning.py`

**Function to Update**: `train_hierarchical_single_pass()` (lines ~1650-1663)

**Key Changes**:

1. **Add batch accumulation before node0 learning**:
```python
# Configuration
NODE0_BATCH_SIZE = 50  # Configurable (25-100 range)

# Accumulate chunks
node0_chunk_batch = []
for chunk in all_chunks:
    node0_chunk_batch.append(chunk)

    # Process batch when full
    if len(node0_chunk_batch) >= NODE0_BATCH_SIZE:
        learned_patterns = batch_learn_node0(node0_chunk_batch, learner, book_metadata)
        for pattern in learned_patterns:
            pattern_buffers['node1'].append(pattern)
            # Check higher levels for readiness
            check_higher_levels_ready()
        node0_chunk_batch = []
```

2. **Implement batch_learn_node0() helper**:
```python
def batch_learn_node0(chunks, learner, metadata):
    """
    Learn multiple chunks in a single API call.

    Args:
        chunks: List of token lists [[tok1, tok2, ...], [tok3, tok4, ...], ...]
        learner: HierarchicalConceptLearner instance
        metadata: Book metadata dict

    Returns:
        List of learned pattern names
    """
    node0 = learner.nodes['node0']
    observations = []
    chunk_boundaries = []  # Track where each chunk ends

    # Build observations with chunk boundaries
    for chunk_idx, chunk in enumerate(chunks):
        chunk_start = len(observations)

        for token in chunk:
            observations.append({'strings': [token]})

        chunk_boundaries.append(len(observations) - 1)  # Mark end of chunk

    # Inject metadata into first observation
    if observations:
        observations[0]['metadata'] = metadata

    # Mark learn boundaries (every chunk_size tokens)
    chunk_size = learner.node_configs[0].chunk_size
    for i in range(chunk_size - 1, len(observations), chunk_size):
        observations[i]['learn'] = True

    # Single API call learns all chunks
    result = node0.observe_sequence(
        observations=observations,
        learn_after_each=True
    )

    # Extract learned patterns
    learned_patterns = result.get('auto_learned_patterns', [])

    # Validate: Should have learned len(chunks) patterns
    if len(learned_patterns) != len(chunks):
        print(f"Warning: Expected {len(chunks)} patterns, got {len(learned_patterns)}")

    return learned_patterns
```

3. **Add configuration parameter to HierarchicalConceptLearner**:
```python
class HierarchicalConceptLearner:
    def __init__(self,
                 nodes: List['HierarchicalNode'] = None,
                 node0_batch_size: int = 50,  # NEW parameter
                 # ... existing parameters
                 ):
        self.node0_batch_size = node0_batch_size
        # ... rest of init
```

### Testing Strategy

**Test Cases**:
1. **Correctness**: Pattern names match non-batched version
2. **Count**: Same number of patterns learned
3. **Frequency**: Pattern frequency distributions unchanged
4. **Higher Levels**: node1+ learning unaffected

**Validation Script**:
```python
# Run both versions on same 10 samples
baseline_patterns = train_without_batching(samples[:10])
optimized_patterns = train_with_batching(samples[:10])

# Compare
assert len(baseline_patterns['node0']) == len(optimized_patterns['node0'])
assert baseline_patterns['node0'] == optimized_patterns['node0']  # Order should match
```

**Benchmark**:
```python
import time

start = time.time()
train_hierarchical_single_pass(learner, sample_data)
duration = time.time() - start

print(f"Time per sample: {duration:.2f}s")
print(f"Throughput: {1/duration:.2f} samples/sec")
```

### Rollback Plan

If batching causes issues:
1. Add flag: `use_node0_batching=True` (default False)
2. Keep old code path available
3. Gradual rollout: Test with 10 samples, then 100, then 1000

### Expected Outcome

**Before**:
- 100 API calls @ ~120ms = 12 seconds
- node0 accounts for ~85% of total time

**After**:
- 2-4 API calls @ ~200-300ms = 0.6-1.2 seconds
- node0 accounts for ~20% of total time

**Speedup**: 10x reduction in node0 time → **4-7x overall speedup**

**New Performance**: 12s → 2-3s per sample

---

## Phase 2: KATO Configuration Tuning

**Priority**: HIGH (quick win, low risk)
**Estimated Speedup**: 1.15x (on top of Phase 1)
**Estimated Time to Implement**: 30 minutes
**Risk**: Low (configuration only)

### Current Configuration

**File**: `/Users/sevakavakians/PROGRAMMING/kato/docker-compose.yml`

```yaml
environment:
  - KATO_BATCH_SIZE=1000
  - CONNECTION_POOL_SIZE=10
  - REQUEST_TIMEOUT=30.0
  - STM_CAPACITY=10000
```

### Recommended Changes

```yaml
environment:
  # Increase batch processing capacity
  - KATO_BATCH_SIZE=10000          # Was: 1000

  # Increase connection pool for concurrent requests
  - CONNECTION_POOL_SIZE=50         # Was: 10

  # Allow longer processing for large batches
  - REQUEST_TIMEOUT=120.0           # Was: 30.0

  # Increase STM for larger observation sequences
  - STM_CAPACITY=50000              # Was: 10000

  # Enable advanced optimizations
  - ENABLE_BATCH_OPTIMIZATION=true
  - ENABLE_PARALLEL_PROCESSING=true
```

### Rationale

1. **KATO_BATCH_SIZE=10000**:
   - With Phase 1 batching, we'll send 50-100 observations per call
   - KATO can process larger batches more efficiently
   - Reduces per-item overhead in KATO's internal processing

2. **CONNECTION_POOL_SIZE=50**:
   - Prepares for Phase 3 parallel processing
   - Allows multiple concurrent training sessions
   - Reduces connection establishment overhead

3. **REQUEST_TIMEOUT=120.0**:
   - Large batches may take longer to process
   - Prevents premature timeouts on 50-100 chunk batches
   - Safety margin for complex pattern matching

4. **STM_CAPACITY=50000**:
   - With batching, STM holds more observations before learning
   - Prevents STM overflow on large observation sequences
   - Ensures all observations fit in memory

### Implementation Steps

1. **Stop KATO**:
```bash
cd /Users/sevakavakians/PROGRAMMING/kato
docker-compose down
```

2. **Edit Configuration**:
```bash
# Edit docker-compose.yml with new values
nano docker-compose.yml
```

3. **Restart KATO**:
```bash
docker-compose up -d
```

4. **Verify**:
```bash
# Check KATO logs
docker-compose logs -f kato

# Test connection
curl http://localhost:8000/health
```

### Testing

**Smoke Test**:
```python
# Simple test to ensure KATO still works
from tools import KATOClient

client = KATOClient(base_url="http://localhost:8000", knowledge_base="test_kb")
result = client.observe(['test', 'observation'])
print(f"KATO responding: {result}")
```

**Load Test**:
```python
# Test large batch
observations = [{'strings': [f'token_{i}']} for i in range(1000)]
result = client.observe_sequence(observations, learn_at_end=True)
print(f"Processed {len(observations)} observations: {result}")
```

### Expected Outcome

- 10-15% throughput improvement from reduced per-item overhead
- Better handling of large batches
- Preparation for Phase 3 parallelization

**New Performance**: 2-3s → 1.8-2.5s per sample (after Phase 1)

---

## Phase 3: Parallel Sample Processing

**Priority**: MEDIUM (advanced optimization)
**Estimated Speedup**: 2-3x (on top of Phases 1-2)
**Estimated Time to Implement**: 4-8 hours
**Risk**: Medium-High (concurrency complexity)

### Current Behavior

```python
# Sequential processing
for sample in dataset:
    train_hierarchical_single_pass(learner, sample)
```

**Problem**: CPU idle while waiting for KATO responses

### Proposed Architecture

```python
from concurrent.futures import ThreadPoolExecutor
import uuid

# Process multiple samples concurrently
def train_parallel(learner, dataset, num_workers=4):
    """
    Train on dataset using parallel workers.
    Each worker processes samples in its own KATO session.
    """

    def process_sample_with_session(sample):
        # Create isolated session for this worker
        session_id = f"session_{uuid.uuid4()}"

        # Clone learner with session-aware clients
        session_learner = create_session_learner(learner, session_id)

        # Train on sample
        train_hierarchical_single_pass(session_learner, sample)

        # Cleanup session
        cleanup_session(session_learner, session_id)

    # Parallel execution
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_sample_with_session, dataset)
```

### KATO Session Support

**Key Insight**: KATO supports isolated sessions for concurrent processing

**Session API** (from `/Users/sevakavakians/PROGRAMMING/kato/kato/api/endpoints/sessions.py`):

1. **Create Session**:
```python
POST /sessions
→ {"session_id": "unique-session-id"}
```

2. **Use Session**:
```python
# All requests include session_id header
headers = {"X-Session-ID": session_id}
client.observe(['test'], headers=headers)
```

3. **Cleanup Session**:
```python
DELETE /sessions/{session_id}
```

**Benefits**:
- Each session has isolated STM
- No lock contention between sessions
- Sessions can process concurrently
- Automatic cleanup on session close

### Implementation Details

**File to Modify**: `/Users/sevakavakians/PROGRAMMING/kato-notebooks/tools/streaming_dataset_loader.py`

**Function to Update**: `train_from_streaming_dataset()` (lines ~407-580)

**Key Changes**:

1. **Add session-aware KATOClient**:
```python
# tools/kato_client.py
class KATOClient:
    def __init__(self, base_url, knowledge_base, session_id=None):
        self.session_id = session_id
        self.base_url = base_url
        self.knowledge_base = knowledge_base

    def _request(self, method, endpoint, **kwargs):
        headers = kwargs.get('headers', {})
        if self.session_id:
            headers['X-Session-ID'] = self.session_id
        kwargs['headers'] = headers

        return requests.request(method, endpoint, **kwargs)
```

2. **Implement session learner creation**:
```python
def create_session_learner(base_learner, session_id):
    """
    Create a session-aware copy of learner.
    All KATO clients use the same session_id.
    """
    session_learner = copy.deepcopy(base_learner)

    # Update all node clients to use session
    for node_key, node in session_learner.nodes.items():
        node.session_id = session_id

    return session_learner
```

3. **Implement parallel training**:
```python
def train_from_streaming_dataset(
    learner,
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    streaming: bool = True,
    max_samples: int = None,
    tokenizer_name: str = "gpt2",
    num_workers: int = 1,  # NEW: parallel workers
    use_sessions: bool = True  # NEW: enable session isolation
):
    # ... existing setup ...

    if num_workers > 1 and use_sessions:
        # Parallel processing with sessions
        train_parallel_with_sessions(
            learner, dataset, num_workers, max_samples
        )
    else:
        # Sequential processing (original behavior)
        for sample in dataset:
            train_hierarchical_single_pass(learner, sample)
```

4. **Implement parallel executor**:
```python
def train_parallel_with_sessions(learner, dataset, num_workers, max_samples):
    """
    Process dataset samples in parallel using KATO sessions.
    """
    import uuid
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker_process_sample(sample):
        session_id = f"worker_{uuid.uuid4()}"

        try:
            # Create session
            create_kato_session(learner.node_configs[0].base_url, session_id)

            # Create session-aware learner
            session_learner = create_session_learner(learner, session_id)

            # Train on sample
            train_hierarchical_single_pass(session_learner, sample)

        finally:
            # Always cleanup session
            cleanup_kato_session(learner.node_configs[0].base_url, session_id)

    # Create thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        futures = []
        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            future = executor.submit(worker_process_sample, sample)
            futures.append(future)

        # Wait for completion with progress bar
        from tqdm import tqdm
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()  # Raise any exceptions
            except Exception as e:
                print(f"Worker error: {e}")
```

5. **Implement session management**:
```python
def create_kato_session(base_url, session_id):
    """Create a new KATO session."""
    response = requests.post(f"{base_url}/sessions", json={"session_id": session_id})
    response.raise_for_status()
    return response.json()

def cleanup_kato_session(base_url, session_id):
    """Delete a KATO session."""
    response = requests.delete(f"{base_url}/sessions/{session_id}")
    response.raise_for_status()
```

### Optimal Worker Count

**Factors to Consider**:
- CPU cores available
- KATO connection pool size (Phase 2: 50)
- Memory for concurrent learners
- Diminishing returns from contention

**Recommended**:
```python
import multiprocessing

# Conservative: 2x CPU cores (I/O bound workload)
num_workers = min(multiprocessing.cpu_count() * 2, 8)

# Aggressive: Match connection pool size
num_workers = min(CONNECTION_POOL_SIZE // 6, 8)  # 6 nodes × ~8 workers = ~48 connections
```

**Example**:
- 8-core CPU → 4-8 workers
- Each worker processes 1 sample at a time
- 4x parallelism → 2-3x speedup (accounting for overhead)

### Testing Strategy

**Test Cases**:
1. **Correctness**: Results match sequential version
2. **Concurrency**: No race conditions or deadlocks
3. **Session Isolation**: Workers don't interfere
4. **Error Handling**: Worker failures don't crash entire job

**Validation**:
```python
# Sequential baseline
sequential_results = train_sequential(dataset[:100])

# Parallel test
parallel_results = train_parallel(dataset[:100], num_workers=4)

# Compare knowledge bases
assert_knowledge_bases_equivalent(sequential_results, parallel_results)
```

**Load Test**:
```python
# Start with 2 workers, gradually increase
for num_workers in [1, 2, 4, 8]:
    start = time.time()
    train_parallel(dataset[:100], num_workers=num_workers)
    duration = time.time() - start
    print(f"{num_workers} workers: {duration:.2f}s ({100/duration:.2f} samples/sec)")
```

### Expected Outcome

**Ideal Speedup**: 4x with 4 workers
**Realistic Speedup**: 2-3x (accounting for overhead)

**New Performance**: 1.8-2.5s → 0.6-1.0s per sample

---

## Combined Performance Projection

### Phase-by-Phase Impact

| Phase | Optimization | Speedup | Time/Sample | Throughput |
|-------|--------------|---------|-------------|------------|
| Baseline | None | 1x | 12-14s | 0.07 samples/sec |
| Phase 1 | Batch node0 | 4-7x | 2-3s | 0.33-0.5 samples/sec |
| Phase 2 | KATO config | 1.15x | 1.8-2.5s | 0.4-0.55 samples/sec |
| Phase 3 | Parallel (4 workers) | 2-3x | 0.6-1.0s | 1-1.7 samples/sec |
| **TOTAL** | **All phases** | **15-28x** | **0.5-1.0s** | **1-2 samples/sec** |

### Real-World Impact

**Training 100K samples**:

| Configuration | Time | Speedup |
|---------------|------|---------|
| Baseline | 14-16 days | - |
| Phase 1 only | 2.3-3.5 days | 4-7x |
| Phases 1+2 | 2.0-3.0 days | 5-8x |
| All phases | **0.6-1.2 days** | **15-28x** |

**Training 1M samples**:

| Configuration | Time | Speedup |
|---------------|------|---------|
| Baseline | 139-162 days (4.6-5.4 months) | - |
| All phases | **5.8-11.6 days** | **15-28x** |

---

## Implementation Roadmap

### Week 1: Phase 1 (Batching)

**Day 1-2**: Implementation
- Implement `batch_learn_node0()` helper
- Add batch accumulation to main loop
- Add `node0_batch_size` parameter

**Day 3**: Testing
- Unit tests for batch learning
- Correctness validation (compare with baseline)
- Pattern frequency analysis

**Day 4**: Optimization
- Tune batch size (test 25, 50, 75, 100)
- Profile for remaining bottlenecks
- Document results

**Day 5**: Integration
- Update notebook examples
- Update documentation
- Create benchmark script

### Week 2: Phase 2 (Config Tuning)

**Day 1**: Configuration
- Update docker-compose.yml
- Test different KATO_BATCH_SIZE values
- Validate stability

**Day 2**: Testing
- Load testing with large batches
- Stress testing with concurrent requests
- Performance benchmarking

**Day 3**: Documentation
- Document optimal settings
- Create tuning guide
- Update troubleshooting section

### Week 3: Phase 3 (Parallel Processing)

**Day 1-2**: Session Support
- Implement session-aware KATOClient
- Implement session creation/cleanup
- Test session isolation

**Day 3-4**: Parallel Executor
- Implement ThreadPoolExecutor wrapper
- Add worker error handling
- Test with 2, 4, 8 workers

**Day 5**: Optimization
- Determine optimal worker count
- Profile for bottlenecks
- Tune for maximum throughput

**Day 6-7**: Testing & Documentation
- Comprehensive correctness testing
- Load testing
- Update all documentation

---

## Risk Mitigation

### Phase 1 Risks

**Risk**: Batching changes pattern names (non-deterministic)
**Mitigation**:
- Validate pattern names match baseline
- KATO's deterministic hashing should ensure consistency
- Add regression tests

**Risk**: Memory overflow with large batches
**Mitigation**:
- Start with small batch_size (25)
- Monitor memory usage
- Add batch_size limits based on available RAM

**Risk**: KATO timeout on large batches
**Mitigation**:
- Increase REQUEST_TIMEOUT (Phase 2)
- Add retry logic with exponential backoff
- Split batches if timeout occurs

### Phase 2 Risks

**Risk**: KATO instability with new config
**Mitigation**:
- Test in isolation before combining with Phase 1
- Monitor KATO logs for errors
- Keep old docker-compose.yml as backup

**Risk**: Memory overflow in KATO
**Mitigation**:
- Monitor Docker container memory
- Increase container memory limits if needed
- Set STM_CAPACITY conservatively

### Phase 3 Risks

**Risk**: Race conditions in concurrent writes
**Mitigation**:
- Use KATO sessions for isolation
- Each worker writes to same KB but in isolated session
- KATO handles thread-safe MongoDB writes

**Risk**: Worker failures crash entire job
**Mitigation**:
- Wrap worker code in try/except
- Log errors, continue processing
- Add sample checkpointing for resume

**Risk**: Resource exhaustion (too many workers)
**Mitigation**:
- Start with conservative worker count
- Monitor CPU/memory/connections
- Add dynamic worker scaling

---

## Monitoring & Metrics

### Key Metrics to Track

**Performance Metrics**:
```python
metrics = {
    'samples_processed': 0,
    'total_time': 0.0,
    'api_calls_made': 0,
    'patterns_learned': {
        'node0': 0,
        'node1': 0,
        'node2': 0,
        # ...
    },
    'average_time_per_sample': 0.0,
    'throughput_samples_per_sec': 0.0
}
```

**Tracking Code**:
```python
import time

class PerformanceTracker:
    def __init__(self):
        self.start_time = time.time()
        self.samples = 0
        self.api_calls = 0

    def record_sample(self):
        self.samples += 1

    def record_api_call(self):
        self.api_calls += 1

    def report(self):
        elapsed = time.time() - self.start_time
        return {
            'samples': self.samples,
            'time': elapsed,
            'samples_per_sec': self.samples / elapsed,
            'api_calls': self.api_calls,
            'api_calls_per_sample': self.api_calls / self.samples
        }

# Usage
tracker = PerformanceTracker()

for sample in dataset:
    train_hierarchical_single_pass(learner, sample)
    tracker.record_sample()

print(tracker.report())
```

### Benchmarking Script

Create `/Users/sevakavakians/PROGRAMMING/kato-notebooks/benchmark_performance.py`:

```python
#!/usr/bin/env python3
"""
Performance benchmarking script for hierarchical training.

Usage:
    python benchmark_performance.py --samples 100 --config baseline
    python benchmark_performance.py --samples 100 --config phase1
    python benchmark_performance.py --samples 100 --config all_phases
"""

import argparse
import time
from tools import (
    HierarchicalConceptLearner,
    HierarchicalNode,
    train_from_streaming_dataset
)

def benchmark_config(config_name, num_samples):
    """Benchmark a specific configuration."""

    # Define configurations
    configs = {
        'baseline': {
            'nodes': [
                HierarchicalNode('node0', chunk_size=5),
                HierarchicalNode('node1', chunk_size=7),
                HierarchicalNode('node2', chunk_size=9),
            ],
            'node0_batch_size': 1,  # No batching
            'num_workers': 1  # Sequential
        },
        'phase1': {
            'nodes': [
                HierarchicalNode('node0', chunk_size=5),
                HierarchicalNode('node1', chunk_size=7),
                HierarchicalNode('node2', chunk_size=9),
            ],
            'node0_batch_size': 50,  # Batching enabled
            'num_workers': 1  # Sequential
        },
        'all_phases': {
            'nodes': [
                HierarchicalNode('node0', chunk_size=5),
                HierarchicalNode('node1', chunk_size=7),
                HierarchicalNode('node2', chunk_size=9),
            ],
            'node0_batch_size': 50,  # Batching
            'num_workers': 4  # Parallel
        }
    }

    config = configs[config_name]

    # Create learner
    learner = HierarchicalConceptLearner(
        nodes=config['nodes'],
        node0_batch_size=config.get('node0_batch_size', 1),
        tokenizer_name='gpt2'
    )

    # Benchmark
    start = time.time()

    train_from_streaming_dataset(
        learner=learner,
        dataset_name="allenai/c4",
        dataset_config="en",
        streaming=True,
        max_samples=num_samples,
        num_workers=config.get('num_workers', 1)
    )

    duration = time.time() - start

    # Report
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"{'='*60}")
    print(f"Samples processed: {num_samples}")
    print(f"Total time: {duration:.2f}s")
    print(f"Time per sample: {duration/num_samples:.2f}s")
    print(f"Throughput: {num_samples/duration:.2f} samples/sec")
    print(f"{'='*60}\n")

    return {
        'config': config_name,
        'samples': num_samples,
        'duration': duration,
        'time_per_sample': duration / num_samples,
        'throughput': num_samples / duration
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--config', choices=['baseline', 'phase1', 'all_phases'],
                       default='baseline')

    args = parser.parse_args()

    benchmark_config(args.config, args.samples)
```

---

## Success Criteria

### Phase 1 Success

- ✅ Same patterns learned as baseline (deterministic)
- ✅ 4-7x speedup measured on 100 sample benchmark
- ✅ All unit tests pass
- ✅ No errors in KATO logs
- ✅ Memory usage stable

### Phase 2 Success

- ✅ KATO stable with new configuration
- ✅ 10-15% additional speedup measured
- ✅ Large batches (100+ observations) process without timeout
- ✅ No performance degradation

### Phase 3 Success

- ✅ Concurrent processing works without errors
- ✅ 2-3x speedup with 4 workers
- ✅ Knowledge bases identical to sequential version
- ✅ No race conditions or deadlocks
- ✅ Worker failures don't crash entire job

### Overall Success

- ✅ **15-28x total speedup achieved**
- ✅ **0.5-1.0s per sample** (target met)
- ✅ **1-2 samples/sec throughput** (target met)
- ✅ All correctness tests pass
- ✅ System stable under load
- ✅ Documentation complete

---

## Troubleshooting Guide

### Issue: Patterns Don't Match Baseline

**Symptoms**: Different pattern names in node0 after batching

**Diagnosis**:
```python
# Compare pattern names
baseline_patterns = get_patterns('node0_level0_kato')
batched_patterns = get_patterns('node0_level0_kato_batch_test')

diff = set(baseline_patterns) - set(batched_patterns)
print(f"Missing patterns: {diff}")
```

**Solutions**:
1. Verify chunk boundaries are marked correctly
2. Check that observations are in same order
3. Ensure metadata is in same position (first observation)
4. Validate KATO version (should be deterministic)

### Issue: KATO Timeout on Large Batches

**Symptoms**: `requests.exceptions.ReadTimeout`

**Solutions**:
1. Increase REQUEST_TIMEOUT in docker-compose.yml
2. Reduce node0_batch_size (try 25 instead of 50)
3. Check KATO logs for processing bottleneck
4. Add retry logic with exponential backoff

### Issue: Memory Overflow

**Symptoms**: Out of memory errors, Docker container crash

**Solutions**:
1. Reduce node0_batch_size
2. Increase Docker container memory limit
3. Reduce STM_CAPACITY in KATO config
4. Process smaller dataset chunks

### Issue: Parallel Workers Failing

**Symptoms**: Worker exceptions, incomplete processing

**Diagnosis**:
```python
# Check KATO sessions
curl http://localhost:8000/sessions

# Check connection pool
# Should see ~6 connections per active worker
```

**Solutions**:
1. Reduce num_workers
2. Increase CONNECTION_POOL_SIZE in KATO
3. Add session cleanup in finally block
4. Check for KATO session limit

### Issue: Slower Than Expected

**Symptoms**: Speedup less than projected

**Diagnosis**:
```python
# Profile code
import cProfile
cProfile.run('train_hierarchical_single_pass(learner, sample)')

# Check API call count
# Should be ~2-4 calls for node0 with batch_size=50
```

**Solutions**:
1. Verify batching is actually enabled
2. Check for network latency (use localhost, not docker hostname)
3. Profile for unexpected bottlenecks
4. Ensure KATO config changes applied (restart Docker)

---

## Future Optimizations (Beyond Phase 3)

### Phase 4: GPU Acceleration (Long-term)

KATO currently doesn't use GPUs, but future optimizations could include:
- GPU-accelerated pattern matching (CUDA)
- Vectorized similarity computations
- Estimated speedup: 2-5x additional

### Phase 5: Distributed Training

For massive datasets (10M+ samples):
- Multiple KATO instances with shared MongoDB
- Distributed task queue (Celery/RabbitMQ)
- Horizontal scaling across machines
- Estimated speedup: 10-100x (depending on cluster size)

### Phase 6: Incremental Learning

Current implementation reprocesses all samples. Future:
- Checkpoint progress every N samples
- Resume from checkpoint on failure
- Incremental knowledge base updates
- Benefit: Fault tolerance, not speed

---

## Conclusion

This 3-phase optimization plan targets **15-28x speedup** through:

1. **Phase 1** (4-7x): Batch node0 API calls to reduce HTTP overhead
2. **Phase 2** (1.15x): Tune KATO configuration for better throughput
3. **Phase 3** (2-3x): Parallel processing with session isolation

**Combined**: 12-14s → 0.5-1.0s per sample

**Implementation order**: Phase 1 → Phase 2 → Phase 3
**Total effort**: 2-3 weeks
**Risk level**: Medium (carefully test each phase)

---

**Document Status**: ✅ COMPLETE
**Last Updated**: 2025-10-17
**Next Step**: Implement Phase 1 (batch node0 API calls)
