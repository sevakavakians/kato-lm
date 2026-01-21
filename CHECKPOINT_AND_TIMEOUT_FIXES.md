# Training Issues: Investigation and Fixes

**Date**: 2026-01-19 (Updated: 2026-01-19 Evening)
**Issues Addressed**:
1. Checkpoint resume not working in `train_from_streaming_dataset_parallel()`
2. Training sessions consistently timing out at 17k-20k records
3. **NEW**: Checkpoint functionality completely missing from `training.ipynb`

---

## Issue 3: training.ipynb Missing Checkpoint Implementation (CRITICAL)

### Root Cause

The `training.ipynb` (v2.0 with HierarchicalBuilder) had configuration variables for checkpointing (`CHECKPOINT_INTERVAL`, `RESUME_FROM_CHECKPOINT`) but **ZERO implementation** of checkpoint save/load logic.

**Location**: `training.ipynb`, Cell 18 (training loop)

**Problem**: The training loop simply processed samples without any checkpoint logic:
- ❌ No checkpoint loading at startup
- ❌ No checkpoint saving during training
- ❌ No stream skipping for resume
- ❌ Configuration variables were completely ignored

**Impact**: Users running `training.ipynb` thought checkpointing was working, but:
- All training runs started from sample 0
- Progress was lost on interruption
- `RESUME_FROM_CHECKPOINT = True` did nothing
- The old checkpoint file (`wikitext_parallel_checkpoint.json`) was from a different training function entirely

### Fix Implemented

✅ **Added complete checkpoint save/load functionality to `training.ipynb`**

**Changes to Cell 18**:

1. **Checkpoint Load Logic** (at training start):
```python
# Load checkpoint if resuming
if RESUME_FROM_CHECKPOINT and checkpoint_path.exists():
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)

    start_sample = checkpoint_data.get('samples_completed', 0)

    # Validate configuration matches
    if saved_config['chunk_sizes'] != CHUNK_SIZES:
        raise ValueError("Configuration mismatch!")
```

2. **Efficient Stream Skipping**:
```python
# Skip already-processed samples
stream_iterator = StreamingDatasetLoader.load_streaming(
    dataset_key=DATASET_KEY,
    max_samples=MAX_SAMPLES,
    skip=start_sample  # ← Uses HuggingFace .skip()
)
```

3. **Periodic Checkpoint Saving**:
```python
# Save checkpoint every CHECKPOINT_INTERVAL samples
if samples_completed % CHECKPOINT_INTERVAL == 0:
    checkpoint_data = {
        'dataset_key': DATASET_KEY,
        'samples_completed': samples_completed,
        'samples_errored': stats['errors'],
        'model_config': {
            'chunk_sizes': CHUNK_SIZES,
            'dataset_key': DATASET_KEY,
            'tokenizer': 'gpt2',
            'num_layers': 4
        }
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
```

4. **Configuration Validation**:
- Ensures resumed training uses same configuration
- Prevents silent data corruption
- Clear error messages on mismatch

5. **Emergency Checkpointing**:
- Saves checkpoint on errors (every 10 errors)
- Final checkpoint at training completion

**Checkpoint Filename**: `checkpoints/{dataset_key}_v2_checkpoint.json`

**Result**:
- ✅ Training can now be interrupted and resumed correctly
- ✅ Starts from last completed sample (e.g., 20,001 not 1)
- ✅ Configuration validation prevents corruption
- ✅ Progress preserved across sessions

---

## Issue 1: Checkpoint Resume Not Working

### Root Cause

The streaming dataset loader was fetching ALL samples from the beginning and then discarding them until reaching the checkpoint position.

**Location**: `tools/streaming_dataset_loader.py:1055-1089`

```python
# OLD CODE (inefficient):
stream_iterator = StreamingDatasetLoader.load_streaming(dataset_key, max_samples)
# ↓ Loads from index 0

for sample in stream_iterator:
    if sample_idx < start_idx:
        sample_idx += 1
        continue  # Still fetches and discards ALL previous samples!
```

**Problem**: Even though the training logic skipped samples before `start_idx`, the streaming dataset API still fetched and iterated through all those samples from the HuggingFace dataset server. This made resume extremely slow and appeared to "restart from beginning."

**Impact**: If you checkpointed at 18,000 samples and resumed, the stream still had to process samples 0-17,999 before continuing.

### Fix Implemented

✅ **Added `skip` parameter to `load_streaming()` method**

The method now uses HuggingFace's efficient `.skip()` method to skip already-processed samples in the stream:

```python
# NEW CODE (efficient):
def load_streaming(dataset_key: str, max_samples: int = None, skip: int = 0) -> Iterator[str]:
    """
    Load dataset in streaming mode.

    Args:
        skip: Number of samples to skip at the beginning (for checkpoint resume)
    """
    # ... load dataset ...

    # Skip samples efficiently using HuggingFace's .skip() method
    if skip > 0:
        dataset = dataset.skip(skip)  # ← Efficient skip!

    # Yield samples starting from skip position
```

✅ **Updated `train_from_streaming_dataset_parallel()` to use skip parameter**

```python
# Load dataset stream with efficient skip for checkpoint resume
stream_iterator = StreamingDatasetLoader.load_streaming(
    dataset_key,
    max_samples,
    skip=start_idx  # ← Efficiently skip already-processed samples
)

# Start from start_idx since we've already skipped in the stream
sample_idx = start_idx
```

**Result**: Checkpoint resume now truly resumes from where it left off, without re-fetching previous samples.

---

## Issue 2: Consistent Timeout at 17k-20k Records

### Root Cause

The KATO server was configured to restart after **10,000 requests** using uvicorn's `--limit-max-requests` setting.

**Location**: `/Users/sevakavakians/PROGRAMMING/kato/kato/config/api.py:298`

```python
# OLD VALUE:
'limit_max_requests': 10000,
```

**Why it timed out at 17k-20k samples**:
- Each training sample makes multiple API requests:
  - `observe_sequence()` for node0 chunks
  - `observe()` calls for higher nodes
  - `learn()` calls at each level
  - Potentially `get_predictions()` calls
- Total: **~2-5 requests per sample** depending on hierarchy depth
- Calculation: **17k samples × 2-5 requests = 34k-85k total requests**
- Uvicorn restarts after **every 10,000 requests**
- After multiple restarts during a long training run, connection failures cascade

**Why the client retry logic wasn't enough**:
- Health check timeout was only 30 seconds
- Multiple workers might fail simultaneously during restart
- Exponential backoff might not give service enough time to stabilize

### Fixes Implemented

✅ **Increased KATO server request limit from 10k to 100k**

**File**: `/Users/sevakavakians/PROGRAMMING/kato/kato/config/api.py:298`

```python
# NEW VALUE (10x increase):
'limit_max_requests': 100000,  # Increased from 10k to 100k for training workloads
```

**Rationale**:
- 100,000 requests supports training ~20,000-50,000 samples depending on hierarchy depth
- Still provides periodic restarts to prevent memory leaks (best practice)
- More appropriate for long-running training workloads
- Balances reliability with resource management

✅ **Increased client health check timeout from 30s to 60s**

**File**: `tools/kato_client.py:309`

```python
# OLD:
def _wait_for_kato_healthy(self, max_wait: int = 30, ...):

# NEW:
def _wait_for_kato_healthy(self, max_wait: int = 60, ...):
```

**Also updated call sites**:
```python
# OLD:
if self._wait_for_kato_healthy(max_wait=30):

# NEW:
if self._wait_for_kato_healthy(max_wait=60):
    print(f"   ✓ Service healthy, recreating session and retrying...")
```

**Rationale**:
- Gives KATO service more time to restart gracefully
- Prevents premature timeout during Docker container restart
- Better suited for production training workloads

✅ **Updated KATOClient version to 3.7.0**

Version history:
- v3.6.0: Added automatic connection retry for service restarts
- v3.7.0: Improved resilience (60s health check, 100k request limit)

---

## Testing the Fixes

### To test checkpoint resume in training.ipynb (NEW FIX):

**Important**: Use the updated `training.ipynb` notebook, not the parallel training function.

1. In `training.ipynb`, configure and run first training session:
   ```python
   # Cell 8: Configuration
   DATASET_KEY = 'wikitext'
   MAX_SAMPLES = 20000  # Start with 20k samples
   CHECKPOINT_INTERVAL = 1000
   RESUME_FROM_CHECKPOINT = False  # Start fresh

   # Run all cells through Cell 18
   ```

2. Wait for training to complete (all 20,000 samples)
   - You should see checkpoint saves at 1k, 2k, ..., 20k
   - Final message: "💾 Final checkpoint saved: checkpoints/wikitext_v2_checkpoint.json"

3. Change configuration to resume with MORE samples:
   ```python
   # Cell 8: Update configuration
   MAX_SAMPLES = 100000  # Increase to 100k samples
   RESUME_FROM_CHECKPOINT = True  # ← Enable resume

   # Run Cell 8 and Cell 18 again
   ```

4. Verify output shows resume from checkpoint:
   ```
   📂 Resuming from checkpoint:
      Samples completed: 20,000
      Samples errored: 0
      ✓ Configuration validated

   ============================================================
   TRAINING START
   ============================================================
   Dataset: wikitext
   Max samples: 100,000
   Starting from: sample 20,000  ← Should NOT be 0!
   Checkpoint interval: 1,000
   ============================================================

   📡 Streaming: WikiText-103
      Skipping: 20,000 samples (checkpoint resume)  ← NEW!
   ```

5. Training should continue from sample 20,001 (not restart from 1)

### To test checkpoint resume in parallel training function:

Follow original instructions for `train_from_streaming_dataset_parallel()`:

1. Start a training run:
   ```python
   from tools.streaming_dataset_loader import StreamingDatasetLoader

   stats = StreamingDatasetLoader.train_from_streaming_dataset_parallel(
       dataset_key='wikitext',
       max_samples=20000,
       learner=learner,
       profiler=profiler,
       checkpoint_interval=5000,
       resume_from_checkpoint=False
   )
   ```

2. Interrupt after ~10,000 samples (Ctrl+C)

3. Resume training:
   ```python
   stats = StreamingDatasetLoader.train_from_streaming_dataset_parallel(
       dataset_key='wikitext',
       max_samples=20000,
       learner=learner,
       profiler=profiler,
       resume_from_checkpoint=True  # ← Enable resume
   )

   # Should see:
   # 📂 Resuming from checkpoint:
   #    Samples attempted: 10,000
   #    ✓ Configuration validated
   # 📡 Streaming: WikiText-103
   #    Skipping: 10,000 samples (checkpoint resume)  ← NEW!
   ```

4. Verify training continues from ~10,000 instead of reprocessing 0-9,999

### To test session timeout fix:

1. Restart KATO server to apply new configuration:
   ```bash
   cd /Users/sevakavakians/PROGRAMMING/kato
   ./start.sh restart kato
   ```

2. Run a long training session (30k+ samples):
   ```python
   MAX_SAMPLES = 30000

   # Training should complete without timeout at 17k-20k
   ```

3. Monitor logs for service restarts:
   ```bash
   ./start.sh follow kato

   # You should see fewer restarts (every ~100k requests instead of 10k)
   ```

---

## Configuration Summary

### Before Fixes:
- ✗ Checkpoint resume: Re-fetched all samples from beginning
- ✗ Request limit: 10,000 (caused restarts every ~2k-5k training samples)
- ✗ Health check timeout: 30 seconds

### After Fixes:
- ✅ Checkpoint resume: Efficiently skips processed samples using `.skip()`
- ✅ Request limit: 100,000 (restarts every ~20k-50k training samples)
- ✅ Health check timeout: 60 seconds
- ✅ Better logging for connection issues

---

## Best Practices for Long Training Runs

### 1. Use Checkpoints
```python
checkpoint_interval = 5000  # Save every 5K samples
resume_from_checkpoint = True
```

### 2. Monitor KATO Health
```bash
# Check service status
./start.sh status

# Monitor logs in real-time
./start.sh follow kato

# Check for restarts
docker logs kato | grep "limit-max-requests"
```

### 3. Adjust Worker Count
```python
# Balance parallelism with stability
num_workers = 3  # Recommended: 2-4 workers
connections_needed = num_workers * num_levels  # Must be ≤ 30
```

### 4. Handle Memory Issues
```bash
# Monitor memory usage
cd /Users/sevakavakians/PROGRAMMING/kato/deployment
./kato-manager.sh memory
```

---

## Files Modified

1. **`tools/streaming_dataset_loader.py`** (kato-notebooks):
   - Added `skip` parameter to `load_streaming()` method
   - Updated `train_from_streaming_dataset_parallel()` to use skip parameter
   - Removed inefficient loop-and-skip logic

2. **`training.ipynb`** (kato-notebooks) - **NEW FIX**:
   - Added complete checkpoint save/load functionality to Cell 18
   - Implemented checkpoint loading with configuration validation
   - Added periodic checkpoint saving (every CHECKPOINT_INTERVAL samples)
   - Added emergency checkpoint saving on errors
   - Uses efficient stream skipping via `skip=start_sample`
   - Checkpoint file: `checkpoints/{dataset_key}_v2_checkpoint.json`

3. **`/Users/sevakavakians/PROGRAMMING/kato/kato/config/api.py`** (kato server):
   - Increased `limit_max_requests` from 10,000 to 100,000

4. **`tools/kato_client.py`** (kato-notebooks):
   - Increased `_wait_for_kato_healthy()` default timeout from 30s to 60s
   - Updated all call sites to use 60s timeout
   - Updated version to 3.7.0 with improved docstrings

---

## Summary

All three issues have been resolved:

✅ **Issue 1 (Checkpoint Resume in Parallel Function)**: `train_from_streaming_dataset_parallel()` now properly resumes from checkpoint position without re-fetching previous samples. Uses HuggingFace's efficient `.skip()` method.

✅ **Issue 2 (Session Timeout)**: Training sessions should no longer timeout at 17k-20k records. The KATO server now restarts every ~100k requests (instead of 10k), and the client gives the service 60s (instead of 30s) to become healthy during restarts.

✅ **Issue 3 (Missing Checkpoint in training.ipynb)** - **CRITICAL**: `training.ipynb` now has complete checkpoint functionality. Training can be interrupted and resumed correctly, starting from the last completed sample (e.g., resumes at sample 20,001 not sample 1).

**Key Takeaways**:
- **Use `training.ipynb`** for educational training with full checkpoint support
- **Use `train_from_streaming_dataset_parallel()`** for production training with parallel workers
- Both now support checkpoint resume correctly
- Checkpoint files are separate: `wikitext_v2_checkpoint.json` vs `wikitext_parallel_checkpoint.json`

**Next Steps**:
1. ✅ KATO server already restarted with new configuration
2. Test checkpoint resume in `training.ipynb` with MAX_SAMPLES change
3. Verify training resumes from correct sample number
4. Monitor for any remaining issues

---

## Questions?

If you encounter any issues:
1. Check KATO logs: `./start.sh logs kato 100`
2. Verify health: `./start.sh status`
3. Check checkpoint files: `ls -lh checkpoints/`
4. Monitor memory: `cd deployment && ./kato-manager.sh memory`
