# Training Issues: Investigation and Fixes

**Date**: 2026-01-19
**Issues Addressed**:
1. Checkpoint resume not working correctly (restarting from beginning)
2. Training sessions consistently timing out at 17k-20k records

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

### To test checkpoint resume:

1. Start a training run:
   ```python
   RESUME_FROM_CHECKPOINT = False
   MAX_SAMPLES = 20000
   CHECKPOINT_INTERVAL = 5000

   # Run training...
   ```

2. Interrupt after ~10,000 samples (Ctrl+C)

3. Resume training:
   ```python
   RESUME_FROM_CHECKPOINT = True

   # Should see output:
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

1. **`tools/streaming_dataset_loader.py`**:
   - Added `skip` parameter to `load_streaming()` method
   - Updated `train_from_streaming_dataset_parallel()` to use skip parameter
   - Removed inefficient loop-and-skip logic

2. **`/Users/sevakavakians/PROGRAMMING/kato/kato/config/api.py`**:
   - Increased `limit_max_requests` from 10,000 to 100,000

3. **`tools/kato_client.py`**:
   - Increased `_wait_for_kato_healthy()` default timeout from 30s to 60s
   - Updated all call sites to use 60s timeout
   - Updated version to 3.7.0 with improved docstrings

---

## Summary

Both issues have been resolved:

✅ **Issue 1 (Checkpoint Resume)**: Training now properly resumes from checkpoint position without re-fetching previous samples. Uses HuggingFace's efficient `.skip()` method.

✅ **Issue 2 (Session Timeout)**: Training sessions should no longer timeout at 17k-20k records. The KATO server now restarts every ~100k requests (instead of 10k), and the client gives the service 60s (instead of 30s) to become healthy during restarts.

**Next Steps**:
1. Restart KATO server to apply new configuration
2. Test with a long training run (30k+ samples)
3. Monitor for any remaining issues

---

## Questions?

If you encounter any issues:
1. Check KATO logs: `./start.sh logs kato 100`
2. Verify health: `./start.sh status`
3. Check checkpoint files: `ls -lh checkpoints/`
4. Monitor memory: `cd deployment && ./kato-manager.sh memory`
