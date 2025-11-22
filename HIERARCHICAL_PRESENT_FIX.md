# Hierarchical Present Field Fix - Complete Documentation

## Issue Description

After the initial fix to use `pred['present']` instead of `pred['name']`, a new issue appeared: the Present field was showing pattern names (e.g., `PTRN|906d23e40d02cadf2793b99de53cc4fb7f292548`) instead of decoded text, and pattern lookups were failing with "Pattern not found via KATO API".

### Root Cause

Two interrelated problems:

1. **The `pred['present']` field structure varies by hierarchical level:**

   - **At node0**: Contains actual tokens
     ```python
     pred['present'] = [['The'], ['cat'], ['sat']]
     ```

   - **At node1+**: Contains pattern names that need unraveling
     ```python
     pred['present'] = [['PTRN|906d23e40d02cadf2793b99de53cc4fb7f292548']]
     ```

2. **CRITICAL: `pred['present']` at level N contains level N-1 patterns!**

   - **node1 prediction**: `pred['present']` contains **node0 patterns**
   - **node2 prediction**: `pred['present']` contains **node1 patterns**
   - **node3 prediction**: `pred['present']` contains **node2 patterns**

   This is because nodeN observes the pattern names from node(N-1) as its input!

The original `extract_tokens_from_present()` function:
1. Assumed `present` always contained tokens (worked for node0 only)
2. Tried to unravel patterns at the wrong level (searched at level N instead of N-1)

## The Complete Solution

### 1. Level-Aware Extraction Function

**File**: `generation.ipynb` (Cell 24, lines ~1565-1630)

```python
def extract_tokens_from_present(present_events, level=0, nodes=None, verbose=False):
    """
    Extract tokens from pred['present'] field (KATO event format).

    The 'present' field structure varies by hierarchical level:
    - At node0: Contains actual tokens [['The'], ['cat'], ['sat']]
    - At node1+: Contains pattern names [['PTRN|abc123'], ['PTRN|def456']]

    CRITICAL INSIGHT: pred['present'] at level N contains level N-1 patterns!
    - node1 prediction: present contains node0 patterns → unravel at level 0
    - node2 prediction: present contains node1 patterns → unravel at level 1
    - node3 prediction: present contains node2 patterns → unravel at level 2

    For higher levels, pattern names must be unraveled recursively to get tokens.

    Args:
        present_events: List of KATO events from pred['present']
        level: Hierarchical level (0 = node0, 1 = node1, etc.)
        nodes: List of KATOClient instances (required for level > 0)
        verbose: Print unraveling details

    Returns:
        List of token strings
    """
    if not present_events:
        return []

    if level == 0:
        # node0: Extract tokens directly (present contains actual tokens)
        tokens = []
        for event in present_events:
            if event and len(event) > 0:
                tokens.append(event[0])
        return tokens
    else:
        # node1+: Unravel pattern names recursively (present contains pattern names)
        if nodes is None:
            raise ValueError("nodes parameter required for hierarchical unraveling (level > 0)")

        all_tokens = []
        for event in present_events:
            if event and len(event) > 0:
                pattern_name = event[0]

                # Strip PTRN| prefix if present
                if pattern_name.startswith('PTRN|'):
                    pattern_name = pattern_name[5:]

                if verbose:
                    print(f"    Unraveling present pattern: {pattern_name[:16]}... (level {level})")

                # Recursively unravel this pattern to tokens
                # CRITICAL: Pattern is from level-1 (the child level that was observed)
                # level-1 is safe because we only reach this branch if level > 0
                tokens = unravel_pattern(
                    pattern_name,
                    level=level-1,  # ← THE FIX: Unravel at child level!
                    nodes=nodes,
                    verbose=verbose,
                    indent=2 if verbose else 0
                )

                if tokens:
                    all_tokens.extend(tokens)
                elif verbose:
                    print(f"      Warning: Failed to unravel pattern {pattern_name[:16]}...")

        return all_tokens
```

### 2. Updated Call Site

**File**: `generation.ipynb` (Cell 24, line ~1802)

```python
# OLD (incomplete fix):
present_tokens = extract_tokens_from_present(present_events)

# NEW (complete fix):
present_tokens = extract_tokens_from_present(
    present_events,
    level=used_level,      # Pass the hierarchical level
    nodes=nodes,           # Pass KATO clients for unraveling
    verbose=verbose_unravel # Pass verbose flag
)
```

### 3. Updated Comment

**File**: `generation.ipynb` (Cell 24, line ~1798)

```python
# OLD:
# First, extract the PRESENT tokens directly (no API lookup needed)

# NEW:
# First, extract the PRESENT tokens (unravel if from higher level)
```

## How It Works

### For node0 Predictions (Tokens)

```python
# Input
pred = {
    'present': [['ĠAmong'], ['Ġfl'], ['ukes'], ['Ġ,'], ['Ġthe']],
    'future': [['Ġmost']]
}

# Extraction (level=0)
present_tokens = extract_tokens_from_present(pred['present'], level=0)
# Returns: ['ĠAmong', 'Ġfl', 'ukes', 'Ġ,', 'Ġthe']

# Output
Present: Among fl ukes , the
Future: most
```

### For node1+ Predictions (Pattern Names)

```python
# Input
pred = {
    'present': [['PTRN|906d23e40d02cadf2793b99de53cc4fb7f292548']],
    'future': [['PTRN|xyz789']]
}

# Extraction (level=1)
present_tokens = extract_tokens_from_present(
    pred['present'],
    level=1,
    nodes=nodes
)
# 1. Extracts pattern name: '906d23e40d02cadf2793b99de53cc4fb7f292548'
# 2. Calls unravel_pattern() recursively
# 3. Returns: ['ĠCompany', 'Ġ)', 'Ġ,', 'Ġand', ...]

# Output
Present: Company ) , and ...
Future: (unraveled future tokens)
```

## Key Changes Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Function signature** | `extract_tokens_from_present(present_events)` | `extract_tokens_from_present(present_events, level=0, nodes=None, verbose=False)` |
| **Level handling** | Assumes tokens only | Checks `level` parameter:<br>• `level == 0`: Extract directly<br>• `level > 0`: Unravel recursively |
| **Unravel level** | N/A (didn't unravel) | **`level-1`** (patterns are from child level!) |
| **Pattern name handling** | N/A | Strips `PTRN\|` prefix<br>Calls `unravel_pattern()` at correct level |
| **Error handling** | None | Raises `ValueError` if `nodes` missing for `level > 0` |
| **Verbose output** | None | Prints unraveling progress when enabled |

## Testing

### Test Coverage

**File**: `test_present_extraction.py`

The test suite includes:

1. **node0 Level Tests**:
   - Normal token sequence
   - Empty present field
   - Events with anomalies
   - Single token

2. **node1+ Level Tests**:
   - Pattern name with PTRN| prefix
   - Multiple pattern names
   - Pattern name without prefix
   - Missing nodes parameter (error handling)

3. **Demonstrations**:
   - Bug scenario (using `pred['name']`)
   - Hierarchical case (node1+ pattern names)

### Running Tests

```bash
python3 test_present_extraction.py
```

**Expected Output**: All tests pass, demonstrating correct handling of both token-level (node0) and pattern-level (node1+) predictions.

## Files Modified

1. ✅ **`generation.ipynb`** (Cell 24)
   - Updated `extract_tokens_from_present()` function
   - Updated call site to pass `level`, `nodes`, `verbose`
   - Updated comment

2. ✅ **`test_present_extraction.py`**
   - Updated function definition
   - Added hierarchical test cases
   - Added demonstration of hierarchical case

3. ✅ **`HIERARCHICAL_PRESENT_FIX.md`** (this file)
   - Complete documentation of the fix

## Expected Behavior After Fix

### Before Fix

```
Present: PTRN|906d23e40d02cadf2793b99de53cc4fb7f292548
Future: Company ) , and Lord Balfour of Burleigh ...
```

❌ Present shows pattern name instead of text

### After Fix

```
Present: (decoded text from unraveled pattern)
Future: Company ) , and Lord Balfour of Burleigh ...
```

✅ Present shows properly decoded text at all hierarchical levels

## The Level N-1 Insight

### Why Patterns Are From the Child Level

This is the most critical insight for understanding hierarchical KATO predictions:

**KATO's Hierarchical Structure:**
1. **node0** observes tokens, learns patterns (token sequences)
2. **node1** observes **node0 pattern names**, learns patterns (pattern name sequences)
3. **node2** observes **node1 pattern names**, learns patterns (pattern name sequences)
4. **node3** observes **node2 pattern names**, learns patterns (pattern name sequences)

**When nodeN makes a prediction:**
- It matched a **nodeN pattern** (stored in its knowledge base)
- `pred['name']` = the matched **nodeN pattern hash**
- `pred['present']` = **the input it received** = node(N-1) pattern names!
- `pred['future']` = **what it predicts comes next** = more node(N-1) pattern names

**Example: node2 Prediction**

```python
# node2's STM contains node1 pattern names:
node2_stm = [
    'PTRN|092dabe0fb4aec91...',  # node1 pattern
    'PTRN|857bb4ca6d675235...',  # node1 pattern
    'PTRN|2178dda8bf735bc1...'   # node1 pattern
]

# node2 learns a pattern from this sequence and makes a prediction:
prediction = {
    'name': '859e0343ab9602d2...',  # This is a node2 pattern hash
    'present': [['092dabe0fb4aec91...'], ['857bb4ca6d675235...']],  # node1 patterns!
    'future': [['2178dda8bf735bc1...']]  # node1 pattern!
}
```

**The Fix:**
```python
# WRONG: Unravel at level 2 (where the prediction came from)
tokens = unravel_pattern('092dabe0fb4aec91...', level=2, ...)
# ❌ Fails: Pattern doesn't exist at node2, it's a node1 pattern!

# CORRECT: Unravel at level 1 (where the pattern actually exists)
tokens = unravel_pattern('092dabe0fb4aec91...', level=1, ...)
# ✓ Success: Found the pattern at node1 and unraveled it!
```

### The Rule

**Always unravel `pred['present']` patterns at `level-1`:**

| Prediction From | pred['present'] Contains | Unravel At |
|----------------|-------------------------|-----------|
| node1 (level=1) | node0 patterns | level 0 |
| node2 (level=2) | node1 patterns | level 1 |
| node3 (level=3) | node2 patterns | level 2 |

This is safe because we only reach the unraveling branch when `level > 0`, so `level-1 >= 0`.

## Why This Matters

This fix ensures that text generation works correctly **at all hierarchical levels**:

- **node0**: Direct token extraction (no change from initial fix)
- **node1**: Paragraph-level patterns → unravel node0 patterns to tokens
- **node2**: Chapter-level patterns → unravel node1 patterns to tokens
- **node3**: Book-level patterns → unravel node2 patterns to tokens

Without this fix, generation would only work at node0, and higher-level predictions would:
1. Show pattern names instead of readable text (display issue)
2. Fail with "Pattern not found via KATO API" (lookup at wrong level)

## Integration with Initial Fix

This fix **builds upon** the initial fix that changed from `pred['name']` to `pred['present']`:

1. **Initial Fix**: Use `pred['present']` instead of `pred['name']`
   - Eliminated token repetition at present/future boundary
   - But assumed `present` always contained tokens

2. **This Fix**: Make `extract_tokens_from_present()` level-aware
   - Handles both tokens (node0) and pattern names (node1+)
   - Recursively unravels pattern names to tokens
   - Preserves the benefit of avoiding repetition

Together, these fixes provide a complete solution for hierarchical text generation.

---

**Fixed by**: Claude
**Date**: 2025-11-21
**Related**: GENERATION_FIX_SUMMARY.md (initial fix documentation)
