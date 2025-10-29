#!/usr/bin/env python3
"""
Quick test to verify process_predictions=False bug is fixed.

This script:
1. Creates a session with process_predictions=False
2. Sends observations
3. Checks if training completes quickly (no prediction overhead)
"""

import time
import sys
sys.path.insert(0, '/Users/sevakavakians/PROGRAMMING/kato-notebooks')

from tools import KATOClient

print("Testing KATO process_predictions fix...")
print("=" * 60)

# Create client with predictions disabled
client = KATOClient(
    node_id='test_predictions_disabled',
    process_predictions=False,  # Should prevent prediction computation
    max_pattern_length=0,
    stm_mode='CLEAR'
)

print("✓ Created KATO session with process_predictions=False")
print(f"  Session ID: {client.session_id}")

# Test with 100 observations to see if predictions are computed
observations = []
for i in range(100):
    observations.append({
        'strings': [f'token{i % 10}', f'word{i % 5}']
    })

print(f"\nSending {len(observations)} observations...")
start = time.time()

result = client.observe_sequence(
    observations=observations,
    learn_at_end=True
)

elapsed = time.time() - start

print(f"✓ Completed in {elapsed:.2f}s")
print(f"  Pattern learned: {result.get('final_learned_pattern', 'N/A')}")

# Performance check
if elapsed < 1.0:
    print(f"\n✅ PASS: Fast completion ({elapsed:.2f}s < 1.0s)")
    print("   Predictions likely NOT being computed (bug is FIXED)")
else:
    print(f"\n⚠️  SLOW: Took {elapsed:.2f}s (expected <1s)")
    print("   Predictions may still be computed (bug NOT fixed)")
    print("   Check KATO logs for 'predictPattern' calls")

# Cleanup
client.close()

print("\n" + "=" * 60)
print("Test complete. Check logs with:")
print("  docker-compose logs kato | grep -i 'predictPattern' | tail -20")
