# Hardware Recommendation Report

## Target Requirements
- **Dataset size**: 100,000,000 samples
- **Desired training time**: 48.0 hours
- **Budget**: $10,000
- **Deployment**: on_premise

## Calculated Requirements
- **CPU cores**: 74
- **Memory**: 128 GB
- **Storage**: 1000 GB
- **Required throughput**: 578.7 samples/sec

## On-Premise Recommendation

### CPU
- **Model**: AMD EPYC 7763
- **Cores/Threads**: 64 cores / 128 threads
- **Clock Speed**: 2.45 GHz (Turbo: 3.50 GHz)
- **TDP**: 280W
- **Price**: $7,200

### Memory
- **Capacity**: 128 GB DDR5
- **Speed**: 5600 MHz
- **Channels**: 2
- **Price**: $640

### Storage
- **Capacity**: 1000 GB
- **Type**: NVMe SSD
- **Read Speed**: 7,000 MB/s
- **Write Speed**: 5,000 MB/s
- **Price**: $100

### Total Cost
**$8,440**

## Performance Estimates
- **Estimated training time**: 87.7 hours
- **Estimated throughput**: 316.8 samples/sec

## ⚠️ Warnings
- Very high throughput required (579 samples/sec). Consider distributed training.
- No CPU found with 74 cores. Using best available.
- Estimated time (87.7h) exceeds desired time (48.0h)

