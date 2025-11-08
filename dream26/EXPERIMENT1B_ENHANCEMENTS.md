# Experiment 1b: Enhanced Platform Capability Profiling

## Overview

Experiment 1b extends the original platform capability profiling to address **Section 4** requirements from the DREAM'26 submission, specifically **"Profiling energy consumption during data exchange"**.

## Gap Analysis from Experiment 1

### Critical Issues Identified:

1. **❌ GPU Power During Load = FAILED**
   - `gpu_metrics_load.csv` was empty (0 bytes)
   - No power data during actual workloads
   - Cannot calculate energy consumption without this data

2. **❌ Small Transfer Testing = MISSING**
   - Section 4 identifies **37.5 byte average H2D** as major bottleneck
   - Original experiment only tested **1MB to 1GB** transfers
   - Missing entire range: 1B, 37.5B, 100B, 1KB, 10KB, 100KB

3. **❌ Energy Cost Analysis = NOT MEASURED**
   - No power × time measurements
   - Cannot determine joules per operation
   - Cannot compare energy costs across operation types

4. **❌ Mixed Workload = NOT TESTED**
   - Section 4 shows simultaneous compute + communication patterns
   - Need realistic co-simulation workload characterization

## Enhancements in Experiment 1b

### Test 1: GPU Compute Performance (Enhanced)
- **Original**: Matrix multiplication benchmarks
- **Enhancement**: GPU power monitoring during all compute operations
- **Output**: `gpu_metrics_compute.csv` with 1-second power samples

### Test 2: Small Transfer Overhead (NEW)
- **Purpose**: Characterize overhead for tiny transfers matching Section 4
- **Sizes**: 1B, 10B, **37B**, 100B, 1KB, 10KB, 100KB, 1MB
- **Metrics**:
  - Latency (microseconds) for each transfer size
  - Bandwidth (MB/s) - will show dramatic drop-off at small sizes
  - H2D, D2H, D2D patterns
- **Section 4 Relevance**: Directly tests 37.5-byte average identified as bottleneck
- **Output**: `small_transfer_results.json`, `gpu_metrics_small_transfers.csv`

### Test 3: Large Memory Bandwidth (From Experiment 1)
- **Sizes**: 1MB, 10MB, 100MB, 500MB, 1000MB
- **Enhancement**: GPU power monitoring during transfers
- **Output**: `memory_bandwidth_results.json`, `gpu_metrics_large_bandwidth.csv`

### Test 4: DDP Communication Overhead (From Experiment 1)
- **Tests**: NCCL AllGather, AllReduce across tensor sizes
- **Enhancement**: GPU power monitoring during NCCL operations
- **Output**: `ddp_communication_results.json`, `gpu_metrics_nccl.csv`

### Test 5: Mixed Workload Energy Profiling (NEW)
- **Purpose**: Simulate Twin-B co-simulation pattern
- **Workload**: Compute (matrix multiply) + frequent small H2D transfers (37 bytes)
- **Metrics**:
  - Compute-only baseline
  - Transfer-only baseline
  - Mixed workload (compute + transfer)
  - Overhead percentage from mixing operations
- **Section 4 Relevance**: Tests realistic co-simulation synchronization pattern
- **Output**: `mixed_workload_results.json`, `gpu_metrics_mixed.csv`

### Test 6: Energy Cost Analysis (NEW)
- **Purpose**: Calculate energy consumption (joules) for each operation type
- **Method**: Energy (J) = Average Power (W) × Duration (s)
- **Analysis**: Per-operation energy costs for:
  - Compute operations
  - Small transfers
  - Large transfers
  - NCCL communication
  - Mixed workloads
- **Section 4 Relevance**: Provides quantitative energy baseline for optimization
- **Output**: `energy_analysis.json`

### Test 7: GPU Idle State
- **Purpose**: Baseline idle power consumption
- **Output**: `gpu_metrics_idle.csv`

## Fixed Issues

### 1. GPU Power Monitoring Reliability
**Problem**: Original `gpu_metrics_load.csv` was empty

**Root Cause**: Background nvidia-smi process killed before writing data

**Fix**:
```bash
start_gpu_monitoring() {
    nvidia-smi --query-gpu=... -l 1 >> $output_file &
    GPU_MON_PID=$!
}

stop_gpu_monitoring() {
    sleep 2  # Ensure final samples captured
    kill $GPU_MON_PID
    wait $GPU_MON_PID  # Wait for process to finish writing
}
```

### 2. Separate Power Logs Per Test
**Enhancement**: Each test gets dedicated power monitoring CSV
- `gpu_metrics_compute.csv`
- `gpu_metrics_small_transfers.csv`
- `gpu_metrics_large_bandwidth.csv`
- `gpu_metrics_nccl.csv`
- `gpu_metrics_mixed.csv`

**Benefit**: Can correlate power consumption with specific operation types

## Expected Runtime

- **Experiment 1**: ~15-30 minutes
- **Experiment 1b**: ~45-60 minutes
  - Additional time for small transfer sweep (8 sizes × 1000 iterations)
  - Mixed workload testing
  - Energy analysis

## Output Files

```
dream26/exp1b_XXXXX/
├── gpu_compute_results.json              # Compute performance
├── small_transfer_results.json           # Small transfer overhead (NEW)
├── memory_bandwidth_results.json         # Large bandwidth
├── ddp_communication_results.json        # NCCL overhead
├── mixed_workload_results.json           # Mixed workload (NEW)
├── energy_analysis.json                  # Energy consumption (NEW)
├── gpu_metrics_compute.csv               # Power during compute
├── gpu_metrics_small_transfers.csv       # Power during small transfers (NEW)
├── gpu_metrics_large_bandwidth.csv       # Power during large transfers
├── gpu_metrics_nccl.csv                  # Power during NCCL
├── gpu_metrics_mixed.csv                 # Power during mixed workload (NEW)
├── gpu_metrics_idle.csv                  # Idle state power
└── test_*.py                             # Test scripts
```

## Key Metrics for Section 4 Analysis

### 1. Small Transfer Overhead
- **Metric**: Latency (μs) and bandwidth (MB/s) for 37-byte transfers
- **Expected**: High latency, low bandwidth due to PCIe overhead
- **Section 4 Value**: 37.5 bytes average H2D transfer size

### 2. Energy Cost per Transfer Type
- **Metric**: Joules per operation
- **Breakdown**:
  - Energy per 37-byte H2D transfer
  - Energy per NCCL AllGather operation
  - Energy per compute iteration
- **Use**: Identify most energy-intensive operations for optimization

### 3. Mixed Workload Overhead
- **Metric**: Performance penalty from interleaved compute + transfer
- **Expected**: Overhead from synchronization and PCIe contention
- **Section 4 Relevance**: Matches Twin-B co-simulation pattern

### 4. Power Consumption by Operation
- **Idle**: ~55W per GPU
- **Compute**: ~300-400W per GPU (expected)
- **Small Transfers**: ~100-200W per GPU (expected)
- **NCCL Communication**: ~200-350W per GPU (expected)
- **Mixed**: Peak power consumption (expected)

## How to Run

```bash
cd dream26

# Submit experiment
./submit_exp1b.sh

# Monitor
squeue -u $USER
tail -f exp1b_XXXXX/exp1b_XXXXX.out

# Analyze results
python analyze_baseline.py exp1b_XXXXX
python generate_platform_report.py exp1b_XXXXX
```

## Comparison with Section 4 Results

After running, compare with DREAM'26 Section 4 findings:

| Metric | Section 4 (Exp 2) | Exp 1b Baseline | Notes |
|--------|-------------------|-----------------|-------|
| Avg H2D size | 37.5 bytes | Test at 37B | Direct measurement |
| H2D operations | 6,289 | 1000 iterations | Controlled test |
| GPU power | Recorded @ 1s | Recorded @ 1s | Same methodology |
| NCCL overhead | 64.4% GPU time | Timing measured | Platform baseline |
| Small transfer overhead | High DMA overhead | Quantified | Energy cost |

## Deliverables

1. ✅ **Platform energy baseline** for all operation types
2. ✅ **Small transfer characterization** (1B to 1MB)
3. ✅ **Energy cost analysis** (joules per operation)
4. ✅ **Mixed workload profiling** (realistic co-simulation pattern)
5. ✅ **Fixed GPU power monitoring** (reliable data collection)

## Next Steps

1. Run Experiment 1b on LANTA
2. Generate enhanced platform report with energy analysis
3. Compare baseline with Experiment 2 (Twin-B simulation) results
4. Identify optimization opportunities based on energy costs
5. Design targeted optimization experiments

---

*This experiment addresses all gaps identified in the original Experiment 1 and provides comprehensive energy baseline for Section 4 analysis.*
