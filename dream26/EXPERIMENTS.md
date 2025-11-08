# Twin-B Initial Experiments

This document describes the setup and execution of two initial experiments for the Twin-B digital building simulation platform, based on the DREAM'26 conference submission.

## Overview

The Twin-B system combines EnergyPlus building energy modeling with Mesa agent-based modeling in a distributed HPC environment. These experiments establish baseline performance metrics and identify optimization opportunities.

---

## Experiment 1: Platform Capability Profiling

### Purpose
Benchmark the HPC platform to understand hardware capabilities and establish performance baselines for:
- GPU compute performance (GFLOPS)
- Memory bandwidth (H2D, D2H, D2D)
- NCCL communication overhead
- System resource availability

### Job Script
`experiment1_platform_capability.slurm`

### How to Run

```bash
# Create logs directory
mkdir -p logs

# Submit job
sbatch experiment1_platform_capability.slurm

# Monitor job
squeue -u $USER

# Check output (replace XXXXX with job ID)
tail -f logs/exp1_platform_XXXXX.out
```

### Expected Runtime
~15-30 minutes

### Tests Performed

1. **GPU Compute Capability Test**
   - Matrix multiplication benchmarks at various sizes (1024×1024 to 8192×8192)
   - Measures GFLOPS performance
   - Output: `gpu_compute_results.json`

2. **Memory Bandwidth Test**
   - Tests Host-to-Device (H2D), Device-to-Host (D2H), and Device-to-Device (D2D) transfers
   - Sizes: 1MB to 1000MB
   - Output: `memory_bandwidth_results.json`

3. **DDP Communication Overhead Test**
   - Measures NCCL AllGather and AllReduce latency
   - Tests with varying tensor sizes
   - Output: `ddp_communication_results.json`

4. **GPU Metrics Collection**
   - Power draw, memory usage, utilization, temperature
   - Collected during idle and load states
   - Output: `gpu_metrics_idle.csv`, `gpu_metrics_load.csv`

### Output Files

All results saved to: `experiment1_results/run_<job_id>/`

```
experiment1_results/run_XXXXX/
├── gpu_compute_results.json          # Compute performance metrics
├── memory_bandwidth_results.json     # Memory transfer speeds
├── ddp_communication_results.json    # NCCL communication overhead
├── gpu_metrics_idle.csv              # GPU metrics at idle
├── gpu_metrics_load.csv              # GPU metrics under load
└── test_*.py                         # Test scripts
```

### Key Metrics to Extract

From `gpu_compute_results.json`:
- GFLOPS for different matrix sizes
- CUDA availability and version

From `memory_bandwidth_results.json`:
- H2D, D2H, D2D bandwidth in MB/s
- Identify PCIe bottlenecks

From `ddp_communication_results.json`:
- AllGather latency (ms)
- AllReduce latency (ms)
- Communication scaling with tensor size

---

## Experiment 2: Baseline Twin-B Simulation

### Purpose
Run baseline building simulation with comprehensive profiling to:
- Identify CPU-GPU synchronization bottlenecks (cudaStreamSynchronize)
- Measure NCCL communication patterns (AllGather, AllReduce)
- Analyze memory transfer inefficiencies
- Establish baseline energy consumption
- Compare against DREAM'26 paper metrics

### Job Script
`experiment2_baseline_simulation.slurm`

### How to Run

```bash
# Create logs directory
mkdir -p logs

# Verify config files exist
ls -l config.yaml agents.json

# Submit job
sbatch experiment2_baseline_simulation.slurm

# Monitor job
squeue -u $USER
watch 'squeue -u $USER'

# Check output (replace XXXXX with job ID)
tail -f logs/exp2_baseline_XXXXX.out

# View errors if any
tail -f logs/exp2_baseline_XXXXX.err
```

### Expected Runtime
~1.5-2 hours (depending on system load)

### Simulation Scenario
- **Type**: Regular semester weekday (baseline)
- **Occupants**: Based on agents.json configuration
- **Steps**: 288 (5-minute intervals for 1 day)
- **Zones**: 65+ building zones

### Execution Phases

**Part 1: Baseline Run (No Profiling)**
- Standard simulation without profiling overhead
- Measures actual runtime
- Collects GPU metrics (power, utilization, memory)
- Output: Agent results, EnergyPlus outputs

**Part 2: Profiled Run (NVIDIA Nsight Systems)**
- Full profiling with nsys
- Captures CUDA API calls, GPU kernels, memory operations
- Includes NVTX markers, OS runtime
- Generates `.nsys-rep` file for GUI analysis

**Part 3: Extract Statistics**
- Automated extraction of profiling metrics
- CSV summaries for CUDA API, kernels, memory
- NVTX marker statistics

**Part 4: Quick Analysis**
- Python script analyzes results
- Generates summary JSON
- Calculates key metrics

### Output Files

All results saved to: `experiment2_results/baseline_<job_id>/`

```
experiment2_results/baseline_XXXXX/
├── EXPERIMENT_SUMMARY.md              # Human-readable summary
├── config_baseline.yaml               # Configuration used
├── agents_baseline.json               # Agent configuration used
│
├── simulation_baseline.log            # Baseline run output
├── simulation_profiled.log            # Profiled run output
├── baseline_runtime.txt               # Total execution time (seconds)
│
├── gpu_metrics_baseline.csv           # GPU metrics during baseline
├── gpu_metrics_profiled.csv           # GPU metrics during profiling
│
├── twinb_baseline_profile.nsys-rep    # Nsight Systems report (MAIN FILE)
├── cuda_api_summary.csv               # CUDA API statistics
├── gpu_kernel_summary.csv             # GPU kernel statistics
├── memory_operation_summary.csv       # Memory transfer statistics
├── nvtx_summary.csv                   # NVTX marker statistics
│
├── mesa_agent_results_baseline.csv    # Agent simulation data
├── mesa_agent_results_profiled.csv    # Agent simulation data (profiled)
├── mesa_out_result_baseline/          # Mesa model outputs
├── eplusout.csv                        # EnergyPlus outputs
│
├── baseline_analysis_summary.json     # Automated analysis results
├── analyze_baseline.py                # Analysis script
└── monitor_gpu.sh                     # GPU monitoring script
```

### Key Bottlenecks to Investigate

Based on DREAM'26 paper findings:

1. **cudaStreamSynchronize Overhead**
   - Expected: ~66% of CUDA API time
   - Location: `cuda_api_summary.csv`
   - Impact: Blocks GPU-CPU concurrency

2. **NCCL Communication Dominance**
   - Expected: AllGather ~32.7%, AllReduce ~31.7% of GPU kernel time
   - Total communication: ~64.4% of GPU kernel time
   - Location: `gpu_kernel_summary.csv`
   - Impact: GPU spends more time syncing than computing

3. **Small Memory Transfers**
   - Expected: Average ~37.5 bytes for H2D transfers
   - Total: ~2.95 GB (2.32 GB D2H, 0.32 GB D2D)
   - Location: `memory_operation_summary.csv`
   - Impact: Poor PCIe bandwidth utilization

4. **CPU Workload Imbalance**
   - Expected: Primary process ~97%, secondary ~2%
   - Location: Simulation logs
   - Impact: Underutilized distributed resources

5. **Host-Side Blocking**
   - Expected: Significant time in pthread/poll operations
   - Location: OS runtime stats in Nsight GUI
   - Impact: CPU threads idle, waiting for GPU

### Analyzing Results

#### 1. Quick Command-Line Analysis

```bash
cd experiment2_results/baseline_XXXXX

# View summary
cat EXPERIMENT_SUMMARY.md

# Check runtime
cat baseline_runtime.txt

# View automated analysis
cat baseline_analysis_summary.json | python -m json.tool

# Check GPU power consumption
awk -F',' 'NR>1 {sum+=$3; count++} END {print "Avg Power:", sum/count, "W"}' gpu_metrics_baseline.csv
```

#### 2. Nsight Systems GUI (Recommended)

```bash
# Download .nsys-rep file to local machine
scp user@lanta:/path/to/experiment2_results/baseline_XXXXX/twinb_baseline_profile.nsys-rep .

# Open in Nsight Systems (on local machine with GUI)
nsys-ui twinb_baseline_profile.nsys-rep
```

**What to look for in Nsight GUI:**
- Timeline view: Identify gaps and synchronization points
- CUDA API calls: cudaStreamSynchronize frequency and duration
- GPU kernels: NCCL operations (AllGather, AllReduce) vs. compute kernels
- Memory operations: Transfer sizes and patterns
- OS runtime: pthread blocking patterns

#### 3. Python Script Analysis

```bash
# Run analysis script
cd dream26
python analyze_baseline.py ../experiment2_results/baseline_XXXXX

# Replace XXXXX with your actual job ID
```

The script provides:
- GPU metrics visualization
- Agent simulation analysis
- Profiling data extraction
- Energy consumption calculation
- Comparison with DREAM'26 metrics
- Optimization recommendations

---

## Expected Outcomes

### Experiment 1 Outcomes
✓ Hardware performance baselines established
✓ GPU compute capability quantified (GFLOPS)
✓ Memory bandwidth characterized (H2D, D2H, D2D)
✓ NCCL communication overhead measured
✓ Platform-specific constraints identified

### Experiment 2 Outcomes
✓ Baseline simulation runtime established
✓ CPU-GPU synchronization bottlenecks identified
✓ NCCL communication patterns analyzed
✓ Memory transfer inefficiencies quantified
✓ Energy consumption measured
✓ Comparison with DREAM'26 paper completed
✓ Optimization opportunities identified

---

## Comparison with DREAM'26 Paper

### Target Metrics to Validate

| Metric | DREAM'26 Paper | Your Results |
|--------|----------------|--------------|
| cudaStreamSynchronize overhead | 66.3% of CUDA API time | Check `cuda_api_summary.csv` |
| NCCL AllGather | 32.7% of GPU kernel time | Check `gpu_kernel_summary.csv` |
| NCCL AllReduce | 31.7% of GPU kernel time | Check `gpu_kernel_summary.csv` |
| Total NCCL communication | 64.4% of GPU kernel time | Sum of above |
| Avg H2D transfer size | 37.5 bytes | Check `memory_operation_summary.csv` |
| Total data transferred | 2.95 GB | Sum from memory summary |
| Primary CPU process | 96.99% CPU | Check simulation logs |
| Secondary CPU process | 2.02% CPU | Check simulation logs |
| GPU kernel runtime | 25.2 ms (4,046 instances) | Check kernel summary |

---

## Troubleshooting

### Common Issues

**Job doesn't start:**
```bash
# Check queue position
squeue -u $USER

# Check resource availability
sinfo -p gpu

# Check account allocation
sacctmgr show assoc where user=$USER
```

**Out of memory:**
- Reduce agent count in `agents.json`
- Reduce simulation steps in `config.yaml`
- Request more memory in SLURM script

**NCCL timeout:**
- Increase `TORCH_NCCL_TIMEOUT` in script
- Check network connectivity between GPUs

**EnergyPlus errors:**
- Check `outEnergyPlusBoonchoo/eplusout.err`
- Verify IDF file path and weather file
- Ensure zone names match between IDF and config.yaml

**Nsight Systems not found:**
```bash
module load nsight-systems  # If available
# Or download from NVIDIA website
```

---

## Next Steps After Baseline

1. **Analyze Results**
   - Run Jupyter notebook analysis
   - Open .nsys-rep in Nsight Systems GUI
   - Compare metrics with DREAM'26 paper

2. **Identify Top Bottlenecks**
   - Rank bottlenecks by impact
   - Prioritize optimization efforts

3. **Design Optimization Experiments**
   - Test reduced synchronization frequency
   - Implement batched memory transfers
   - Evaluate async communication patterns
   - Test different GPU configurations

4. **Implement Energy-Aware Policies**
   - As described in DREAM'26 paper:
     - Minimum room activation criteria
     - Extended temperature ranges
     - Mid-term HVAC breaks
     - Early HVAC shutdown

5. **Validate Optimizations**
   - Re-run with optimizations
   - Compare against baseline
   - Measure energy savings

---

## References

- DREAM'26 Conference Submission: "A Co-Simulation Framework for Building Energy Management as a Testbed for Energy-Aware Data Movement Analysis"
- NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems
- PyTorch DDP: https://pytorch.org/docs/stable/distributed.html
- EnergyPlus: https://energyplus.net/
- Mesa ABM: https://mesa.readthedocs.io/

---

## Contact & Support

For issues or questions:
1. Check experiment logs in `logs/` directory
2. Review error files (`.err`)
3. Consult CLAUDE.md for architecture details
4. Review DREAM'26 paper for methodology

---

*Last updated: 2025-01-08*
