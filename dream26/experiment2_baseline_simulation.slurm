#!/bin/bash
#SBATCH --job-name=exp2-baseline
#SBATCH --output=exp2_%j.out
#SBATCH --error=exp2_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --mem=64G
#SBATCH -A lt200291

# ==============================================================================
# EXPERIMENT 2: Baseline Twin-B Simulation with Profiling
# ==============================================================================
# Purpose: Run baseline building simulation to establish performance metrics:
#   - Identify CPU-GPU synchronization bottlenecks
#   - Measure NCCL communication overhead (AllGather, AllReduce)
#   - Analyze memory transfer patterns (H2D, D2H, D2D)
#   - Profile cudaStreamSynchronize overhead
#   - Measure energy consumption during data exchange
#   - Establish baseline for comparison with optimized scenarios
#
# Based on DREAM'26 submission - Regular semester weekday scenario
# Resources: 2 GPUs, 4 CPUs per GPU, 64GB RAM
# ==============================================================================

echo "=========================================="
echo "EXPERIMENT 2: Baseline Twin-B Simulation"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

# ---------------------------
# Project folder
# ---------------------------
PROJECT_DIR=/project/lt200291-ignite/Project_chomwong/project

# ---------------------------
# Create output directory (simple structure in dream26)
# ---------------------------
RESULTS_DIR=$PROJECT_DIR/dream26/exp2_${SLURM_JOB_ID}
mkdir -p $RESULTS_DIR

# ---------------------------
# Load environment
# ---------------------------
module load cray-python/3.10.10

# Activate shared virtual environment
VENV_PATH=/project/lt200291-ignite/Project_chomwong/.venv
if [ -f $VENV_PATH/bin/activate ]; then
    source $VENV_PATH/bin/activate
    echo "Activated shared virtual environment from $VENV_PATH"
else
    echo "ERROR: Shared virtual environment not found at $VENV_PATH/bin/activate"
    echo "Please ensure pwongta has set up the shared environment."
    exit 1
fi

# ---------------------------
# EnergyPlus paths
# ---------------------------
EP_HOME=/project/lt200291-ignite/Project_chomwong/energyplus/EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64
export PATH=$EP_HOME:$PATH
export PYTHONPATH=$PYTHONPATH:$EP_HOME
export ENERGYPLUS_EXE=$EP_HOME/energyplus

# ---------------------------
# Enable detailed profiling
# ---------------------------
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
export TORCH_NCCL_TIMEOUT=1200
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ---------------------------
# Change to project directory
# ---------------------------
cd $PROJECT_DIR

# ---------------------------
# Backup original config
# ---------------------------
cp config.yaml $RESULTS_DIR/config_baseline.yaml
cp agents.json $RESULTS_DIR/agents_baseline.json

# ---------------------------
# System information
# ---------------------------
echo ""
echo "========== SYSTEM INFORMATION =========="
echo "Hostname: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo ""

nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
echo ""

# ---------------------------
# Create GPU monitoring script
# ---------------------------
cat > $RESULTS_DIR/monitor_gpu.sh << 'EOF'
#!/bin/bash
OUTPUT_FILE=$1
INTERVAL=${2:-1}

echo "timestamp,gpu_id,power_draw_w,memory_used_mb,memory_total_mb,gpu_utilization_pct,temperature_c" > $OUTPUT_FILE

while true; do
    nvidia-smi --query-gpu=timestamp,index,power.draw,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits >> $OUTPUT_FILE
    sleep $INTERVAL
done
EOF

chmod +x $RESULTS_DIR/monitor_gpu.sh

# ---------------------------
# Part 1: Run WITHOUT profiling (for timing baseline)
# ---------------------------
echo "=========================================="
echo "PART 1: Baseline Run (No Profiling)"
echo "=========================================="

# Start GPU monitoring
$RESULTS_DIR/monitor_gpu.sh $RESULTS_DIR/gpu_metrics_baseline.csv 1 &
GPU_MON_PID=$!

# Record start time
START_TIME=$(date +%s)

# Run simulation
echo "Running baseline simulation..."
torchrun --standalone --nproc_per_node=2 main.py 2>&1 | tee $RESULTS_DIR/simulation_baseline.log

# Record end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Stop GPU monitoring
kill $GPU_MON_PID 2>/dev/null || true

echo "Baseline simulation completed in ${ELAPSED} seconds"
echo $ELAPSED > $RESULTS_DIR/baseline_runtime.txt

# Save results
if [ -f "mesa_agent_results.csv" ]; then
    cp mesa_agent_results.csv $RESULTS_DIR/mesa_agent_results_baseline.csv
    echo "Agent results saved"
fi

if [ -d "mesa_out_result" ]; then
    cp -r mesa_out_result $RESULTS_DIR/mesa_out_result_baseline
    echo "Mesa output results saved"
fi

if [ -d "outEnergyPlusBoonchoo" ]; then
    cp outEnergyPlusBoonchoo/*.csv $RESULTS_DIR/ 2>/dev/null || true
    echo "EnergyPlus output saved"
fi

echo ""

# ---------------------------
# Part 2: Run WITH NVIDIA Nsight Systems profiling
# ---------------------------
echo "=========================================="
echo "PART 2: Profiled Run (Nsight Systems)"
echo "=========================================="

# Clean previous outputs
rm -f mesa_agent_results.csv
rm -rf mesa_out_result
rm -rf outEnergyPlusBoonchoo

# Start GPU monitoring for profiled run
$RESULTS_DIR/monitor_gpu.sh $RESULTS_DIR/gpu_metrics_profiled.csv 1 &
GPU_MON_PID=$!

# Run with Nsight Systems profiling
echo "Running profiled simulation with Nsight Systems..."

nsys profile \
    --output=$RESULTS_DIR/twinb_baseline_profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --sample=cpu \
    --cpuctxsw=none \
    --backtrace=dwarf \
    torchrun --standalone --nproc_per_node=2 main.py 2>&1 | tee $RESULTS_DIR/simulation_profiled.log

# Stop GPU monitoring
kill $GPU_MON_PID 2>/dev/null || true

# Save profiled results
if [ -f "mesa_agent_results.csv" ]; then
    cp mesa_agent_results.csv $RESULTS_DIR/mesa_agent_results_profiled.csv
fi

if [ -d "mesa_out_result" ]; then
    cp -r mesa_out_result $RESULTS_DIR/mesa_out_result_profiled
fi

echo "Nsight Systems profiling completed"
echo ""

# ---------------------------
# Part 3: Extract profiling statistics
# ---------------------------
echo "=========================================="
echo "PART 3: Extract Profiling Statistics"
echo "=========================================="

# Generate Nsight Systems report (if nsys stats is available)
if command -v nsys &> /dev/null; then
    echo "Generating Nsight Systems statistics reports..."

    # CUDA API stats
    nsys stats --report cuda_api_sum \
        --format csv \
        --output $RESULTS_DIR/cuda_api_summary \
        $RESULTS_DIR/twinb_baseline_profile.nsys-rep 2>/dev/null || true

    # GPU kernel stats
    nsys stats --report cuda_gpu_kern_sum \
        --format csv \
        --output $RESULTS_DIR/gpu_kernel_summary \
        $RESULTS_DIR/twinb_baseline_profile.nsys-rep 2>/dev/null || true

    # Memory operation stats
    nsys stats --report cuda_gpu_mem_size_sum \
        --format csv \
        --output $RESULTS_DIR/memory_operation_summary \
        $RESULTS_DIR/twinb_baseline_profile.nsys-rep 2>/dev/null || true

    # NVTX stats
    nsys stats --report nvtx_sum \
        --format csv \
        --output $RESULTS_DIR/nvtx_summary \
        $RESULTS_DIR/twinb_baseline_profile.nsys-rep 2>/dev/null || true

    echo "Statistics reports generated"
fi

echo ""

# ---------------------------
# Part 4: Analyze results with Python
# ---------------------------
echo "=========================================="
echo "PART 4: Quick Analysis"
echo "=========================================="

cat > $RESULTS_DIR/analyze_baseline.py << 'EOF'
import pandas as pd
import json
import os

def analyze_gpu_metrics(csv_file):
    """Analyze GPU metrics from monitoring"""
    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)

    # Clean up column names (remove spaces)
    df.columns = df.columns.str.strip()

    stats = {
        'avg_power_w': df['power_draw_w'].mean(),
        'max_power_w': df['power_draw_w'].max(),
        'avg_memory_used_mb': df['memory_used_mb'].mean(),
        'max_memory_used_mb': df['memory_used_mb'].max(),
        'avg_gpu_util_pct': df['gpu_utilization_pct'].mean(),
        'max_gpu_util_pct': df['gpu_utilization_pct'].max(),
        'avg_temperature_c': df['temperature_c'].mean(),
        'max_temperature_c': df['temperature_c'].max(),
        'total_samples': len(df)
    }

    return stats

def analyze_agent_results(csv_file):
    """Analyze agent simulation results"""
    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)

    stats = {
        'total_agent_records': len(df),
        'unique_agents': df['agent_id'].nunique() if 'agent_id' in df.columns else 0,
        'unique_zones': df['room'].nunique() if 'room' in df.columns else 0,
        'total_steps': df['step'].max() if 'step' in df.columns else 0,
        'avg_comfort_level': df['comfort_level'].mean() if 'comfort_level' in df.columns else None,
        'ac_usage_rate': df['using_ac'].mean() if 'using_ac' in df.columns else None,
        'avg_current_temp': df['current_temp'].mean() if 'current_temp' in df.columns else None,
        'avg_preferred_temp': df['preferred_temp'].mean() if 'preferred_temp' in df.columns else None
    }

    return stats

def main():
    results = {
        'baseline_run': {},
        'profiled_run': {}
    }

    # Analyze baseline run
    print("Analyzing baseline run...")
    gpu_stats = analyze_gpu_metrics('gpu_metrics_baseline.csv')
    if gpu_stats:
        results['baseline_run']['gpu_metrics'] = gpu_stats
        print(f"  Avg Power: {gpu_stats['avg_power_w']:.2f}W")
        print(f"  Avg GPU Utilization: {gpu_stats['avg_gpu_util_pct']:.2f}%")

    agent_stats = analyze_agent_results('mesa_agent_results_baseline.csv')
    if agent_stats:
        results['baseline_run']['agent_simulation'] = agent_stats
        print(f"  Total agent records: {agent_stats['total_agent_records']}")
        print(f"  Unique agents: {agent_stats['unique_agents']}")
        print(f"  AC usage rate: {agent_stats['ac_usage_rate']*100:.2f}%")

    # Analyze profiled run
    print("\nAnalyzing profiled run...")
    gpu_stats = analyze_gpu_metrics('gpu_metrics_profiled.csv')
    if gpu_stats:
        results['profiled_run']['gpu_metrics'] = gpu_stats
        print(f"  Avg Power: {gpu_stats['avg_power_w']:.2f}W")
        print(f"  Avg GPU Utilization: {gpu_stats['avg_gpu_util_pct']:.2f}%")

    agent_stats = analyze_agent_results('mesa_agent_results_profiled.csv')
    if agent_stats:
        results['profiled_run']['agent_simulation'] = agent_stats

    # Save results
    with open('baseline_analysis_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nAnalysis complete. Results saved to baseline_analysis_summary.json")

if __name__ == "__main__":
    main()
EOF

cd $RESULTS_DIR
python analyze_baseline.py
cd -

echo ""

# ---------------------------
# Generate summary report
# ---------------------------
echo "=========================================="
echo "EXPERIMENT 2 SUMMARY"
echo "=========================================="

cat > $RESULTS_DIR/EXPERIMENT_SUMMARY.md << EOF
# Experiment 2: Baseline Twin-B Simulation Results

**Job ID:** $SLURM_JOB_ID
**Date:** $(date)
**Scenario:** Regular semester weekday (baseline)

## Configuration
- Nodes: $SLURM_JOB_NUM_NODES
- Tasks: $SLURM_NTASKS
- CPUs per task: $SLURM_CPUS_PER_TASK
- GPUs: 2
- Memory: 64GB

## Simulation Parameters
- See: config_baseline.yaml
- Agent configuration: agents_baseline.json

## Results Files

### Performance Data
- \`simulation_baseline.log\` - Baseline run output
- \`simulation_profiled.log\` - Profiled run output
- \`baseline_runtime.txt\` - Total execution time

### GPU Metrics
- \`gpu_metrics_baseline.csv\` - GPU metrics during baseline run
- \`gpu_metrics_profiled.csv\` - GPU metrics during profiled run

### Profiling Data
- \`twinb_baseline_profile.nsys-rep\` - Nsight Systems report (open with Nsight Systems GUI)
- \`cuda_api_summary.csv\` - CUDA API call statistics
- \`gpu_kernel_summary.csv\` - GPU kernel execution statistics
- \`memory_operation_summary.csv\` - Memory transfer statistics
- \`nvtx_summary.csv\` - NVTX marker statistics

### Simulation Results
- \`mesa_agent_results_baseline.csv\` - Agent-level simulation data
- \`mesa_out_result_baseline/\` - Mesa model output
- EnergyPlus output files (*.csv)

### Analysis
- \`baseline_analysis_summary.json\` - Automated analysis results
- \`analyze_baseline.py\` - Analysis script

## Key Bottlenecks to Investigate

Based on DREAM'26 paper findings, focus on:

1. **cudaStreamSynchronize Overhead**
   - Expected: ~66% of CUDA API time
   - Check: cuda_api_summary.csv

2. **NCCL Communication**
   - Expected: AllGather ~32.7%, AllReduce ~31.7% of GPU kernel time
   - Check: gpu_kernel_summary.csv for ncclDevKernel_* entries

3. **Memory Transfer Patterns**
   - Expected: Many small transfers (avg ~37.5 bytes H2D)
   - Check: memory_operation_summary.csv

4. **CPU Workload Imbalance**
   - Expected: Primary process ~97%, secondary ~2%
   - Check: simulation logs for CPU usage patterns

5. **Host-Side Blocking**
   - Expected: Significant time in pthread/poll operations
   - Check: OS runtime stats in Nsight Systems GUI

## Next Steps

1. Open \`twinb_baseline_profile.nsys-rep\` in Nsight Systems GUI
2. Analyze timeline for synchronization patterns
3. Review JSON summary for quantitative metrics
4. Compare against DREAM'26 paper metrics
5. Identify optimization opportunities

## Viewing Nsight Systems Report

\`\`\`bash
# On local machine with Nsight Systems installed:
nsys-ui twinb_baseline_profile.nsys-rep
\`\`\`

EOF

cat $RESULTS_DIR/EXPERIMENT_SUMMARY.md

echo ""
echo "=========================================="
echo "All results saved to: $RESULTS_DIR"
echo ""
echo "Files generated:"
ls -lh $RESULTS_DIR/ | head -20
echo ""
echo "Completed: $(date)"
echo "=========================================="
