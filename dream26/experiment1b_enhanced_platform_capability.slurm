#!/bin/bash
#SBATCH --job-name=exp1b-platform-enhanced
#SBATCH --output=exp1b_%j/exp1b_%j.out
#SBATCH --error=exp1b_%j/exp1b_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --mem=64G
#SBATCH -A lt200291

# ==============================================================================
# EXPERIMENT 1B: Enhanced Platform Capability Profiling with Energy Analysis
# ==============================================================================
# Purpose: Address Section 4 requirements for energy consumption baseline:
#   - GPU power monitoring during all operations
#   - Small transfer overhead (1B to 1MB, including 37.5 byte transfers)
#   - Energy cost analysis (joules per operation)
#   - Mixed workload profiling (compute + communication + transfers)
#
# Enhancements over Experiment 1:
#   - Fixed GPU power monitoring (longer duration, proper barriers)
#   - Small transfer range: 1B, 10B, 37.5B, 100B, 1KB, 10KB, 100KB, 1MB
#   - Energy profiling: power × time for each operation type
#   - Mixed workload tests: realistic Twin-B patterns
# ==============================================================================

echo "=================================================="
echo "EXPERIMENT 1B: Enhanced Platform Capability"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=================================================="

# ---------------------------
# Create output directory
# ---------------------------
RESULTS_DIR=exp1b_${SLURM_JOB_ID}
# Note: Directory must already exist for SLURM to write .out/.err files there

# ---------------------------
# Load environment (use system-provided PyTorch)
# ---------------------------
module load Mamba

echo "Loaded modules:"
echo "  - Mamba/23.11.0-0"
echo ""

# Activate system PyTorch conda environment
echo "Activating system PyTorch environment..."
source activate pytorch-2.2.2

if [ $? -eq 0 ]; then
    echo "✓ Activated pytorch-2.2.2 conda environment"
else
    echo "ERROR: Failed to activate pytorch-2.2.2 environment"
    echo "Available environments:"
    conda env list
    exit 1
fi

# ---------------------------
# Environment info collection
# ---------------------------
echo ""
echo "========== SYSTEM INFORMATION =========="
echo "Hostname: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total CPUs: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "GPUs allocated: $SLURM_GPUS"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo ""

# ---------------------------
# CPU Information
# ---------------------------
echo "========== CPU INFORMATION =========="
lscpu | grep -E "Model name|Socket|Core|Thread|CPU MHz|Cache"
echo ""

# ---------------------------
# GPU Information
# ---------------------------
echo "========== GPU INFORMATION =========="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv
echo ""

# ---------------------------
# Memory Information
# ---------------------------
echo "========== MEMORY INFORMATION =========="
free -h
echo ""

# ---------------------------
# Python and Library Versions
# ---------------------------
echo "========== PYTHON ENVIRONMENT =========="
python --version
echo ""
echo "Key packages:"
conda list | grep -E "torch|numpy|pandas|pyyaml" || echo "Some packages not found"
echo ""
echo "PyTorch verification:"
python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'  Device count: {torch.cuda.device_count()}')"
echo ""

# ---------------------------
# Helper function: Start GPU monitoring
# ---------------------------
start_gpu_monitoring() {
    local output_file=$1
    local label=$2

    echo "timestamp,gpu_id,power_draw_w,memory_used_mb,memory_total_mb,gpu_utilization_pct,temperature_c" > $output_file

    nvidia-smi --query-gpu=timestamp,index,power.draw,memory.used,memory.total,utilization.gpu,temperature.gpu \
               --format=csv,noheader,nounits -l 1 >> $output_file &

    GPU_MON_PID=$!
    echo "Started GPU monitoring (PID: $GPU_MON_PID) -> $output_file"
}

stop_gpu_monitoring() {
    if [ ! -z "$GPU_MON_PID" ]; then
        sleep 2  # Ensure final samples are captured
        kill $GPU_MON_PID 2>/dev/null || true
        wait $GPU_MON_PID 2>/dev/null || true
        echo "Stopped GPU monitoring (PID: $GPU_MON_PID)"
        GPU_MON_PID=""
    fi
}

# ---------------------------
# Test 1: GPU Compute Capability
# ---------------------------
echo "=================================================="
echo "TEST 1: GPU Compute Performance"
echo "=================================================="

cat > $RESULTS_DIR/test_gpu_compute.py << 'EOF'
import torch
import time
import json

def test_gpu_compute():
    """Test GPU compute performance with matrix multiplication"""
    results = {}

    if torch.cuda.is_available():
        device = torch.device("cuda")
        results['cuda_available'] = True
        results['device_count'] = torch.cuda.device_count()
        results['device_name'] = torch.cuda.get_device_name(0)
        results['cuda_version'] = torch.version.cuda

        # Test different matrix sizes
        sizes = [1024, 2048, 4096, 8192]
        compute_times = {}

        for size in sizes:
            # Warmup
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            torch.mm(a, b)
            torch.cuda.synchronize()

            # Actual test
            start = time.time()
            for _ in range(10):
                c = torch.mm(a, b)
            torch.cuda.synchronize()
            end = time.time()

            avg_time = (end - start) / 10
            flops = 2 * size**3  # Matrix multiplication FLOPs
            gflops = (flops / avg_time) / 1e9

            compute_times[f'size_{size}'] = {
                'avg_time_sec': avg_time,
                'gflops': gflops
            }
            print(f"Matrix size {size}x{size}: {avg_time:.4f}s, {gflops:.2f} GFLOPS")

        results['compute_performance'] = compute_times
    else:
        results['cuda_available'] = False
        print("CUDA not available")

    return results

if __name__ == "__main__":
    results = test_gpu_compute()
    with open('gpu_compute_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to gpu_compute_results.json")
EOF

start_gpu_monitoring "$RESULTS_DIR/gpu_metrics_compute.csv" "compute"
cd $RESULTS_DIR
python test_gpu_compute.py
cd ..
stop_gpu_monitoring

echo ""

# ---------------------------
# Test 2: Small Transfer Overhead (NEW - addresses Section 4)
# ---------------------------
echo "=================================================="
echo "TEST 2: Small Transfer Overhead Analysis"
echo "=================================================="

cat > $RESULTS_DIR/test_small_transfers.py << 'EOF'
import torch
import time
import json
import numpy as np

def test_small_transfers():
    """Test overhead of small transfers matching Section 4 analysis"""
    results = {}

    if not torch.cuda.is_available():
        results['error'] = 'CUDA not available'
        return results

    device = torch.device("cuda")

    # Test sizes in bytes (including 37.5 byte average from Section 4)
    # Convert bytes to number of float32 elements (4 bytes each)
    test_sizes_bytes = [1, 10, 37, 100, 1024, 10240, 102400, 1048576]  # 1B to 1MB
    transfer_results = {}

    print(f"{'Size (bytes)':<15} {'Elements':<10} {'H2D (us)':<12} {'D2H (us)':<12} {'D2D (us)':<12} {'H2D (MB/s)':<12}")
    print("-" * 80)

    for size_bytes in test_sizes_bytes:
        # Calculate number of float32 elements
        num_elements = max(1, size_bytes // 4)
        actual_bytes = num_elements * 4

        # Host to Device
        host_tensor = torch.randn(num_elements)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(10):
            device_tensor = host_tensor.to(device)
            torch.cuda.synchronize()

        # Measure H2D
        num_iterations = 1000
        start = time.time()
        for _ in range(num_iterations):
            device_tensor = host_tensor.to(device)
            torch.cuda.synchronize()
        h2d_time = (time.time() - start) / num_iterations
        h2d_bandwidth_mbps = (actual_bytes / h2d_time) / (1024 * 1024)

        # Device to Host
        torch.cuda.synchronize()
        num_iterations = 1000
        start = time.time()
        for _ in range(num_iterations):
            result_tensor = device_tensor.cpu()
            torch.cuda.synchronize()
        d2h_time = (time.time() - start) / num_iterations
        d2h_bandwidth_mbps = (actual_bytes / d2h_time) / (1024 * 1024)

        # Device to Device
        torch.cuda.synchronize()
        num_iterations = 1000
        start = time.time()
        for _ in range(num_iterations):
            device_tensor2 = device_tensor.clone()
            torch.cuda.synchronize()
        d2d_time = (time.time() - start) / num_iterations
        d2d_bandwidth_mbps = (actual_bytes / d2d_time) / (1024 * 1024)

        transfer_results[f'{size_bytes}B'] = {
            'bytes': actual_bytes,
            'elements': num_elements,
            'host_to_device_us': h2d_time * 1e6,
            'device_to_host_us': d2h_time * 1e6,
            'device_to_device_us': d2d_time * 1e6,
            'host_to_device_mbps': h2d_bandwidth_mbps,
            'device_to_host_mbps': d2h_bandwidth_mbps,
            'device_to_device_mbps': d2d_bandwidth_mbps
        }

        print(f"{size_bytes:<15} {num_elements:<10} {h2d_time*1e6:<12.2f} {d2h_time*1e6:<12.2f} {d2d_time*1e6:<12.2f} {h2d_bandwidth_mbps:<12.2f}")

    results['small_transfer_overhead'] = transfer_results
    return results

if __name__ == "__main__":
    results = test_small_transfers()
    with open('small_transfer_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to small_transfer_results.json")
EOF

start_gpu_monitoring "$RESULTS_DIR/gpu_metrics_small_transfers.csv" "small_transfers"
cd $RESULTS_DIR
python test_small_transfers.py
cd ..
stop_gpu_monitoring

echo ""

# ---------------------------
# Test 3: Large Memory Bandwidth (from original Exp1)
# ---------------------------
echo "=================================================="
echo "TEST 3: Large Memory Bandwidth"
echo "=================================================="

cat > $RESULTS_DIR/test_memory_bandwidth.py << 'EOF'
import torch
import time
import json

def test_memory_bandwidth():
    """Test GPU memory bandwidth with large data transfers"""
    results = {}

    if not torch.cuda.is_available():
        results['error'] = 'CUDA not available'
        return results

    device = torch.device("cuda")

    # Test sizes in MB
    sizes_mb = [1, 10, 100, 500, 1000]
    transfer_results = {}

    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        size_elements = size_bytes // 4  # float32 = 4 bytes

        # Host to Device
        host_tensor = torch.randn(size_elements)
        torch.cuda.synchronize()
        start = time.time()
        device_tensor = host_tensor.to(device)
        torch.cuda.synchronize()
        h2d_time = time.time() - start
        h2d_bandwidth = size_mb / h2d_time  # MB/s

        # Device to Host
        torch.cuda.synchronize()
        start = time.time()
        result_tensor = device_tensor.cpu()
        torch.cuda.synchronize()
        d2h_time = time.time() - start
        d2h_bandwidth = size_mb / d2h_time  # MB/s

        # Device to Device
        torch.cuda.synchronize()
        start = time.time()
        device_tensor2 = device_tensor.clone()
        torch.cuda.synchronize()
        d2d_time = time.time() - start
        d2d_bandwidth = size_mb / d2d_time  # MB/s

        transfer_results[f'{size_mb}MB'] = {
            'host_to_device_mbps': h2d_bandwidth,
            'device_to_host_mbps': d2h_bandwidth,
            'device_to_device_mbps': d2d_bandwidth
        }

        print(f"{size_mb}MB - H2D: {h2d_bandwidth:.2f} MB/s, D2H: {d2h_bandwidth:.2f} MB/s, D2D: {d2d_bandwidth:.2f} MB/s")

    results['memory_bandwidth'] = transfer_results
    return results

if __name__ == "__main__":
    results = test_memory_bandwidth()
    with open('memory_bandwidth_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to memory_bandwidth_results.json")
EOF

start_gpu_monitoring "$RESULTS_DIR/gpu_metrics_large_bandwidth.csv" "large_bandwidth"
cd $RESULTS_DIR
python test_memory_bandwidth.py
cd ..
stop_gpu_monitoring

echo ""

# ---------------------------
# Test 4: DDP Communication Overhead
# ---------------------------
echo "=================================================="
echo "TEST 4: DDP Communication Overhead"
echo "=================================================="

cat > $RESULTS_DIR/test_ddp_communication.py << 'EOF'
import os
import torch
import torch.distributed as dist
import time
import json

def init_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    else:
        backend = "gloo"

    device = torch.device(f"cuda:{local_rank}") if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(local_rank)

    return rank, world_size, device, backend

def test_nccl_operations():
    rank, world_size, device, backend = init_distributed()

    if rank == 0:
        print(f"Testing with backend={backend}, world_size={world_size}, device={device}")

    results = {
        'rank': rank,
        'world_size': world_size,
        'backend': backend,
        'device': str(device)
    }

    if world_size == 1:
        results['note'] = 'Single process, no communication test'
        return results

    # Test different tensor sizes
    sizes = [100, 1000, 10000, 100000]
    comm_results = {}

    for size in sizes:
        tensor = torch.randn(size, device=device)

        # Test AllGather
        if world_size > 1:
            gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.barrier()
            start = time.time()
            for _ in range(100):
                dist.all_gather(gathered, tensor)
            dist.barrier()
            allgather_time = (time.time() - start) / 100

            # Test AllReduce
            dist.barrier()
            start = time.time()
            for _ in range(100):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            dist.barrier()
            allreduce_time = (time.time() - start) / 100

            if rank == 0:
                comm_results[f'size_{size}'] = {
                    'allgather_time_ms': allgather_time * 1000,
                    'allreduce_time_ms': allreduce_time * 1000
                }
                print(f"Size {size}: AllGather={allgather_time*1000:.3f}ms, AllReduce={allreduce_time*1000:.3f}ms")

    if rank == 0:
        results['communication_overhead'] = comm_results

    if world_size > 1:
        dist.destroy_process_group()

    return results

if __name__ == "__main__":
    results = test_nccl_operations()
    if results['rank'] == 0:
        with open('ddp_communication_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to ddp_communication_results.json")
EOF

start_gpu_monitoring "$RESULTS_DIR/gpu_metrics_nccl.csv" "nccl"
cd $RESULTS_DIR
torchrun --standalone --nproc_per_node=2 test_ddp_communication.py
cd ..
stop_gpu_monitoring

echo ""

# ---------------------------
# Test 5: Mixed Workload Energy Profiling (NEW)
# ---------------------------
echo "=================================================="
echo "TEST 5: Mixed Workload Energy Profiling"
echo "=================================================="

cat > $RESULTS_DIR/test_mixed_workload.py << 'EOF'
import torch
import time
import json
import threading

def mixed_compute_and_transfer():
    """Simulate Twin-B pattern: compute + frequent small transfers"""
    results = {}

    if not torch.cuda.is_available():
        results['error'] = 'CUDA not available'
        return results

    device = torch.device("cuda")

    # Setup
    matrix_size = 4096
    small_transfer_size = 37  # bytes (matching Section 4)
    num_elements_small = max(1, small_transfer_size // 4)

    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    small_host = torch.randn(num_elements_small)

    # Test 1: Pure compute baseline
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    compute_only_time = time.time() - start

    # Test 2: Compute + frequent small H2D transfers (Twin-B pattern)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        c = torch.mm(a, b)
        small_device = small_host.to(device)  # Frequent small transfer
        torch.cuda.synchronize()
    mixed_time = time.time() - start

    # Test 3: Transfer-only baseline
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        small_device = small_host.to(device)
        torch.cuda.synchronize()
    transfer_only_time = time.time() - start

    overhead = mixed_time - compute_only_time - transfer_only_time

    results['mixed_workload'] = {
        'compute_only_ms': compute_only_time * 1000,
        'transfer_only_ms': transfer_only_time * 1000,
        'mixed_workload_ms': mixed_time * 1000,
        'overhead_ms': overhead * 1000,
        'overhead_pct': (overhead / compute_only_time) * 100 if compute_only_time > 0 else 0
    }

    print(f"Compute only:     {compute_only_time*1000:.2f} ms")
    print(f"Transfer only:    {transfer_only_time*1000:.2f} ms")
    print(f"Mixed workload:   {mixed_time*1000:.2f} ms")
    print(f"Overhead:         {overhead*1000:.2f} ms ({results['mixed_workload']['overhead_pct']:.1f}%)")

    return results

if __name__ == "__main__":
    results = mixed_compute_and_transfer()
    with open('mixed_workload_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to mixed_workload_results.json")
EOF

start_gpu_monitoring "$RESULTS_DIR/gpu_metrics_mixed.csv" "mixed_workload"
cd $RESULTS_DIR
python test_mixed_workload.py
cd ..
stop_gpu_monitoring

echo ""

# ---------------------------
# Test 6: Energy Cost Analysis (NEW)
# ---------------------------
echo "=================================================="
echo "TEST 6: Energy Cost Analysis"
echo "=================================================="

cat > $RESULTS_DIR/analyze_energy.py << 'EOF'
import pandas as pd
import json
import glob
from pathlib import Path

def analyze_energy_costs():
    """Calculate energy consumption (joules) for each operation type"""

    results = {
        'energy_analysis': {},
        'methodology': 'Energy (J) = Average Power (W) × Duration (s)'
    }

    # Find all GPU metrics files
    csv_files = glob.glob('gpu_metrics_*.csv')

    for csv_file in csv_files:
        operation_name = csv_file.replace('gpu_metrics_', '').replace('.csv', '')

        try:
            df = pd.read_csv(csv_file)

            if len(df) == 0:
                print(f"⚠️  {operation_name}: No data")
                continue

            # Clean column names
            df.columns = df.columns.str.strip()

            # Parse timestamps to calculate duration
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            duration_sec = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()

            # Calculate average power across both GPUs
            avg_power_w = df['power_draw_w'].mean()
            max_power_w = df['power_draw_w'].max()
            min_power_w = df['power_draw_w'].min()

            # Calculate energy
            energy_joules = avg_power_w * duration_sec
            energy_kwh = energy_joules / (3.6e6)  # Convert J to kWh

            results['energy_analysis'][operation_name] = {
                'duration_sec': duration_sec,
                'avg_power_w': avg_power_w,
                'max_power_w': max_power_w,
                'min_power_w': min_power_w,
                'energy_joules': energy_joules,
                'energy_kwh': energy_kwh,
                'samples': len(df)
            }

            print(f"✓ {operation_name:20s}: {avg_power_w:6.1f}W avg, {duration_sec:6.1f}s, {energy_joules:8.1f}J ({energy_kwh:.6f} kWh)")

        except Exception as e:
            print(f"✗ {operation_name}: Error - {e}")

    return results

if __name__ == "__main__":
    results = analyze_energy_costs()
    with open('energy_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Energy analysis saved to energy_analysis.json")
EOF

cd $RESULTS_DIR
python analyze_energy.py
cd ..

echo ""

# ---------------------------
# Test 7: GPU Metrics - Idle State
# ---------------------------
echo "=================================================="
echo "TEST 7: GPU Idle State Metrics"
echo "=================================================="

nvidia-smi --query-gpu=timestamp,index,power.draw,memory.used,memory.total,utilization.gpu,temperature.gpu \
           --format=csv > $RESULTS_DIR/gpu_metrics_idle.csv

echo "Idle GPU metrics collected"
echo ""

# ---------------------------
# Move output files into results directory
# ---------------------------
if [ -f "exp1b_${SLURM_JOB_ID}.out" ]; then
    mv exp1b_${SLURM_JOB_ID}.out $RESULTS_DIR/
fi
if [ -f "exp1b_${SLURM_JOB_ID}.err" ]; then
    mv exp1b_${SLURM_JOB_ID}.err $RESULTS_DIR/
fi

# ---------------------------
# Summary
# ---------------------------
echo "=================================================="
echo "EXPERIMENT 1B COMPLETE"
echo "=================================================="
echo "All results in: $RESULTS_DIR"
echo ""
echo "Files generated:"
ls -lh $RESULTS_DIR/
echo ""
echo "Key outputs:"
echo "  - gpu_compute_results.json         : Compute performance"
echo "  - small_transfer_results.json      : Small transfer overhead (1B-1MB)"
echo "  - memory_bandwidth_results.json    : Large bandwidth (1MB-1GB)"
echo "  - ddp_communication_results.json   : NCCL communication"
echo "  - mixed_workload_results.json      : Mixed workload analysis"
echo "  - energy_analysis.json             : Energy consumption by operation"
echo "  - gpu_metrics_*.csv                : Power monitoring logs"
echo ""
echo "Completed: $(date)"
echo "=================================================="
