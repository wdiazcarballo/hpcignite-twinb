#!/bin/bash
#SBATCH --job-name=exp1-platform
#SBATCH --output=exp1_%j.out
#SBATCH --error=exp1_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --mem=64G
#SBATCH -A lt200291

# ==============================================================================
# EXPERIMENT 1: Platform Capability Profiling at Runtime
# ==============================================================================
# Purpose: Benchmark HPC platform capabilities to understand:
#   - GPU compute capability and memory bandwidth
#   - CPU-GPU data transfer performance
#   - NCCL communication overhead
#   - Memory availability and allocation patterns
#   - Baseline performance metrics for comparison
#
# Resources: 2 GPUs, 4 CPUs per GPU, 64GB RAM
# ==============================================================================

echo "=========================================="
echo "EXPERIMENT 1: Platform Capability Profiling"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

# ---------------------------
# Create output directory (simple structure)
# ---------------------------
RESULTS_DIR=exp1_${SLURM_JOB_ID}
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
pip list | grep -E "torch|numpy|pandas|PyYAML" || echo "Some packages not found"
echo ""

# ---------------------------
# Test 1: GPU Compute Capability
# ---------------------------
echo "=========================================="
echo "TEST 1: GPU Compute Capability Test"
echo "=========================================="

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

cd $RESULTS_DIR
python test_gpu_compute.py || echo "ERROR: GPU compute test failed"
cd ..

echo ""

# ---------------------------
# Test 2: GPU Memory Bandwidth
# ---------------------------
echo "=========================================="
echo "TEST 2: GPU Memory Bandwidth Test"
echo "=========================================="

cat > $RESULTS_DIR/test_memory_bandwidth.py << 'EOF'
import torch
import time
import json

def test_memory_bandwidth():
    """Test GPU memory bandwidth with data transfers"""
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

cd $RESULTS_DIR
python test_memory_bandwidth.py || echo "ERROR: Memory bandwidth test failed"
cd ..

echo ""

# ---------------------------
# Test 3: DDP Communication Overhead
# ---------------------------
echo "=========================================="
echo "TEST 3: DDP Communication Overhead Test"
echo "=========================================="

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

cd $RESULTS_DIR
torchrun --standalone --nproc_per_node=2 test_ddp_communication.py || echo "ERROR: DDP communication test failed (torchrun not available)"
cd ..

echo ""

# ---------------------------
# Test 4: Monitor GPU metrics during idle and load
# ---------------------------
echo "=========================================="
echo "TEST 4: GPU Metrics Collection"
echo "=========================================="

# Collect idle state
nvidia-smi --query-gpu=timestamp,index,power.draw,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv > $RESULTS_DIR/gpu_metrics_idle.csv
echo "Idle GPU metrics collected"

# Collect under load
nvidia-smi --query-gpu=timestamp,index,power.draw,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv -l 1 -f $RESULTS_DIR/gpu_metrics_load.csv &
GPU_MON_PID=$!

python $RESULTS_DIR/test_gpu_compute.py > /dev/null 2>&1

sleep 2
kill $GPU_MON_PID 2>/dev/null || true
echo "Load GPU metrics collected"

echo ""

# ---------------------------
# Summary
# ---------------------------
echo "=========================================="
echo "EXPERIMENT 1 COMPLETE"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files generated:"
ls -lh $RESULTS_DIR/
echo ""
echo "Completed: $(date)"
echo "=========================================="
