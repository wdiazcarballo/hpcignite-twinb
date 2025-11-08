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
