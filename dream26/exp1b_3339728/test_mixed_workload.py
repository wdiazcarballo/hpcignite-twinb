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
