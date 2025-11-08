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
