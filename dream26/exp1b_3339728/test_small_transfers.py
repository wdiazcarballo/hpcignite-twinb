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
