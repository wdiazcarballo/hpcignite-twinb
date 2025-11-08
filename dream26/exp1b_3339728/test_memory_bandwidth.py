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
