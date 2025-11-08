#!/usr/bin/env python3
"""
Baseline Twin-B Experiment Analysis

This script analyzes the results from Experiment 2: Baseline Twin-B simulation with profiling.

Analysis Goals:
1. Identify CPU-GPU synchronization bottlenecks
2. Measure NCCL communication overhead
3. Analyze memory transfer patterns
4. Compare against DREAM'26 paper metrics
5. Establish baseline for future optimizations

Usage:
    python analyze_baseline.py <results_directory>

Example:
    python analyze_baseline.py ../experiment2_results/baseline_12345
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def analyze_gpu_metrics(results_dir):
    """Analyze GPU metrics from monitoring"""
    print_section("1. GPU METRICS ANALYSIS")

    baseline_file = results_dir / 'gpu_metrics_baseline.csv'
    profiled_file = results_dir / 'gpu_metrics_profiled.csv'

    if not baseline_file.exists():
        print("GPU metrics baseline file not found. Skipping.")
        return None

    # Load GPU metrics
    gpu_baseline = pd.read_csv(baseline_file)
    gpu_baseline.columns = gpu_baseline.columns.str.strip()

    print(f"GPU Metrics - Baseline Run")
    print(f"Samples collected: {len(gpu_baseline)}")
    print(f"Duration: ~{len(gpu_baseline)} seconds (~{len(gpu_baseline)/60:.1f} minutes)")
    print(f"\nAverage metrics:")
    print(f"  Power draw: {gpu_baseline['power_draw_w'].mean():.2f}W (max: {gpu_baseline['power_draw_w'].max():.2f}W)")
    print(f"  GPU utilization: {gpu_baseline['gpu_utilization_pct'].mean():.2f}%")
    print(f"  Memory used: {gpu_baseline['memory_used_mb'].mean():.2f}MB")
    print(f"  Temperature: {gpu_baseline['temperature_c'].mean():.2f}°C")

    # Plot GPU metrics over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for gpu_id in gpu_baseline['gpu_id'].unique():
        gpu_data = gpu_baseline[gpu_baseline['gpu_id'] == gpu_id]

        axes[0, 0].plot(gpu_data['power_draw_w'], label=f'GPU {gpu_id}', alpha=0.7)
        axes[0, 1].plot(gpu_data['gpu_utilization_pct'], label=f'GPU {gpu_id}', alpha=0.7)
        axes[1, 0].plot(gpu_data['memory_used_mb'], label=f'GPU {gpu_id}', alpha=0.7)
        axes[1, 1].plot(gpu_data['temperature_c'], label=f'GPU {gpu_id}', alpha=0.7)

    axes[0, 0].set_title('Power Draw Over Time')
    axes[0, 0].set_ylabel('Power (W)')
    axes[0, 0].legend()

    axes[0, 1].set_title('GPU Utilization Over Time')
    axes[0, 1].set_ylabel('Utilization (%)')
    axes[0, 1].legend()

    axes[1, 0].set_title('Memory Usage Over Time')
    axes[1, 0].set_ylabel('Memory (MB)')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].legend()

    axes[1, 1].set_title('Temperature Over Time')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].legend()

    plt.tight_layout()
    output_file = results_dir / 'gpu_metrics_timeline.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()

    return gpu_baseline


def analyze_agent_results(results_dir):
    """Analyze agent simulation results"""
    print_section("2. AGENT SIMULATION RESULTS")

    agents_file = results_dir / 'mesa_agent_results_baseline.csv'
    if not agents_file.exists():
        print("Agent results file not found. Skipping.")
        return None

    agents_df = pd.read_csv(agents_file)

    print("Agent Simulation Summary")
    print(f"Total records: {len(agents_df):,}")
    print(f"Unique agents: {agents_df['agent_id'].nunique()}")
    print(f"Unique zones: {agents_df['room'].nunique()}")
    print(f"Simulation steps: {agents_df['step'].max() + 1}")
    print(f"\nAgent type distribution:")
    print(agents_df['agent_type'].value_counts())
    print(f"\nComfort metrics:")
    print(f"  Average comfort level: {agents_df['comfort_level'].mean():.2f}")
    print(f"  AC usage rate: {agents_df['using_ac'].mean()*100:.2f}%")
    print(f"  Average current temp: {agents_df['current_temp'].mean():.2f}°C")
    print(f"  Average preferred temp: {agents_df['preferred_temp'].mean():.2f}°C")

    # Plot AC usage over time
    ac_usage_by_step = agents_df.groupby('step')['using_ac'].mean()

    plt.figure(figsize=(14, 5))
    plt.plot(ac_usage_by_step.index, ac_usage_by_step.values * 100)
    plt.title('AC Usage Rate Over Simulation Time')
    plt.xlabel('Step')
    plt.ylabel('AC Usage Rate (%)')
    plt.grid(True, alpha=0.3)
    output_file = results_dir / 'ac_usage_over_time.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nAC usage plot saved to: {output_file}")
    plt.close()

    # Temperature distribution by zone
    zone_temps = agents_df.groupby('room').agg({
        'current_temp': ['mean', 'std'],
        'using_ac': 'mean'
    }).round(2)

    zone_temps.columns = ['avg_temp', 'std_temp', 'ac_usage_rate']
    zone_temps = zone_temps.sort_values('ac_usage_rate', ascending=False)

    print("\nTop 10 zones by AC usage:")
    print(zone_temps.head(10))

    return agents_df


def analyze_profiling_data(results_dir):
    """Analyze profiling data from Nsight Systems"""
    print_section("3. PROFILING ANALYSIS")

    # CUDA API Summary
    print("\n3.1 CUDA API Summary")
    cuda_api_file = results_dir / 'cuda_api_summary.csv'
    if cuda_api_file.exists():
        cuda_api = pd.read_csv(cuda_api_file)
        print("CUDA API Summary (Top 20 by time):")
        print(cuda_api.head(20))

        # Look for cudaStreamSynchronize
        if 'Function' in cuda_api.columns or 'Name' in cuda_api.columns:
            name_col = 'Function' if 'Function' in cuda_api.columns else 'Name'
            sync_calls = cuda_api[cuda_api[name_col].str.contains('Synchronize', na=False)]
            if not sync_calls.empty:
                print("\n=== cudaStreamSynchronize Bottleneck Analysis ===")
                print(sync_calls)
                print("\nPer DREAM'26 paper: Expected ~66% of CUDA API time")
    else:
        print("CUDA API summary not found.")

    # GPU Kernel Summary
    print("\n\n3.2 GPU Kernel Summary")
    kernel_file = results_dir / 'gpu_kernel_summary.csv'
    if kernel_file.exists():
        kernels = pd.read_csv(kernel_file)
        print("GPU Kernel Summary (Top 20 by time):")
        print(kernels.head(20))

        # Look for NCCL kernels
        if 'Name' in kernels.columns or 'Kernel' in kernels.columns:
            name_col = 'Name' if 'Name' in kernels.columns else 'Kernel'
            nccl_kernels = kernels[kernels[name_col].str.contains('nccl', case=False, na=False)]
            if not nccl_kernels.empty:
                print("\n=== NCCL Communication Analysis ===")
                print(nccl_kernels)
                print("\nPer DREAM'26 paper:")
                print("  Expected AllGather: ~32.7% of GPU kernel time")
                print("  Expected AllReduce: ~31.7% of GPU kernel time")
    else:
        print("GPU kernel summary not found.")

    # Memory Operations
    print("\n\n3.3 Memory Operations")
    mem_file = results_dir / 'memory_operation_summary.csv'
    if mem_file.exists():
        mem_ops = pd.read_csv(mem_file)
        print("Memory Operation Summary (Top 20):")
        print(mem_ops.head(20))
        print("\nPer DREAM'26 paper:")
        print("  Expected: Many small transfers (~37.5 bytes avg for H2D)")
        print("  Expected total: ~2.95 GB (2.32 GB D2H, 0.32 GB D2D)")
    else:
        print("Memory operation summary not found.")


def analyze_energy_consumption(results_dir, gpu_baseline):
    """Analyze energy consumption"""
    print_section("4. ENERGY CONSUMPTION ANALYSIS")

    if gpu_baseline is None:
        print("No GPU metrics available. Skipping energy analysis.")
        return

    # Assuming 1 sample per second
    runtime_hours = len(gpu_baseline) / 3600

    energy_data = []
    for gpu_id in gpu_baseline['gpu_id'].unique():
        gpu_data = gpu_baseline[gpu_baseline['gpu_id'] == gpu_id]
        avg_power = gpu_data['power_draw_w'].mean()
        energy_kwh = (avg_power * runtime_hours) / 1000

        print(f"\nGPU {gpu_id}:")
        print(f"  Average power: {avg_power:.2f}W")
        print(f"  Runtime: {runtime_hours:.2f} hours")
        print(f"  Energy consumed: {energy_kwh:.4f} kWh")

        energy_data.append({
            'gpu_id': gpu_id,
            'avg_power_w': avg_power,
            'runtime_hours': runtime_hours,
            'energy_kwh': energy_kwh
        })

    # Total energy
    total_energy = (gpu_baseline['power_draw_w'].sum() * runtime_hours) / (1000 * len(gpu_baseline['gpu_id'].unique()))
    print(f"\nTotal GPU energy consumption: {total_energy:.4f} kWh")

    # Load runtime info
    runtime_file = results_dir / 'baseline_runtime.txt'
    if runtime_file.exists():
        with open(runtime_file) as f:
            runtime_sec = int(f.read().strip())
            print(f"\nTotal simulation runtime: {runtime_sec} seconds ({runtime_sec/60:.2f} minutes)")

    return energy_data


def generate_comparison_table():
    """Generate comparison table with DREAM'26 paper"""
    print_section("5. COMPARISON WITH DREAM'26 PAPER METRICS")

    comparison_data = {
        'Metric': [
            'cudaStreamSynchronize overhead',
            'NCCL AllGather (% of GPU kernel time)',
            'NCCL AllReduce (% of GPU kernel time)',
            'Total NCCL communication',
            'Average H2D transfer size',
            'Primary CPU process usage',
            'Secondary CPU process usage'
        ],
        'DREAM\'26 Paper': [
            '66.3% of CUDA API time',
            '32.7%',
            '31.7%',
            '64.4%',
            '37.5 bytes',
            '96.99%',
            '2.02%'
        ],
        'This Experiment': [
            'Check cuda_api_summary.csv',
            'Check gpu_kernel_summary.csv',
            'Check gpu_kernel_summary.csv',
            'Sum of AllGather + AllReduce',
            'Check memory_operation_summary.csv',
            'Check simulation logs',
            'Check simulation logs'
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print("\nNote: Open the .nsys-rep file in Nsight Systems GUI for detailed analysis")


def print_recommendations():
    """Print optimization recommendations"""
    print_section("6. OPTIMIZATION RECOMMENDATIONS")

    print("Based on the DREAM'26 paper and this baseline analysis:\n")

    print("KEY BOTTLENECKS IDENTIFIED:")
    print("1. cudaStreamSynchronize overhead - Blocks GPU-CPU concurrency")
    print("2. NCCL communication dominates GPU time - More time syncing than computing")
    print("3. Small, fragmented memory transfers - Poor PCIe bandwidth utilization")
    print("4. CPU workload imbalance - Secondary process underutilized")
    print("5. Host-side blocking - CPU threads waiting instead of working")

    print("\nOPTIMIZATION OPPORTUNITIES:")
    print("1. Batch data transfers - Reduce number of small transfers")
    print("2. Overlap communication with computation - Use asynchronous operations")
    print("3. Reduce synchronization frequency - Balance accuracy vs. performance")
    print("4. Load balancing - Better distribute work across CPU processes")
    print("5. Optimize NCCL collective operations - Tune AllGather/AllReduce patterns")

    print("\nNEXT EXPERIMENTS:")
    print("1. Test different data exchange frequencies")
    print("2. Implement batched memory transfers")
    print("3. Evaluate async communication patterns")
    print("4. Profile with different GPU counts")
    print("5. Test energy-aware scheduling strategies")


def generate_summary_report(results_dir, gpu_baseline, agents_df, energy_data):
    """Generate a summary JSON report"""
    print_section("7. GENERATING SUMMARY REPORT")

    summary = {
        'experiment': 'Baseline Twin-B Simulation',
        'results_directory': str(results_dir),
        'gpu_metrics': {},
        'agent_simulation': {},
        'energy_consumption': {}
    }

    if gpu_baseline is not None:
        summary['gpu_metrics'] = {
            'total_samples': len(gpu_baseline),
            'duration_seconds': len(gpu_baseline),
            'avg_power_w': float(gpu_baseline['power_draw_w'].mean()),
            'max_power_w': float(gpu_baseline['power_draw_w'].max()),
            'avg_gpu_util_pct': float(gpu_baseline['gpu_utilization_pct'].mean()),
            'avg_memory_used_mb': float(gpu_baseline['memory_used_mb'].mean()),
            'avg_temperature_c': float(gpu_baseline['temperature_c'].mean())
        }

    if agents_df is not None:
        summary['agent_simulation'] = {
            'total_records': len(agents_df),
            'unique_agents': int(agents_df['agent_id'].nunique()),
            'unique_zones': int(agents_df['room'].nunique()),
            'total_steps': int(agents_df['step'].max() + 1),
            'avg_comfort_level': float(agents_df['comfort_level'].mean()),
            'ac_usage_rate': float(agents_df['using_ac'].mean()),
            'avg_current_temp': float(agents_df['current_temp'].mean()),
            'avg_preferred_temp': float(agents_df['preferred_temp'].mean())
        }

    if energy_data:
        summary['energy_consumption'] = {
            'per_gpu': energy_data,
            'total_kwh': sum(d['energy_kwh'] for d in energy_data)
        }

    # Save to file
    output_file = results_dir / 'analysis_summary.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary report saved to: {output_file}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


def main():
    """Main analysis function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_baseline.py <results_directory>")
        print("Example: python analyze_baseline.py ../experiment2_results/baseline_12345")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print("=" * 80)
    print("BASELINE TWIN-B EXPERIMENT ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing results from: {results_dir}")

    # Run analyses
    gpu_baseline = analyze_gpu_metrics(results_dir)
    agents_df = analyze_agent_results(results_dir)
    analyze_profiling_data(results_dir)
    energy_data = analyze_energy_consumption(results_dir, gpu_baseline)
    generate_comparison_table()
    print_recommendations()
    generate_summary_report(results_dir, gpu_baseline, agents_df, energy_data)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {results_dir}")
    print("\nGenerated files:")
    print(f"  - gpu_metrics_timeline.png")
    print(f"  - ac_usage_over_time.png")
    print(f"  - analysis_summary.json")
    print("\nFor detailed profiling analysis, open the .nsys-rep file in Nsight Systems GUI")


if __name__ == "__main__":
    main()
