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
