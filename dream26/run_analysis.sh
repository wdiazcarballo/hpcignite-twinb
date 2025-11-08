#!/bin/bash
# Quick script to run analysis on the most recent experiment 2 results

if [ "$#" -eq 1 ]; then
    # User provided job ID
    JOB_ID=$1
    RESULTS_DIR="../experiment2_results/baseline_${JOB_ID}"
else
    # Find most recent results directory
    RESULTS_DIR=$(ls -dt ../experiment2_results/baseline_* 2>/dev/null | head -1)

    if [ -z "$RESULTS_DIR" ]; then
        echo "Error: No experiment2 results found."
        echo ""
        echo "Usage:"
        echo "  $0 <job_id>        # Analyze specific job"
        echo "  $0                 # Analyze most recent job"
        echo ""
        echo "Example:"
        echo "  $0 12345"
        exit 1
    fi
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

echo "=========================================="
echo "Running analysis on:"
echo "$RESULTS_DIR"
echo "=========================================="
echo ""

# Run analysis
python analyze_baseline.py "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo ""
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Generated files:"
echo "  - analysis_summary.json"
echo "  - gpu_metrics_timeline.png"
echo "  - ac_usage_over_time.png"
