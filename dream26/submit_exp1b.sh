#!/bin/bash
# Helper script to submit Experiment 1b with directory pre-creation

# Create results directory
RESULTS_DIR="exp1b_temp"
mkdir -p $RESULTS_DIR

echo "Submitting Experiment 1b: Enhanced Platform Capability Profiling"
echo "Results will be in: $RESULTS_DIR (renamed after job starts)"
echo ""

# Submit job
JOB_ID=$(sbatch experiment1b_enhanced_platform_capability.slurm | awk '{print $NF}')

if [ ! -z "$JOB_ID" ]; then
    echo "Job submitted: $JOB_ID"

    # Rename directory to actual job ID
    if [ -d "$RESULTS_DIR" ]; then
        NEW_DIR="exp1b_${JOB_ID}"
        mv $RESULTS_DIR $NEW_DIR
        echo "Results directory: $NEW_DIR"
    fi

    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f exp1b_${JOB_ID}.out"
else
    echo "ERROR: Job submission failed"
    exit 1
fi
