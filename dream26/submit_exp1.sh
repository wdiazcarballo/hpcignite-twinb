#!/bin/bash
# Helper script to submit Experiment 1 with directory pre-creation

# Get the next job ID (approximate)
NEXT_JOB_ID=$(squeue -u $USER | tail -1 | awk '{print $1}' | sed 's/[^0-9]//g')
if [ -z "$NEXT_JOB_ID" ]; then
    NEXT_JOB_ID=1
else
    NEXT_JOB_ID=$((NEXT_JOB_ID + 1))
fi

# Create results directory
RESULTS_DIR="exp1_temp"
mkdir -p $RESULTS_DIR

echo "Submitting Experiment 1: Platform Capability Profiling"
echo "Results will be in: $RESULTS_DIR (renamed after job starts)"
echo ""

# Submit job
JOB_ID=$(sbatch experiment1_platform_capability.slurm | awk '{print $NF}')

if [ ! -z "$JOB_ID" ]; then
    echo "Job submitted: $JOB_ID"

    # Rename directory to actual job ID
    if [ -d "$RESULTS_DIR" ]; then
        NEW_DIR="exp1_${JOB_ID}"
        mv $RESULTS_DIR $NEW_DIR
        echo "Results directory: $NEW_DIR"
    fi

    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f exp1_${JOB_ID}.out"
else
    echo "ERROR: Job submission failed"
    exit 1
fi
