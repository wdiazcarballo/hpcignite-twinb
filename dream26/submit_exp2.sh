#!/bin/bash
# Helper script to submit Experiment 2 with directory pre-creation

PROJECT_DIR=/project/lt200291-ignite/Project_chomwong/project

# Create results directory
RESULTS_DIR="$PROJECT_DIR/dream26/exp2_temp"
mkdir -p $RESULTS_DIR

echo "Submitting Experiment 2: Baseline Twin-B Simulation"
echo "Results will be in: $RESULTS_DIR (renamed after job starts)"
echo ""

# Submit job
JOB_ID=$(sbatch experiment2_baseline_simulation.slurm | awk '{print $NF}')

if [ ! -z "$JOB_ID" ]; then
    echo "Job submitted: $JOB_ID"

    # Rename directory to actual job ID
    if [ -d "$RESULTS_DIR" ]; then
        NEW_DIR="$PROJECT_DIR/dream26/exp2_${JOB_ID}"
        mv $RESULTS_DIR $NEW_DIR
        echo "Results directory: $NEW_DIR"
    fi

    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f $PROJECT_DIR/dream26/exp2_${JOB_ID}.out"
else
    echo "ERROR: Job submission failed"
    exit 1
fi
