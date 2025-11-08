#!/bin/bash
#SBATCH --job-name=boonchu-full-profile
#SBATCH --output=logs/boonchu_full_%j.out
#SBATCH --error=logs/boonchu_full_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2                 # จำนวน process = GPU
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --mem=64G
#SBATCH -A lt200291

# ---------------------------
# Load environment
# ---------------------------
module load cray-python/3.10.10
source ~/mesa_env/bin/activate

# ---------------------------
# EnergyPlus paths
# ---------------------------
EP_HOME=/project/lt200291-ignite/Project_chomwong/energyplus/EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64
export PATH=$EP_HOME:$PATH
export PYTHONPATH=$PYTHONPATH:$EP_HOME
export ENERGYPLUS_EXE=$EP_HOME/energyplus

# ---------------------------
# Project folder
# ---------------------------
PROJECT_DIR=/project/lt200291-ignite/Project_chomwong/project
OUT_DIR=$PROJECT_DIR/outEnergyPlus
mkdir -p $OUT_DIR
cd $PROJECT_DIR

# ---------------------------
# EnergyPlus IDF and Weather 
# ---------------------------
IDF_PATH=$PROJECT_DIR/EnergyPlus_BP_Boonchoo/output/expanded.idf
WEATHER_PATH=$PROJECT_DIR/EnergyPlus_BP_Boonchoo/output/in.epw

# ---------------------------
# Run EnergyPlus 
# ---------------------------
echo "Running EnergyPlus..."
$EP_HOME/energyplus -r -d $OUT_DIR -w $WEATHER_PATH $IDF_PATH

if [ ! -f "$OUT_DIR/eplusout.csv" ]; then
    echo "ERROR: EnergyPlus CSV not found!"
    exit 1
fi
echo "EnergyPlus finished successfully."

# ---------------------------
# Nsight Systems profiling
# ---------------------------
NSYS_BIN=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin/nsys
$NSYS_BIN --version || { echo "ERROR: nsys not found"; exit 1; }

TRACE_NAME="boonchu_trace_${SLURM_JOB_ID}"

# ตั้ง environment สำหรับ distributed
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

echo "Starting distributed run with Nsight Systems..."
$NSYS_BIN profile \
    --trace=cuda,nvtx,osrt,openmp \
    --gpu-metrics-devices=all \
    --force-overwrite=true \
    --output=${TRACE_NAME} \
    torchrun --standalone --nproc_per_node=2 main.py

# ---------------------------
# Export trace (only rank 0)
# ---------------------------
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Exporting Nsight trace to SQLite..."
    $NSYS_BIN export --force-overwrite=true --type=sqlite \
        --output ${TRACE_NAME}.sqlite \
        ${TRACE_NAME}.nsys-rep

    echo "Exporting summary reports to JSON..."
    for rpt in cuda_api_sum cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum nvtx_sum osrt_sum openmp_sum
    do
        echo "Exporting $rpt ..."
        $NSYS_BIN stats ${TRACE_NAME}.nsys-rep --report $rpt --format json --output ${TRACE_NAME}_${rpt}.json
    done
fi

echo "Profiling finished successfully."
