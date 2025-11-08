#!/bin/bash
#SBATCH --job-name=boonchu-full
#SBATCH --output=logs/boonchu_full_%j.out
#SBATCH --error=logs/boonchu_full_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
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
# Run Mesa DDP
# ---------------------------
echo "Running Mesa DDP..."
torchrun --standalone --nproc_per_node=2 main.py
