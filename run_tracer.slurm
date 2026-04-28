#!/bin/bash
#SBATCH --job-name=triosim-trace
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -n 1
#SBATCH --cpus-per-task=9
#SBATCH -t 12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=weh7xp@virginia.edu
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/GPUArch/triosim_project/triosim"
CONDA_ENV="triosim"

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"
echo "=========================================="

cd "$PROJECT_DIR"
mkdir -p logs

# --- Modules ---
module purge
module load miniforge/25.3.1-py3.21
module load cuda/12.8.1
module load cudnn/9.10.1.4-CUDA-12.8.1

# --- Conda env ---
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo "Creating conda environment '$CONDA_ENV'..."
    conda create -n "$CONDA_ENV" python=3.12 -y
    conda run -n "$CONDA_ENV" pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128
    conda run -n "$CONDA_ENV" pip install tqdm pandas numpy pyyaml
fi

conda activate "$CONDA_ENV"

# --- GPU info ---
echo ""
nvidia-smi
echo ""

# --- Run tracer ---
export PYTORCH_ALLOC_CONF=expandable_segments:True
python tracer/generate_trace.py

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
