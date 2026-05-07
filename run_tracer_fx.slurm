#!/bin/bash
#SBATCH --job-name=triosim-trace-fx
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH -t 6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=weh7xp@virginia.edu
#SBATCH -o logs/slurm-fx-%j.out
#SBATCH -e logs/slurm-fx-%j.err

# FX-method tracer: torch.fx graph extraction + per-operator GPU microbenchmarks.
#
# Key differences from run_tracer.slurm (original method):
#   - No dataset required            (graph extracted symbolically on CPU)
#   - Much lower GPU memory usage    (one layer at a time, not the full model)
#   - Fewer CPUs needed              (no DataLoader workers)
#   - Shorter walltime               (~10-15 min/model vs hours)
#
# Dependencies already in the triosim conda env:
#   torch (>=1.8, includes torch.fx and ShapeProp), torchvision, pandas, pyyaml
#   No tqdm or dataset required.

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
module load miniforge/25.3.1-py3.12
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
    conda run -n "$CONDA_ENV" pip install pandas pyyaml
fi

conda activate "$CONDA_ENV"

# --- Verify key dependencies are present ---
python - <<'EOF'
import importlib, sys
missing = []
for pkg, import_as in [
    ('torch',        'torch'),
    ('torchvision',  'torchvision'),
    ('torch.fx',     'torch.fx'),
    ('torch.fx.passes.shape_prop', 'torch.fx.passes.shape_prop'),
    ('pandas',       'pandas'),
    ('yaml',         'yaml'),
]:
    try:
        importlib.import_module(import_as)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"ERROR: missing dependencies: {missing}", file=sys.stderr)
    sys.exit(1)
import torch
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
EOF

# --- GPU info ---
echo ""
nvidia-smi
echo ""

# --- Run FX tracer ---
python tracer/generate_trace.py \
    --method fx \
    --config tracer/trace_config_fx.yaml

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
