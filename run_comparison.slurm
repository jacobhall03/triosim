#!/bin/bash
#SBATCH --job-name=triosim-compare
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=weh7xp@virginia.edu
#SBATCH -o logs/slurm-compare-%j.out
#SBATCH -e logs/slurm-compare-%j.err

# Comparison job: collect traces with both the original and FX methods for
# vgg13, densenet161, and resnet50 at batch size 32, then run triosim with
# every parallelism type and interconnect and write comparison_results.csv.
#
# Step 1: Original tracer (full-model GPU profiling, ~30 min for 3 models)
# Step 2: FX tracer      (per-op microbenchmarks,   ~15 min for 3 models)
# Step 3: Build triosim binary
# Step 4: Run run_comparison.py  (~5 min, 60 simulations)

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

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
module purge
module load miniforge/25.3.1-py3.21
module load cuda/12.8.1
module load cudnn/9.10.1.4-CUDA-12.8.1

# Load Go; fall back to a conda-forge install if the module is unavailable.
if module load go 2>/dev/null; then
    echo "Go loaded via module: $(go version)"
else
    echo "Go module not found; installing via conda-forge..."
    conda run -n "$CONDA_ENV" conda install -c conda-forge go -y 2>/dev/null || true
    export PATH="$HOME/.conda/envs/$CONDA_ENV/bin:$PATH"
    go version || { echo "ERROR: Go not available"; exit 1; }
fi

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo "Creating conda environment '$CONDA_ENV'..."
    conda create -n "$CONDA_ENV" python=3.12 -y
    conda run -n "$CONDA_ENV" pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128
    conda run -n "$CONDA_ENV" pip install pandas pyyaml tqdm
fi

conda activate "$CONDA_ENV"

# Ensure tqdm is present (needed by the original tracer).
python -c "import tqdm" 2>/dev/null || pip install tqdm

# ---------------------------------------------------------------------------
# Verify key Python dependencies
# ---------------------------------------------------------------------------
python - <<'EOF'
import importlib, sys
missing = []
for pkg in ['torch', 'torchvision', 'torch.fx',
            'torch.fx.passes.shape_prop', 'pandas', 'yaml', 'tqdm']:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"ERROR: missing Python packages: {missing}", file=sys.stderr)
    sys.exit(1)
import torch
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
EOF

echo ""
nvidia-smi
echo ""

# ---------------------------------------------------------------------------
# Step 1: Original tracer
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Step 1/4: Original tracer (3 models, bs=32)"
echo "=========================================="
python tracer/generate_trace.py \
    --method original \
    --config tracer/trace_config_compare_orig.yaml

echo ""
echo "Validating original traces..."
python tracer/validate_trace.py tracer/data_compare_orig/middledata/trace

# ---------------------------------------------------------------------------
# Step 2: FX tracer
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 2/4: FX tracer (3 models, bs=32)"
echo "=========================================="
# --no-validate because validate_all only scans one level; we validate manually below.
python tracer/generate_trace.py \
    --method fx \
    --no-validate \
    --config tracer/trace_config_compare_fx.yaml

echo ""
echo "Validating FX traces..."
python tracer/validate_trace.py tracer/data_compare_fx/middledata/trace/bs32

# ---------------------------------------------------------------------------
# Step 3: Build triosim binary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 3/4: Building triosim binary"
echo "=========================================="
# go mod download fetches dependencies; requires network access.
# On compute nodes without internet, pre-run 'go mod download' from a login node.
go mod download
go build -o triosim_bin ./triosim/
echo "Built: $(ls -lh triosim_bin)"

# ---------------------------------------------------------------------------
# Step 4: Run comparison
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 4/4: Running triosim comparisons"
echo "=========================================="
python run_comparison.py

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "Results: $PROJECT_DIR/comparison_results.csv"
echo "=========================================="
