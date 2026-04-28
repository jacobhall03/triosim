#!/usr/bin/env bash
set -e

# Load required modules
module load miniforge/25.3.1-py3.21
module load cuda/12.8.1
module load cudnn/9.10.1.4-CUDA-12.8.1

# Initialize conda for this shell session
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create conda environment (skip if it already exists)
conda create -n triosim python=3.12 -y

# Install packages directly into the env without activating it
# Fall back to --index-url .../cu126 if cu128 wheels are not yet available
conda run -n triosim pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda run -n triosim pip install tqdm pandas numpy pyyaml

echo ""
echo "Setup complete. To activate the environment in your shell, run:"
echo "  module load miniforge/25.3.1-py3.21"
echo "  conda activate triosim"
