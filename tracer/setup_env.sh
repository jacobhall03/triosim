#!/usr/bin/env bash
set -e

# Load required modules
module load miniforge/25.3.1-py12
module load cuda/12.8.1
module load cudnn/9.10.1.4-CUDA-12.8.1

# Create and activate conda environment
conda create -n triosim python=3.12 -y
source activate triosim

# Install PyTorch with CUDA 12.8 support
# Fall back to --index-url .../cu126 if cu128 wheels are not yet available
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install tqdm pandas numpy pyyaml

echo "Environment setup complete. Activate with: conda activate triosim"
