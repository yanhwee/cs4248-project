#!/bin/bash
#SBATCH --job-name=setup_venv
#SBATCH --partition=normal
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/setup_env.log
#SBATCH --error=runs/setup_env.err

echo "=== Setting up Python Virtual Environment on Compute Node ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

cd "$HOME/cs4248"
mkdir -p runs weights figures

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -c "import torch, transformers, datasets, scipy; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "=== Environment Setup Complete at $(date) ==="
