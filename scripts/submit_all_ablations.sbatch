#!/bin/bash
#SBATCH --job-name=cs4248_all_ablations
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --exclude=xgpd0,xgpd1,xgpd2,xgpd3,xgpd4,xgpd5,xgpd6,xgpd7,xgpd8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=runs/slurm_all_%j.log
#SBATCH --error=runs/slurm_all_%j.err

echo "=== CS4248 All-Ablations Slurm Job Started ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
which nvidia-smi && nvidia-smi

# 1. Activate Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/cs4248/.venv" ]; then
    source "$HOME/cs4248/.venv/bin/activate"
fi

# 2. Run the sequential benchmark suite
chmod +x scripts/run_all_ablations.sh
./scripts/run_all_ablations.sh 1000 16 2

echo "=== All Ablations Slurm Job Completed at $(date) ==="
