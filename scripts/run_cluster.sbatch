#!/bin/bash
#SBATCH --job-name=cs4248_single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=xgpd0,xgpd1,xgpd2,xgpd3,xgpd4,xgpd5,xgpd6,xgpd7,xgpd8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm_%j.log
#SBATCH --error=runs/slurm_%j.err

echo "=== CS4248 SQuAD Consistency Training Job ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
which nvidia-smi && nvidia-smi

# 1. Activate Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/cs4248/.venv" ]; then
    source "$HOME/cs4248/.venv/bin/activate"
fi

# 2. Execute Training Engine
# Default to Dual Regularization (Ours) with CUDA FP16 AMP
python train.py \
    --model "google-bert/bert-base-uncased" \
    --loss_type "${1:-dual}" \
    --samples "${2:-0}" \
    --epochs 2 \
    --batch_size 16 \
    --lr 3e-5 \
    --lambda_ot 0.5 \
    --lambda_kl 0.5 \
    --mu_hung 0.1 \
    --top_k 3 \
    --fp16

echo "=== Training Finished at $(date) ==="
