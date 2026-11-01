#!/usr/bin/env bash
# ==============================================================================
# CS4248 Group G17: SQuAD Paraphrase-Consistent QA Ablation Suite
# Runs all 5 experimental configurations sequentially on a single allocated GPU.
# Total estimated runtime on SoC cluster (RTX/V100/A100) with FP16: ~60-80 mins.
# ==============================================================================

set -euo pipefail

echo "========================================================================"
echo "Starting CS4248 5-Condition Empirical Ablation Benchmark"
echo "Host: $(hostname) | Date: $(date)"
echo "========================================================================"

# Activate virtual environment if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "$HOME/cs4248/.venv" ]; then
        source "$HOME/cs4248/.venv/bin/activate"
    fi
fi

SAMPLES="${1:-0}"
BATCH_SIZE="${2:-16}"
EPOCHS="${3:-2}"

ABLATIONS=("ce_only" "kl" "ot" "hungarian" "dual")

for LOSS in "${ABLATIONS[@]}"; do
    echo ""
    echo "===================================================================="
    echo ">>> Running Ablation: ${LOSS} (Samples: ${SAMPLES}, Epochs: ${EPOCHS}) <<<"
    echo "===================================================================="
    python train.py \
        --model "google-bert/bert-base-uncased" \
        --loss_type "$LOSS" \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr 3e-5 \
        --lambda_ot 0.5 \
        --lambda_kl 0.5 \
        --mu_hung 0.1 \
        --top_k 3 \
        --fp16
    echo "Completed ablation: ${LOSS}"
done

echo ""
echo "========================================================================"
echo "All 5 Ablations Completed! Generating Report Figures & LaTeX Tables..."
echo "========================================================================"
python visualize.py
echo "Artifacts successfully generated in figures/ and runs/."
