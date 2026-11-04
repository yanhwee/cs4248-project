#!/usr/bin/env bash
# One GPU job per extractor for the reranker ablation (existing checkpoints only; nothing is retrained).
#   bash scripts/submit_reranker_variants.sh
#   RUNS="bert_kl_lam1.0" EVAL_SETS="dev gan_dev gan_adv addsent addonesent" bash scripts/submit_reranker_variants.sh
#   GATE_MARGINS="0.1 0.25 0.5" PAIR_RESIDUAL=1 bash scripts/submit_reranker_variants.sh     # optional extras
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-h100-47}
mkdir -p logs/slurm
for R in ${RUNS:-bert_kl_lam1.0 deb_kl}; do
  echo "$R -> job $(sbatch --parsable --job-name="g17_rr_$R" --gpus="$GPU:1" \
    --export=ALL,RUNS="$R",EVAL_SETS="${EVAL_SETS:-dev gan_dev}",GATE_MARGINS="${GATE_MARGINS:-}",PAIR_RESIDUAL="${PAIR_RESIDUAL:-}" scripts/job_reranker_variants.sbatch)"
done
echo "monitor: squeue --me   logs: logs/slurm/g17_rr_<run>-<jobid>.out"
