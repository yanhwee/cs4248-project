#!/usr/bin/env bash
# One GPU job for the consistency baseline (eval only). Reuses scripts/job_reranker_variants.sbatch's environment setup.
#   bash scripts/submit_consistency_baseline.sh
#   RUNS=deb_large_kl EVAL_SETS="dev gan_dev gan_adv addsent addonesent" bash scripts/submit_consistency_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-h100-47}
mkdir -p logs/slurm
for R in ${RUNS:-deb_large_kl}; do
  echo "$R -> job $(sbatch --parsable --job-name="g17_cb_$R" --time=03:00:00 --gpus="$GPU:1" \
    --export=ALL,RUNS="$R",EVAL_SETS="${EVAL_SETS:-dev gan_dev addsent addonesent}",PIPELINE=scripts/run_consistency_baseline.sh scripts/job_reranker_variants.sbatch)"
done
echo "monitor: squeue --me   logs: logs/slurm/g17_cb_<run>-<jobid>.out"
