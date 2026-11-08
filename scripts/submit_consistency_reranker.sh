#!/usr/bin/env bash
# One GPU job: train + evaluate consistency-trained rerankers on an existing extractor (nothing else is retrained).
#   bash scripts/submit_consistency_reranker.sh
#   RUNS=deb_large_kl WEIGHTS="0.5 1.0 2.0" SELECT_BY=em+agreement bash scripts/submit_consistency_reranker.sh
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-h100-47}
mkdir -p logs/slurm
for R in ${RUNS:-deb_large_kl}; do
  echo "$R -> job $(sbatch --parsable --job-name="g17_cr_$R" --time=04:00:00 --gpus="$GPU:1" \
    --export=ALL,RUNS="$R",WEIGHTS="${WEIGHTS:-0.5 1.0 2.0}",EVAL_SETS="${EVAL_SETS:-dev gan_dev addsent addonesent}",SELECT_BY="${SELECT_BY:-em}",PIPELINE=scripts/run_consistency_reranker.sh scripts/job_reranker_variants.sbatch)"
done
echo "monitor: squeue --me   logs: logs/slurm/g17_cr_<run>-<jobid>.out"
