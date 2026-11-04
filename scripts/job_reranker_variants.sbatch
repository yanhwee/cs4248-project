#!/bin/bash
# Slurm job: reranker ablation (pair_score on/off, residual fusion) on existing extractor checkpoints.
# Submit from the repository root on xlogin; see scripts/run_reranker_variants.sh for what it does.
#
#   sbatch --job-name=g17_rr_bert --export=ALL,RUNS=bert_kl_lam1.0 scripts/job_reranker_variants.sbatch
#   sbatch --job-name=g17_rr_deb  --export=ALL,RUNS=deb_kl         scripts/job_reranker_variants.sbatch
#   bash scripts/submit_reranker_variants.sh                       # both, one job each
#SBATCH --job-name=g17_rr_variants
#SBATCH --partition=gpu-long
#SBATCH --time=04:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000

set -euo pipefail
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-cs4248}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-local}.out" 2>&1
cd "$REPO_ROOT"
echo "job ${SLURM_JOB_ID:-?} (${SLURM_JOB_NAME:-?}) started on $(hostname) at $(date)"
nvidia-smi || echo "nvidia-smi not available"
set +u; source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; set -u
export PYTHONUNBUFFERED=1 PYTHON=python DEVICE=cuda
export PRECISION="${PRECISION:-bf16}"
bash scripts/run_reranker_variants.sh "$@"
echo "job finished at $(date)"
