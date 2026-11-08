#!/bin/bash
# Slurm job for the SoC compute cluster, following the course's job_template.sbatch
# (the course's SoC cluster quick-start guide on Canvas, https://github.com/isaacNingLee/Unofficial_SoC_Cluster_Guide).
#
# Submit from the repository root on xlogin. Settings are passed as environment variables or sbatch flags:
#
#   sbatch scripts/job.sbatch                                                  # bert-base: ce, aug, kl + reranker + all evals
#   sbatch --job-name=g17_deb --export=ALL,MODEL=microsoft/deberta-v3-base,RUN_PREFIX=deb,MODES="ce kl" scripts/job.sbatch
#   sbatch --job-name=g17_para --time=02:00:00 --export=ALL,STAGES=paraphrases scripts/job.sbatch   # caches only
#   sbatch --gpus=a100-40:1 scripts/job.sbatch                                 # a different GPU type
#   sbatch scripts/job.sbatch --help                                           # extra args go to the pipeline script
#
# Every pipeline stage is skipped when its output already exists, so a job can be resubmitted after a time-out.
#SBATCH --job-name=g17_pipeline
#SBATCH --partition=gpu-long
#SBATCH --time=06:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-cs4248}"
PIPELINE_SCRIPT="scripts/run_pipeline.sh"

# SLURM_SUBMIT_DIR is the directory from which sbatch was invoked (the repository root).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR" "$REPO_ROOT/runs" "$REPO_ROOT/weights"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-local}.out" 2>&1
cd "$REPO_ROOT"

echo "job ${SLURM_JOB_ID:-?} (${SLURM_JOB_NAME:-?}) started on $(hostname) at $(date)"
nvidia-smi || echo "nvidia-smi not available (no GPU allocated?)"

# conda's activation scripts reference unset variables, so temporarily disable `set -u`.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
echo "python: $(which python) ($(python --version 2>&1))"

# defaults for the pipeline (override with --export=ALL,VAR=value)
export PYTHONUNBUFFERED=1   # stream Python's progress lines to the log as they happen instead of in 8 KB bursts
export PYTHON=python
export DEVICE=cuda
export PRECISION="${PRECISION:-bf16}"     # bf16 on H100/A100; use fp16 on T4/V100 (bf16 falls back to fp32 with a warning)
export MODEL="${MODEL:-bert-base-uncased}"
export MODES="${MODES:-ce aug kl}"
export RUN_PREFIX="${RUN_PREFIX:-bert}"
export STAGES="${STAGES:-paraphrases train reranker eval figures}"

bash "$PIPELINE_SCRIPT" "$@"
echo "job finished at $(date)"
