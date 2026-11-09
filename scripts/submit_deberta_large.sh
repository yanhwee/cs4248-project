#!/usr/bin/env bash
# Submits the DeBERTa-v3-large runs (ce and kl with lambda = 1) as separate Slurm jobs through scripts/job.sbatch.
# Reuses the paraphrase caches built by scripts/submit_all_ablations.sh and fails early if they are missing.
#
#   bash scripts/submit_deberta_large.sh                  # from the repository root on xlogin
#   GPU=<type> bash scripts/submit_deberta_large.sh       # another GPU type listed by gpufree (h100-47 is a 47 GB MIG slice)
#   GRAD_CKPT=1 bash scripts/submit_deberta_large.sh      # activation checkpointing: ~30 % slower, far less memory
#   MODES="ce" bash scripts/submit_deberta_large.sh       # a subset of the modes
#
# Defaults (override through the environment): BATCH=8 GRAD_ACCUM=2 (effective batch 16, the same as the base runs),
# LR=1e-5 (the usual rate for large encoders; the base runs' 3e-5 risks divergence), EPOCHS=2, PRECISION=bf16,
# TIME=10:00:00. Expected on one h100-47 slice: ~1 h training for ce, ~2 h for kl, plus ~15 min reranker + evaluation.
# Each run writes ~3.5 GB of checkpoints to weights/<run>/ (best.pt and last.pt); check your disk quota first.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-h100-47}
MODEL=${MODEL:-microsoft/deberta-v3-large}
RUN_PREFIX=${RUN_PREFIX:-deb_large}
MODES=${MODES:-"ce kl"}
BATCH=${BATCH:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-2}
PRECISION=${PRECISION:-bf16}
TIME=${TIME:-10:00:00}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
if [ -n "${GRAD_CKPT:-}" ]; then EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:+$EXTRA_TRAIN_ARGS }--grad_checkpointing"; fi
mkdir -p logs/slurm runs weights

for F in train dev addsent addonesent; do
  if [ ! -s "data/paraphrases/$F.jsonl" ]; then
    echo "missing data/paraphrases/$F.jsonl -- build the paraphrase caches first:"
    echo "  sbatch --job-name=g17_para --time=02:00:00 --gpus=$GPU:1 --export=ALL,STAGES=paraphrases scripts/job.sbatch"
    exit 1
  fi
done

for MODE in $MODES; do
  RUN="${RUN_PREFIX}_${MODE}"
  JOB=$(sbatch --parsable --job-name="g17_$RUN" --gpus="$GPU:1" --time="$TIME" \
    --export=ALL,MODEL="$MODEL",RUN_PREFIX="$RUN_PREFIX",MODES="$MODE",BATCH="$BATCH",GRAD_ACCUM="$GRAD_ACCUM",LR="$LR",EPOCHS="$EPOCHS",PRECISION="$PRECISION",EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS",STAGES="train reranker eval" \
    scripts/job.sbatch)
  echo "$RUN  ($MODEL, mode $MODE, batch ${BATCH}x${GRAD_ACCUM}, lr $LR${EXTRA_TRAIN_ARGS:+, $EXTRA_TRAIN_ARGS})  -> job $JOB"
done
echo
echo "monitor:   squeue --me        log: tail -f logs/slurm/g17_${RUN_PREFIX}_<mode>-<job-id>.out"
echo "progress:  grep -E 'val @|done in|peak GPU' logs/slurm/g17_${RUN_PREFIX}_*.out"
echo "afterwards: python visualize.py --best bert_kl_lam1.0 && python scripts/make_results_report.py   (locally, after rsync of runs/)"
