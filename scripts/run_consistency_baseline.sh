#!/usr/bin/env bash
# Learning-free paraphrase-ensemble baseline vs the learned reranker, with both-views-correct rates and 95% bootstrap CIs.
# Eval only: reuses weights/<run>/best.pt and the rerankers trained by scripts/run_reranker_variants.sh. Nothing is retrained.
#
#   bash scripts/submit_consistency_baseline.sh                       # sbatch, default RUNS=deb_large_kl
#   RUNS=deb_large_kl EVAL_SETS="dev gan_dev" bash scripts/run_consistency_baseline.sh    # inside an allocation
#
# Per run and eval set it writes (existing artefacts untouched; each step skipped if its output exists):
#   runs/<R>/rerank_variants/CB_pair_argmax_ensemble/eval_<set>/   pair-feature reranker (argmax) AND the --pair_ensemble baseline
#   runs/<R>/rerank_variants/CB_nopair_argmax/eval_<set>/          no-pair reranker (argmax)
#   runs/<R>/rerank_variants/CB_nopair_res1.0/eval_<set>/          no-pair reranker, residual alpha=1
# and a table runs/<R>/rerank_variants/summary_<set>.md (via scripts/summarize_reranker_variants.py).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
RUNS=${RUNS:-deb_large_kl}
EVAL_SETS=${EVAL_SETS:-"dev gan_dev addsent addonesent"}   # gan_adv has 56 questions: add it only for completeness
PRECISION=${PRECISION:-bf16}
DEVICE=${DEVICE:-auto}
BETA=${BETA:-1.0}

evaluate() {  # evaluate RUN SET TAG RERANKER [extra evaluate.py args...]
  local RUN=$1 S=$2 TAG=$3 RR=$4; shift 4
  if [ -f "runs/$RUN/rerank_variants/$TAG/eval_$S/results.json" ]; then
    echo "   runs/$RUN/rerank_variants/$TAG/eval_$S/results.json exists, skipping"
  else
    echo "-- $RUN | $S | $TAG"
    $PY evaluate.py --weights "weights/$RUN/best.pt" --eval_set "$S" --reranker "$RR" --tag "$TAG" --precision "$PRECISION" --device "$DEVICE" "$@"
  fi
}

echo "== consistency baseline: runs=[$RUNS] sets=[$EVAL_SETS] beta=$BETA =="
echo "   started on $(hostname) at $(date)"
$PY download_eval_sets.py
for RUN in $RUNS; do
  for F in "weights/$RUN/best.pt" "weights/$RUN/reranker_pair.pt" "weights/$RUN/reranker_nopair.pt"; do
    [ -f "$F" ] || { echo "missing $F (run scripts/run_reranker_variants.sh for $RUN first)"; exit 1; }
  done
  for S in $EVAL_SETS; do
    evaluate "$RUN" "$S" CB_pair_argmax_ensemble "weights/$RUN/reranker_pair.pt"   --select_mode reranker --pair_ensemble --ensemble_beta "$BETA"
    evaluate "$RUN" "$S" CB_nopair_argmax        "weights/$RUN/reranker_nopair.pt" --select_mode reranker
    evaluate "$RUN" "$S" CB_nopair_res1.0        "weights/$RUN/reranker_nopair.pt" --select_mode residual --alpha 1.0
    $PY scripts/summarize_reranker_variants.py --run "$RUN" --eval_set "$S"
  done
done
echo "consistency baseline finished at $(date)"
