#!/usr/bin/env bash
# Consistency-trained reranker (symmetric KL between a question's and its paraphrase's reranker distributions).
# Reuses weights/<run>/best.pt (no extractor training). Trains WITHOUT the paraphrase-derived features so that, unlike the
# pair_score variant, the reranker needs no paraphrase at test time; the consistency term is the only paraphrase signal.
#
#   bash scripts/submit_consistency_reranker.sh                  # sbatch, default RUNS=deb_large_kl
#   RUNS=deb_large_kl WEIGHTS="0.5 1.0 2.0" bash scripts/run_consistency_reranker.sh     # inside an allocation
#
# For each consistency weight W it trains   weights/<R>/reranker_nopair_cw<W>.pt   (runs/<R>/reranker_summary_nopair_cw<W>.json, with a
# held-out grid that includes question/paraphrase agreement), then evaluates on the official scorer:
#   runs/<R>/rerank_variants/CR_cw<W>_argmax/eval_<set>/    reranker argmax
#   runs/<R>/rerank_variants/CR_cw<W>_res1.0/eval_<set>/    residual fusion, alpha=1
# Compare with B_nopair_reranker / E_nopair_res1.0 (same features, W=0) from scripts/run_reranker_variants.sh. Choose W on the
# held-out grid in the summary json, not on dev. SELECT_BY=em+agreement switches the best-epoch rule (default em, comparable with B).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
RUNS=${RUNS:-deb_large_kl}
WEIGHTS=${WEIGHTS:-"0.5 1.0 2.0"}
EVAL_SETS=${EVAL_SETS:-"dev gan_dev addsent addonesent"}
SELECT_BY=${SELECT_BY:-em}
PRECISION=${PRECISION:-bf16}
DEVICE=${DEVICE:-auto}
PARA_TRAIN=${PARA_TRAIN:-data/paraphrases/train.jsonl}
EXTRA_RR_ARGS=${EXTRA_RR_ARGS:-}

evaluate() {  # evaluate RUN SET TAG RERANKER [extra evaluate.py args...]
  local RUN=$1 S=$2 TAG=$3 RR=$4; shift 4
  if [ -f "runs/$RUN/rerank_variants/$TAG/eval_$S/results.json" ]; then
    echo "   runs/$RUN/rerank_variants/$TAG/eval_$S/results.json exists, skipping"
  else
    echo "-- $RUN | $S | $TAG"
    $PY evaluate.py --weights "weights/$RUN/best.pt" --eval_set "$S" --reranker "$RR" --tag "$TAG" --precision "$PRECISION" --device "$DEVICE" "$@"
  fi
}

echo "== consistency reranker: runs=[$RUNS] weights=[$WEIGHTS] sets=[$EVAL_SETS] select_by=$SELECT_BY =="
echo "   started on $(hostname) at $(date)"
$PY download_eval_sets.py
for RUN in $RUNS; do
  [ -f "weights/$RUN/best.pt" ] || { echo "missing weights/$RUN/best.pt"; exit 1; }
  [ -f "$PARA_TRAIN" ] || { echo "missing $PARA_TRAIN (the consistency term needs the training paraphrase cache)"; exit 1; }
  for W in $WEIGHTS; do
    if [ -f "weights/$RUN/reranker_nopair_cw$W.pt" ]; then echo "   reranker_nopair_cw$W.pt exists, skipping"; else
      $PY train_reranker.py --weights "weights/$RUN/best.pt" --paraphrase_cache "$PARA_TRAIN" --no_pair_score --consistency_weight "$W" \
        --select_by "$SELECT_BY" --tag "_nopair_cw$W" --precision "$PRECISION" --device "$DEVICE" $EXTRA_RR_ARGS
    fi
    for S in $EVAL_SETS; do
      evaluate "$RUN" "$S" "CR_cw${W}_argmax" "weights/$RUN/reranker_nopair_cw$W.pt" --select_mode reranker
      evaluate "$RUN" "$S" "CR_cw${W}_res1.0"  "weights/$RUN/reranker_nopair_cw$W.pt" --select_mode residual --alpha 1.0
    done
  done
  for S in $EVAL_SETS; do $PY scripts/summarize_reranker_variants.py --run "$RUN" --eval_set "$S" || true; done
done
echo "consistency reranker finished at $(date)"
