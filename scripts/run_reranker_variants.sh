#!/usr/bin/env bash
# Reranker ablation on EXISTING extractor checkpoints (no extractor retraining).
#
#   RUNS="bert_kl_lam1.0 deb_kl" bash scripts/run_reranker_variants.sh
#   RUNS=bert_kl_lam1.0 EVAL_SETS="dev" bash scripts/run_reranker_variants.sh
#
# For every run <R> (needs weights/<R>/best.pt and the train paraphrase cache) it
#   1. trains two rerankers on the held-out training articles (never dev), same code / seed / graded targets:
#        weights/<R>/reranker_pair.pt     with the paraphrase-derived features (pair_score, pair_gap, pair_missing)
#        weights/<R>/reranker_nopair.pt   without them
#      summaries: runs/<R>/reranker_summary_{pair,nopair}.json (incl. a held-out grid of alpha / gate settings)
#   2. evaluates the variants with the official scorer; outputs go to runs/<R>/rerank_variants/<variant>/eval_<set>/
#        A_pair_reranker      pair_score yes, pure reranker           (current behaviour)
#        B_nopair_reranker    pair_score no,  pure reranker
#        C_nopair_res0.25     pair_score no,  residual alpha 0.25
#        D_nopair_res0.5      pair_score no,  residual alpha 0.5
#        E_nopair_res1.0      pair_score no,  residual alpha 1.0
#      optional (set PAIR_RESIDUAL=1): F/G/H_pair_res<alpha>  = residual with pair_score yes
#      optional (set GATE_MARGINS="0.1 0.25 0.5"): <variant>_gate<m> for the pure-reranker B and residual-1.0 E
#   3. writes runs/<R>/rerank_variants/summary_<set>.md (generated from the results.json files)
# Existing artefacts (weights/<R>/reranker.pt, runs/<R>/eval_<set>/, reranker_summary.json) are never touched.
# Every step is skipped when its output exists, so the script can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
RUNS=${RUNS:-"bert_kl_lam1.0 deb_kl"}
EVAL_SETS=${EVAL_SETS:-"dev gan_dev"}     # add gan_adv addsent addonesent for the robustness sets
PRECISION=${PRECISION:-bf16}
DEVICE=${DEVICE:-auto}
PARA_TRAIN=${PARA_TRAIN:-data/paraphrases/train.jsonl}
ALPHAS=${ALPHAS:-"0.25 0.5 1.0"}
PAIR_RESIDUAL=${PAIR_RESIDUAL:-}
GATE_MARGINS=${GATE_MARGINS:-}
EXTRA_RR_ARGS=${EXTRA_RR_ARGS:-}          # e.g. "--epochs 30"

echo "== reranker variants: runs=[$RUNS] sets=[$EVAL_SETS] precision=$PRECISION device=$DEVICE =="
echo "   started on $(hostname) at $(date)"
$PY scripts/download_eval_sets.py

evaluate() {  # evaluate RUN SET TAG RERANKER [extra evaluate.py args...]
  local RUN=$1 S=$2 TAG=$3 RR=$4; shift 4
  if [ -f "runs/$RUN/rerank_variants/$TAG/eval_$S/results.json" ]; then
    echo "   runs/$RUN/rerank_variants/$TAG/eval_$S/results.json exists, skipping"
  else
    echo "-- $RUN | $S | $TAG"
    $PY scripts/evaluate.py --weights "weights/$RUN/best.pt" --eval_set "$S" --reranker "$RR" --tag "$TAG" --precision "$PRECISION" --device "$DEVICE" "$@"
  fi
}

for RUN in $RUNS; do
  [ -f "weights/$RUN/best.pt" ] || { echo "missing weights/$RUN/best.pt (this script reuses trained extractors)"; exit 1; }
  echo "== $RUN: train rerankers =="
  if [ -f "weights/$RUN/reranker_pair.pt" ]; then echo "   reranker_pair.pt exists, skipping"; else
    $PY scripts/train_reranker.py --weights "weights/$RUN/best.pt" --paraphrase_cache "$PARA_TRAIN" --use_pair_score --tag _pair --precision "$PRECISION" --device "$DEVICE" $EXTRA_RR_ARGS
  fi
  if [ -f "weights/$RUN/reranker_nopair.pt" ]; then echo "   reranker_nopair.pt exists, skipping"; else
    $PY scripts/train_reranker.py --weights "weights/$RUN/best.pt" --paraphrase_cache "$PARA_TRAIN" --no_pair_score --tag _nopair --precision "$PRECISION" --device "$DEVICE" $EXTRA_RR_ARGS
  fi
  PAIR="weights/$RUN/reranker_pair.pt"; NOPAIR="weights/$RUN/reranker_nopair.pt"

  echo "== $RUN: evaluate variants =="
  for S in $EVAL_SETS; do
    evaluate "$RUN" "$S" A_pair_reranker   "$PAIR"   --select_mode reranker
    evaluate "$RUN" "$S" B_nopair_reranker "$NOPAIR" --select_mode reranker
    i=0
    for A in $ALPHAS; do   # C, D, E
      L=$(echo C D E F G H | cut -d' ' -f$((i+1))); i=$((i+1))
      evaluate "$RUN" "$S" "${L}_nopair_res${A}" "$NOPAIR" --select_mode residual --alpha "$A"
    done
    if [ -n "$PAIR_RESIDUAL" ]; then
      for A in $ALPHAS; do evaluate "$RUN" "$S" "pair_res${A}" "$PAIR" --select_mode residual --alpha "$A"; done
    fi
    for M in $GATE_MARGINS; do
      evaluate "$RUN" "$S" "B_nopair_reranker_gate${M}" "$NOPAIR" --select_mode reranker --override_margin "$M"
      evaluate "$RUN" "$S" "E_nopair_res1.0_gate${M}"   "$NOPAIR" --select_mode residual --alpha 1.0 --override_margin "$M"
    done
    $PY scripts/summarize_reranker_variants.py --run "$RUN" --eval_set "$S"
  done
done
echo "reranker variants finished at $(date)"
