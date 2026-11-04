#!/usr/bin/env bash
# Full experiment pipeline, sequential. Works locally (GPU/MPS/CPU) or inside one Slurm allocation.
#
#   MODEL=bert-base-uncased MODES="ce aug kl" bash scripts/run_pipeline.sh
#   MODEL=microsoft/deberta-v3-base MODES="kl" PRECISION=bf16 RUN_PREFIX=deb bash scripts/run_pipeline.sh
#   MODEL=microsoft/deberta-v3-large MODES="ce kl" RUN_PREFIX=deb_large BATCH=8 GRAD_ACCUM=2 LR=1e-5 PRECISION=bf16 \
#     EXTRA_TRAIN_ARGS=--grad_checkpointing bash scripts/run_pipeline.sh      # large encoder (see scripts/submit_deberta_large.sh)
#
# Stages (each is skipped when its output already exists, so the script can be re-run after an interruption):
#   1. paraphrase caches for train and dev (MarianMT + similarity filter)
#   2. extractor training for every mode in $MODES, 5% article-level validation split, best checkpoint kept
#   3. learned span reranker on the validation articles
#   4. official-scorer evaluation on dev, Gan & Ng paraphrased sets and Adversarial SQuAD, with and without reranker
#   5. figures and tables from the collected results
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
MODEL=${MODEL:-bert-base-uncased}
MODES=${MODES:-"ce aug kl"}
RUN_PREFIX=${RUN_PREFIX:-bert}
EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}         # optimiser step every GRAD_ACCUM batches (effective batch = BATCH x GRAD_ACCUM)
LR=${LR:-3e-5}                      # 3e-5 for base encoders; 1e-5 for microsoft/deberta-v3-large
PRECISION=${PRECISION:-fp16}        # fp16 for T4/V100, bf16 for A100/CPU, fp32 otherwise
DEVICE=${DEVICE:-auto}
LAMBDA_KL=${LAMBDA_KL:-1.0}         # single lambda for the kl run ...
KL_LAMBDAS=${KL_LAMBDAS:-}          # ... or a sweep, e.g. "0.5 1.0 2.0": one run per value, named <prefix>_kl_lam<value>;
                                    # compare best_val_f1 in runs/*/train_summary.json (held-out articles, not dev)
MAX_EXAMPLES=${MAX_EXAMPLES:-0}     # 0 = full training set
EVAL_SETS=${EVAL_SETS:-"dev gan_dev gan_adv addsent addonesent"}
STAGES=${STAGES:-"paraphrases train reranker eval figures"}   # subset to run, e.g. STAGES=paraphrases
PARA_TRAIN=data/paraphrases/train.jsonl
PARA_DEV=data/paraphrases/dev.jsonl
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
has_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

echo "== pipeline: model=$MODEL modes=[$MODES] epochs=$EPOCHS batch=$BATCH x$GRAD_ACCUM lr=$LR precision=$PRECISION device=$DEVICE stages=[$STAGES] =="
echo "   started on $(hostname) at $(date)"
$PY download_eval_sets.py

if has_stage paraphrases; then
  echo "== 1. paraphrase caches =="
  $PY prepare_paraphrases.py --input provided/train-v1.1.json --output "$PARA_TRAIN" --device "$DEVICE"
  $PY prepare_paraphrases.py --input provided/dev-v1.1.json --output "$PARA_DEV" --device "$DEVICE"
  for S in addsent addonesent; do
    F=data/external/adversarial_squad/$( [ "$S" = addsent ] && echo AddSent || echo AddOneSent ).json
    $PY prepare_paraphrases.py --input "$F" --output "data/paraphrases/$S.jsonl" --device "$DEVICE"
  done
fi
if ! has_stage train && ! has_stage reranker && ! has_stage eval; then
  has_stage figures && $PY visualize.py || true
  echo "pipeline finished at $(date)"; exit 0
fi

# expand "kl" into one job per lambda when KL_LAMBDAS is set
JOBS=""
for MODE in $MODES; do
  if [ "$MODE" = "kl" ] && [ -n "$KL_LAMBDAS" ]; then
    for L in $KL_LAMBDAS; do JOBS="$JOBS kl:$L:${RUN_PREFIX}_kl_lam${L}"; done
  else
    JOBS="$JOBS $MODE:$LAMBDA_KL:${RUN_PREFIX}_${MODE}"
  fi
done

for JOB in $JOBS; do
  MODE=${JOB%%:*}; REST=${JOB#*:}; LAM=${REST%%:*}; RUN=${REST#*:}
  if has_stage train; then
    echo "== 2. train $RUN (mode=$MODE lambda_kl=$LAM) =="
    # train_summary.json is written only when training completes; best.pt alone may come from an interrupted run
    if [ -f "runs/$RUN/train_summary.json" ] && [ -f "weights/$RUN/best.pt" ]; then
      echo "   runs/$RUN/train_summary.json exists, skipping training"
    else
      if [ -f "weights/$RUN/best.pt" ]; then
        echo "   found a checkpoint from an interrupted run; retraining $RUN from scratch"
        rm -rf "weights/$RUN" "runs/$RUN"
      fi
      $PY train.py --run_name "$RUN" --model "$MODEL" --loss_mode "$MODE" --paraphrase_cache "$PARA_TRAIN" \
        --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" --lr "$LR" --precision "$PRECISION" --device "$DEVICE" --lambda_kl "$LAM" \
        --max_examples "$MAX_EXAMPLES" $EXTRA_TRAIN_ARGS
    fi
  fi
  if has_stage reranker; then
    echo "== 3. reranker for $RUN =="
    if [ -f "weights/$RUN/reranker.pt" ]; then
      echo "   weights/$RUN/reranker.pt exists, skipping"
    else
      $PY train_reranker.py --weights "weights/$RUN/best.pt" --paraphrase_cache "$PARA_TRAIN" --precision "$PRECISION" --device "$DEVICE"
    fi
  fi
  if has_stage eval; then
    echo "== 4. evaluation for $RUN =="
    RR_ARGS=""; [ -f "weights/$RUN/reranker.pt" ] && RR_ARGS="--reranker weights/$RUN/reranker.pt"
    for S in $EVAL_SETS; do
      if [ -f "runs/$RUN/eval_$S/results.json" ]; then
        echo "   runs/$RUN/eval_$S/results.json exists, skipping"
      else
        $PY evaluate.py --weights "weights/$RUN/best.pt" --eval_set "$S" $RR_ARGS --precision "$PRECISION" --device "$DEVICE"
      fi
    done
  fi
done

if has_stage figures; then
  echo "== 5. figures and tables =="
  $PY visualize.py || true
fi
echo "pipeline finished at $(date)"
