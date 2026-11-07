#!/usr/bin/env bash
# End-to-end CPU smoke test of the whole pipeline with a tiny BERT (a few minutes on a laptop).
# It exercises: paraphrase cache -> consistency training with validation split -> reranker -> official-scorer
# evaluation on dev, Gan & Ng paraphrases and Adversarial SQuAD. Numbers are meaningless (tiny model, 1k questions).
#
#   bash scripts/smoke_test_cpu.sh            # default: google/bert_uncased_L-2_H-128_A-2 (BERT-tiny), 1000 questions
#   N=1000 MODEL=bert-base-uncased bash scripts/smoke_test_cpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
MODEL=${MODEL:-google/bert_uncased_L-2_H-128_A-2}
N=${N:-1000}
EPOCHS=${EPOCHS:-2}
RUN=${RUN:-smoke}
DEV=${DEVICE:-cpu}

echo "== 1/6 paraphrase cache (rule-based, all training questions; MarianMT is used for real runs) =="
$PY prepare_paraphrases.py --input provided/train-v1.1.json --output data/paraphrases/train_heuristic.jsonl --method heuristic --device "$DEV"

echo "== 2/6 extractor training (loss_mode=kl) =="
$PY train.py --run_name "$RUN" --model "$MODEL" --loss_mode kl --paraphrase_cache data/paraphrases/train_heuristic.jsonl \
  --max_examples "$N" --epochs "$EPOCHS" --batch_size 8 --max_seq_len 256 --val_max_examples 200 --device "$DEV"

echo "== 3/6 reranker =="
$PY train_reranker.py --weights "weights/$RUN/best.pt" --paraphrase_cache data/paraphrases/train_heuristic.jsonl --max_examples 300 --epochs 3 --device "$DEV"

echo "== 4/6 evaluate: dev (100 questions, rule-based paraphrases for the consistency metric) =="
$PY evaluate.py --weights "weights/$RUN/best.pt" --eval_set dev --max_examples 100 --paraphrase_method heuristic \
  --paraphrase_cache data/paraphrases/dev_heuristic.jsonl --reranker "weights/$RUN/reranker.pt" --pair_ensemble --n_boot 200 --device "$DEV"

echo "== 5/6 evaluate: Gan & Ng paraphrased dev (50 questions) =="
$PY evaluate.py --weights "weights/$RUN/best.pt" --eval_set gan_dev --max_examples 50 --reranker "weights/$RUN/reranker.pt" --device "$DEV"

echo "== 6/6 evaluate: Adversarial SQuAD AddOneSent (50 questions, rule-based paraphrases) =="
$PY evaluate.py --weights "weights/$RUN/best.pt" --eval_set addonesent --max_examples 50 --paraphrase_method heuristic \
  --paraphrase_cache data/paraphrases/addonesent_heuristic.jsonl --device "$DEV"

echo "smoke test finished; see runs/$RUN/"
