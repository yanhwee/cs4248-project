#!/usr/bin/env python3
"""Train the learned span reranker (proposal item 3) on top of a trained extractor.

Data: the same 5% held-out training articles the extractor used for validation (read from the checkpoint
config, so the official dev set is never used). Those articles are split 80/20 by article into reranker
train / early-stopping sets. For every question the extractor's top-k distinct candidates are extracted
together with their hidden-state features and, when a paraphrase is cached, the paraphrase view's scores.

Example
  python train_reranker.py --weights weights/bert_kl/best.pt --paraphrase_cache data/paraphrases/train.jsonl
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Any, Dict


from g17qa.data import attach_paraphrases, gold_answers, load_squad, split_by_article
from g17qa.decode import run_extractor
from g17qa.model import load_checkpoint
from g17qa.reranker import build_items, candidate_ceiling, evaluate_items, save_reranker, train_reranker
from g17qa.utils import amp_dtype_for, ensure_dir, get_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="extractor checkpoint (weights/<run>/best.pt)")
    p.add_argument("--train_file", default=None, help="defaults to the file recorded in the checkpoint")
    p.add_argument("--paraphrase_cache", default=None, help="train paraphrase cache; enables the pair-score feature")
    p.add_argument("--out", default=None, help="defaults to weights/<run>/reranker.pt")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--holdout_frac", type=float, default=0.2, help="share of validation articles used for early stopping")
    p.add_argument("--max_examples", type=int, default=0, help="cap on validation questions (0 = all)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--proj", type=int, default=128)
    p.add_argument("--target_mode", choices=["graded", "exact", "f1"], default="graded",
                   help="graded: EM=1, partial-F1 candidates get partial_weight*F1; exact: EM candidates only; f1: legacy best-F1 fallback")
    p.add_argument("--partial_weight", type=float, default=0.3, help="graded mode: weight of a partial-F1 candidate relative to an exact match")
    p.add_argument("--extract_batch_size", type=int, default=32)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs_dir", default="runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    amp = amp_dtype_for(device, args.precision)
    t0 = time.time()

    model, tokenizer, cfg = load_checkpoint(args.weights, device)
    run_name = cfg.get("run_name") or os.path.basename(os.path.dirname(os.path.abspath(args.weights)))
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.weights)), "reranker.pt")
    run_dir = ensure_dir(os.path.join(args.runs_dir, run_name))
    train_file = args.train_file or cfg.get("train_file", "provided/train-v1.1.json")
    val_frac, split_seed = cfg.get("val_frac", 0.05), cfg.get("split_seed", 42)
    print(f"extractor {args.weights} ({cfg.get('model')}) | device {device} | validation split: {val_frac} of articles, seed {split_seed}")

    all_ex = load_squad(train_file)
    _, val_ex = split_by_article(all_ex, val_frac, split_seed)
    if not val_ex:
        raise SystemExit("the checkpoint was trained with val_frac=0; the reranker needs held-out articles")
    if args.max_examples and len(val_ex) > args.max_examples:
        val_ex = list(val_ex)
        random.Random(args.seed).shuffle(val_ex)  # keep several articles when capping
        val_ex = val_ex[: args.max_examples]
    rr_train, rr_dev = split_by_article(val_ex, args.holdout_frac, split_seed + 1)
    if not rr_train or not rr_dev:
        print("warning: too few held-out articles for an article-level reranker split; using a question-level split")
        idx = list(range(len(val_ex)))
        random.Random(split_seed + 1).shuffle(idx)
        n_dev = max(1, int(round(args.holdout_frac * len(val_ex))))
        rr_dev = [val_ex[i] for i in idx[:n_dev]]
        rr_train = [val_ex[i] for i in idx[n_dev:]]
    use_pairs = False
    if args.paraphrase_cache and os.path.exists(args.paraphrase_cache):
        st = attach_paraphrases(val_ex, args.paraphrase_cache)
        use_pairs = st["attached"] > 0
        print(f"paraphrases attached to {st['attached']} / {st['total']} validation questions")
    print(f"reranker train questions {len(rr_train)} | early-stop questions {len(rr_dev)} | top_k {args.top_k}")

    golds = gold_answers(val_ex)
    common = dict(
        batch_size=args.extract_batch_size, max_seq_len=cfg.get("max_seq_len", 384), doc_stride=cfg.get("doc_stride", 128),
        max_query_len=cfg.get("max_query_len", 64), top_k=args.top_k, max_answer_len=cfg.get("max_answer_len", 30),
        with_pairs=use_pairs, want_vecs=True, amp_dtype=amp,
    )
    print("extracting candidates + features (train part) ...")
    out_train = run_extractor(model, tokenizer, rr_train, device, **common)
    print("extracting candidates + features (early-stop part) ...")
    out_dev = run_extractor(model, tokenizer, rr_dev, device, **common)
    items_train = build_items(out_train, golds, use_pair_view=True)
    items_dev = build_items(out_dev, golds, use_pair_view=False)

    reranker, history = train_reranker(
        items_train, items_dev, golds, model.hidden_size, device, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, proj=args.proj, seed=args.seed, target_mode=args.target_mode, partial_weight=args.partial_weight,
    )
    before = evaluate_items(None, items_dev, golds, device)
    after = evaluate_items(reranker, items_dev, golds, device)
    summary: Dict[str, Any] = {
        "extractor": args.weights, "reranker": out_path, "top_k": args.top_k, "pair_features": use_pairs,
        "target_mode": args.target_mode, "partial_weight": args.partial_weight,
        "ceiling_train": candidate_ceiling(items_train), "ceiling_dev": candidate_ceiling(items_dev),
        "n_train_items": len(items_train), "n_dev_items": len(items_dev),
        "dev_top1": before, "dev_reranked": after, "history": history, "elapsed_sec": round(time.time() - t0, 1),
    }
    save_reranker(reranker, out_path, {"extractor": args.weights, "top_k": args.top_k, "pair_features": use_pairs, "target_mode": args.target_mode, "summary": {k: v for k, v in summary.items() if k != "history"}})
    write_json(summary, os.path.join(run_dir, "reranker_summary.json"))
    print(f"held-out articles: top-1 EM {before['exact']:.2f} F1 {before['f1']:.2f} -> reranked EM {after['exact']:.2f} F1 {after['f1']:.2f} ")
    print(f"ceiling@{args.top_k} on held-out articles: Recall@{args.top_k} EM {before['recall_em']:.2f} | best-F1@{args.top_k} {before['best_f1']:.2f}  (reranker EM/F1 cannot exceed these)")
    print(f"saved {out_path} | summary {os.path.join(run_dir, 'reranker_summary.json')}")


if __name__ == "__main__":
    main()
