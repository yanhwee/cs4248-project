#!/usr/bin/env python3
"""Train the span extractor (proposal items 1 and 2).

Protocol
  * 5% of the training *articles* are held out (--val_frac, --split_seed) for model selection and tuning; the
    official dev set is never touched here.
  * --loss_mode selects the objective (see g17qa/losses.py):
        ce         CE(q)                                plain extractor, paraphrases unused
        aug        CE(q) + CE(q')                       paraphrase augmentation only
        kl         CE(q) + CE(q') + lambda_kl * KL_sym  proposed consistency training
        ot | hungarian | dual                           1-D optimal-transport / span-matching variants (item 2a)
  * The checkpoint with the best validation F1 is kept as weights/<run_name>/best.pt.

Examples
  python train.py --run_name bert_ce  --loss_mode ce
  python train.py --run_name bert_kl  --loss_mode kl --paraphrase_cache data/paraphrases/train.jsonl --lambda_kl 1.0
  python train.py --run_name deb_kl   --model microsoft/deberta-v3-base --loss_mode kl \
      --paraphrase_cache data/paraphrases/train.jsonl --precision bf16
  # CPU smoke test
  python train.py --run_name smoke --model google/bert_uncased_L-2_H-128_A-2 --loss_mode kl \
      --paraphrase_cache data/paraphrases/smoke.jsonl --max_examples 300 --epochs 1 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from g17qa.data import WindowDataset, attach_paraphrases, build_features, gold_answers, load_squad, make_collate_fn, split_by_article
from g17qa.decode import run_extractor
from g17qa.losses import LOSS_MODES, PAIRED_MODES, compute_total_loss
from g17qa.metrics import summarize_outputs
from g17qa.model import SpanExtractor, load_tokenizer, save_checkpoint, tokenizer_uses_token_type_ids
from g17qa.utils import Timer, amp_dtype_for, count_trainable_parameters, ensure_dir, get_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", required=True)
    p.add_argument("--model", default="bert-base-uncased", help="HF encoder: bert-base-uncased | microsoft/deberta-v3-base | ...")
    p.add_argument("--train_file", default="provided/train-v1.1.json")
    p.add_argument("--paraphrase_cache", default=None, help="JSONL from prepare_paraphrases.py (required for paired modes)")
    p.add_argument("--loss_mode", choices=LOSS_MODES, default="kl")
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--lambda_ot", type=float, default=0.5)
    p.add_argument("--mu_hung", type=float, default=0.1)
    p.add_argument("--hung_topk", type=int, default=5)
    p.add_argument("--max_seq_len", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128, help="step between window starts (BERT run_squad convention)")
    p.add_argument("--max_query_len", type=int, default=64)
    p.add_argument("--max_answer_len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_examples", type=int, default=0, help="cap on training questions after the split (0 = all)")
    p.add_argument("--val_frac", type=float, default=0.05, help="fraction of training articles held out for validation")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_max_examples", type=int, default=2000, help="validation questions used during training (0 = all)")
    p.add_argument("--eval_every", type=int, default=0, help="validate every N optimizer steps (0 = once per epoch)")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--weights_dir", default="weights")
    p.add_argument("--log_interval", type=int, default=20)
    return p.parse_args()


def validate(model, tokenizer, examples, device, args, amp_dtype) -> Dict[str, float]:
    """EM / F1 on the held-out questions and, when their paraphrases are attached, paraphrase agreement
    (both views decoded), so that lambda can be chosen on accuracy *and* consistency."""
    with_pairs = any(e.has_pair for e in examples)
    outputs = run_extractor(
        model, tokenizer, examples, device, batch_size=max(8, args.batch_size * 2), max_seq_len=args.max_seq_len,
        doc_stride=args.doc_stride, max_query_len=args.max_query_len, top_k=5, max_answer_len=args.max_answer_len,
        with_pairs=with_pairs, want_vecs=False, amp_dtype=amp_dtype, progress=False, num_workers=args.num_workers,
    )
    model.train()
    return summarize_outputs(outputs, gold_answers(examples))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    amp_dtype = amp_dtype_for(device, args.precision)
    run_dir = ensure_dir(os.path.join(args.runs_dir, args.run_name))
    wdir = ensure_dir(os.path.join(args.weights_dir, args.run_name))
    timer = Timer()
    paired = args.loss_mode in PAIRED_MODES

    print("=" * 78)
    print(f"G17 extractor training | run={args.run_name} | model={args.model} | loss={args.loss_mode} | device={device} | precision={args.precision if amp_dtype else 'fp32'}")
    print("=" * 78)

    # ---- data ------------------------------------------------------------------------------
    all_ex = load_squad(args.train_file)
    train_ex, val_ex = split_by_article(all_ex, args.val_frac, args.split_seed)
    if args.max_examples:
        train_ex = train_ex[: args.max_examples]
    if args.val_max_examples and len(val_ex) > args.val_max_examples:
        val_eval = list(val_ex)
        random.Random(args.split_seed).shuffle(val_eval)  # a sample across the held-out articles, not the first few
        val_eval = val_eval[: args.val_max_examples]
    else:
        val_eval = val_ex
    print(f"articles: {len({e.title for e in all_ex})} | train questions {len(train_ex)} | val questions {len(val_ex)} (using {len(val_eval)} during training)")

    pair_stats: Dict[str, int] = {}
    cache_ok = bool(args.paraphrase_cache) and os.path.exists(args.paraphrase_cache)
    if paired and not cache_ok:
        raise SystemExit(f"--loss_mode {args.loss_mode} needs --paraphrase_cache (run prepare_paraphrases.py first)")
    if cache_ok:
        # Held-out questions get their paraphrases in every mode (including the plain baseline) so that validation
        # reports paraphrase agreement comparably across runs; training questions only in the paired modes.
        val_stats = attach_paraphrases(val_ex, args.paraphrase_cache)
        print(f"validation paraphrases: attached {val_stats['attached']} / {val_stats['total']} (for the agreement metric)")
        if paired:
            pair_stats = attach_paraphrases(train_ex, args.paraphrase_cache)
            print(f"training paraphrases: attached {pair_stats['attached']} / {pair_stats['total']} (rejected {pair_stats['rejected']}, missing {pair_stats['missing']})")
            if pair_stats["attached"] == 0:
                raise SystemExit("no accepted paraphrases for the training questions; check the cache / threshold")
    elif args.loss_mode == "ce":
        print("note: no --paraphrase_cache given, so validation will report EM/F1 only (no paraphrase agreement)")

    tokenizer = load_tokenizer(args.model)
    use_tt = tokenizer_uses_token_type_ids(tokenizer)
    feats, _ = build_features(train_ex, tokenizer, args.max_seq_len, args.doc_stride, args.max_query_len, with_pairs=paired)
    loader = DataLoader(
        WindowDataset(feats), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=make_collate_fn(tokenizer.pad_token_id, use_tt), drop_last=False,
    )
    n_pair_feats = sum(f.has_pair for f in feats)
    print(f"features: {len(feats)} windows ({n_pair_feats} with a paraphrase view) | {len(loader)} batches/epoch")

    # ---- model / optimiser -------------------------------------------------------------------
    model = SpanExtractor(args.model).to(device)
    print(f"parameters: {count_trainable_parameters(model) / 1e6:.1f}M trainable")
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    use_scaler = amp_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    config: Dict[str, Any] = {**vars(args), "n_train_questions": len(train_ex), "n_val_questions": len(val_ex), "n_features": len(feats), "pair_stats": pair_stats, "total_steps": total_steps}
    write_json(config, os.path.join(run_dir, "config.json"))

    history: List[Dict[str, Any]] = []
    val_history: List[Dict[str, Any]] = []
    best_f1 = -1.0
    best_entry: Dict[str, Any] = {}
    global_step = 0

    def do_validation(tag: str) -> None:
        nonlocal best_f1, best_entry
        if not val_eval:
            return
        m = validate(model, tokenizer, val_eval, device, args, amp_dtype)
        entry = {
            "step": global_step, "tag": tag, "exact": round(m["exact"], 3), "f1": round(m["f1"], 3), "n": m["total"],
            "pairs": m["pairs"], "agreement_em": round(m["agreement_em"], 3), "agreement_f1": round(m["agreement_f1"], 3),
            "elapsed_sec": round(timer.elapsed(), 1),
        }
        val_history.append(entry)
        write_json(val_history, os.path.join(run_dir, "val_history.json"))
        improved = m["f1"] > best_f1  # checkpoint selection within a run is by F1; lambda is chosen across runs
        if improved:
            best_f1 = m["f1"]
            best_entry = entry
            save_checkpoint(model, os.path.join(wdir, "best.pt"), config, {"val": entry})
        agr = f" | agreement {m['agreement_em']:.2f} over {m['pairs']} pairs" if m["pairs"] else ""
        print(f"  [val @ step {global_step} ({tag})] EM {m['exact']:.2f} | F1 {m['f1']:.2f} | n={m['total']}{agr}{'  *best*' if improved else ''}")

    # ---- training loop -----------------------------------------------------------------------
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(loader):
            kw = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
            if use_tt:
                kw["token_type_ids"] = batch["token_type_ids"].to(device)
            bt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out_q = model(**kw)
                out_p = None
                if paired and "input_ids_p" in batch:
                    kwp = {"input_ids": bt["input_ids_p"], "attention_mask": bt["attention_mask_p"]}
                    if use_tt:
                        kwp["token_type_ids"] = bt["token_type_ids_p"]
                    out_p = model(**kwp)
                comps = compute_total_loss(
                    args.loss_mode, out_q, bt, out_p, lambda_kl=args.lambda_kl, lambda_ot=args.lambda_ot,
                    mu_hung=args.mu_hung, hung_topk=args.hung_topk, max_answer_len=args.max_answer_len,
                )
                loss = comps["total"] / args.grad_accum
            scaler.scale(loss).backward()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                entry = {"step": global_step, "epoch": epoch, "lr": scheduler.get_last_lr()[0], "elapsed_sec": round(timer.elapsed(), 1)}
                entry.update({k: round(float(v.item()), 5) for k, v in comps.items()})
                history.append(entry)
                if global_step % args.log_interval == 0 or global_step == total_steps:
                    parts = " | ".join(f"{k} {entry[k]:.4f}" for k in ("total", "ce_q", "ce_p", "kl", "ot", "hung") if k in entry)
                    rate = global_step / max(1e-6, timer.elapsed())
                    eta = (total_steps - global_step) / max(1e-6, rate)
                    print(f"ep {epoch} step {global_step}/{total_steps} | {parts} | lr {entry['lr']:.2e} | {rate:.2f} it/s | eta {eta / 60:.1f} min")
                    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
                        json.dump(history, f)
                if args.eval_every and global_step % args.eval_every == 0:
                    do_validation("periodic")
        do_validation(f"epoch{epoch}")

    # ---- wrap-up --------------------------------------------------------------------------------
    save_checkpoint(model, os.path.join(wdir, "last.pt"), config)
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f)
    final = {}
    if val_ex and (args.val_max_examples and len(val_ex) > len(val_eval)):
        print(f"final validation on all {len(val_ex)} held-out questions with the best checkpoint ...")
        best_path = os.path.join(wdir, "best.pt")
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
            model.load_state_dict(state)
        final = validate(model, tokenizer, val_ex, device, args, amp_dtype)
        agr = f" | agreement {final['agreement_em']:.2f} over {final['pairs']} pairs" if final["pairs"] else ""
        print(f"  full val: EM {final['exact']:.2f} | F1 {final['f1']:.2f}{agr}")
    summary = {
        "run_name": args.run_name, "model": args.model, "loss_mode": args.loss_mode, "lambda_kl": args.lambda_kl,
        "best_val_f1": round(best_f1, 3), "best_val_exact": best_entry.get("exact"),
        "best_val_agreement_em": best_entry.get("agreement_em"), "best_val_agreement_f1": best_entry.get("agreement_f1"),
        "best_val_pairs": best_entry.get("pairs"), "selection_note": "best checkpoint within the run by validation F1; choose lambda across runs on validation F1 and agreement",
        "val_history": val_history, "final_full_val": final, "train_time_sec": round(timer.elapsed(), 1),
        "steps": global_step, "best_checkpoint": os.path.join(wdir, "best.pt"), "last_checkpoint": os.path.join(wdir, "last.pt"),
    }
    write_json(summary, os.path.join(run_dir, "train_summary.json"))
    agr = f" | agreement {best_entry['agreement_em']:.2f}" if best_entry.get("pairs") else ""
    print(f"done in {timer.elapsed() / 60:.1f} min | best val F1 {best_f1:.2f}{agr} | checkpoints in {wdir}")


if __name__ == "__main__":
    main()
