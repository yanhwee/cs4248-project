#!/usr/bin/env python3
"""Measure how long the real pipeline takes on this machine (CPU, and Apple MPS if present) and extrapolate to
full SQuAD, so the group can decide what to run locally vs. on the SoC cluster.

  python scripts/benchmark_cpu.py --model bert-base-uncased --steps 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g17qa.data import build_features, load_squad, make_collate_fn, WindowDataset  # noqa: E402
from g17qa.losses import compute_total_loss  # noqa: E402
from g17qa.model import SpanExtractor, load_tokenizer, tokenizer_uses_token_type_ids  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def time_training(model, loader, device, use_tt, steps, paired):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    times = []
    it = iter(loader)
    for s in range(steps + 1):
        batch = next(it)
        bt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        kw = {"input_ids": bt["input_ids"], "attention_mask": bt["attention_mask"]}
        if use_tt:
            kw["token_type_ids"] = bt["token_type_ids"]
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        out_q = model(**kw)
        out_p = None
        if paired:
            kwp = {"input_ids": bt["input_ids_p"], "attention_mask": bt["attention_mask_p"]}
            if use_tt:
                kwp["token_type_ids"] = bt["token_type_ids_p"]
            out_p = model(**kwp)
        loss = compute_total_loss("kl" if paired else "ce", out_q, bt, out_p)["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        if s > 0:  # first step is warm-up
            times.append(time.time() - t0)
    return sum(times) / len(times)


@torch.no_grad()
def time_inference(model, loader, device, use_tt, steps):
    model.eval()
    times = []
    it = iter(loader)
    for s in range(steps + 1):
        batch = next(it)
        kw = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
        if use_tt:
            kw["token_type_ids"] = batch["token_type_ids"].to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        model(**kw)
        if device.type == "mps":
            torch.mps.synchronize()
        if s > 0:
            times.append(time.time() - t0)
    return sum(times) / len(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_seq_len", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--devices", default="cpu,mps")
    p.add_argument("--skip_marian", action="store_true")
    p.add_argument("--only_marian", action="store_true", help="measure only the paraphrase pipeline")
    args = p.parse_args()

    tok = load_tokenizer(args.model)
    use_tt = tokenizer_uses_token_type_ids(tok)
    print(f"model {args.model} | torch {torch.__version__} | threads {torch.get_num_threads()}")

    # how many windows does the full training set produce at this configuration?
    t0 = time.time()
    train = load_squad(os.path.join(ROOT, "provided", "train-v1.1.json"))
    feats_all, _ = build_features(train, tok, args.max_seq_len, args.doc_stride, with_pairs=False)
    n_train_windows = len(feats_all)
    dev = load_squad(os.path.join(ROOT, "provided", "dev-v1.1.json"))
    feats_dev, _ = build_features(dev, tok, args.max_seq_len, args.doc_stride, with_pairs=False)
    n_dev_windows = len(feats_dev)
    print(f"full train: {len(train)} questions -> {n_train_windows} windows; dev: {len(dev)} -> {n_dev_windows} windows (tokenised in {time.time() - t0:.0f}s)")
    del feats_all

    sample = dev[:256]
    for e in sample:
        e.paraphrase = "Please tell me, " + e.question
    feats_pair, _ = build_features(sample, tok, args.max_seq_len, args.doc_stride, with_pairs=True)
    feats_single, _ = build_features(sample, tok, args.max_seq_len, args.doc_stride, with_pairs=False)
    coll = make_collate_fn(tok.pad_token_id, use_tt)
    ld_pair = DataLoader(WindowDataset(feats_pair), batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    ld_single = DataLoader(WindowDataset(feats_single), batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    ld_inf = DataLoader(WindowDataset(feats_single), batch_size=64, shuffle=False, collate_fn=coll)

    for dev_name in ([] if args.only_marian else args.devices.split(",")):
        dev_name = dev_name.strip()
        if dev_name == "mps" and not torch.backends.mps.is_available():
            continue
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(dev_name)
        steps_per_epoch = math.ceil(n_train_windows / args.batch_size)
        print(f"\n[{dev_name}] batch {args.batch_size} x {args.max_seq_len} tokens")
        model = SpanExtractor(args.model).to(device)
        for label, fn in (
            ("train step  ce (1 forward)", lambda: time_training(model, ld_single, device, use_tt, args.steps, paired=False)),
            ("train step  kl (2 forwards)", lambda: time_training(model, ld_pair, device, use_tt, args.steps, paired=True)),
            ("inference (batch 64)", lambda: time_inference(model, ld_inf, device, use_tt, args.steps)),
        ):
            try:
                t = fn()
            except (RuntimeError, torch.OutOfMemoryError) as e:  # e.g. MPS out of memory
                print(f"  {label:28s}: FAILED ({type(e).__name__}: {str(e).splitlines()[0][:90]}) -> try a smaller --batch_size")
                if device.type == "mps":
                    torch.mps.empty_cache()
                continue
            if label.startswith("inference"):
                print(f"  {label:28s}: {t:6.2f} s  -> {t * math.ceil(n_dev_windows / 64) / 60:6.1f} min for the dev set (x2 with paraphrases)")
            else:
                print(f"  {label:28s}: {t:6.2f} s  -> {t * steps_per_epoch / 3600:6.1f} h per epoch over full SQuAD")
        model = None
        if device.type == "mps":
            torch.mps.empty_cache()

    if not args.skip_marian:
        from g17qa.paraphrase import MarianBackTranslator, SimilarityFilter

        for dev_name in args.devices.split(","):
            dev_name = dev_name.strip()
            if dev_name == "mps" and not torch.backends.mps.is_available():
                continue
            device = torch.device(dev_name)
            qs = [e.question for e in dev[:128]]
            bt = MarianBackTranslator(device, batch_size=32)
            bt.back_translate(qs[:32])  # warm-up
            t0 = time.time()
            out = bt.back_translate(qs)
            dt = time.time() - t0
            sf = SimilarityFilter(device)
            t1 = time.time()
            sims = sf.similarities(qs, out)
            dsim = time.time() - t1
            n_changed = sum(1 for q, o in zip(qs, out) if q.strip().lower() != o.strip().lower())
            print(f"\n[{dev_name}] MarianMT back-translation: {len(qs) / dt:5.1f} questions/s -> {len(train) / (len(qs) / dt) / 3600:5.1f} h for the full training set; "
                  f"{n_changed}/{len(qs)} changed, mean similarity {sims.mean():.3f}, >=0.85: {(sims >= 0.85).mean() * 100:.0f}% | similarity filter {len(qs) / dsim:.0f} q/s")
            for q, o, s in list(zip(qs, out, sims))[:3]:
                print(f"     {s:.2f} | {q} -> {o}")
            del bt, sf


if __name__ == "__main__":
    main()
