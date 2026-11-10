#!/usr/bin/env python3
"""t-SNE of (question, paraphrase) representations (proposal item 2b).

For each dev question with a cached paraphrase, both views are encoded on the question's first passage window
and the [CLS] representation is taken. The plot shows t-SNE of all 2n points with a line joining each pair,
coloured by whether the extractor gives the pair the same answer. The JSON next to the figure records the
mean cosine distance between pair representations, which should shrink after consistency training.

  python scripts/tsne_pairs.py --weights weights/bert_ce/best.pt --paraphrase_cache data/paraphrases/dev.jsonl --n 300
  python scripts/tsne_pairs.py --weights weights/bert_kl/best.pt --paraphrase_cache data/paraphrases/dev.jsonl --n 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from g17qa.data import WindowDataset, attach_paraphrases, build_features, load_squad, make_collate_fn  # noqa: E402
from g17qa.decode import run_extractor  # noqa: E402
from g17qa.metrics import compute_exact  # noqa: E402
from g17qa.model import load_checkpoint, tokenizer_uses_token_type_ids  # noqa: E402
from g17qa.utils import get_device, set_seed  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data_file", default="provided/dev-v1.1.json")
    p.add_argument("--paraphrase_cache", default="data/paraphrases/dev.jsonl")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", default=None, help="default figures/tsne_<run>.pdf")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    model, tok, cfg = load_checkpoint(args.weights, device)
    run = cfg.get("run_name") or os.path.basename(os.path.dirname(args.weights))
    out = args.out or os.path.join("figures", f"tsne_{run}.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ex = load_squad(args.data_file)
    attach_paraphrases(ex, args.paraphrase_cache)
    ex = [e for e in ex if e.has_pair][: args.n]
    if len(ex) < 10:
        raise SystemExit("need at least 10 questions with cached paraphrases (run prepare_paraphrases.py on the dev file)")

    # answer agreement per pair
    outs = run_extractor(model, tok, ex, device, max_seq_len=cfg.get("max_seq_len", 384), doc_stride=cfg.get("doc_stride", 128), top_k=1, with_pairs=True, progress=False)
    agree = np.array([compute_exact(outs[e.qid].pred, outs[e.qid].pred_p) for e in ex], dtype=bool)

    # [CLS] representations of both views on the first window
    feats, _ = build_features(ex, tok, cfg.get("max_seq_len", 384), cfg.get("doc_stride", 128), with_pairs=True)
    first = {}
    for f in feats:
        first.setdefault(f.qid, f)
    feats = [first[e.qid] for e in ex]
    use_tt = tokenizer_uses_token_type_ids(tok)
    loader = DataLoader(WindowDataset(feats), batch_size=32, shuffle=False, collate_fn=make_collate_fn(tok.pad_token_id, use_tt))
    reps_q, reps_p = [], []
    with torch.no_grad():
        for b in loader:
            for key, store in (("", reps_q), ("_p", reps_p)):
                kw = {"input_ids": b["input_ids" + key].to(device), "attention_mask": b["attention_mask" + key].to(device)}
                if use_tt:
                    kw["token_type_ids"] = b["token_type_ids" + key].to(device)
                store.append(model(**kw, return_hidden=True)["hidden"][:, 0].float().cpu().numpy())
    Q, P = np.concatenate(reps_q), np.concatenate(reps_p)
    cos = 1 - (Q * P).sum(1) / (np.linalg.norm(Q, axis=1) * np.linalg.norm(P, axis=1) + 1e-9)

    from sklearn.manifold import TSNE

    Z = TSNE(n_components=2, perplexity=min(30, len(ex) - 1), random_state=args.seed, init="pca").fit_transform(np.concatenate([Q, P]))
    zq, zp = Z[: len(ex)], Z[len(ex):]
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    for i in range(len(ex)):
        ax.plot([zq[i, 0], zp[i, 0]], [zq[i, 1], zp[i, 1]], color="#2a9d8f" if agree[i] else "#e76f51", lw=0.5, alpha=0.6)
    ax.scatter(zq[:, 0], zq[:, 1], s=6, c="#1b4965", label="question")
    ax.scatter(zp[:, 0], zp[:, 1], s=6, marker="^", c="#f4a261", label="paraphrase")
    ax.set_title(f"{run}: [CLS] t-SNE, pairs agree {agree.mean() * 100:.0f}%", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=200, bbox_inches="tight")
    stats = {"run": run, "weights": args.weights, "n_pairs": len(ex), "mean_cosine_distance": float(cos.mean()), "median_cosine_distance": float(np.median(cos)), "answer_agreement": float(agree.mean() * 100)}
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
