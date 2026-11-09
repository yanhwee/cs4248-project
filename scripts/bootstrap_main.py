#!/usr/bin/env python3
"""Paired-bootstrap 95% intervals for CE vs CE+aug.+KL (extractor alone) on SQuAD v1.1 dev, from the committed
per-question predictions (runs/<run>/eval_dev/predictions.json). Same procedure as g17qa.consistency.bootstrap_delta (2,000
resamples of questions, seed 0). The intervals cover evaluation-sample uncertainty only, not training-seed variance.

Agreement and both-correct are NOT bootstrapped: they need the paraphrase-view predictions of the CE run, which
were not saved (only the 200 disagreements are in disagreements.jsonl).

  python scripts/bootstrap_main.py --dev provided/dev-v1.1.json   # writes docs/bootstrap_main.json
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from g17qa.metrics import exact_match_score, f1_score  # noqa: E402



def bootstrap_delta(base, variant, n_boot=2000, seed=0, ci=95.0):
    """Verbatim copy of g17qa.consistency.bootstrap_delta (copied so the script needs no torch/transformers)."""
    d = np.asarray(variant, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(1) * 100.0
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"delta": 100.0 * float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(d.size)}


PAIRS = [("deb_large_ce", "deb_large_kl")]  # only the large runs have per-question predictions committed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=os.path.join(ROOT, "provided", "dev-v1.1.json"))
    a = ap.parse_args()
    gold = {q["id"]: [x["text"] for x in q["answers"]]
            for art in json.load(open(a.dev, encoding="utf-8"))["data"] for p in art["paragraphs"] for q in p["qas"]}
    ids = sorted(gold)
    out = {}
    for base, var in PAIRS:
        pb, pv = (json.load(open(os.path.join(ROOT, "runs", r, "eval_dev", "predictions.json"))) for r in (base, var))
        res = {}
        for name, fn in (("em", exact_match_score), ("f1", f1_score)):
            x = [fn(pb.get(i, ""), gold[i]) for i in ids]
            y = [fn(pv.get(i, ""), gold[i]) for i in ids]
            res[name] = bootstrap_delta(x, y)
        out[f"{var}_vs_{base}"] = res
        print(var, "vs", base, {k: (round(v["delta"], 2), round(v["ci_low"], 2), round(v["ci_high"], 2)) for k, v in res.items()})
    json.dump(out, open(os.path.join(ROOT, "docs", "bootstrap_main.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
