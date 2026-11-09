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
from g17qa.consistency import bootstrap_delta  # noqa: E402
from g17qa.metrics import exact_match_score, f1_score  # noqa: E402


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
