#!/usr/bin/env python3
"""Tabulate the reranker variants of one run from their results.json files (nothing is hard-coded).

    python scripts/summarize_reranker_variants.py --run bert_kl_lam1.0 --eval_set dev

Reads  runs/<run>/rerank_variants/*/eval_<set>/results.json  and writes  runs/<run>/rerank_variants/summary_<set>.md
(also printed). Columns: extractor-only vs reranked EM / F1 / paraphrase agreement, the deltas, and the
FIX / BREAK / SAME counts (correct = exact match).
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval_set", default="dev")
    ap.add_argument("--runs_dir", default="runs")
    a = ap.parse_args()
    base = os.path.join(a.runs_dir, a.run, "rerank_variants")
    files = sorted(glob.glob(os.path.join(base, "*", f"eval_{a.eval_set}", "results.json")))
    if not files:
        raise SystemExit(f"no results under {base}/*/eval_{a.eval_set}/")

    def f(x, d=2, sign=False):
        return "n/a" if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")

    rows, ext_line = [], None
    for p in files:
        r = json.load(open(p, encoding="utf-8"))
        v, ext, rr = r.get("reranker_variant"), r["extractor"], r.get("reranker")
        if not rr:
            continue
        ea = (ext.get("consistency") or {}).get("agreement_em")
        ra = (rr.get("consistency") or {}).get("agreement_em")
        if ext_line is None:
            rk = ext.get("recall_at_k") or {}
            ext_line = (f"Extractor only: EM {f(ext['official']['exact'])} / F1 {f(ext['official']['f1'])} / agreement {f(ea)} (n={ext['official']['total']}). "
                        + "Candidate ceilings: " + ", ".join(f"R@{k.split('@')[1]} EM {f(x)}" for k, x in rk.items() if k.startswith("recall_em"))
                        + (f"; best-F1@{ext['candidate_ceiling']['top_k']} {f(ext['candidate_ceiling']['best_f1'])}" if ext.get("candidate_ceiling") else ""))
        c, d = rr["changes"], rr["delta"]
        rows.append("| {tag} | {pair} | {mode} | {alpha} | {gate} | {em} | {f1} | {ag} | {dem} | {df1} | {dag} | {fix} | {brk} | {sc} | {sw} |".format(
            tag=v["tag"], pair="yes" if v["use_pair_score"] else "no", mode=v["select_mode"], alpha=v["alpha"] if v["select_mode"] == "residual" else "-",
            gate=v["override_margin"] if v["override_margin"] is not None else "-", em=f(rr["official"]["exact"]), f1=f(rr["official"]["f1"]), ag=f(ra),
            dem=f(d["d_em"], sign=True), df1=f(d["d_f1"], sign=True), dag=f(d["d_agreement_em"], sign=True),
            fix=f"{c['FIX']} ({c['FIX_pct']:.2f}%)", brk=f"{c['BREAK']} ({c['BREAK_pct']:.2f}%)", sc=c["SAME_CORRECT"], sw=c["SAME_WRONG"]))
    head = ["| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dF1 | dAgreement | FIX | BREAK | SAME_CORRECT | SAME_WRONG |",
            "| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |"]
    out = f"### {a.run} | {a.eval_set}\n\n{ext_line}\n\n" + "\n".join(head + rows) + "\n"
    print(out)
    with open(os.path.join(base, f"summary_{a.eval_set}.md"), "w", encoding="utf-8") as fh:
        fh.write(out)


if __name__ == "__main__":
    main()
