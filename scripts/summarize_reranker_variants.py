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

    def ci(x):
        return "n/a" if not x else f"[{x['ci_low']:+.2f}, {x['ci_high']:+.2f}]{'*' if x['excludes_zero'] else ''}"

    rows, ext_line = [], None
    for p in files:
        r = json.load(open(p, encoding="utf-8"))
        v, ext, rr = r.get("reranker_variant"), r["extractor"], r.get("reranker")
        ea = (ext.get("consistency") or {}).get("agreement_em")
        if ext_line is None:
            rk = ext.get("recall_at_k") or {}
            pa = ext.get("pair_accuracy")
            ext_line = (f"Extractor only: EM {f(ext['official']['exact'])} / F1 {f(ext['official']['f1'])} / agreement {f(ea)} (n={ext['official']['total']})"
                        + (f"; both views correct {f(pa['both_correct'])}" if pa else "") + ". Candidate ceilings: "
                        + ", ".join(f"R@{k.split('@')[1]} EM {f(x)}" for k, x in rk.items() if k.startswith("recall_em"))
                        + (f"; best-F1@{ext['candidate_ceiling']['top_k']} {f(ext['candidate_ceiling']['best_f1'])}" if ext.get("candidate_ceiling") else ""))

        def add(tag, pair, mode, alpha, gate, blk):
            c, d, pacc, dci = blk["changes"], blk["delta"], blk.get("pair_accuracy"), blk.get("delta_ci") or {}
            ra = (blk.get("consistency") or {}).get("agreement_em")
            rows.append("| {tag} | {pair} | {mode} | {alpha} | {gate} | {em} | {f1} | {ag} | {dem} | {demci} | {df1} | {dag} | {dagci} | {both} | {fix} | {brk} |".format(
                tag=tag, pair=pair, mode=mode, alpha=alpha, gate=gate, em=f(blk["official"]["exact"]), f1=f(blk["official"]["f1"]), ag=f(ra),
                dem=f(d["d_em"], sign=True), demci=ci(dci.get("em")), df1=f(d["d_f1"], sign=True), dag=f(d["d_agreement_em"], sign=True), dagci=ci(dci.get("agreement_em")),
                both=f(pacc["both_correct"]) if pacc else "n/a", fix=f"{c['FIX']} ({c['FIX_pct']:.2f}%)", brk=f"{c['BREAK']} ({c['BREAK_pct']:.2f}%)"))

        if rr and v:
            add(v["tag"], "yes" if v["use_pair_score"] else "no", v["select_mode"], v["alpha"] if v["select_mode"] == "residual" else "-",
                v["override_margin"] if v["override_margin"] is not None else "-", rr)
        if r.get("pair_ensemble"):
            add(f"{(v or {}).get('tag', 'run')} / pair-ensemble", "yes (extractor's)", "ensemble", f"beta={r['pair_ensemble']['beta']}", "-", r["pair_ensemble"])
    if not rows:
        raise SystemExit("no reranker or ensemble results found")
    head = ["| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dEM 95% CI | dF1 | dAgreement | dAgreement 95% CI | Both views correct | FIX | BREAK |",
            "| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | :-- | --: | --: | :-- | --: | --: | --: |"]
    out = (f"### {a.run} | {a.eval_set}\n\n{ext_line}\n\n" + "\n".join(head + rows)
           + "\n\n95% paired-bootstrap CIs (resampling questions; agreement: pairs); `*` = interval excludes 0. Both views correct = share of pairs where the "
             "question's and the paraphrase's answers are both exact matches.\n")
    print(out)
    with open(os.path.join(base, f"summary_{a.eval_set}.md"), "w", encoding="utf-8") as fh:
        fh.write(out)


if __name__ == "__main__":
    main()
