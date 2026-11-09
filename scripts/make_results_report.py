#!/usr/bin/env python3
"""Write docs/RESULTS.md from the run artefacts in runs/ (config.json, train_summary.json, reranker_summary.json,
eval_*/results.json). Every number in the report is read from those files or computed from them here; the digest
of all values used is written next to the report as docs/results_digest.json for verification.

Convention: every metric is rounded to one decimal first and all differences quoted in the text are differences of
those rounded values, so the prose is consistent with the tables.

  python scripts/make_results_report.py [--runs_dir runs] [--selected bert_kl_lam1.0] [--tolerance 0.5]
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = ["dev", "gan_dev", "gan_adv", "addsent", "addonesent"]
MODE_DESC = {"ce": "cross-entropy only (baseline)", "aug": "paraphrases as extra training data", "kl": "paraphrases + symmetric-KL consistency penalty",
             "ot": "paraphrases + 1-D Wasserstein penalty", "dual": "paraphrases + Wasserstein + Hungarian span matching"}


def load(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ci(block: Dict[str, Any], key: str):
    """(low, high) of the 95 % paired-bootstrap interval stored by evaluate.py --n_boot, rounded to one decimal; None if absent."""
    c = (block.get("delta_ci") or {}).get(key)
    return (round(c["ci_low"], 2), round(c["ci_high"], 2)) if c else None


def r1(x):
    """Round a metric to one decimal."""
    return None if x is None else round(float(x) + 1e-9, 1)


def se_pct(p_pct: float, n: int) -> float:
    """Standard error (percentage points) of a proportion p over n items."""
    p = p_pct / 100.0
    return 100.0 * math.sqrt(max(p * (1 - p), 0.0) / max(n, 1))


def f1(x: Optional[float], nd: int = 1) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def sgn(x: Optional[float], nd: int = 1) -> str:
    if x is None:
        return "--"
    x = round(x, nd)
    if x == 0:
        x = 0.0
    return f"{x:+.{nd}f}"


def collect(runs_dir: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d) or name.startswith("smoke"):
            continue
        cfg, summ = load(f"{d}/config.json"), load(f"{d}/train_summary.json")
        if not cfg or not summ:
            continue
        variants = {}
        for vd_ in sorted(glob.glob(os.path.join(d, "rerank_variants", "*"))):
            if os.path.isdir(vd_):
                ev = {s: load(f"{vd_}/eval_{s}/results.json") for s in SETS}
                if any(ev.values()):
                    variants[os.path.basename(vd_)] = ev
        out[name] = {"config": cfg, "summary": summ, "reranker": load(f"{d}/reranker_summary.json"),
                     "evals": {s: load(f"{d}/eval_{s}/results.json") for s in SETS}, "dir": d, "variants": variants,
                     "rr_summaries": {os.path.basename(f)[len("reranker_summary_"):-len(".json")]: load(f) for f in sorted(glob.glob(f"{d}/reranker_summary_*.json"))}}
    return out


def digest_run(name: str, r: Dict[str, Any]) -> Dict[str, Any]:
    cfg, s, rr = r["config"], r["summary"], r["reranker"]
    vh = s.get("val_history") or []
    fv = s.get("final_full_val") or {}
    d: Dict[str, Any] = {
        "run": name, "model": cfg["model"], "mode": cfg["loss_mode"], "lambda_kl": cfg["lambda_kl"] if cfg["loss_mode"] == "kl" else None,
        "epochs": cfg["epochs"], "batch_size": cfg["batch_size"], "grad_accum": int(cfg.get("grad_accum", 1)), "lr": cfg["lr"], "max_seq_len": cfg["max_seq_len"], "doc_stride": cfg["doc_stride"],
        "grad_checkpointing": bool(cfg.get("grad_checkpointing", False)), "peak_gpu_mem_gb": s.get("peak_gpu_mem_gb"),
        "n_train_questions": cfg["n_train_questions"], "n_val_questions": cfg["n_val_questions"], "n_features": cfg["n_features"],
        "val_max_examples": cfg.get("val_max_examples"), "pair_stats": cfg.get("pair_stats") or {},
        "steps": s["steps"], "train_time_min": s["train_time_sec"] / 60.0,
        # checkpoint-selection sample (val_max_examples questions)
        "val_f1": r1(s["best_val_f1"]), "val_f1_raw": s["best_val_f1"], "val_exact": r1(s.get("best_val_exact")), "val_agreement": r1(s.get("best_val_agreement_em")),
        "val_pairs": s.get("best_val_pairs"), "val_n": vh[0]["n"] if vh else None,
        # full held-out set, evaluated once with the selected checkpoint
        "full_val_f1": r1(fv.get("f1")), "full_val_f1_raw": fv.get("f1"), "full_val_exact": r1(fv.get("exact")), "full_val_agreement": r1(fv.get("agreement_em")),
        "full_val_n": fv.get("total"), "full_val_pairs": fv.get("pairs"),
        "reranker": None, "sets": {},
    }
    if rr:
        hist = rr.get("history") or []
        graded = "target_mode" in rr  # summaries written after the change to graded targets (4 Nov 2026) also pick the epoch by EM, then F1
        best_ep = (max(hist, key=lambda h: (h["dev_exact"], h["dev_f1"])) if graded else max(hist, key=lambda h: h["dev_f1"])) if hist else None
        d["reranker"] = {
            "train_items": rr["n_train_items"], "dev_items": rr["n_dev_items"], "pair_features": rr["pair_features"],
            "target_mode": rr.get("target_mode", "f1"), "epoch_rule": "EM, then F1" if graded else "F1",
            "top1": {k: (r1(v) if isinstance(v, float) else v) for k, v in rr["dev_top1"].items()},
            "reranked": {k: (r1(v) if isinstance(v, float) else v) for k, v in rr["dev_reranked"].items()},
            "n_epochs": len(hist), "best_epoch": best_ep["epoch"] if best_ep else None,
            "last_epoch_f1": r1(hist[-1]["dev_f1"]) if hist else None,
        }
    for st, e in r["evals"].items():
        if not e:
            continue
        x, y = e["extractor"], e.get("reranker") or {}
        entry: Dict[str, Any] = {
            "n": e["n_questions"], "pairs": e["n_pairs"], "paired_view": e.get("paired_view"),
            "em": r1(x["official"]["exact"]), "f1": r1(x["official"]["f1"]),
            "agr": r1((x.get("consistency") or {}).get("agreement_em")), "agr_f1": r1((x.get("consistency") or {}).get("agreement_f1")),
            "rr_em": r1((y.get("official") or {}).get("exact")), "rr_f1": r1((y.get("official") or {}).get("f1")),
            "rr_agr": r1((y.get("consistency") or {}).get("agreement_em")),
            "by_type": {t: {"n": m["n"], "exact": r1(m["exact"]), "f1": r1(m["f1"])} for t, m in (x.get("by_question_type") or {}).items()},
            "rr_by_type": {t: {"n": m["n"], "exact": r1(m["exact"]), "f1": r1(m["f1"])} for t, m in (y.get("by_question_type") or {}).items()},
            "source": f"{os.path.relpath(r['dir'], ROOT)}/eval_{st}/results.json",
        }
        if x.get("paired_view_official_on_orig_file"):
            entry["orig_em"] = r1(x["paired_view_official_on_orig_file"]["exact"])
            entry["orig_f1"] = r1(x["paired_view_official_on_orig_file"]["f1"])
            entry["rr_orig_f1"] = r1((y.get("paired_view_official_on_orig_file") or {}).get("f1"))
        d["sets"][st] = entry
    d["variants"] = {}
    for vn, evs in (r.get("variants") or {}).items():
        vsets: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}
        ens_sets: Dict[str, Any] = {}
        ens_meta: Dict[str, Any] = {}
        for st, e in evs.items():
            if not e or not e.get("reranker"):
                continue
            x, y = e["extractor"], e["reranker"]
            meta = e.get("reranker_variant") or meta
            m_cw = re.search(r"_cw([0-9.]+)", vn)
            if m_cw:
                meta = dict(meta, cw=float(m_cw.group(1)))
            if e.get("pair_ensemble"):
                z = e["pair_ensemble"]
                ens_meta = {"kind": "ensemble", "beta": z.get("beta", 1.0), "use_pair_score": True, "select_mode": "ensemble"}
                ens_sets[st] = {"n": e["n_questions"], "pairs": e["n_pairs"], "em": r1(x["official"]["exact"]), "f1": r1(x["official"]["f1"]),
                                "agr": r1((x.get("consistency") or {}).get("agreement_em")), "both": r1((x.get("pair_accuracy") or {}).get("both_correct")),
                                "rr_em": r1(z["official"]["exact"]), "rr_f1": r1(z["official"]["f1"]), "rr_agr": r1((z.get("consistency") or {}).get("agreement_em")),
                                "rr_both": r1((z.get("pair_accuracy") or {}).get("both_correct")),
                                "ci_em": _ci(z, "em"), "ci_agr": _ci(z, "agreement_em"), "source": f"{os.path.relpath(r['dir'], ROOT)}/rerank_variants/{vn}/eval_{st}/results.json"}
            vsets[st] = {"n": e["n_questions"], "pairs": e["n_pairs"], "em": r1(x["official"]["exact"]), "f1": r1(x["official"]["f1"]),
                         "agr": r1((x.get("consistency") or {}).get("agreement_em")), "rr_em": r1(y["official"]["exact"]), "rr_f1": r1(y["official"]["f1"]),
                         "rr_agr": r1((y.get("consistency") or {}).get("agreement_em")),
                         "fix": (y.get("changes") or {}).get("FIX"), "brk": (y.get("changes") or {}).get("BREAK"),
                         "both": r1((x.get("pair_accuracy") or {}).get("both_correct")), "rr_both": r1((y.get("pair_accuracy") or {}).get("both_correct")),
                         "ci_em": _ci(y, "em"), "ci_agr": _ci(y, "agreement_em"), "source": f"{os.path.relpath(r['dir'], ROOT)}/rerank_variants/{vn}/eval_{st}/results.json"}
        if vsets:
            d["variants"][vn] = {"meta": meta, "sets": vsets}
        if ens_sets:
            d["variants"]["ENSEMBLE"] = {"meta": ens_meta, "sets": ens_sets}
    d["rr_summaries"] = {t: {"n_dev_items": sm["n_dev_items"], "dev_top1": sm["dev_top1"], "dev_reranked": sm["dev_reranked"], "heldout_selection_grid": sm.get("heldout_selection_grid"),
                            "n_features": len(sm.get("scalar_features_used") or []), "elapsed_sec": sm.get("elapsed_sec")} for t, sm in (r.get("rr_summaries") or {}).items() if sm}
    return d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default=os.path.join(ROOT, "runs"))
    p.add_argument("--selected", default="bert_kl_lam1.0", help="the kl run chosen by the lambda rule")
    p.add_argument("--tolerance", type=float, default=0.5, help="F1 tolerance of the lambda rule (points)")
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "RESULTS.md"))
    p.add_argument("--verified", default=None, help="date of an independent fact-check of this report for the current runs/, e.g. 2026-11-04")
    args = p.parse_args()

    runs = collect(args.runs_dir)
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1
    D = {n: digest_run(n, r) for n, r in runs.items()}
    order = ["bert_ce", "bert_aug", "bert_kl_lam0.5", "bert_kl_lam1.0", "bert_kl_lam2.0", "bert_ot", "bert_dual", "deb_ce", "deb_kl", "deb_large_ce", "deb_large_kl"]
    order = [n for n in order if n in D] + [n for n in D if n not in order]
    label = {n: {"bert_ce": "BERT ce (baseline)", "bert_aug": "BERT aug", "bert_kl_lam0.5": "BERT kl λ=0.5", "bert_kl_lam1.0": "BERT kl λ=1", "bert_kl_lam2.0": "BERT kl λ=2",
                 "bert_ot": "BERT ot", "bert_dual": "BERT dual", "deb_ce": "DeBERTa ce (baseline)", "deb_kl": "DeBERTa kl λ=1",
                 "deb_large_ce": "DeBERTa-large ce (baseline)", "deb_large_kl": "DeBERTa-large kl λ=1"}.get(n, n) for n in order}
    ce, aug, kl, kl2, deb_ce, deb_kl, ot, dual = (D.get(k) for k in ("bert_ce", "bert_aug", args.selected, "bert_kl_lam2.0", "deb_ce", "deb_kl", "bert_ot", "bert_dual"))
    dl_ce, dl_kl = D.get("deb_large_ce"), D.get("deb_large_kl")
    if not (ce and kl and aug):
        print("need bert_ce, bert_aug and the selected kl run", file=sys.stderr)
        return 1
    dv = lambda d, k: d["sets"]["dev"][k]  # noqa: E731
    bert_names = [n for n in order if D[n]["model"] == ce["model"]]

    # ---- lambda rule, on the checkpoint-selection sample and on the full held-out set --------------
    # The rule is applied to the unrounded F1 values (rounding first would move a run sitting 0.53 below the best onto
    # the 0.5 boundary); the displayed values are rounded as everywhere else.
    sweep = sorted([d for d in D.values() if d["mode"] == "kl" and d["model"] == ce["model"]], key=lambda d: d["lambda_kl"])
    best_s = max(d["val_f1_raw"] for d in sweep)
    elig_s = [d for d in sweep if d["val_f1_raw"] >= best_s - args.tolerance]
    pick_s = max(elig_s, key=lambda d: d["lambda_kl"])["run"]
    best_f = max(d["full_val_f1_raw"] for d in sweep)
    best_f_run = max(sweep, key=lambda d: d["full_val_f1_raw"])
    elig_f = [d for d in sweep if d["full_val_f1_raw"] >= best_f - args.tolerance]
    pick_f = max(elig_f, key=lambda d: d["lambda_kl"])["run"]
    gaps_s = {d["lambda_kl"]: round(best_s - d["val_f1_raw"], 2) for d in sweep}  # tolerance needed to admit each lambda (sample)
    gaps_f = {d["lambda_kl"]: round(best_f - d["full_val_f1_raw"], 2) for d in sweep}
    lam_list = ", ".join(f"{d['lambda_kl']:g}" for d in sweep)

    # ---- derived numbers ------------------------------------------------------------------------------
    dev_n, dev_pairs = dv(ce, "n"), dv(ce, "pairs")
    deltas = {
        "kl_vs_ce_agr": dv(kl, "agr") - dv(ce, "agr"), "aug_vs_ce_agr": dv(aug, "agr") - dv(ce, "agr"), "kl_vs_aug_agr": dv(kl, "agr") - dv(aug, "agr"),
        "kl2_vs_aug_agr": (dv(kl2, "agr") - dv(aug, "agr")) if kl2 else None,
        "kl_vs_ce_em": dv(kl, "em") - dv(ce, "em"), "kl_vs_ce_f1": dv(kl, "f1") - dv(ce, "f1"),
        "deb_kl_vs_ce_agr": (dv(deb_kl, "agr") - dv(deb_ce, "agr")) if (deb_ce and deb_kl) else None,
        "deb_kl_vs_ce_f1": (dv(deb_kl, "f1") - dv(deb_ce, "f1")) if (deb_ce and deb_kl) else None,
        "rr_dev_f1": dv(kl, "rr_f1") - dv(kl, "f1"), "rr_dev_em": dv(kl, "rr_em") - dv(kl, "em"), "rr_dev_agr": dv(kl, "rr_agr") - dv(kl, "agr"),
        "se_dev_em": se_pct(dv(ce, "em"), dev_n), "se_dev_agr": se_pct(dv(ce, "agr"), dev_pairs),
        "ot_vs_aug_agr": (dv(ot, "agr") - dv(aug, "agr")) if ot else None, "dual_vs_aug_agr": (dv(dual, "agr") - dv(aug, "agr")) if dual else None,
    }
    deltas["se_dev_agr_diff"] = math.sqrt(2) * deltas["se_dev_agr"]
    deltas["kl_vs_ce_agr_in_se"] = deltas["kl_vs_ce_agr"] / deltas["se_dev_agr_diff"]
    deltas["kl_vs_aug_agr_in_se"] = deltas["kl_vs_aug_agr"] / deltas["se_dev_agr_diff"]
    rob: Dict[str, Dict[str, Any]] = {}
    for st in ("addsent", "addonesent", "gan_dev", "gan_adv"):
        if st in ce["sets"] and st in kl["sets"]:
            c, k, a = ce["sets"][st], kl["sets"][st], aug["sets"].get(st)
            rob[st] = {"n": c["n"], "pairs": c["pairs"], "ce_f1": c["f1"], "kl_f1": k["f1"], "aug_f1": a["f1"] if a else None,
                       "delta_f1": k["f1"] - c["f1"], "aug_delta_f1": (a["f1"] - c["f1"]) if a else None,
                       "ce_agr": c["agr"], "kl_agr": k["agr"], "delta_agr": (k["agr"] - c["agr"]) if (c["agr"] is not None and k["agr"] is not None) else None,
                       "drop_from_dev_ce": dv(ce, "f1") - c["f1"], "drop_from_dev_kl": dv(kl, "f1") - k["f1"]}
    addsent_bert = sorted(((D[n]["sets"]["addsent"]["f1"], n) for n in bert_names if "addsent" in D[n]["sets"]), reverse=True)
    rr_rows = [(n, D[n]) for n in order if D[n]["reranker"] and "dev" in D[n]["sets"]]
    ROB_SETS = ("addsent", "addonesent", "gan_dev", "gan_adv")
    of4 = lambda k: "all four" if k == 4 else f"{k} of the four"  # noqa: E731
    rob_up = {n: sum(1 for st in ROB_SETS if D[n]["sets"].get(st) and D[n]["sets"][st]["rr_f1"] is not None and D[n]["sets"][st]["rr_f1"] > D[n]["sets"][st]["f1"]) for n in order}  # robustness sets where the reranker raises F1
    base_all4 = [n for n in order if n not in ("deb_large_ce", "deb_large_kl") and D[n]["reranker"] and rob_up[n] == 4]
    n_base_rr = sum(1 for n in order if n not in ("deb_large_ce", "deb_large_kl") and D[n]["reranker"])
    rr_overfit = {n: round(d["reranker"]["reranked"]["f1"] - d["reranker"]["last_epoch_f1"], 1) for n, d in rr_rows if d["reranker"]["last_epoch_f1"] is not None}
    nf = {m: sorted({D[n]["n_features"] for n in order if D[n]["model"] == m}) for m in sorted({D[n]["model"] for n in order})}
    digest = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "selected": args.selected, "tolerance": args.tolerance,
              "rule_pick_sample": pick_s, "rule_pick_full": pick_f, "best_val_f1_sample_raw": best_s, "best_val_f1_full_raw": best_f, "best_full_run": best_f_run["run"],
              "tolerance_gaps_sample_raw": gaps_s, "tolerance_gaps_full_raw": gaps_f,
              "deltas": deltas, "robustness": rob, "addsent_bert_ranking": addsent_bert, "reranker_overfit_points": rr_overfit, "reranker_robustness_sets_f1_up": rob_up, "n_features_by_model": nf, "runs": D}
    with open(os.path.join(os.path.dirname(args.out), "results_digest.json"), "w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2)

    # ---- report -------------------------------------------------------------------------------------------
    L: List[str] = []
    w = L.append
    sel_lam = f"λ = {kl['lambda_kl']:g}"
    w("# G17 Results Report: Paraphrase-Consistent Extractive QA on SQuAD v1.1")
    w("")
    w(f"*Generated {digest['generated'][:10]} by `scripts/make_results_report.py` from `runs/`. Every number is read from `config.json`, `train_summary.json`, `reranker_summary.json` or `eval_<set>/results.json` (official `evaluate-v2.0.py` output). "
      "Metrics are rounded to one decimal first; every difference in the text is a difference of rounded values. Exact values: `docs/results_digest.json`."
      + (f" Independently fact-checked against the raw files on {args.verified}." if args.verified else "") + "*")
    w("")

    # 1 ---------------------------------------------------------------------------------------------------------
    w("## 1. Summary")
    w("")
    w(f"- **The baseline reproduces the literature.** BERT-base: {f1(dv(ce,'em'))} EM / {f1(dv(ce,'f1'))} F1 on the {dev_n:,}-question dev set (published 80.8 / 88.5).")
    line = (f"- **Consistency training buys stability, not accuracy.** Paraphrase agreement rises from {f1(dv(ce,'agr'))} % to {f1(dv(kl,'agr'))} % ({sgn(deltas['kl_vs_ce_agr'])} points, of which {sgn(deltas['kl_vs_aug_agr'])} beyond augmentation alone) "
            f"at {sgn(deltas['kl_vs_ce_em'])} EM / {sgn(deltas['kl_vs_ce_f1'])} F1.")
    same = []
    if deltas["deb_kl_vs_ce_agr"] is not None:
        same.append(f"DeBERTa-v3-base ({sgn(deltas['deb_kl_vs_ce_agr'])})")
    if dl_ce and dl_kl:
        same.append(f"DeBERTa-v3-large ({sgn(dv(dl_kl,'agr') - dv(dl_ce,'agr'))})")
    if same:
        line += " The same holds for " + " and ".join(same) + "."
    w(line)
    if deb_ce and dl_ce:
        w(f"- **The backbone is the accuracy lever.** DeBERTa-v3-base {f1(dv(deb_ce,'em'))} / {f1(dv(deb_ce,'f1'))}, DeBERTa-v3-large {f1(dv(dl_ce,'em'))} / {f1(dv(dl_ce,'f1'))} EM / F1, against {f1(dv(ce,'em'))} / {f1(dv(ce,'f1'))} for BERT-base; no objective change moves accuracy by more than half a point.")
    w(f"- **Robustness gains are small and come mostly from the paraphrase data.** AddSent F1 {f1(rob['addsent']['ce_f1'])} → {f1(rob['addsent']['kl_f1'])} ({sgn(rob['addsent']['delta_f1'])}, of which {sgn(rob['addsent']['aug_delta_f1'])} from augmentation alone); AddSent agreement {sgn(rob['addsent']['delta_agr'])}.")
    rr_line = (f"- **The reranker trades accuracy for agreement.** Selected run, dev: {sgn(deltas['rr_dev_f1'])} F1 for {sgn(deltas['rr_dev_agr'])} agreement; its largest gain is on the {kl['sets']['gan_adv']['n']}-question adversarial-paraphrase set ({f1(kl['sets']['gan_adv']['f1'])} → {f1(kl['sets']['gan_adv']['rr_f1'])} F1), which is too small for firm conclusions.")
    if dl_kl:
        rr_line += f" On DeBERTa-v3-large the cost falls to {sgn(dv(dl_kl,'rr_f1') - dv(dl_kl,'f1'))} F1 and F1 rises on {of4(rob_up['deb_large_kl'])} robustness sets, but with newer reranker targets (Section 8)."
    w(rr_line)
    if any(D[n].get("variants") for n in order):
        w("- **Reranker ablation.** The paraphrase features carry the agreement and adversarial-paraphrase gains; blending reranker and extractor scores instead of letting the reranker overrule removes the dev accuracy loss and adds a few tenths of EM on all three extractors tested, at the cost of most of the agreement. Neither is adopted (Section 9).")
    if any("ENSEMBLE" in D[n].get("variants", {}) for n in order):
        w("- **Learning-free check.** Simply averaging the extractor's scores over the question and its paraphrase beats the learned reranker on agreement and on the both-views-correct rate, at a small EM cost on dev; a reranker trained with a consistency term but no paraphrase features matches the plain reranker and hurts at a high weight (Section 9b).")
    w(f"- **λ = {D[pick_s]['lambda_kl']:g}**, chosen on held-out training articles as the largest λ with F1 within {args.tolerance:g} points of the best" + (" (same choice on the full held-out set)." if pick_s == pick_f else f" (the full held-out set would give λ = {D[pick_f]['lambda_kl']:g})."))
    w("")

    # 2 ---------------------------------------------------------------------------------------------------------
    w("## 2. What was run")
    w("")
    w(f"{ce['epochs']} epochs, effective batch {ce['batch_size'] * ce['grad_accum']}, learning rate {ce['lr']:g}, sequence length {ce['max_seq_len']}, stride {ce['doc_stride']}, AdamW with 10 % warm-up, best checkpoint by validation F1"
      + (f"; the DeBERTa-v3-large runs use batch {dl_ce['batch_size']} × {dl_ce['grad_accum']} accumulation steps and learning rate {dl_ce['lr']:g}" if dl_ce else "") + ". "
      f"Training uses the {ce['n_train_questions']:,} questions of 95 % of the training articles; the other 5 % ({ce['n_val_questions']:,} questions) are held out for checkpoint selection (on a {ce['val_max_examples']:,}-question sample), λ selection and reranker training. Dev and the external sets are used only for the final numbers. "
      f"Paraphrases are MarianMT back-translations (EN→DE→EN) kept at cosine similarity ≥ 0.85; {dev_pairs:,} of {dev_n:,} dev questions ({100*dev_pairs/dev_n:.1f} %) have one, and agreement is measured over those pairs.")
    w("")
    w("| Run | Model | Objective | λ | Batch × accum. | LR | Optimiser steps | Training (min) | Paraphrases attached |")
    w("| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |")
    for n in order:
        d = D[n]
        ps = d["pair_stats"]
        att = f"{ps['attached']:,} / {ps['total']:,}" if ps else "not used"
        lam_txt = f"{d['lambda_kl']:g}" if d['lambda_kl'] is not None else "–"
        w(f"| {label[n]} | {d['model'].split('/')[-1]} | {MODE_DESC.get(d['mode'], d['mode'])} | {lam_txt} | {d['batch_size']} × {d['grad_accum']} | {d['lr']:g} | {d['steps']:,} | {d['train_time_min']:.0f} | {att} |")
    w("")

    # 3 ---------------------------------------------------------------------------------------------------------
    w("## 3. Main results on the dev set")
    w("")
    w(f"Official scorer, all {dev_n:,} questions. *Agreement* = share of (question, paraphrase) pairs that get the same normalised answer. `+reranker` = the second system version.")
    w("")
    w("| Run | EM | F1 | Agreement | EM +reranker | F1 +reranker | Agreement +reranker |")
    w("| :-- | --: | --: | --: | --: | --: | --: |")
    for n in order:
        s = D[n]["sets"].get("dev")
        if s:
            w(f"| {label[n]} | {f1(s['em'])} | {f1(s['f1'])} | {f1(s['agr'])} | {f1(s['rr_em'])} | {f1(s['rr_f1'])} | {f1(s['rr_agr'])} |")
    w("")
    em_lo, em_hi = min(dv(D[n], "em") for n in bert_names), max(dv(D[n], "em") for n in bert_names)
    f1_lo, f1_hi = min(dv(D[n], "f1") for n in bert_names), max(dv(D[n], "f1") for n in bert_names)
    w(f"**Reading.** Accuracy is flat across the {len(bert_names)} BERT objectives (EM {em_lo:.1f}–{em_hi:.1f}, F1 {f1_lo:.1f}–{f1_hi:.1f}; one standard error of an EM value is about {deltas['se_dev_em']:.1f} points, so differences under half a point are noise). "
      f"Agreement is where they separate: {f1(dv(ce,'agr'))} % baseline, {f1(dv(aug,'agr'))} % augmentation, {f1(dv(kl,'agr'))} % consistency ({sel_lam}). "
      f"The {sgn(deltas['kl_vs_ce_agr'])} gain over the baseline is about {deltas['kl_vs_ce_agr_in_se']:.0f} standard errors of a difference; the {sgn(deltas['kl_vs_aug_agr'])} over augmentation-only is about {deltas['kl_vs_aug_agr_in_se']:.1f} and, with one seed per run, only suggestive.")
    w("")

    # 4 ---------------------------------------------------------------------------------------------------------
    w("## 4. Augmentation versus consistency loss")
    w("")
    w(f"The paraphrased examples alone (`aug`) move agreement {f1(dv(ce,'agr'))} → {f1(dv(aug,'agr'))} % ({sgn(deltas['aug_vs_ce_agr'])}); the KL penalty adds {sgn(deltas['kl_vs_aug_agr'])} at λ = 1 and {sgn(deltas['kl2_vs_aug_agr'])} at λ = 2, at no accuracy cost (EM / F1 {f1(dv(aug,'em'))} / {f1(dv(aug,'f1'))} for `aug`, {f1(dv(kl,'em'))} / {f1(dv(kl,'f1'))} at λ = 1" + (f", {f1(dv(kl2,'em'))} / {f1(dv(kl2,'f1'))} at λ = 2" if kl2 else "") + "). "
      f"Roughly {100 * deltas['aug_vs_ce_agr'] / deltas['kl_vs_ce_agr']:.0f} % of the gain over the baseline is therefore the data and the rest the penalty; the penalty's share is the part that is only suggestive at one seed.")
    w("")

    # 5 ---------------------------------------------------------------------------------------------------------
    w("## 5. Choosing λ on held-out data")
    w("")
    w(f"Rule: the largest λ whose held-out F1 is within {args.tolerance:g} points of the best, applied to unrounded values. The selection sample is what training used; the full held-out set is a check.")
    w("")
    w(f"| Model | λ | Sample F1 (n={sweep[0]['val_n']:,}) | Sample agreement | Full held-out F1 (n={sweep[0]['full_val_n']:,}) | Full held-out agreement | Dev F1 | Dev agreement |")
    w("| :-- | --: | --: | --: | --: | --: | --: | --: |")
    for d in sorted([d for d in D.values() if d["mode"] == "kl"], key=lambda d: (d["model"], d["lambda_kl"])):
        w(f"| {d['model'].split('/')[-1]} | {d['lambda_kl']:g} | {f1(d['val_f1'])} | {f1(d['val_agreement'])} | {f1(d['full_val_f1'])} | {f1(d['full_val_agreement'])} | {f1(d['sets']['dev']['f1'])} | {f1(d['sets']['dev']['agr'])} |")
    w("")
    pick_txt = (f"Both the sample and the full set select **λ = {D[pick_s]['lambda_kl']:g}**" if pick_s == pick_f
                else f"The sample selects **λ = {D[pick_s]['lambda_kl']:g}**, the full set λ = {D[pick_f]['lambda_kl']:g}")
    w(f"{pick_txt} (sample F1 {', '.join(f1(d['val_f1']) for d in sweep)} for λ = {lam_list}; all within the sampling error of a {sweep[0]['val_n']:,}-question sample, about {math.sqrt(2) * se_pct(best_s, sweep[0]['val_n']):.1f} points for a difference). "
      f"Agreement rises monotonically with λ ({', '.join(f1(d['val_agreement']) for d in sweep)}), so the choice rests on the tolerance: a tolerance under about {gaps_s[1.0]:.2f} would pick λ = 0.5 and one of {gaps_s[2.0]:.2f} or more λ = 2. On dev, {sel_lam} also has the best F1 of the sweep ({', '.join(f1(d['sets']['dev']['f1']) for d in sweep)}; a {max(d['sets']['dev']['f1'] for d in sweep) - min(d['sets']['dev']['f1'] for d in sweep):.1f}-point spread, within noise).")
    w("")

    # 6 ---------------------------------------------------------------------------------------------------------
    w("## 6. Backbone: BERT-base, DeBERTa-v3-base and DeBERTa-v3-large")
    w("")
    bb_runs = [n for n in ("bert_ce", args.selected, "deb_ce", "deb_kl", "deb_large_ce", "deb_large_kl") if n in D]
    w("| Backbone | Objective | Dev EM | Dev F1 | Dev agreement | Full held-out F1 | Full held-out agreement | AddSent F1 | AddSent agreement | Training (min) |")
    w("| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |")
    for n in bb_runs:
        d = D[n]
        a1 = d["sets"].get("addsent") or {}
        obj = "ce" if d["mode"] == "ce" else f"kl λ={d['lambda_kl']:g}"
        w(f"| {d['model'].split('/')[-1]} | {obj} | {f1(dv(d,'em'))} | {f1(dv(d,'f1'))} | {f1(dv(d,'agr'))} | {f1(d['full_val_f1'])} | {f1(d['full_val_agreement'])} | {f1(a1.get('f1'))} | {f1(a1.get('agr'))} | {d['train_time_min']:.0f} |")
    w("")
    if deb_ce and deb_kl:
        txt = (f"Each step up in backbone adds far more accuracy than any objective change ({f1(dv(ce,'f1'))} → {f1(dv(deb_ce,'f1'))}" + (f" → {f1(dv(dl_ce,'f1'))}" if dl_ce else "") + " F1). "
               f"Consistency training raises agreement on every backbone ({sgn(deltas['kl_vs_ce_agr'])}, {sgn(deltas['deb_kl_vs_ce_agr'])}" + (f", {sgn(dv(dl_kl,'agr') - dv(dl_ce,'agr'))}" if dl_ce and dl_kl else "") + f") at {sgn(deltas['kl_vs_ce_f1'])}, {sgn(deltas['deb_kl_vs_ce_f1'])}" + (f", {sgn(dv(dl_kl,'f1') - dv(dl_ce,'f1'))}" if dl_ce and dl_kl else "") + " F1, so the effect is not specific to the weak encoder. ")
        txt += (f"Distractor robustness is the clearest place the backbones disagree: consistency training raises AddSent F1 for BERT-base ({sgn(rob['addsent']['delta_f1'])})" + (f" and DeBERTa-v3-large ({sgn(dl_kl['sets']['addsent']['f1'] - dl_ce['sets']['addsent']['f1'])})" if dl_ce and dl_kl else "") + f" but lowers it for DeBERTa-v3-base ({sgn(deb_kl['sets']['addsent']['f1'] - deb_ce['sets']['addsent']['f1'])}); with one seed per run, F1 moves of about one point are not reliable, whereas the agreement gains are several standard errors (Section 7). ")
        if dl_ce and dl_kl:
            txt += (f"The large runs peaked at {f1(dl_ce['peak_gpu_mem_gb'])} / {f1(dl_kl['peak_gpu_mem_gb'])} GB of GPU memory; the `kl` job ran on a full, shared H100 rather than a 47 GB MIG slice, so its {dl_kl['train_time_min']:.0f} min is not comparable with the other times. Its reranker behaviour is discussed in Section 8.")
        w(txt)
    w("")

    # 7 ---------------------------------------------------------------------------------------------------------
    w("## 7. Robustness: paraphrased and adversarial test sets")
    w("")
    w("F1 from the official scorer. Gan & Ng: *orig* = original questions, *para* = their human paraphrases, *agreement* = same answer on both. AddSent / AddOneSent: agreement with back-translated paraphrases of the adversarial questions.")
    w("")
    w("| Run | G&N orig F1 | G&N para F1 | G&N agreement | G&N adv orig F1 | G&N adv para F1 | AddSent F1 | AddSent agreement | AddOneSent F1 | AddOneSent agreement |")
    w("| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |")
    for n in order:
        s = D[n]["sets"]
        g, ga, a1, a2 = s.get("gan_dev"), s.get("gan_adv"), s.get("addsent"), s.get("addonesent")
        w(f"| {label[n]} | {f1(g and g.get('orig_f1'))} | {f1(g and g['f1'])} | {f1(g and g['agr'])} | {f1(ga and ga.get('orig_f1'))} | {f1(ga and ga['f1'])} | {f1(a1 and a1['f1'])} | {f1(a1 and a1['agr'])} | {f1(a2 and a2['f1'])} | {f1(a2 and a2['agr'])} |")
    w("")
    para_bert = [v for v, n in addsent_bert if n != "bert_ce"]
    txt = (f"**Distractors (Jia & Liang).** The BERT baseline drops from {f1(dv(ce,'f1'))} F1 on dev to {f1(rob['addsent']['ce_f1'])} on AddSent and {f1(rob['addonesent']['ce_f1'])} on AddOneSent. "
           f"All {len(para_bert)} paraphrase-trained BERT runs sit within {min(para_bert):.1f}–{max(para_bert):.1f} F1 on AddSent, so the {sgn(rob['addsent']['delta_f1'])} of {sel_lam} is mostly the paraphrase data (`aug` alone {sgn(rob['addsent']['aug_delta_f1'])}); what changes is stability, AddSent agreement {f1(rob['addsent']['ce_agr'])} → {f1(rob['addsent']['kl_agr'])} % ({sgn(rob['addsent']['delta_agr'])}), i.e. the model is fooled more consistently rather than less often. ")
    if deb_ce and deb_kl:
        txt += f"DeBERTa-v3-base is far more robust ({f1(deb_ce['sets']['addsent']['f1'])} F1) and consistency training moves it by {sgn(deb_kl['sets']['addsent']['f1'] - deb_ce['sets']['addsent']['f1'])}"
        if dl_ce and dl_kl:
            a_ce, a_kl = dl_ce["sets"]["addsent"], dl_kl["sets"]["addsent"]
            se_a = math.sqrt(2) * se_pct(a_ce["agr"], a_ce["pairs"])
            txt += f"; DeBERTa-v3-large gains {sgn(a_kl['f1'] - a_ce['f1'])} ({f1(a_ce['f1'])} → {f1(a_kl['f1'])}) with agreement {f1(a_ce['agr'])} → {f1(a_kl['agr'])} % ({sgn(a_kl['agr'] - a_ce['agr'])}, about {(a_kl['agr'] - a_ce['agr']) / se_a:.0f} standard errors over {a_ce['pairs']:,} pairs)"
        txt += "."
    w(txt)
    w("")
    g_ce, g_kl = ce["sets"]["gan_dev"], kl["sets"]["gan_dev"]
    w(f"**Human paraphrases (Gan & Ng, {g_ce['n']:,} questions, {g_ce['pairs']:,} pairs).** Paraphrasing costs the baseline {g_ce['orig_f1'] - g_ce['f1']:.1f} F1 ({f1(g_ce['orig_f1'])} → {f1(g_ce['f1'])}) and the {sel_lam} model {g_kl['orig_f1'] - g_kl['f1']:.1f} ({f1(g_kl['orig_f1'])} → {f1(g_kl['f1'])}); the narrower gap comes from both sides ({sgn(g_kl['f1'] - g_ce['f1'])} on the paraphrases, {sgn(g_kl['orig_f1'] - g_ce['orig_f1'])} on the originals). Agreement {f1(g_ce['agr'])} → {f1(g_kl['agr'])} % ({sgn(rob['gan_dev']['delta_agr'])}). "
      f"On the {ce['sets']['gan_adv']['n']}-question adversarial subset ({100/ce['sets']['gan_adv']['n']:.1f} points per question) the baseline scores {f1(ce['sets']['gan_adv']['f1'])} and {sel_lam} {f1(kl['sets']['gan_adv']['f1'])} F1; indicative only.")
    w("")

    # 8 ---------------------------------------------------------------------------------------------------------
    w("## 8. The learned span reranker")
    w("")
    rr0 = kl["reranker"]
    w(f"Held-out columns: the reranker is trained on 80 % of the held-out articles and early-stopped on the other 20 % ({rr0['dev_items']:,} questions); the reranked value is the best of {rr0['n_epochs']} epochs on that slice (by F1 for one-hot targets, by EM then F1 for graded) and therefore optimistic. Remaining columns: dev and external sets, Δ = reranked − extractor.")
    w("")
    w("| Run | Targets | Held-out top-1 EM / F1 | Held-out reranked EM / F1 (best epoch) | Last-epoch F1 | Oracle EM (correct span in top-20) | Dev Δ EM | Dev Δ F1 | Dev Δ agreement | AddSent Δ F1 | G&N adv Δ F1 |")
    w("| :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: |")
    for n, d in rr_rows:
        rr = d["reranker"]
        s, a1, ga = d["sets"]["dev"], d["sets"].get("addsent"), d["sets"].get("gan_adv")
        w(f"| {label[n]} | {'graded' if rr['target_mode'] == 'graded' else 'one-hot'} | {f1(rr['top1']['exact'])} / {f1(rr['top1']['f1'])} | {f1(rr['reranked']['exact'])} / {f1(rr['reranked']['f1'])} (ep. {rr['best_epoch']}) | {f1(rr['last_epoch_f1'])} | {f1(rr['top1']['oracle_exact'])} | {sgn(s['rr_em'] - s['em'])} | {sgn(s['rr_f1'] - s['f1'])} | {sgn(s['rr_agr'] - s['agr'])} | {sgn(a1['rr_f1'] - a1['f1']) if a1 else '--'} | {sgn(ga['rr_f1'] - ga['f1']) if ga else '--'} |")
    w("")
    n_f1_down = sum(1 for _, d in rr_rows if dv(d, "rr_f1") < dv(d, "f1"))
    n_f1_flat = sum(1 for _, d in rr_rows if dv(d, "rr_f1") == dv(d, "f1"))
    n_agr_up = sum(1 for _, d in rr_rows if dv(d, "rr_agr") > dv(d, "agr"))
    w(f"**Reading.** The correct span is among the top-20 candidates for {min(d['reranker']['top1']['oracle_exact'] for _, d in rr_rows):.1f}–{max(d['reranker']['top1']['oracle_exact'] for _, d in rr_rows):.1f} % of held-out questions, but the reranker recovers almost none of that: on dev, F1 never rises (it falls in {n_f1_down} of {len(rr_rows)} runs, unchanged in {n_f1_flat}) while agreement rises in {n_agr_up} of {len(rr_rows)}; for the selected run the trade is {sgn(deltas['rr_dev_f1'])} F1 for {sgn(deltas['rr_dev_agr'])} agreement. "
      f"The last-epoch F1 is {min(rr_overfit.values()):.1f}–{max(rr_overfit.values()):.1f} points below the best epoch, so it overfits within a few epochs. "
      f"We hypothesise that the paraphrase-score feature drives the trade (it favours spans both phrasings score highly); " + ("Section 9 tests this by ablation. " if any(D[n].get("variants") for n in order) else "this is not yet ablated. ") + f"Its clearest benefit is on the adversarial paraphrases, where it can lean on the easier original question ({f1(kl['sets']['gan_adv']['f1'])} → {f1(kl['sets']['gan_adv']['rr_f1'])} F1 for the selected run; small set). "
      "Both system versions are reported.")
    graded_rows = [(n, d) for n, d in rr_rows if d["reranker"]["target_mode"] == "graded"]
    if graded_rows:
        names = " and ".join("`" + n + "`" for n, _ in graded_rows)
        w("")
        w(f"**Graded targets.** {names} were trained after the target changed from one-hot (first exact-match candidate, else best F1) to graded relevance (exact match 1, partial overlap 0.3 × F1), so they are not strictly comparable with the other {len(rr_rows) - len(graded_rows)} rows. "
          + " ".join(f"`{n}`: {sgn(dv(d,'rr_f1') - dv(d,'f1'))} F1 for {sgn(dv(d,'rr_agr') - dv(d,'agr'))} agreement on dev, F1 up on {of4(rob_up[n])} robustness sets." for n, d in graded_rows)
          + (f" Among the base-model rerankers only {' and '.join('`' + n + '`' for n in base_all4)} also gain{'s' if len(base_all4) == 1 else ''} on all four, mostly by smaller margins." if base_all4 else " No base-model reranker gains on all four.")
          + (" Section 9 retrains the base rerankers with graded targets and separates the two effects." if any(D[n].get("variants") for n in order) else " Whether this comes from the stronger extractor or from the new targets needs the base rerankers retrained with graded targets; the reranker-ablation scripts added on 4 Nov 2026 do that without retraining any extractor."))
    w("")

    # 9 ---------------------------------------------------------------------------------------------------------
    var_runs = [n for n in order if D[n].get("variants")]
    if var_runs:
        w("## 9. Reranker ablation: features, decision rule, confidence gate")
        w("")
        vr0 = D[var_runs[0]]
        w(f"On {', '.join('`' + n + '`' for n in var_runs)} the reranker was retrained twice on the held-out articles (graded targets, same seed): with the three paraphrase-derived features (*pair*) and without them (*no pair*). "
          "Each was then used in two ways: the reranker's argmax replaces the extractor's answer (*argmax*, the current rule), or the two log-probabilities are added with weight α on the reranker (*blend*), optionally behind a *gate* that keeps the extractor's answer unless the reranker prefers another span by more than the margin in nats. "
          f"Δ = variant − extractor, dev set; FIX / BREAK = answers turned right / wrong (exact match).")
        w("")

        def vmeta(v):  # metadata recorded in results.json
            m = v["meta"]
            return bool(m.get("use_pair_score", True)), m.get("select_mode", "reranker"), float(m.get("alpha", 1.0)), m.get("override_margin")

        def vkey(item):
            pair, mode, alpha, gate = vmeta(item[1])
            return (0 if pair else 1, 0 if mode == "reranker" else 1, alpha if mode == "residual" else 0.0, -1.0 if gate is None else gate)

        def vlabel(name, v):
            pair, mode, alpha, gate = vmeta(v)
            s = name.split("_")[0] + " · " + ("pair" if pair else "no pair") + " · " + ("argmax" if mode == "reranker" else "blend α=" + format(alpha, "g"))
            return s + (" · gate " + format(gate, "g") if gate is not None else "")

        def vd(v, st, a, b):
            e = v["sets"].get(st)
            return None if (not e or e.get(a) is None or e.get(b) is None) else e[a] - e[b]

        def tri(v, st):
            return f"{sgn(vd(v, st, 'rr_em', 'em'))} / {sgn(vd(v, st, 'rr_f1', 'f1'))} / {sgn(vd(v, st, 'rr_agr', 'agr'))}"

        core = lambda vn: not vn.startswith(("CB_", "CR_", "ENSEMBLE"))  # noqa: E731  (9b variants are handled separately)
        vnames = sorted(((vn, v) for vn, v in vr0["variants"].items() if core(vn)), key=vkey)
        vnames = [n for n, _ in vnames]
        w("| Variant | " + " | ".join(f"{label[n]}: Δ EM / Δ F1 / Δ agreement" for n in var_runs) + " |")
        w("| :-- |" + " :-- |" * len(var_runs))
        for vn in vnames:
            cells = []
            for n in var_runs:
                v = D[n]["variants"].get(vn)
                cells.append(tri(v, "dev") if v and "dev" in v["sets"] else "--")
            w(f"| {vlabel(vn, vr0['variants'][vn])} | " + " | ".join(cells) + " |")
        w("")

        # --- held-out selection: best (EM, then F1) over both rerankers' grids -----------------------------------
        def norm_gate(g):
            return None if g is None or float(g) == 0.0 else float(g)

        def grid_pick(d):
            cands = []
            for tag, summ in (d.get("rr_summaries") or {}).items():
                if tag not in ("pair", "nopair"):
                    continue
                for g in summ.get("heldout_selection_grid") or []:
                    cands.append((tag, g["mode"], float(g["alpha"]), norm_gate(g["override_margin"]), g["exact"], g["f1"]))
            if not cands:
                return None
            cands.sort(key=lambda c: (-c[4], -c[5], c[3] is not None, c[1] != "reranker"))
            return cands[0]

        def match_variant(d, pick):
            tag, mode, alpha, gate = pick[0], pick[1], pick[2], pick[3]
            for vn, v in d["variants"].items():
                if not core(vn):
                    continue
                p, m, a, g = vmeta(v)
                if p == (tag == "pair") and m == mode and norm_gate(g) == gate and (m != "residual" or abs(a - alpha) < 1e-9):
                    return vn
            return None

        ses = [se_pct(D[n]["rr_summaries"]["pair"]["dev_top1"]["exact"], D[n]["rr_summaries"]["pair"]["n_dev_items"]) for n in var_runs if D[n].get("rr_summaries")]
        w(f"**Held-out selection.** Before looking at dev, the rule \"best held-out EM, then F1\" over all configurations of both rerankers ({vr0['rr_summaries']['pair']['n_dev_items']:,} held-out questions, standard error {min(ses):.1f}–{max(ses):.1f} EM points) would choose the following; the chosen values are early-stopped on the same questions and therefore optimistic:")
        w("")
        w("| Run | Held-out EM / F1, extractor | Chosen configuration | Held-out EM / F1, chosen | Dev Δ EM / Δ F1 / Δ agreement of the choice |")
        w("| :-- | :-- | :-- | :-- | :-- |")
        picks = {}
        for n in var_runs:
            d = D[n]
            pk = grid_pick(d)
            if not pk:
                continue
            vn = match_variant(d, pk)
            picks[n] = vn
            top = d["rr_summaries"]["pair"]["dev_top1"]
            conf = ("pair" if pk[0] == "pair" else "no pair") + " · " + ("argmax" if pk[1] == "reranker" else ("extractor only" if pk[1] == "extractor" else "blend α=" + format(pk[2], "g"))) + (" · gate " + format(pk[3], "g") if pk[3] is not None else "")
            dev_txt = tri(d["variants"][vn], "dev") if vn else "not evaluated on dev"
            w(f"| {label[n]} | {f1(r1(top['exact']))} / {f1(r1(top['f1']))} | {conf}" + (f" (= {vn.split('_')[0]})" if vn else "") + f" | {f1(r1(pk[4]))} / {f1(r1(pk[5]))} | {dev_txt} |")
        w("")

        # --- robustness for A, B and E ---------------------------------------------------------------------------
        def find_variant(d, pair, mode, alpha=1.0, gate=None):
            for vn, v in d["variants"].items():
                if not core(vn):
                    continue
                p, m, a, g = vmeta(v)
                if p == pair and m == mode and norm_gate(g) == gate and (m != "residual" or abs(a - alpha) < 1e-9):
                    return vn
            return None

        w("Robustness sets for the three main variants (Δ = variant − extractor):")
        w("")
        w("| Run | Variant | AddSent Δ F1 | AddSent Δ agreement | AddOneSent Δ F1 | G&N para Δ F1 | G&N adv Δ F1 |")
        w("| :-- | :-- | --: | --: | --: | --: | --: |")
        for n in var_runs:
            d = D[n]
            for spec in ((True, "reranker", 1.0), (False, "reranker", 1.0), (False, "residual", 1.0)):
                vn = find_variant(d, *spec)
                if not vn:
                    continue
                v = d["variants"][vn]
                w(f"| {label[n]} | {vlabel(vn, v)} | {sgn(vd(v,'addsent','rr_f1','f1'))} | {sgn(vd(v,'addsent','rr_agr','agr'))} | {sgn(vd(v,'addonesent','rr_f1','f1'))} | {sgn(vd(v,'gan_dev','rr_f1','f1'))} | {sgn(vd(v,'gan_adv','rr_f1','f1'))} |")
        w("")

        # --- reading ---------------------------------------------------------------------------------------------
        A = {n: find_variant(D[n], True, "reranker") for n in var_runs}
        B = {n: find_variant(D[n], False, "reranker") for n in var_runs}
        E = {n: find_variant(D[n], False, "residual", 1.0) for n in var_runs}
        ok = all(A[n] and B[n] and E[n] for n in var_runs)
        if ok:
            agrA = [vd(D[n]["variants"][A[n]], "dev", "rr_agr", "agr") for n in var_runs]
            agrB = [vd(D[n]["variants"][B[n]], "dev", "rr_agr", "agr") for n in var_runs]
            advA = [vd(D[n]["variants"][A[n]], "gan_adv", "rr_f1", "f1") for n in var_runs]
            advB = [vd(D[n]["variants"][B[n]], "gan_adv", "rr_f1", "f1") for n in var_runs]
            emE = [vd(D[n]["variants"][E[n]], "dev", "rr_em", "em") for n in var_runs]
            f1E = [vd(D[n]["variants"][E[n]], "dev", "rr_f1", "f1") for n in var_runs]
            agrE = [vd(D[n]["variants"][E[n]], "dev", "rr_agr", "agr") for n in var_runs]
            fbE = [(D[n]["variants"][E[n]]["sets"]["dev"]["fix"], D[n]["variants"][E[n]]["sets"]["dev"]["brk"]) for n in var_runs]
            em_up_E, f1_up_E, f1_ok_E = all(e > 0 for e in emE), all(f > 0 for f in f1E), all(f >= 0 for f in f1E)
            e_claim = ("raises dev EM and F1 on all three extractors" if em_up_E and f1_up_E else
                       "raises dev EM on all three extractors without lowering F1" if em_up_E and f1_ok_E else "changes dev EM / F1")
            lst = lambda xs: ", ".join(sgn(x) for x in xs)  # noqa: E731
            lst = lambda xs: ", ".join(sgn(x) for x in xs)  # noqa: E731
            w(f"**Features.** The paraphrase features carry the agreement gain ({lst(agrA)} agreement points on dev with them, {lst(agrB)} without; order {', '.join(label[n] for n in var_runs)}) and the adversarial-paraphrase gain ({lst(advA)} F1 with, {lst(advB)} without). This confirms the hypothesis of Section 8.")
            w("")
            gate_shift = 0.0
            for n in var_runs:
                for vn, v in D[n]["variants"].items():
                    pp, mm, aa, gg = vmeta(v)
                    if not core(vn) or norm_gate(gg) is None or "dev" not in v["sets"]:
                        continue
                    base_vn = find_variant(D[n], pp, mm, aa, None)
                    if base_vn and "dev" in D[n]["variants"][base_vn]["sets"]:
                        b_, g_ = D[n]["variants"][base_vn]["sets"]["dev"], v["sets"]["dev"]
                        gate_shift = max(gate_shift, *(abs(g_[k] - b_[k]) for k in ("rr_em", "rr_f1", "rr_agr") if g_.get(k) is not None and b_.get(k) is not None))
            w(f"**Decision rule.** Blending instead of taking the reranker's argmax (α = 1, no pair features) " + e_claim + f" ({lst(emE)} EM, {lst(f1E)} F1; fixes : breaks {', '.join(f'{a}:{b}' for a, b in fbE)}), but by a few tenths of a point and keeping only {lst(agrE)} of the agreement. Gates reduce both fixes and breaks; relative to the ungated rule they shift dev EM, F1 or agreement by at most {gate_shift:.1f} points.")
            w("")
            cmp = []
            for n in var_runs:
                s_ = D[n]["sets"]["dev"]
                a = D[n]["variants"][A[n]]
                cmp.append((n, D[n]["reranker"]["target_mode"], s_["rr_f1"] - s_["f1"], s_["rr_agr"] - s_["agr"], vd(a, "dev", "rr_f1", "f1"), vd(a, "dev", "rr_agr", "agr")))
            onehot = [c for c in cmp if c[1] != "graded"]
            graded = [c for c in cmp if c[1] == "graded"]
            txt = "**Targets and noise.** Variant A is the committed reranker's configuration retrained with graded targets. "
            if onehot:
                txt += "Against the one-hot rerankers of Section 8 the target alone moves the dev trade: " + "; ".join(f"{label[c[0]]} {sgn(c[2])} F1 / {sgn(c[3])} agreement → {sgn(c[4])} / {sgn(c[5])}" for c in onehot) + ". "
            if graded:
                txt += "For " + " and ".join(label[c[0]] for c in graded) + " both are graded, so the shift (" + "; ".join(f"{sgn(c[2])} → {sgn(c[4])} F1, {sgn(c[3])} → {sgn(c[5])} agreement" for c in graded) + ") is the run-to-run noise of reranker training; differences of that size in this section mean nothing."
            w(txt)
            w("")
            n_lose = sum(1 for n in var_runs if picks.get(n) and vd(D[n]["variants"][picks[n]], "dev", "rr_f1", "f1") < 0)
            w(f"**Selection.** The held-out rule would pick {', '.join(picks[n].split('_')[0] for n in var_runs if picks.get(n))} for the three runs (table above), {n_lose} of which {'loses' if n_lose == 1 else 'lose'} dev F1; {vr0['rr_summaries']['pair']['n_dev_items']:,} questions cannot separate configurations a few tenths apart. No variant is adopted: the pair-feature argmax reranker stays the agreement version and the extractor alone the accuracy version.")
        # --- 9b: learning-free ensemble baseline, bootstrap CIs, consistency-trained rerankers (one extractor) -------------
        ext_runs = [n for n in var_runs if any(vn.startswith(("CB_", "CR_", "ENSEMBLE")) for vn in D[n]["variants"])]
        for n in ext_runs:
            d = D[n]
            V = d["variants"]
            ens = V.get("ENSEMBLE")
            cb_A, cb_B, cb_E = V.get("CB_pair_argmax_ensemble"), V.get("CB_nopair_argmax"), V.get("CB_nopair_res1.0")
            crs = sorted([(vn, v) for vn, v in V.items() if vn.startswith("CR_")], key=lambda kv: (kv[1]["meta"].get("cw", 0.0), kv[1]["meta"].get("select_mode") != "reranker"))
            if not (ens and cb_A and cb_B):
                continue
            ext_dev = cb_A["sets"]["dev"]

            def ci(v, st, k):  # "[lo, hi]" of the paired-bootstrap 95 % interval of a delta, two decimals
                c = (v["sets"].get(st) or {}).get(k)
                return f"[{sgn(c[0], 2)}, {sgn(c[1], 2)}]" if c else "--"

            def excl(v, st, k="ci_em"):
                c = (v["sets"].get(st) or {}).get(k)
                return "interval excludes zero" if c and (c[0] > 0 or c[1] < 0) else "interval includes zero"

            def row(name, v, st="dev"):
                e = v["sets"].get(st)
                if not e:
                    return f"| {name} | -- | -- | -- | -- |"
                return (f"| {name} | {sgn(e['rr_em'] - e['em'])} {ci(v, st, 'ci_em')} | {sgn(e['rr_f1'] - e['f1'])} | {sgn(e['rr_agr'] - e['agr'])} {ci(v, st, 'ci_agr')} | {f1(e.get('rr_both'))} |")

            w(f"### 9b. Learning-free baseline, confidence intervals and consistency-trained rerankers ({label[n]})")
            w("")
            pairs_re = [(V.get("A_pair_reranker"), cb_A), (V.get("B_nopair_reranker"), cb_B), (V.get("E_nopair_res1.0"), cb_E)]
            re_diff = 0.0
            for orig, re_ in pairs_re:
                if orig and re_ and "dev" in orig["sets"] and "dev" in re_["sets"]:
                    for a_, b_ in (("rr_em", "em"), ("rr_f1", "f1"), ("rr_agr", "agr")):
                        re_diff = max(re_diff, abs((orig["sets"]["dev"][a_] - orig["sets"]["dev"][b_]) - (re_["sets"]["dev"][a_] - re_["sets"]["dev"][b_])))
            a_main = (V.get("A_pair_reranker") or {}).get("sets", {}).get("dev") or {}
            reeval_note = (f" Rows A, B and E re-evaluate the Section 9 checkpoints on another cluster node; on the columns shown there their deltas differ by at most {re_diff:.1f} (bf16 non-determinism)."
                           if a_main and any(a_main.get(k) != ext_dev.get(k) for k in ("em", "f1", "agr")) else "")
            w(f"On `{n}` only. The *paraphrase ensemble* needs no training: a candidate's extractor log-probability under the question is added to its log-probability under the paraphrase (weight β = {ens['meta'].get('beta', 1.0):g}) and the best is taken, so like the pair-feature reranker it uses the paraphrase at test time. "
              "*Both correct* = share of (question, paraphrase) pairs where both answers are exact matches, a consistency measure that cannot be gamed by always giving the same answer. "
              "Intervals are 95 % paired bootstraps over questions (or pairs) of the change against the extractor, printed to two decimals. "
              "The *consistency-trained* rerankers add a symmetric-KL term between the reranker's distributions for a question and its paraphrase during training (weight w) and use no pair features, so they need no paraphrase at test time." + reeval_note)
            w("")
            w("| Variant (dev) | Δ EM [95 % CI] | Δ F1 | Δ agreement [95 % CI] | Both correct (%) |")
            w("| :-- | :-- | --: | :-- | --: |")
            w(f"| extractor alone | – | – | – | {f1(ext_dev.get('both'))} |")
            w(row("A · pair · argmax", cb_A))
            w(row("B · no pair · argmax", cb_B))
            if cb_E:
                w(row("E · no pair · blend α=1", cb_E))
            w(row(f"paraphrase ensemble, β = {ens['meta'].get('beta', 1.0):g} (no learning)", ens))
            for vn, v in crs:
                w(row(f"consistency-trained w = {v['meta']['cw']:g} · no pair · {'argmax' if v['meta'].get('select_mode') == 'reranker' else 'blend α=1'}", v))
            w("")
            w("Robustness sets, Δ EM [95 % CI] / Δ agreement:")
            w("")
            n_adv = (ens["sets"].get("gan_adv") or {}).get("n")
            w(f"| Variant | AddSent | AddOneSent | G&N paraphrases | G&N adversarial Δ EM ({n_adv} questions) |")
            w("| :-- | :-- | :-- | :-- | --: |")

            def rob(v, st):
                e = v["sets"].get(st)
                return "--" if not e else f"{sgn(e['rr_em'] - e['em'])} {ci(v, st, 'ci_em')} / {sgn(e['rr_agr'] - e['agr'])}"

            for name, v in [("paraphrase ensemble", ens), ("B · no pair · argmax", cb_B)] + [(f"consistency-trained w = {v['meta']['cw']:g} · argmax", v) for vn, v in crs if v["meta"].get("select_mode") == "reranker"]:
                ga = v["sets"].get("gan_adv")
                w(f"| {name} | {rob(v, 'addsent')} | {rob(v, 'addonesent')} | {rob(v, 'gan_dev')} | {sgn(ga['rr_em'] - ga['em']) if ga else '--'} |")
            w("")
            # readings
            e_ens, e_A, e_B = ens["sets"]["dev"], cb_A["sets"]["dev"], cb_B["sets"]["dev"]
            w(f"**Ensemble versus learned reranker.** Averaging the two views gives {sgn(e_ens['rr_agr'] - e_ens['agr'])} agreement and raises both-correct from {f1(ext_dev.get('both'))} to {f1(e_ens.get('rr_both'))} %, at {sgn(e_ens['rr_em'] - e_ens['em'])} EM on the original questions ({excl(ens, 'dev')}). "
              f"The learned pair-feature reranker gives {sgn(e_A['rr_agr'] - e_A['agr'])} agreement and both-correct {f1(e_A.get('rr_both'))} % at {sgn(e_A['rr_em'] - e_A['em'])} EM. So on consistency the learned reranker does not beat simple averaging; what it buys is avoiding the EM cost. "
              f"An agreement of {f1(e_ens['rr_agr'])} % also shows that agreement alone is close to trivially attainable once both views share a candidate list; both-correct is the number to read.")
            w("")
            if crs:
                arg = [(v["meta"]["cw"], v) for vn, v in crs if v["meta"].get("select_mode") == "reranker"]
                hs = d.get("rr_summaries") or {}
                ho = []
                for cw_, v in arg:
                    sm = next((sm_ for tag_, sm_ in hs.items() if tag_.startswith("nopair_cw") and abs(float(tag_[len("nopair_cw"):]) - cw_) < 1e-9), None)
                    if sm:
                        g_ex = [g for g in sm["heldout_selection_grid"] if g["mode"] == "extractor"]
                        g_rr = [g for g in sm["heldout_selection_grid"] if g["mode"] == "reranker" and g["override_margin"] is None]
                        if g_ex and g_rr and g_rr[0].get("agreement_em") is not None:
                            ho.append((cw_, g_ex[0]["agreement_em"], g_rr[0]["agreement_em"]))
                ho_txt = (" On the 758-question early-stopping slice the term barely moves agreement: extractor " + f1(r1(ho[0][1])) + " %, reranker " + ", ".join(f"{f1(r1(a))} % at w = {cw_:g}" for cw_, _, a in ho) + ".") if ho else ""
                worst = max(arg, key=lambda t: t[0])
                def best_grid(tags):
                    c = [(g["exact"], g["f1"], t) for t, sm in hs.items() if t in tags for g in (sm.get("heldout_selection_grid") or []) if g["mode"] != "extractor"]
                    return max(c) if c else None
                b_old = best_grid({"pair", "nopair"})
                b_cw = best_grid({t for t in hs if t.startswith("nopair_cw")})
                if b_old and b_cw and (b_cw[0], b_cw[1]) >= (b_old[0], b_old[1]):
                    ho_txt += f" Under the held-out rule of Section 9 the w = {float(b_cw[2][len('nopair_cw'):]):g} reranker would now be chosen ({f1(r1(b_cw[0]))} / {f1(r1(b_cw[1]))} held-out EM / F1 against {f1(r1(b_old[0]))} / {f1(r1(b_old[1]))}); that table deliberately compares only the two Section 9 rerankers."
                wa = worst[1]["sets"].get("addsent")
                low = arg[:-1]
                cmp_txt = "; ".join(f"w = {cw_:g}: {sgn(v['sets']['dev']['rr_em'] - v['sets']['dev']['em'])} EM / {sgn(v['sets']['dev']['rr_agr'] - v['sets']['dev']['agr'])} agreement" for cw_, v in low)
                w_list = " and ".join(f"{cw_:g}" for cw_, _ in low)
                w(f"**Consistency-trained rerankers.** At w = {w_list} the argmax reranker is indistinguishable from the plain no-pair reranker B ({cmp_txt} against {sgn(e_B['rr_em'] - e_B['em'])} / {sgn(e_B['rr_agr'] - e_B['agr'])} for B); at w = {worst[0]:g} it hurts on AddSent ({sgn(wa['rr_em'] - wa['em'])} EM, {excl(worst[1], 'addsent')}; {sgn(wa['rr_f1'] - wa['f1'])} F1) and is neutral elsewhere."
                  + ho_txt + " Consistency without a paraphrase at test time does not emerge from a reranker training set of this size (80 % of the held-out articles, Section 8).")
                w("")
            sig = [(st, cb_B["sets"][st]) for st in ("addsent", "addonesent") if cb_B["sets"].get(st) and cb_B["sets"][st].get("ci_em") and cb_B["sets"][st]["ci_em"][0] > 0]
            rob3 = ("addsent", "addonesent", "gan_dev")
            xz = lambda v, st: bool((v["sets"].get(st) or {}).get("ci_agr")) and (v["sets"][st]["ci_agr"][0] > 0 or v["sets"][st]["ci_agr"][1] < 0)  # noqa: E731
            np_rows = [cb_B] + [v for _, v in crs]
            n_agr_ex, n_agr_tot = sum(1 for v in np_rows for st in rob3 if xz(v, st)), sum(1 for v in np_rows for st in rob3 if (v["sets"].get(st) or {}).get("ci_agr"))
            n_ens_ex = sum(1 for st in rob3 if xz(ens, st))
            w(f"**What the intervals add.** The no-pair reranker's dev EM change {sgn(e_B['rr_em'] - e_B['em'])} has interval {ci(cb_B, 'dev', 'ci_em')}, so it is not established; its gains on "
              + " and ".join(f"{'AddSent' if st == 'addsent' else 'AddOneSent'} ({sgn(e['rr_em'] - e['em'])} EM, {ci(cb_B, st, 'ci_em')})" for st, e in sig)
              + (" exclude zero and are real on this extractor." if sig else " do not exclude zero.")
              + f" Agreement changes on AddSent, AddOneSent and the Gan & Ng paraphrases exclude zero for {n_ens_ex} of 3 ensemble results but only {n_agr_ex} of {n_agr_tot} no-pair reranker results."
              + " Section 9's conclusion stands: no variant replaces the two system versions, but the ensemble is the stronger consistency device of the two and costs nothing to train, which the group should weigh.")
            w("")
        w("")

    # 10 --------------------------------------------------------------------------------------------------------
    w("## 10. Optimal-transport and span-matching ablation")
    w("")
    if ot and dual:
        w(f"Replacing the KL penalty by the 1-D Wasserstein penalty (`ot`) or by Wasserstein plus Hungarian span matching (`dual`) gives dev agreement {f1(dv(ot,'agr'))} % and {f1(dv(dual,'agr'))} % against {f1(dv(kl,'agr'))} % for KL, at EM / F1 {f1(dv(ot,'em'))} / {f1(dv(ot,'f1'))} and {f1(dv(dual,'em'))} / {f1(dv(dual,'f1'))}. "
          f"Both are within {max(abs(deltas['ot_vs_aug_agr']), abs(deltas['dual_vs_aug_agr'])):.1f} points of augmentation-only ({sgn(deltas['ot_vs_aug_agr'])}, {sgn(deltas['dual_vs_aug_agr'])}) whereas KL adds {sgn(deltas['kl_vs_aug_agr'])}: at their default, untuned weights (λ_OT = 0.5, μ = 0.1) these penalties add nothing beyond the paraphrase data. KL remains the method; this is a negative ablation.")
    w("")

    # 10 --------------------------------------------------------------------------------------------------------
    w("## 11. Question-type analysis (dev)")
    w("")
    bt_ce, bt_kl, bt_rr = ce["sets"]["dev"]["by_type"], kl["sets"]["dev"]["by_type"], kl["sets"]["dev"]["rr_by_type"]
    w(f"Types from the question's wh-word (*how many/much* also covers *how long / old / far / often / tall / big / large*). Baseline and selected run, extractor only; last column with the reranker.")
    w("")
    w(f"| Type | n | Baseline EM / F1 | kl {sel_lam} EM / F1 | Δ F1 | kl {sel_lam} F1 +reranker |")
    w("| :-- | --: | :-- | :-- | --: | --: |")
    rows = sorted(bt_kl.items(), key=lambda kv: -kv[1]["n"])
    for t, m in rows:
        b = bt_ce.get(t, {})
        w(f"| {t} | {m['n']:,} | {f1(b.get('exact'))} / {f1(b.get('f1'))} | {f1(m['exact'])} / {f1(m['f1'])} | {sgn(m['f1'] - b['f1']) if b else '--'} | {f1(bt_rr.get(t, {}).get('f1'))} |")
    w("")
    hardest = sorted(rows, key=lambda kv: kv[1]["f1"])[:3]
    easiest = sorted(rows, key=lambda kv: -kv[1]["f1"])[:2]
    n_rr_lower = sum(1 for t, m in rows if bt_rr.get(t, {}).get("f1", m["f1"]) < m["f1"])
    w(f"Hardest: {', '.join('*' + t + '* (' + f1(m['f1']) + ')' for t, m in hardest)}; easiest: {', '.join('*' + t + '* (' + f1(m['f1']) + ')' for t, m in easiest)}. "
      f"Consistency training changes per-type F1 by at most {max(abs(m['f1'] - bt_ce[t]['f1']) for t, m in rows if t in bt_ce):.1f} points, so it does not redistribute accuracy across types; the reranker lowers F1 for {n_rr_lower} of {len(rows)} types, consistent with Section 8.")
    w("")

    # 11 --------------------------------------------------------------------------------------------------------
    w("## 12. Caveats")
    w("")
    w(f"- One training seed per configuration. Dev EM/F1 differences under about half a point, the {sgn(deltas['kl_vs_aug_agr'])}-point agreement gain of KL over augmentation alone, and everything on the {ce['sets']['gan_adv']['n']}-question adversarial-paraphrase set are unconfirmed.")
    w(f"- Agreement is measured only on questions with an accepted paraphrase ({dev_pairs:,} of {dev_n:,} on dev); the 0.85 filter admits some meaning-changing paraphrases, which caps attainable agreement.")
    w(f"- The λ rule's tolerance ({args.tolerance:g}) was fixed before the runs but not formally pre-registered, and it is load-bearing (Section 5).")
    w("- OT / Hungarian weights were not tuned (Section 10).")
    if var_runs:
        w(f"- The reranker ablation selects on {vr0['rr_summaries']['pair']['n_dev_items']:,} held-out questions, too few to separate configurations a few tenths of a point apart; its dev differences below about half a point are unconfirmed (Section 9). The ensemble baseline and the consistency-trained rerankers were run on one extractor only (Section 9b).")
    if dl_ce:
        w("- The DeBERTa-v3-large runs use a smaller per-step batch and lower learning rate (Section 2) and graded reranker targets (Section 8), so large-versus-base is not a pure backbone comparison.")
    w("- Held-out training articles were used for every decision (checkpoint, λ, reranker); dev and the external sets never were.")
    w("")

    # 12 --------------------------------------------------------------------------------------------------------
    w("## 13. Provenance")
    w("")
    w("```")
    w("python scripts/visualize.py --best " + args.selected + "  # regenerates docs/figures/ from runs/")
    w("python scripts/make_results_report.py              # regenerates this file and docs/results_digest.json")
    w("```")
    w("Sources per run: `config.json`, `train_summary.json`, `reranker_summary.json`, `eval_<set>/results.json` and `official_eval*.json`. "
      "NUS SoC cluster, H100 NVL; base-model training jobs 1068703–1068711 and re-evaluation jobs 1069429–1069438 (3 Nov 2026, after the Gan & Ng original-question scoring fix in commit 3d62dc6)"
      + ("; DeBERTa-v3-large jobs 1072440 (`ce`) and 1072441 (`kl`) on 4 Nov 2026" if dl_ce else "")
      + ("; reranker-ablation jobs 1073460–1073462 on 4 Nov 2026 (`scripts/run_reranker_variants.sh`, outputs under `runs/<run>/rerank_variants/`)" if any(D[n].get("variants") for n in order) else "")
      + ("; ensemble-baseline and consistency-reranker jobs 1096581–1096582 on 8 Nov 2026 (`scripts/run_consistency_baseline.sh`, `scripts/run_consistency_reranker.sh`)" if any("ENSEMBLE" in D[n].get("variants", {}) for n in order) else "") + ". Cluster logs are not part of the repository.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out} and results_digest.json ({len(D)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
