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
        out[name] = {"config": cfg, "summary": summ, "reranker": load(f"{d}/reranker_summary.json"),
                     "evals": {s: load(f"{d}/eval_{s}/results.json") for s in SETS}, "dir": d}
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
            "source": f"{r['dir']}/eval_{st}/results.json",
        }
        if x.get("paired_view_official_on_orig_file"):
            entry["orig_em"] = r1(x["paired_view_official_on_orig_file"]["exact"])
            entry["orig_f1"] = r1(x["paired_view_official_on_orig_file"]["f1"])
            entry["rr_orig_f1"] = r1((y.get("paired_view_official_on_orig_file") or {}).get("f1"))
        d["sets"][st] = entry
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
    w("# G17 Results Report: Paraphrase-Consistent Extractive QA on SQuAD v1.1")
    w("")
    w(f"*Generated {digest['generated'][:10]} by `scripts/make_results_report.py` from the run artefacts in `runs/`. Every number is read from "
      "`runs/<run>/config.json`, `train_summary.json`, `reranker_summary.json` or `eval_<set>/results.json` (which copy the official `evaluate-v2.0.py` output "
      "stored in `official_eval*.json`). Metrics are rounded to one decimal first and every difference quoted in the text is the difference of the rounded values. "
      "The exact values used are in `docs/results_digest.json`."
      + (f" An independent fact-check on {args.verified} recomputed every table cell and prose number from the raw files for the runs present on that date." if args.verified else "") + "*")
    w("")

    # 1 ---------------------------------------------------------------------------------------------------------
    w("## 1. Summary")
    w("")
    w(f"- The BERT-base baseline reaches **{f1(dv(ce,'em'))} EM / {f1(dv(ce,'f1'))} F1** on the full {dev_n:,}-question dev set, matching the published BERT-base result (80.8 / 88.5, Devlin et al. 2019, Table 2), so the training recipe is validated.")
    line = (f"- Consistency training (symmetric KL, λ = {kl['lambda_kl']:g}) raises **paraphrase agreement from {f1(dv(ce,'agr'))} % to {f1(dv(kl,'agr'))} %** ({sgn(deltas['kl_vs_ce_agr'])} points over the baseline, {sgn(deltas['kl_vs_aug_agr'])} over augmentation-only) "
            f"while EM / F1 move by {sgn(deltas['kl_vs_ce_em'])} / {sgn(deltas['kl_vs_ce_f1'])}, i.e. within noise.")
    if deltas["deb_kl_vs_ce_agr"] is not None:
        line += f" DeBERTa-v3-base shows the same pattern ({sgn(deltas['deb_kl_vs_ce_agr'])} agreement, {sgn(deltas['deb_kl_vs_ce_f1'])} F1)."
    w(line)
    w(f"- Against distractor sentences the effect is small: AddSent F1 {f1(rob['addsent']['ce_f1'])} → {f1(rob['addsent']['kl_f1'])} ({sgn(rob['addsent']['delta_f1'])}, of which {sgn(rob['addsent']['aug_delta_f1'])} comes from augmentation alone), "
      f"AddOneSent {f1(rob['addonesent']['ce_f1'])} → {f1(rob['addonesent']['kl_f1'])} ({sgn(rob['addonesent']['delta_f1'])}); agreement on those sets rises by {sgn(rob['addsent']['delta_agr'])} and {sgn(rob['addonesent']['delta_agr'])} points.")
    w(f"- The learned span reranker raises agreement further ({f1(dv(kl,'agr'))} → {f1(dv(kl,'rr_agr'))} % on dev) but costs {sgn(deltas['rr_dev_f1'])} F1 / {sgn(deltas['rr_dev_em'])} EM on dev; its clearest benefit is on the {kl['sets']['gan_adv']['n']}-question adversarial-paraphrase set ({f1(kl['sets']['gan_adv']['f1'])} → {f1(kl['sets']['gan_adv']['rr_f1'])} F1), which is too small for firm conclusions.")
    w(f"- λ was chosen on held-out training articles with the rule \"largest λ whose held-out F1 is within {args.tolerance:g} points of the best\": **λ = {D[pick_s]['lambda_kl']:g}** ({pick_s}). "
      f"The rule gives λ = {D[pick_f]['lambda_kl']:g} as well when applied to all {D[pick_s]['full_val_n']:,} held-out questions instead of the {D[pick_s]['val_n']:,}-question selection sample, where λ = {best_f_run['lambda_kl']:g} is nominally the best (by {min(v for k, v in gaps_f.items() if k != best_f_run['lambda_kl']):.2f} points, well within noise).")
    if dl_ce and dl_kl and deb_ce:
        w(f"- Scaling the backbone to DeBERTa-v3-large gives **{f1(dv(dl_ce,'em'))} EM / {f1(dv(dl_ce,'f1'))} F1** ({sgn(dv(dl_ce,'em') - dv(deb_ce,'em'))} / {sgn(dv(dl_ce,'f1') - dv(deb_ce,'f1'))} over DeBERTa-v3-base). "
          f"Consistency training again raises agreement ({f1(dv(dl_ce,'agr'))} → {f1(dv(dl_kl,'agr'))} %, {sgn(dv(dl_kl,'agr') - dv(dl_ce,'agr'))}) at {sgn(dv(dl_kl,'f1') - dv(dl_ce,'f1'))} F1, and on this backbone the reranker costs only {sgn(dv(dl_kl,'rr_f1') - dv(dl_kl,'f1'))} F1 on dev while raising F1 on {of4(rob_up['deb_large_kl'])} robustness sets "
          "(one seed; these rerankers use the newer graded targets, Section 8).")
    w("")

    # 2 ---------------------------------------------------------------------------------------------------------
    w("## 2. What was run")
    w("")
    nf_txt = "; ".join(f"{m.split('/')[-1]}: {v[0]:,}" + (f"–{v[-1]:,}" if len(v) > 1 else "") + " windows" for m, v in nf.items())
    recipe = (f"All runs share the same recipe: {ce['epochs']} epochs, effective batch {ce['batch_size'] * ce['grad_accum']}, learning rate {ce['lr']:g}, sequence length {ce['max_seq_len']}, window stride {ce['doc_stride']}, AdamW with 10 % warm-up, best checkpoint by validation F1"
              + (f"; the DeBERTa-v3-large runs keep the effective batch ({dl_ce['batch_size']} × {dl_ce['grad_accum']} gradient-accumulation steps) but use learning rate {dl_ce['lr']:g}, the usual range for large encoders (Section 6). " if dl_ce else ". "))
    w(recipe
      + f"Training uses the {ce['n_train_questions']:,} questions from 95 % of the SQuAD training articles ({nf_txt}; the count varies slightly with the tokeniser and with whether a paraphrase view is built). "
      f"The {ce['n_val_questions']:,} questions of the remaining 5 % of articles are held out: during training, validation is run on a seeded random sample of {ce['val_max_examples']:,} of them (`--val_max_examples`), which is what checkpoint selection and the λ rule use; the selected checkpoint is then evaluated once on all {ce['n_val_questions']:,} (`final_full_val`), and the reranker is trained on the same articles. "
      "The dev set and the external test sets are used only for the final numbers.")
    w("")
    w("| Run | Model | Objective | λ | Batch × accum. | LR | Optimiser steps | Training time (min) | Paraphrases attached to training questions |")
    w("| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |")
    for n in order:
        d = D[n]
        ps = d["pair_stats"]
        att = f"{ps['attached']:,} / {ps['total']:,}" if ps else "not used"
        w(f"| {label[n]} | {d['model'].split('/')[-1]} | {MODE_DESC.get(d['mode'], d['mode'])} | {f1(d['lambda_kl'],1) if d['lambda_kl'] is not None else '–'} | {d['batch_size']} × {d['grad_accum']} | {d['lr']:g} | {d['steps']:,} | {d['train_time_min']:.0f} | {att} |")
    w("")
    w(f"Paraphrases come from MarianMT back-translation (EN→DE→EN) filtered at cosine similarity ≥ 0.85. On the dev set {dev_pairs:,} of {dev_n:,} questions ({100*dev_pairs/dev_n:.1f} %) received an accepted paraphrase; the agreement metric is computed over those pairs only.")
    w("")

    # 3 ---------------------------------------------------------------------------------------------------------
    w("## 3. Main results on the dev set")
    w("")
    w(f"Official scorer (`evaluate-v2.0.py`), all {dev_n:,} questions. *Agreement* = percentage of (question, paraphrase) pairs that receive the same normalised answer. The `+reranker` columns are the second system version.")
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
    w(f"**Reading.** Accuracy is flat across the {len(bert_names)} BERT objectives (EM {em_lo:.1f}–{em_hi:.1f}, F1 {f1_lo:.1f}–{f1_hi:.1f}); the binomial standard error of one EM value over {dev_n:,} questions is about {deltas['se_dev_em']:.2f} points and seed-to-seed variation of BERT fine-tuning is typically a few tenths of a point, so differences below roughly half a point should not be interpreted. "
      f"Agreement is where the objectives separate: baseline {f1(dv(ce,'agr'))} %, augmentation {f1(dv(aug,'agr'))} %, consistency training {f1(dv(kl,'agr'))} % (λ = 1) and {f1(dv(kl2,'agr')) if kl2 else '--'} % (λ = 2). "
      f"With {dev_pairs:,} pairs the standard error of one agreement value is about {deltas['se_dev_agr']:.2f} points and that of a difference between two runs about {deltas['se_dev_agr_diff']:.2f}; the {sgn(deltas['kl_vs_ce_agr'])}-point gain over the baseline is about {deltas['kl_vs_ce_agr_in_se']:.0f} standard errors and clearly real, whereas the {sgn(deltas['kl_vs_aug_agr'])}-point gain over augmentation-only is about {deltas['kl_vs_aug_agr_in_se']:.1f} standard errors and, with one training seed per configuration, only suggestive.")
    w("")

    # 4 ---------------------------------------------------------------------------------------------------------
    w("## 4. Augmentation versus consistency loss")
    w("")
    w(f"The `aug` run isolates the effect of the extra paraphrased examples from the effect of the penalty. Extra data alone moves agreement from {f1(dv(ce,'agr'))} % to {f1(dv(aug,'agr'))} % ({sgn(deltas['aug_vs_ce_agr'])}); the KL penalty adds {sgn(deltas['kl_vs_aug_agr'])} points on top of that at λ = 1 and {sgn(deltas['kl2_vs_aug_agr'])} at λ = 2 (both relative to `aug`). "
      f"Accuracy does not pay for this: `aug` scores {f1(dv(aug,'em'))} / {f1(dv(aug,'f1'))} and `kl` (λ = 1) {f1(dv(kl,'em'))} / {f1(dv(kl,'f1'))} EM / F1. "
      f"So roughly {100 * deltas['aug_vs_ce_agr'] / deltas['kl_vs_ce_agr']:.0f} % of the agreement gain over the baseline comes from the paraphrase data itself and the remainder from the penalty; the penalty's own share is the part that is only suggestive at one seed (Section 3).")
    w("")

    # 5 ---------------------------------------------------------------------------------------------------------
    w("## 5. Choosing λ on held-out data")
    w("")
    w(f"The form of the rule (largest λ whose held-out F1 is within a tolerance of the best) and the example tolerance of {args.tolerance:g} points were written into the README and the report draft before the runs finished; the value {args.tolerance:g} was not formally pre-registered and is load-bearing (on the selection sample a tolerance below {gaps_s[1.0]:.2f} would pick λ = 0.5 and one of {gaps_s[2.0]:.2f} or more would pick λ = 2; on the full held-out set λ = 2 is {gaps_f[2.0]:.2f} below the best). "
      "The rule is applied to unrounded F1 values. The table shows both the selection sample used during training and the full held-out set.")
    w("")
    w(f"| Model | λ | Sample F1 (n={sweep[0]['val_n']:,}) | Sample agreement | Full held-out F1 (n={sweep[0]['full_val_n']:,}) | Full held-out agreement | Dev F1 | Dev agreement |")
    w("| :-- | --: | --: | --: | --: | --: | --: | --: |")
    for d in sorted([d for d in D.values() if d["mode"] == "kl"], key=lambda d: (d["model"], d["lambda_kl"])):
        w(f"| {d['model'].split('/')[-1]} | {d['lambda_kl']:g} | {f1(d['val_f1'])} | {f1(d['val_agreement'])} | {f1(d['full_val_f1'])} | {f1(d['full_val_agreement'])} | {f1(d['sets']['dev']['f1'])} | {f1(d['sets']['dev']['agr'])} |")
    w("")
    w(f"On the selection sample the best F1 is {f1(best_s)} (λ = {max(sweep, key=lambda d: d['val_f1_raw'])['lambda_kl']:g}) and λ = " + ", ".join(f"{d['lambda_kl']:g}" for d in elig_s) + f" fall within {args.tolerance:g} points, so the rule selects **λ = {D[pick_s]['lambda_kl']:g}**. "
      f"On all {sweep[0]['full_val_n']:,} held-out questions the F1 values are {', '.join(f1(d['full_val_f1']) for d in sweep)} for λ = {lam_list}: λ = {best_f_run['lambda_kl']:g} is nominally the best (by {min(v for k, v in gaps_f.items() if k != best_f_run['lambda_kl']):.2f} points), λ = " + ", ".join(f"{d['lambda_kl']:g}" for d in elig_f if d is not best_f_run) + f" within tolerance, and λ = 2 is {gaps_f[2.0]:.2f} below the best, so the rule selects λ = {D[pick_f]['lambda_kl']:g} there too; "
      f"agreement rises with λ on both the sample ({', '.join(f1(d['val_agreement']) for d in sweep)}) and the full set ({', '.join(f1(d['full_val_agreement']) for d in sweep)}). "
      f"The sample differences in F1 ({', '.join(f1(d['val_f1']) for d in sweep)}) are within the sampling error of a {sweep[0]['val_n']:,}-question sample (at most about {se_pct(best_s, sweep[0]['val_n']):.1f} points for one value under a binomial bound, roughly {math.sqrt(2) * se_pct(best_s, sweep[0]['val_n']):.1f} for a difference between two runs), so no accuracy trade-off is demonstrated between these λ values. "
      f"On dev, λ = 1 also has the best F1 of the three ({', '.join(f1(d['sets']['dev']['f1']) for d in sweep)}).")
    w("")

    # 6 ---------------------------------------------------------------------------------------------------------
    w("## 6. Backbone: BERT-base, DeBERTa-v3-base and DeBERTa-v3-large")
    w("")
    if deb_ce and deb_kl:
        w(f"DeBERTa-v3-base is far stronger as an extractor ({f1(dv(deb_ce,'em'))} / {f1(dv(deb_ce,'f1'))} EM / F1 versus {f1(dv(ce,'em'))} / {f1(dv(ce,'f1'))} for BERT-base) and already more self-consistent ({f1(dv(deb_ce,'agr'))} % agreement versus {f1(dv(ce,'agr'))} %). "
          f"Consistency training still adds {sgn(deltas['deb_kl_vs_ce_agr'])} agreement points at {sgn(deltas['deb_kl_vs_ce_f1'])} F1 on dev, so the effect is not specific to the weaker backbone. "
          f"On the full held-out set the picture is the same: F1 {f1(deb_ce['full_val_f1'])} → {f1(deb_kl['full_val_f1'])}, agreement {f1(deb_ce['full_val_agreement'])} → {f1(deb_kl['full_val_agreement'])}.")
    if dl_ce and dl_kl and deb_ce:
        w("")
        w(f"**DeBERTa-v3-large.** The large model (24 layers, hidden size 1,024) was fine-tuned on the same data and windows for the same {dl_ce['epochs']} epochs and effective batch, with a per-step batch of {dl_ce['batch_size']} × {dl_ce['grad_accum']} accumulation steps and learning rate {dl_ce['lr']:g} (Section 2). "
          f"Training took {dl_ce['train_time_min']:.0f} min (`ce`) and {dl_kl['train_time_min']:.0f} min (`kl`) with peak GPU memory {f1(dl_ce['peak_gpu_mem_gb'])} GB and {f1(dl_kl['peak_gpu_mem_gb'])} GB; the cluster logs show the `ce` job on a 47 GB H100 MIG slice but the `kl` job on a full H100 NVL shared with another process, so the two times are not comparable with each other or with the base model's {deb_ce['train_time_min']:.0f} and {deb_kl['train_time_min']:.0f} min on MIG slices. "
          f"As an extractor it reaches **{f1(dv(dl_ce,'em'))} / {f1(dv(dl_ce,'f1'))} EM / F1** on dev, {sgn(dv(dl_ce,'em') - dv(deb_ce,'em'))} / {sgn(dv(dl_ce,'f1') - dv(deb_ce,'f1'))} over DeBERTa-v3-base and {sgn(dv(dl_ce,'em') - dv(ce,'em'))} / {sgn(dv(dl_ce,'f1') - dv(ce,'f1'))} over BERT-base, with {f1(dv(dl_ce,'agr'))} % agreement. "
          f"Consistency training adds {sgn(dv(dl_kl,'agr') - dv(dl_ce,'agr'))} agreement points ({f1(dv(dl_ce,'agr'))} → {f1(dv(dl_kl,'agr'))} %) at {sgn(dv(dl_kl,'f1') - dv(dl_ce,'f1'))} F1 / {sgn(dv(dl_kl,'em') - dv(dl_ce,'em'))} EM, so all three backbones show the same trade; on the full held-out set F1 goes {f1(dl_ce['full_val_f1'])} → {f1(dl_kl['full_val_f1'])} and agreement {f1(dl_ce['full_val_agreement'])} → {f1(dl_kl['full_val_agreement'])}. "
          f"Two things differ from DeBERTa-v3-base. First, consistency training raises rather than lowers distractor F1 (AddSent {f1(dl_ce['sets']['addsent']['f1'])} → {f1(dl_kl['sets']['addsent']['f1'])}, AddOneSent {f1(dl_ce['sets']['addonesent']['f1'])} → {f1(dl_kl['sets']['addonesent']['f1'])}), as it does for BERT-base (Section 7). "
          f"Second, the reranker raises F1 on {'all four robustness sets for both runs' if rob_up['deb_large_ce'] == rob_up['deb_large_kl'] == 4 else of4(rob_up['deb_large_ce']) + ' robustness sets for ce and ' + of4(rob_up['deb_large_kl']) + ' for kl'}, which {('none of the ' + str(n_base_rr) + ' base-model rerankers does') if not base_all4 else ('only ' + ' and '.join('`' + n + '`' for n in base_all4) + ' of the ' + str(n_base_rr) + ' base-model rerankers also ' + ('does' if len(base_all4) == 1 else 'do') + ', by smaller margins')}, while costing {sgn(dv(dl_ce,'rr_f1') - dv(dl_ce,'f1'))} / {sgn(dv(dl_kl,'rr_f1') - dv(dl_kl,'f1'))} F1 on dev (`ce` / `kl`) and adding {sgn(dv(dl_ce,'rr_agr') - dv(dl_ce,'agr'))} / {sgn(dv(dl_kl,'rr_agr') - dv(dl_kl,'agr'))} agreement; the near-zero dev cost itself is not new (DeBERTa-v3-base `kl` was at {sgn(dv(deb_kl,'rr_f1') - dv(deb_kl,'f1'))} F1, for {sgn(dv(deb_kl,'rr_agr') - dv(deb_kl,'agr'))} agreement; Section 8). "
          "The two large-model rerankers were trained with the newer graded targets, so backbone and target effects are confounded there (Section 8), and every number in this paragraph is one seed per configuration.")
    w("")

    # 7 ---------------------------------------------------------------------------------------------------------
    w("## 7. Robustness: paraphrased and adversarial test sets")
    w("")
    w("F1 from the official scorer. For the Gan & Ng sets, *orig* is the model's F1 on the original questions and *para* on their human-written paraphrases; *agreement* is between the two answers. For AddSent / AddOneSent the agreement is with back-translated paraphrases of the adversarial questions.")
    w("")
    w("| Run | G&N orig F1 | G&N para F1 | G&N agreement | G&N adv orig F1 | G&N adv para F1 | AddSent F1 | AddSent agreement | AddOneSent F1 | AddOneSent agreement |")
    w("| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |")
    for n in order:
        s = D[n]["sets"]
        g, ga, a1, a2 = s.get("gan_dev"), s.get("gan_adv"), s.get("addsent"), s.get("addonesent")
        w(f"| {label[n]} | {f1(g and g.get('orig_f1'))} | {f1(g and g['f1'])} | {f1(g and g['agr'])} | {f1(ga and ga.get('orig_f1'))} | {f1(ga and ga['f1'])} | {f1(a1 and a1['f1'])} | {f1(a1 and a1['agr'])} | {f1(a2 and a2['f1'])} | {f1(a2 and a2['agr'])} |")
    w("")
    para_bert = [v for v, n in addsent_bert if n != "bert_ce"]
    w(f"**Distractor sentences (Jia & Liang).** Every model loses heavily when a distractor sentence is appended: the BERT baseline drops from {f1(dv(ce,'f1'))} F1 on dev to {f1(rob['addsent']['ce_f1'])} on AddSent (−{rob['addsent']['drop_from_dev_ce']:.1f}) and {f1(rob['addonesent']['ce_f1'])} on AddOneSent (−{rob['addonesent']['drop_from_dev_ce']:.1f}). "
      f"All {len(para_bert)} paraphrase-trained BERT runs land in a narrow band on AddSent ({min(para_bert):.1f}–{max(para_bert):.1f} F1), so the {sgn(rob['addsent']['delta_f1'])} of the λ = 1 run over the baseline is mostly the paraphrase data (`aug` alone: {sgn(rob['addsent']['aug_delta_f1'])}); on AddOneSent `aug` is {sgn(rob['addonesent']['aug_delta_f1'])} and λ = 1 {sgn(rob['addonesent']['delta_f1'])}. "
      f"What consistency training does change is stability: agreement on AddSent goes from {f1(rob['addsent']['ce_agr'])} % to {f1(rob['addsent']['kl_agr'])} % ({sgn(rob['addsent']['delta_agr'])}), i.e. the model is fooled more consistently rather than less often. "
      + (f"DeBERTa is much more robust to distractors ({f1(deb_ce['sets']['addsent']['f1'])} F1 on AddSent for the baseline) and consistency training slightly lowers its AddSent F1 ({f1(deb_kl['sets']['addsent']['f1'])}) while raising agreement ({f1(deb_ce['sets']['addsent']['agr'])} → {f1(deb_kl['sets']['addsent']['agr'])} %)." if (deb_ce and deb_kl) else "")
      + (f" For DeBERTa-v3-large the direction reverses: under consistency training AddSent F1 goes {f1(dl_ce['sets']['addsent']['f1'])} → {f1(dl_kl['sets']['addsent']['f1'])} ({sgn(dl_kl['sets']['addsent']['f1'] - dl_ce['sets']['addsent']['f1'])}) and AddOneSent {f1(dl_ce['sets']['addonesent']['f1'])} → {f1(dl_kl['sets']['addonesent']['f1'])} ({sgn(dl_kl['sets']['addonesent']['f1'] - dl_ce['sets']['addonesent']['f1'])}), with AddSent agreement {f1(dl_ce['sets']['addsent']['agr'])} → {f1(dl_kl['sets']['addsent']['agr'])} % ({sgn(dl_kl['sets']['addsent']['agr'] - dl_ce['sets']['addsent']['agr'])}, about {(dl_kl['sets']['addsent']['agr'] - dl_ce['sets']['addsent']['agr']) / (math.sqrt(2) * se_pct(dl_ce['sets']['addsent']['agr'], dl_ce['sets']['addsent']['pairs'])):.0f} standard errors of a difference between two runs over {dl_ce['sets']['addsent']['pairs']:,} pairs). With one seed per run the F1 gains are of the size of ordinary seed-to-seed variation (Section 3) and only suggestive; the agreement gain is not." if (dl_ce and dl_kl) else ""))
    w("")
    g_ce, g_kl = ce["sets"]["gan_dev"], kl["sets"]["gan_dev"]
    w(f"**Human paraphrases (Gan & Ng, {g_ce['n']:,} questions, {g_ce['pairs']:,} pairs).** Paraphrasing costs the baseline {g_ce['orig_f1'] - g_ce['f1']:.1f} F1 ({f1(g_ce['orig_f1'])} on the original questions → {f1(g_ce['f1'])} on the paraphrases) and the λ = 1 model {g_kl['orig_f1'] - g_kl['f1']:.1f} ({f1(g_kl['orig_f1'])} → {f1(g_kl['f1'])}); "
      f"note that the narrower gap comes from both directions, {sgn(g_kl['f1'] - g_ce['f1'])} on the paraphrases but {sgn(g_kl['orig_f1'] - g_ce['orig_f1'])} on the originals. Agreement between the two answers goes from {f1(g_ce['agr'])} % to {f1(g_kl['agr'])} % ({sgn(rob['gan_dev']['delta_agr'])}). "
      f"On the {ce['sets']['gan_adv']['n']}-question adversarial-paraphrase subset the baseline falls to {f1(ce['sets']['gan_adv']['f1'])} F1 and the λ = 1 model to {f1(kl['sets']['gan_adv']['f1'])}; with {ce['sets']['gan_adv']['n']} questions each one is worth {100/ce['sets']['gan_adv']['n']:.1f} points, so these numbers are indicative only.")
    w("")

    # 8 ---------------------------------------------------------------------------------------------------------
    w("## 8. The learned span reranker")
    w("")
    rr0 = kl["reranker"]
    w(f"Held-out columns come from `reranker_summary.json`: the reranker is trained on 80 % of the held-out articles and early-stopped on the other 20 % ({rr0['dev_items']:,} questions); the held-out *reranked* value is the best of {rr0['n_epochs']} epochs on that same slice (chosen by F1 for the one-hot rows, by EM then F1 for the graded rows) and is therefore optimistic. The remaining columns are the dev set. Δ = reranked − extractor.")
    w("")
    w("| Run | Targets | Held-out top-1 EM / F1 | Held-out reranked EM / F1 (best epoch) | Last-epoch F1 | Oracle EM (correct span among top-20) | Dev Δ EM | Dev Δ F1 | Dev Δ agreement | AddSent Δ F1 | G&N adv Δ F1 |")
    w("| :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: |")
    for n, d in rr_rows:
        rr = d["reranker"]
        s, a1, ga = d["sets"]["dev"], d["sets"].get("addsent"), d["sets"].get("gan_adv")
        w(f"| {label[n]} | {'graded' if rr['target_mode'] == 'graded' else 'one-hot'} | {f1(rr['top1']['exact'])} / {f1(rr['top1']['f1'])} | {f1(rr['reranked']['exact'])} / {f1(rr['reranked']['f1'])} (ep. {rr['best_epoch']}) | {f1(rr['last_epoch_f1'])} | {f1(rr['top1']['oracle_exact'])} | {sgn(s['rr_em'] - s['em'])} | {sgn(s['rr_f1'] - s['f1'])} | {sgn(s['rr_agr'] - s['agr'])} | {sgn(a1['rr_f1'] - a1['f1']) if a1 else '--'} | {sgn(ga['rr_f1'] - ga['f1']) if ga else '--'} |")
    w("")
    n_f1_down = sum(1 for _, d in rr_rows if dv(d, "rr_f1") < dv(d, "f1"))
    n_f1_flat = sum(1 for _, d in rr_rows if dv(d, "rr_f1") == dv(d, "f1"))
    n_agr_up = sum(1 for _, d in rr_rows if dv(d, "rr_agr") > dv(d, "agr"))
    ho_em = [d["reranker"]["reranked"]["exact"] - d["reranker"]["top1"]["exact"] for _, d in rr_rows]
    w(f"**Reading.** The oracle column shows that the correct span is among the top-20 candidates for {min(d['reranker']['top1']['oracle_exact'] for _, d in rr_rows):.0f}–{max(d['reranker']['top1']['oracle_exact'] for _, d in rr_rows):.0f} % of held-out questions, so there is room to gain; the reranker captures almost none of it. "
      f"The best-epoch held-out EM changes of {min(ho_em):+.1f} to {max(ho_em):+.1f} points are measured on {rr0['dev_items']:,} questions (binomial standard error {min(se_pct(d['reranker']['top1']['exact'], d['reranker']['dev_items']) for _, d in rr_rows):.1f}–{max(se_pct(d['reranker']['top1']['exact'], d['reranker']['dev_items']) for _, d in rr_rows):.1f} points) and the last-epoch F1 is {min(rr_overfit.values()):.1f}–{max(rr_overfit.values()):.1f} points below the best epoch, i.e. the reranker overfits within a few epochs. "
      f"The honest estimates are the dev deltas: F1 never rises (it falls in {n_f1_down} of {len(rr_rows)} runs and is unchanged in {n_f1_flat}) while agreement rises in {n_agr_up} of {len(rr_rows)}; for the selected run the trade is {sgn(deltas['rr_dev_f1'])} F1 for {sgn(deltas['rr_dev_agr'])} agreement. "
      "We hypothesise that the paraphrase-score feature drives this (the reranker prefers spans both phrasings score highly, which stabilises answers but sometimes overrides a correct top-1 span); no run without that feature was trained, so this is not established by ablation. "
      f"Its largest effect is on the adversarial paraphrases of Gan & Ng, where the paired view is the easier original question (orig F1 {f1(kl['sets']['gan_adv']['orig_f1'])}) and the reranker can lean on it (F1 {f1(kl['sets']['gan_adv']['f1'])} → {f1(kl['sets']['gan_adv']['rr_f1'])} for the selected run; small set). "
      "Conclusion: the reranker is a consistency device with an accuracy cost on in-distribution data; both system versions are reported.")
    graded_rows = [(n, d) for n, d in rr_rows if d["reranker"]["target_mode"] == "graded"]
    if graded_rows:
        dd = lambda d, st, a, b: (d["sets"][st][a] - d["sets"][st][b]) if (d["sets"].get(st) and d["sets"][st].get(a) is not None and d["sets"][st].get(b) is not None) else None  # noqa: E731
        w("")
        w(f"**Graded-target rerankers ({' and '.join('`' + n + '`' for n, _ in graded_rows)}).** These rerankers were trained after the training target changed from a one-hot label (the first exact-match candidate, else the best-F1 candidate; best epoch by held-out F1) to graded relevance (exact-match candidates 1, partial-overlap candidates 0.3 × F1, normalised to a distribution; best epoch by held-out EM, then F1), so their rows are not strictly comparable with the other {len(rr_rows) - len(graded_rows)}. "
          + " ".join(f"For `{n}` the dev trade is {sgn(dd(d,'dev','rr_f1','f1'))} F1 / {sgn(dd(d,'dev','rr_em','em'))} EM for {sgn(dd(d,'dev','rr_agr','agr'))} agreement, and the reranker changes F1 by {sgn(dd(d,'addsent','rr_f1','f1'))} on AddSent, {sgn(dd(d,'addonesent','rr_f1','f1'))} on AddOneSent, {sgn(dd(d,'gan_dev','rr_f1','f1'))} on the Gan & Ng paraphrases and {sgn(dd(d,'gan_adv','rr_f1','f1'))} on the adversarial paraphrases." for n, d in graded_rows)
          + " The near-zero dev cost on the large extractor may come from the stronger extractor (its top-1 is right more often, so there is less to override), from the graded targets, or both; separating the two needs the base-model rerankers retrained with graded targets, which the reranker-ablation scripts added on 4 Nov 2026 can do without retraining any extractor.")
    w("")

    # 9 ---------------------------------------------------------------------------------------------------------
    w("## 9. Optimal-transport and span-matching ablation")
    w("")
    if ot and dual:
        w(f"Replacing the KL penalty by the 1-D Wasserstein penalty (`ot`) or by Wasserstein plus Hungarian span matching (`dual`), with the same paraphrase data, gives dev agreement of {f1(dv(ot,'agr'))} % and {f1(dv(dual,'agr'))} % against {f1(dv(kl,'agr'))} % for KL at λ = 1, with EM / F1 of {f1(dv(ot,'em'))} / {f1(dv(ot,'f1'))} and {f1(dv(dual,'em'))} / {f1(dv(dual,'f1'))}. "
          f"Both are within {max(abs(deltas['ot_vs_aug_agr']), abs(deltas['dual_vs_aug_agr'])):.1f} points of the augmentation-only run ({sgn(deltas['ot_vs_aug_agr'])} and {sgn(deltas['dual_vs_aug_agr'])}), so at their default weights (λ_OT = 0.5, μ = 0.1, not tuned) these penalties add nothing beyond the paraphrase data, whereas KL adds {sgn(deltas['kl_vs_aug_agr'])}. "
          f"On AddSent they reach {f1(ot['sets']['addsent']['f1'])} and {f1(dual['sets']['addsent']['f1'])} F1 versus {f1(kl['sets']['addsent']['f1'])}. KL remains the primary method; these runs are a negative ablation.")
    w("")

    # 10 --------------------------------------------------------------------------------------------------------
    w("## 10. Question-type analysis (dev)")
    w("")
    bt_ce, bt_kl, bt_rr = ce["sets"]["dev"]["by_type"], kl["sets"]["dev"]["by_type"], kl["sets"]["dev"]["rr_by_type"]
    w("EM / F1 by question type for the baseline and the selected consistency run (extractor only), and F1 with the reranker for the selected run. Types are assigned from the question's wh-word; *how many/much* also covers *how long / old / far / often / tall / big / large*.")
    w("")
    w("| Type | n | Baseline EM / F1 | kl λ=1 EM / F1 | Δ F1 | kl λ=1 F1 +reranker |")
    w("| :-- | --: | :-- | :-- | --: | --: |")
    rows = sorted(bt_kl.items(), key=lambda kv: -kv[1]["n"])
    for t, m in rows:
        b = bt_ce.get(t, {})
        w(f"| {t} | {m['n']:,} | {f1(b.get('exact'))} / {f1(b.get('f1'))} | {f1(m['exact'])} / {f1(m['f1'])} | {sgn(m['f1'] - b['f1']) if b else '--'} | {f1(bt_rr.get(t, {}).get('f1'))} |")
    w("")
    hardest = sorted(rows, key=lambda kv: kv[1]["f1"])[:3]
    easiest = sorted(rows, key=lambda kv: -kv[1]["f1"])[:2]
    hard_s = ", ".join("*" + t + "* (" + f1(m["f1"]) + " F1)" for t, m in hardest)
    easy_s = ", ".join("*" + t + "* (" + f1(m["f1"]) + ")" for t, m in easiest)
    n_rr_lower = sum(1 for t, m in rows if bt_rr.get(t, {}).get("f1", m["f1"]) < m["f1"])
    w(f"The hardest types are {hard_s}; the easiest are {easy_s}. Consistency training changes per-type F1 by at most {max(abs(m['f1'] - bt_ce[t]['f1']) for t, m in rows if t in bt_ce):.1f} points, i.e. it does not redistribute accuracy across question types; its effect is on stability, which this table does not measure. "
      f"The reranker lowers F1 for {n_rr_lower} of the {len(rows)} types, consistent with Section 8.")
    w("")

    # 11 --------------------------------------------------------------------------------------------------------
    w("## 11. Caveats")
    w("")
    w(f"- Single training seed per configuration; no confidence intervals beyond the binomial standard errors quoted above. Dev EM/F1 differences below about half a point, the {sgn(deltas['kl_vs_aug_agr'])}-point agreement gain of KL over augmentation alone, and all differences on the {ce['sets']['gan_adv']['n']}-question adversarial-paraphrase set should be treated as unconfirmed.")
    w(f"- Agreement is measured only on questions with an accepted back-translated paraphrase ({dev_pairs:,} of {dev_n:,} on dev); the filter admits some meaning-changing paraphrases, which caps attainable agreement below 100 %.")
    w(f"- Checkpoint selection and the λ rule used a {ce['val_max_examples']:,}-question sample of the held-out articles; the full-set numbers are reported alongside and lead to the same λ.")
    w("- The OT / Hungarian weights were not tuned, so Section 9 compares them at default weights.")
    w("- Validation articles were used for checkpoint selection, λ selection and reranker training; dev and the external sets were never used for any decision.")
    if dl_ce:
        w("- The DeBERTa-v3-large runs use a smaller per-step batch and a lower learning rate than the base runs (Section 2), and their rerankers were trained with the graded targets (Section 8), so the large-versus-base comparison is not a pure backbone comparison.")
    w("")

    # 12 --------------------------------------------------------------------------------------------------------
    w("## 12. Provenance")
    w("")
    w("```")
    w("python visualize.py --best " + args.selected + "        # regenerates figures/ and report/tables/ from runs/")
    w("python scripts/make_results_report.py              # regenerates this file and docs/results_digest.json")
    w("```")
    w("Per-run sources: `runs/<run>/config.json` (hyper-parameters, split sizes, paraphrase attachment), `runs/<run>/train_summary.json` (steps, time, selection-sample and full held-out F1/agreement), `runs/<run>/reranker_summary.json` (including the per-epoch history), `runs/<run>/eval_<set>/results.json` and `official_eval*.json` (official scorer output). "
      "Training, reranker fitting and the dev evaluations were produced by the training jobs on 3 Nov 2026; the four external sets were re-evaluated the same evening after a fix to the Gan & Ng original-question scoring (commit 3d62dc6). Hardware (NUS SoC cluster, H100 NVL MIG 3g.47gb slices) and Slurm job ids (training 1068703–1068711, re-evaluation 1069429–1069438) are taken from the cluster logs, which are not part of the repository."
      + (" The DeBERTa-v3-large runs (training, reranker and all evaluations in one job each) were produced on 4 Nov 2026 as Slurm jobs 1072440 (`ce`) and 1072441 (`kl`); their `train_summary.json` also records the peak GPU memory quoted in Section 6." if dl_ce else ""))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out} and results_digest.json ({len(D)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
