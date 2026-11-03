#!/usr/bin/env python3
"""Build report figures and LaTeX/Markdown tables from *real* run artefacts only.

Reads   runs/<run>/history.json            (train.py loss telemetry)
        runs/<run>/train_summary.json      (validation history)
        runs/<run>/reranker_summary.json   (train_reranker.py)
        runs/<run>/eval_<set>/results.json (evaluate.py, official-scorer numbers)
Writes  figures/*.pdf|png, figures/*.md, report/figures/*.pdf, report/tables/*.tex

There are deliberately no fallbacks: if a run or evaluation is missing, the corresponding figure/table cell is
simply absent, and the script exits non-zero when it finds nothing at all.

  python visualize.py                  # all runs
  python visualize.py --runs bert_ce bert_aug bert_kl --best bert_kl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 200})

EVAL_SETS = ["dev", "gan_dev", "gan_adv", "addsent", "addonesent"]
SET_LABELS = {"dev": "SQuAD dev", "gan_dev": "Gan\\&Ng para.", "gan_adv": "Gan\\&Ng adv.", "addsent": "AddSent", "addonesent": "AddOneSent"}
SET_LABELS_PLAIN = {k: v.replace("\\&", "&") for k, v in SET_LABELS.items()}
MODE_LABELS = {"ce": "CE", "aug": "CE + aug.", "kl": "CE + aug. + KL", "ot": "CE + aug. + OT", "hungarian": "CE + aug. + Hung.", "dual": "CE + aug. + OT + Hung."}


def _load(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect(runs_dir: str, only: Optional[List[str]], exclude: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    import fnmatch

    runs: Dict[str, Dict[str, Any]] = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d) or (only and name not in only):
            continue
        if not only and exclude and any(fnmatch.fnmatch(name, pat) for pat in exclude):
            continue
        cfg = _load(os.path.join(d, "config.json")) or {}
        entry = {
            "name": name, "config": cfg, "history": _load(os.path.join(d, "history.json")),
            "summary": _load(os.path.join(d, "train_summary.json")), "reranker": _load(os.path.join(d, "reranker_summary.json")),
            "evals": {s: _load(os.path.join(d, f"eval_{s}", "results.json")) for s in EVAL_SETS},
        }
        if entry["history"] or any(entry["evals"].values()):
            runs[name] = entry
    return runs


def _save(fig, out_dirs: List[str], stem: str) -> None:
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dirs[0], stem + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def _ema(x: np.ndarray, w: float = 0.9) -> np.ndarray:
    out, v = [], float(x[0])
    for a in x:
        v = w * v + (1 - w) * float(a)
        out.append(v)
    return np.asarray(out)


# ---------------------------------------------------------------------------------------------
def fig_loss_curves(runs, out_dirs) -> bool:
    with_hist = [r for r in runs.values() if r["history"]]
    if not with_hist:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for r in with_hist:
        h = r["history"]
        steps = np.array([e["step"] for e in h])
        axes[0].plot(steps, _ema(np.array([e["total"] for e in h])), lw=1.2, label=r["name"])
        reg = next((k for k in ("kl", "ot", "hung") if k in h[-1]), None)
        if reg:
            axes[1].plot(steps, _ema(np.array([e.get(reg, 0.0) for e in h])), lw=1.2, label=f"{r['name']} ({reg})")
    axes[0].set_title("Total training loss (EMA)")
    axes[0].set_xlabel("optimizer step")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend()
    axes[1].set_title("Consistency penalty (EMA)")
    axes[1].set_xlabel("optimizer step")
    axes[1].grid(True, ls=":", alpha=0.5)
    if axes[1].lines:
        axes[1].legend()
    fig.tight_layout()
    _save(fig, out_dirs, "loss_curves")
    return True


def _dev_numbers(r) -> Optional[Dict[str, float]]:
    ev = r["evals"].get("dev")
    if not ev:
        return None
    out = {"em": ev["extractor"]["official"]["exact"], "f1": ev["extractor"]["official"]["f1"]}
    if r.get("summary") and r["summary"].get("best_val_f1") is not None:
        out["val_f1"] = r["summary"]["best_val_f1"]
        if r["summary"].get("best_val_agreement_em") is not None and r["summary"].get("best_val_pairs"):
            out["val_agr"] = r["summary"]["best_val_agreement_em"]
    if ev["extractor"].get("consistency"):
        out["cons"] = ev["extractor"]["consistency"]["agreement_em"]
    if ev.get("reranker"):
        out["rr_em"] = ev["reranker"]["official"]["exact"]
        out["rr_f1"] = ev["reranker"]["official"]["f1"]
        if ev["reranker"].get("consistency"):
            out["rr_cons"] = ev["reranker"]["consistency"]["agreement_em"]
    return out


def fig_ablation(runs, out_dirs) -> bool:
    rows = [(r["name"], _dev_numbers(r)) for r in runs.values()]
    rows = [(n, d) for n, d in rows if d]
    if not rows:
        return False
    names = [n for n, _ in rows]
    x = np.arange(len(rows))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(x - w, [d["em"] for _, d in rows], w, label="EM")
    ax.bar(x, [d["f1"] for _, d in rows], w, label="F1")
    ax.bar(x + w, [d.get("cons", 0) for _, d in rows], w, label="paraphrase agreement")
    for xi, (_, d) in zip(x, rows):
        for off, key in ((-w, "em"), (0, "f1"), (w, "cons")):
            if key in d:
                ax.annotate(f"{d[key]:.1f}", (xi + off, d[key]), ha="center", va="bottom", fontsize=6.5, xytext=(0, 1), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("%")
    lo = min(min(d["em"], d.get("cons", d["em"])) for _, d in rows)
    ax.set_ylim(max(0, lo - 15), 100)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(ncol=3, fontsize=7)
    ax.set_title("SQuAD v1.1 dev (official scorer) and paraphrase agreement")
    fig.tight_layout()
    _save(fig, out_dirs, "ablation_dev")
    return True


def fig_robustness(runs, out_dirs) -> bool:
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    drawn = False
    for r in runs.values():
        xs, ys = [], []
        for i, s in enumerate(EVAL_SETS):
            ev = r["evals"].get(s)
            if ev:
                xs.append(i)
                ys.append(ev["extractor"]["official"]["f1"])
        if len(xs) >= 2:
            ax.plot(xs, ys, marker="o", lw=1.2, label=r["name"])
            drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.set_xticks(range(len(EVAL_SETS)))
    ax.set_xticklabels([SET_LABELS_PLAIN[s] for s in EVAL_SETS])
    ax.set_ylabel("F1 (%)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=7)
    ax.set_title("Robustness across evaluation sets (extractor top-1)")
    fig.tight_layout()
    _save(fig, out_dirs, "robustness")
    return True


# ---------------------------------------------------------------------------------------------
def _lambda_runs(runs):
    rows = []
    for r in runs.values():
        cfg, sm = r["config"], r.get("summary")
        if cfg.get("loss_mode") == "kl" and sm and sm.get("best_val_f1") is not None:
            rows.append((float(cfg.get("lambda_kl", 0.0)), r["name"], sm, _dev_numbers(r)))
    return sorted(rows)


def fig_lambda_sweep(runs, out_dirs) -> bool:
    rows = _lambda_runs(runs)
    if len(rows) < 2:
        return False
    lams = [x[0] for x in rows]
    f1 = [x[2]["best_val_f1"] for x in rows]
    agr = [x[2].get("best_val_agreement_em") for x in rows]
    fig, ax1 = plt.subplots(figsize=(3.3, 2.3))
    ax1.plot(lams, f1, marker="o", color="#1b4965", label="val F1")
    ax1.set_xlabel(r"$\lambda$ (consistency weight)")
    ax1.set_ylabel("held-out F1 (%)", color="#1b4965")
    ax1.grid(True, ls=":", alpha=0.5)
    if all(a is not None for a in agr):
        ax2 = ax1.twinx()
        ax2.plot(lams, agr, marker="s", color="#2a9d8f", label="val agreement")
        ax2.set_ylabel("paraphrase agreement (%)", color="#2a9d8f")
    ax1.set_title("Accuracy / consistency trade-off on held-out articles", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dirs, "lambda_sweep")
    return True


def table_lambda(runs, fig_dir, tab_dir) -> bool:
    rows = _lambda_runs(runs)
    if not rows:
        return False
    tex = ["\\begin{table}[t]", "\\centering\\small", "\\begin{tabular}{lcccc}", "\\toprule",
           "$\\lambda$ & \\textbf{Val F1} & \\textbf{Val Agr.} & \\textbf{Dev F1} & \\textbf{Dev Agr.} \\\\", "\\midrule"]
    md = ["| λ | run | Val F1 | Val Agreement | Dev F1 | Dev Agreement |", "| --: | :-- | --: | --: | --: | --: |"]
    for lam, name, sm, d in rows:
        dev_f1 = d["f1"] if d else None
        dev_agr = d.get("cons") if d else None
        tex.append(f"{lam:g} & {_f(sm['best_val_f1'])} & {_f(sm.get('best_val_agreement_em'))} & {_f(dev_f1)} & {_f(dev_agr)} \\\\")
        md.append(f"| {lam:g} | {name} | {_f(sm['best_val_f1'])} | {_f(sm.get('best_val_agreement_em'))} | {_f(dev_f1)} | {_f(dev_agr)} |")
    tex += ["\\bottomrule", "\\end{tabular}", "\\caption{Consistency weight $\\lambda$: F1 and paraphrase agreement on the held-out training articles (used to choose $\\lambda$) and on the dev set.}", "\\label{tab:lambda}", "\\end{table}"]
    _write(os.path.join(tab_dir, "table_lambda.tex"), "\n".join(tex))
    _write(os.path.join(fig_dir, "table_lambda.md"), "\n".join(md))
    return True


def _tex(s: str) -> str:
    return s.replace("_", "\\_")


def _f(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.1f}"


def table_ablation(runs, fig_dir, tab_dir) -> bool:
    rows = []
    for r in runs.values():
        d = _dev_numbers(r)
        if not d:
            continue
        cfg = r["config"]
        rows.append((r["name"], cfg.get("model", "?").split("/")[-1], MODE_LABELS.get(cfg.get("loss_mode"), cfg.get("loss_mode", "?")), d))
    if not rows:
        return False
    tex = ["\\begin{table}[t]", "\\centering\\small", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{llcccccccc}", "\\toprule",
           "\\textbf{Run} & \\textbf{Objective} & \\textbf{Val F1} & \\textbf{Val Agr.} & \\textbf{EM} & \\textbf{F1} & \\textbf{Agr.} & \\textbf{EM$_{rr}$} & \\textbf{F1$_{rr}$} & \\textbf{Agr.$_{rr}$} \\\\", "\\midrule"]
    md = ["| Run | Model | Objective | Val F1 | Val Agreement | EM | F1 | Agreement | EM +reranker | F1 +reranker | Agreement +reranker |", "| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |"]
    for name, model, mode, d in rows:
        tex.append(f"{_tex(name)} ({model}) & {mode} & {_f(d.get('val_f1'))} & {_f(d.get('val_agr'))} & {_f(d['em'])} & {_f(d['f1'])} & {_f(d.get('cons'))} & {_f(d.get('rr_em'))} & {_f(d.get('rr_f1'))} & {_f(d.get('rr_cons'))} \\\\")
        md.append(f"| {name} | {model} | {mode} | {_f(d.get('val_f1'))} | {_f(d.get('val_agr'))} | {_f(d['em'])} | {_f(d['f1'])} | {_f(d.get('cons'))} | {_f(d.get('rr_em'))} | {_f(d.get('rr_f1'))} | {_f(d.get('rr_cons'))} |")
    tex += ["\\bottomrule", "\\end{tabular}}", "\\caption{Val F1 / Val Agr.\\ are on the held-out training articles used for model selection and $\\lambda$ tuning; the remaining columns are SQuAD v1.1 dev results from the official scorer. Agr.\\ is the share of (question, back-translated paraphrase) pairs receiving the same answer; $rr$ columns use the learned span reranker.}", "\\label{tab:ablation}", "\\end{table}"]
    _write(os.path.join(tab_dir, "table_ablation.tex"), "\n".join(tex))
    _write(os.path.join(fig_dir, "table_ablation.md"), "\n".join(md))
    return True


def table_robustness(runs, fig_dir, tab_dir) -> bool:
    rows = []
    for r in runs.values():
        cells = {}
        for s in EVAL_SETS:
            ev = r["evals"].get(s)
            if ev:
                cells[s] = (ev["extractor"]["official"]["f1"], (ev.get("reranker") or {}).get("official", {}).get("f1"))
                if s.startswith("gan_") and ev["extractor"].get("paired_view_official_on_orig_file"):
                    cells[s + "_orig"] = (ev["extractor"]["paired_view_official_on_orig_file"]["f1"], None)
                    cells[s + "_agr"] = (ev["extractor"]["consistency"]["agreement_em"], None)
        if cells:
            rows.append((r["name"], cells))
    if not rows:
        return False
    cols = ["dev", "gan_dev_orig", "gan_dev", "gan_dev_agr", "gan_adv", "addsent", "addonesent"]
    heads = {"dev": "dev", "gan_dev_orig": "G\\&N orig.", "gan_dev": "G\\&N para.", "gan_dev_agr": "G\\&N agr.", "gan_adv": "G\\&N adv.", "addsent": "AddSent", "addonesent": "AddOneSent"}
    tex = ["\\begin{table}[t]", "\\centering\\small", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
           "\\textbf{Run} & " + " & ".join(f"\\textbf{{{heads[c]}}}" for c in cols) + " \\\\", "\\midrule"]
    md = ["| Run | " + " | ".join(heads[c].replace("\\&", "&") for c in cols) + " |", "| :-- |" + " --: |" * len(cols)]
    for name, cells in rows:
        vals = []
        for c in cols:
            v = cells.get(c)
            if v is None:
                vals.append("--")
            elif v[1] is not None:
                vals.append(f"{v[0]:.1f} / {v[1]:.1f}")
            else:
                vals.append(f"{v[0]:.1f}")
        tex.append(f"{_tex(name)} & " + " & ".join(vals) + " \\\\")
        md.append(f"| {name} | " + " | ".join(vals) + " |")
    tex += ["\\bottomrule", "\\end{tabular}}", "\\caption{F1 (\\%) from the official scorer on the dev set, the Gan \\& Ng (2019) paraphrased sets (original questions, paraphrases, and agreement between them) and Adversarial SQuAD. Cells with two numbers are extractor / +reranker.}", "\\label{tab:robustness}", "\\end{table}"]
    _write(os.path.join(tab_dir, "table_robustness.tex"), "\n".join(tex))
    _write(os.path.join(fig_dir, "table_robustness.md"), "\n".join(md))
    return True


def table_question_types(runs, best: Optional[str], fig_dir, tab_dir) -> bool:
    cand = runs.get(best) if best else None
    if cand is None:
        for r in runs.values():
            if r["evals"].get("dev"):
                cand = r
                break
    if not cand or not cand["evals"].get("dev"):
        return False
    ev = cand["evals"]["dev"]
    bt = ev["extractor"]["by_question_type"]
    rr = (ev.get("reranker") or {}).get("by_question_type", {})
    tex = ["\\begin{table}[t]", "\\centering\\small", "\\begin{tabular}{lrccc}", "\\toprule", "\\textbf{Type} & $n$ & \\textbf{EM} & \\textbf{F1} & \\textbf{F1$_{rr}$} \\\\", "\\midrule"]
    md = [f"Question-type breakdown for run `{cand['name']}` (dev):", "", "| Type | n | EM | F1 | F1 +reranker |", "| :-- | --: | --: | --: | --: |"]
    for t, m in sorted(bt.items(), key=lambda kv: -kv[1]["n"]):
        rrf = rr.get(t, {}).get("f1")
        tex.append(f"{t} & {m['n']} & {m['exact']:.1f} & {m['f1']:.1f} & {_f(rrf)} \\\\")
        md.append(f"| {t} | {m['n']} | {m['exact']:.1f} | {m['f1']:.1f} | {_f(rrf)} |")
    tex += ["\\bottomrule", "\\end{tabular}", f"\\caption{{Dev-set results by question type for run {_tex(cand['name'])}.}}", "\\label{tab:qtypes}", "\\end{table}"]
    _write(os.path.join(tab_dir, "table_question_types.tex"), "\n".join(tex))
    _write(os.path.join(fig_dir, "table_question_types.md"), "\n".join(md))
    return True


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--runs", nargs="*", default=None, help="restrict to these run names (overrides --exclude)")
    p.add_argument("--exclude", nargs="*", default=["smoke*"], help="glob patterns of run names to skip (default: smoke*)")
    p.add_argument("--best", default=None, help="run used for the question-type table")
    p.add_argument("--fig_dir", default="figures")
    p.add_argument("--report_dir", default="report")
    args = p.parse_args()
    runs = collect(args.runs_dir, args.runs, args.exclude)
    if not runs:
        print(f"no run artefacts found under {args.runs_dir}/ (train.py / evaluate.py write them); nothing to plot")
        return 1
    out_dirs = [args.fig_dir, os.path.join(args.report_dir, "figures")]
    tab_dir = os.path.join(args.report_dir, "tables")
    made = {
        "loss_curves": fig_loss_curves(runs, out_dirs), "ablation_dev": fig_ablation(runs, out_dirs), "robustness": fig_robustness(runs, out_dirs),
        "table_ablation": table_ablation(runs, args.fig_dir, tab_dir), "table_robustness": table_robustness(runs, args.fig_dir, tab_dir),
        "table_question_types": table_question_types(runs, args.best, args.fig_dir, tab_dir),
        "lambda_sweep": fig_lambda_sweep(runs, out_dirs), "table_lambda": table_lambda(runs, args.fig_dir, tab_dir),
    }
    for k, v in made.items():
        print(f"{'generated' if v else 'skipped  '} {k}")
    print(f"runs considered: {', '.join(runs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
