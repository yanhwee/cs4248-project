#!/usr/bin/env python3
"""Tables for report/main.tex, generated from docs/results_digest.json (itself written by make_results_report.py
from runs/). Nothing in the paper's tables is typed by hand.

  python scripts/make_results_report.py        # refresh the digest first
  python scripts/make_report_tables.py         # writes report/tables/rep_*.tex

Conventions follow docs/RESULTS.md: metrics are rounded to one decimal first and every difference is a difference
of rounded values; bootstrap intervals are printed to two decimals.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIGEST = os.path.join(ROOT, "docs", "results_digest.json")
OUT = os.path.join(ROOT, "report", "tables")

SYSTEMS = [  # (run, backbone label, objective label, is_proposed)
    ("bert_ce", "BERT-base", "CE", False),
    ("bert_aug", "BERT-base", "CE + aug.", False),
    ("bert_kl_lam1.0", "BERT-base", "CE + aug. + KL", False),
    ("deb_ce", "DeBERTa-v3-base", "CE", False),
    ("deb_kl", "DeBERTa-v3-base", "CE + aug. + KL", False),
    ("deb_large_ce", "DeBERTa-v3-large", "CE", False),
    ("deb_large_kl", "DeBERTa-v3-large", "CE + aug. + KL", True),
]


def f(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def delta(a, b):
    """Difference of one-decimal-rounded values, printed with sign."""
    if a is None or b is None:
        return "--"
    d = round(round(a, 1) - round(b, 1), 1)
    return f"{d:+.1f}".replace("-0.0", "+0.0")


def ci(v):
    return "--" if not v else f"[{v[0]:+.2f}, {v[1]:+.2f}]"


def bold(s, on):
    return f"\\textbf{{{s}}}" if on else s


def table_main(runs):
    rows = []
    for run, bb, obj, prop in SYSTEMS:
        s = runs[run]["sets"]
        dev, gd, ga, ads, aos = s["dev"], s["gan_dev"], s["gan_adv"], s["addsent"], s["addonesent"]
        cells = [bb, obj, f(dev["em"]), f(dev["f1"]), f(dev["agr"]), f(ads["f1"]), f(ads["agr"]), f(aos["f1"]),
                 f(gd["orig_f1"]), f(gd["f1"]), f(gd["agr"]), f(ga["f1"])]
        rows.append(" & ".join(bold(c, prop) for c in cells) + " \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[t]
\centering\small
\setlength{{\tabcolsep}}{{3.2pt}}
\begin{{tabular}}{{llccc@{{\hspace{{8pt}}}}cc@{{\hspace{{8pt}}}}c@{{\hspace{{8pt}}}}ccc@{{\hspace{{8pt}}}}c}}
\toprule
& & \multicolumn{{3}}{{c}}{{\textbf{{SQuAD dev}}}} & \multicolumn{{2}}{{c}}{{\textbf{{AddSent}}}} & \textbf{{AddOneSent}} & \multicolumn{{3}}{{c}}{{\textbf{{G\&N paraphrases}}}} & \textbf{{G\&N adv.}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-7}} \cmidrule(lr){{8-8}} \cmidrule(lr){{9-11}} \cmidrule(lr){{12-12}}
\textbf{{Backbone}} & \textbf{{Objective}} & EM & F1 & Agr. & F1 & Agr. & F1 & F1$_{{\mathrm{{orig}}}}$ & F1$_{{\mathrm{{para}}}}$ & Agr. & F1 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Extractor alone, official scorer. \emph{{Agr.}}: share of (question, paraphrase) pairs with the same normalised answer; back-translated paraphrases on dev ({runs['deb_large_kl']['sets']['dev']['pairs']:,} pairs) and AddSent ({runs['deb_large_kl']['sets']['addsent']['pairs']:,}), human paraphrase vs.\ original question on the \citet{{gan2019improving}} sets (1,061 pairs; 56 adversarial questions). Bold: proposed system.}}
\label{{tab:main}}
\end{{table*}}
"""


def sig(d, interval):
    """Delta with a star when the 95% bootstrap interval excludes zero."""
    star = interval and (interval[0] > 0 or interval[1] < 0)
    return d + (r"$^{*}$" if star else "")


def table_second(runs):
    r = runs["deb_large_kl"]
    V = r["variants"]
    base = V["CB_pair_argmax_ensemble"]["sets"]
    rows = [[r"Extractor alone", "--", "--", "--", f(base["dev"]["both"]), "--", "--"]]
    specs = [
        (r"Reranker (pair features)", "CB_pair_argmax_ensemble", True),
        (r"Reranker (no pair features)", "CB_nopair_argmax", False),
        (r"Reranker (no pair), blended $\alpha{=}1$", "CB_nopair_res1.0", False),
        (r"Reranker (no pair), cons.-trained $w{=}1$", "CR_cw1.0_argmax", False),
        (r"Paraphrase ensemble, training-free", "ENSEMBLE", True),
    ]
    for label, key, needs_para in specs:
        s = V[key]["sets"]
        dv, ads, aos = s["dev"], s["addsent"], s["addonesent"]
        rows.append([label + (r"$^\dagger$" if needs_para else ""),
                     sig(delta(dv["rr_em"], dv["em"]), dv.get("ci_em")),
                     delta(dv["rr_f1"], dv["f1"]),
                     sig(delta(dv["rr_agr"], dv["agr"]), dv.get("ci_agr")),
                     f(dv["rr_both"]),
                     delta(ads["rr_f1"], ads["f1"]),
                     delta(aos["rr_f1"], aos["f1"])])
    body = "\n".join(" & ".join(c) + " \\\\" for c in rows)
    return rf"""\begin{{table*}}[t]
\centering\small
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{lcccc@{{\hspace{{14pt}}}}cc}}
\toprule
& \multicolumn{{4}}{{c}}{{\textbf{{SQuAD dev}}}} & \textbf{{AddSent}} & \textbf{{AddOneSent}} \\
\cmidrule(lr){{2-5}} \cmidrule(lr){{6-7}}
\textbf{{Second stage on the proposed extractor}} & $\Delta$EM & $\Delta$F1 & $\Delta$Agreement & Both correct & $\Delta$F1 & $\Delta$F1 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Second-stage selection among the top-20 spans of the proposed extractor. $\Delta$ = second stage $-$ extractor; $^{{*}}$: 95\% paired-bootstrap interval excludes zero (dev EM/agreement only; no F1 intervals). \emph{{Both correct}}: pairs where both answers are exact matches. $^\dagger$needs a paraphrase at test time.}}
\label{{tab:second}}
\end{{table*}}
"""


def main() -> None:
    d = json.load(open(DIGEST, encoding="utf-8"))
    runs = d["runs"]
    os.makedirs(OUT, exist_ok=True)
    for name, tex in [("rep_main", table_main(runs)), ("rep_second", table_second(runs))]:
        with open(os.path.join(OUT, f"{name}.tex"), "w", encoding="utf-8") as fh:
            fh.write(tex)
        print(f"wrote report/tables/{name}.tex")


if __name__ == "__main__":
    main()
