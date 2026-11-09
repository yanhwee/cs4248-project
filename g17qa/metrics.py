"""Answer metrics.

`normalize_answer`, `compute_exact`, `compute_f1` are line-for-line ports of the course-provided
`evaluate-v2.0.py` so that local numbers agree with the official scorer. `metric_max_over_ground_truths`
reproduces the scorer's max over all gold answers. `run_official_scorer` shells out to the real script so the
reported numbers are literally its output.
"""

from __future__ import annotations

import collections
import json
import os
import re
import string
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

from .utils import PROJECT_ROOT, ensure_dir

OFFICIAL_SCRIPT = os.path.join(PROJECT_ROOT, "provided", "evaluate-v2.0.py")


# ---------------------------------------------------------------------------------------------
# Ports of evaluate-v2.0.py
# ---------------------------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: Sequence[str]) -> float:
    golds = [g for g in ground_truths if normalize_answer(g)] or [""]
    return max(metric_fn(g, prediction) for g in golds)


def exact_match_score(prediction: str, ground_truths: Sequence[str]) -> float:
    return float(metric_max_over_ground_truths(compute_exact, prediction, ground_truths))


def f1_score(prediction: str, ground_truths: Sequence[str]) -> float:
    return float(metric_max_over_ground_truths(compute_f1, prediction, ground_truths))


def aggregate_em_f1(preds: Dict[str, str], golds: Dict[str, Sequence[str]]) -> Dict[str, float]:
    """Mean EM / F1 (in %) over the question ids present in `golds`; missing predictions count as ''."""
    ems, f1s = [], []
    for qid, gts in golds.items():
        p = preds.get(qid, "")
        ems.append(exact_match_score(p, gts))
        f1s.append(f1_score(p, gts))
    n = max(1, len(ems))
    return {"exact": 100.0 * sum(ems) / n, "f1": 100.0 * sum(f1s) / n, "total": len(ems)}


# ---------------------------------------------------------------------------------------------
# Paraphrase consistency
# ---------------------------------------------------------------------------------------------
def consistency_metrics(preds_q: Dict[str, str], preds_qp: Dict[str, str]) -> Dict[str, float]:
    """Share of paraphrase pairs that receive the same normalised answer (EM agreement) and mean F1 agreement."""
    ids = [i for i in preds_q if i in preds_qp]
    if not ids:
        return {"pairs": 0, "agreement_em": 0.0, "agreement_f1": 0.0}
    em = [compute_exact(preds_q[i], preds_qp[i]) for i in ids]
    f1 = [compute_f1(preds_q[i], preds_qp[i]) for i in ids]
    return {
        "pairs": len(ids),
        "agreement_em": 100.0 * sum(em) / len(ids),
        "agreement_f1": 100.0 * sum(f1) / len(ids),
    }


def summarize_outputs(outputs, golds: Dict[str, Sequence[str]]) -> Dict[str, float]:
    """EM / F1 (max over gold answers, identical to the official scorer) of the top-1 predictions in a
    {qid: QuestionOutput} dict, plus paraphrase agreement over the questions that carried a paired view."""
    preds = {q: o.pred for q, o in outputs.items()}
    out = aggregate_em_f1(preds, {q: golds[q] for q in preds if q in golds})
    preds_p = {q: o.pred_p for q, o in outputs.items() if o.paired}
    cons = consistency_metrics({q: preds[q] for q in preds_p}, preds_p)
    out.update({"pairs": cons["pairs"], "agreement_em": cons["agreement_em"], "agreement_f1": cons["agreement_f1"]})
    return out


# ---------------------------------------------------------------------------------------------
# Question-type buckets for error analysis
# ---------------------------------------------------------------------------------------------
_QTYPE_START = [
    ("how many/much", re.compile(r"^(about |approximately |roughly )?how (many|much|long|old|far|often|tall|big|large)\b")),
    ("how", re.compile(r"^how\b")),
    ("who", re.compile(r"^(who|whom|whose)\b")),
    ("what", re.compile(r"^(what|whats)\b")),
    ("when", re.compile(r"^when\b")),
    ("where", re.compile(r"^where\b")),
    ("why", re.compile(r"^why\b")),
    ("which", re.compile(r"^which\b")),
]
# wh-word appearing later, e.g. "In what year ...", "By whom ...", "During which era ..."
_QTYPE_ANYWHERE = [
    ("how many/much", re.compile(r"\bhow (many|much)\b")),
    ("who", re.compile(r"\b(who|whom|whose)\b")),
    ("when", re.compile(r"\b(when|what year|which year|what date|what time)\b")),
    ("where", re.compile(r"\b(where|what place|which place|what city|which city|what country|which country)\b")),
    ("why", re.compile(r"\bwhy\b")),
    ("which", re.compile(r"\bwhich\b")),
    ("what", re.compile(r"\bwhat\b")),
    ("how", re.compile(r"\bhow\b")),
]


def question_type(question: str) -> str:
    """Coarse question-type bucket used for error analysis."""
    q = question.strip().lower()
    for name, pat in _QTYPE_START:
        if pat.search(q):
            return name
    for name, pat in _QTYPE_ANYWHERE:
        if pat.search(q):
            return name
    return "other"


def per_bucket_metrics(
    preds: Dict[str, str], golds: Dict[str, Sequence[str]], buckets: Dict[str, str]
) -> Dict[str, Dict[str, float]]:
    """EM / F1 per bucket label (e.g. question type)."""
    acc: Dict[str, List[List[float]]] = collections.defaultdict(lambda: [[], []])
    for qid, gts in golds.items():
        b = buckets.get(qid, "other")
        p = preds.get(qid, "")
        acc[b][0].append(exact_match_score(p, gts))
        acc[b][1].append(f1_score(p, gts))
    out = {}
    for b, (ems, f1s) in sorted(acc.items()):
        n = len(ems)
        out[b] = {"n": n, "exact": 100.0 * sum(ems) / n, "f1": 100.0 * sum(f1s) / n}
    return out


# ---------------------------------------------------------------------------------------------
# Official scorer
# ---------------------------------------------------------------------------------------------
def run_official_scorer(data_file: str, pred_file: str, out_file: str, script: str = OFFICIAL_SCRIPT) -> Dict[str, float]:
    """Run the course-provided evaluate-v2.0.py and return its JSON output."""
    if not os.path.exists(script):
        raise FileNotFoundError(f"official scorer not found at {script}")
    ensure_dir(os.path.dirname(os.path.abspath(out_file)))
    cmd = [sys.executable, "-I", script, data_file, pred_file, "-o", out_file]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"official scorer failed:\n{proc.stdout}\n{proc.stderr}")
    with open(out_file, "r", encoding="utf-8") as f:
        return json.load(f)
