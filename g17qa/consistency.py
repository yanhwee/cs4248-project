"""Learning-free paraphrase-ensemble baseline and paired consistency / accuracy statistics.

pair-ensemble baseline
    The extractor scores every candidate under the question view and (when a paraphrase is available) under the
    paraphrase view (`Candidate.pair_score`, see decode.py). Without any learning, a candidate is scored by the sum
    of its two per-question log-softmaxes over the candidate list,
        final_i = log p_view(i) + beta * log p_other(i),
    i.e. the same normalisation as the reranker's residual fusion (reranker.select_index). It needs a paraphrase at
    test time, exactly like the learned reranker's `pair_score` features, so it is the fair reference for "does the
    learned reranker add anything beyond averaging the two views?".

pair accuracy
    Agreement alone can be gamed (always answering the same span scores 100 %). `pair_accuracy` therefore reports
    accuracy of each view and the share of pairs where BOTH answers are exact matches.

paired bootstrap
    `bootstrap_delta` gives a percentile confidence interval for the mean difference of two aligned 0/1 (or real)
    arrays by resampling questions (or pairs) with replacement.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .decode import QuestionOutput
from .metrics import exact_match_score
from .reranker import log_softmax_np


def _ensemble_pick(cands, beta: float) -> int:
    if not cands or any(c.pair_score is None for c in cands):
        return 0  # no paraphrase view for this question: keep the extractor top-1
    own = log_softmax_np(np.array([c.score for c in cands], dtype=np.float64))
    other = log_softmax_np(np.array([c.pair_score for c in cands], dtype=np.float64))
    return int(np.argmax(own + beta * other))


def ensemble_predictions(outputs: Dict[str, QuestionOutput], beta: float = 1.0) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(predictions for q, predictions for q') of the learning-free pair-ensemble. beta=0 is the extractor top-1."""
    preds_q: Dict[str, str] = {}
    preds_p: Dict[str, str] = {}
    for qid, o in outputs.items():
        preds_q[qid] = o.candidates[_ensemble_pick(o.candidates, beta)].text if o.candidates else ""
        if o.candidates_p:
            preds_p[qid] = o.candidates_p[_ensemble_pick(o.candidates_p, beta)].text
    return preds_q, preds_p


def pair_accuracy(preds_q: Dict[str, str], preds_p: Dict[str, str], golds: Dict[str, Sequence[str]]) -> Dict[str, float]:
    """Over the questions that have a paired view: EM of each view, and the share of pairs where both / either is an exact match."""
    ids = [i for i in preds_q if i in preds_p]
    if not ids:
        return {"pairs": 0, "em_q": 0.0, "em_paraphrase": 0.0, "both_correct": 0.0, "either_correct": 0.0}
    a = np.array([exact_match_score(preds_q[i], golds[i]) > 0 for i in ids])
    b = np.array([exact_match_score(preds_p[i], golds[i]) > 0 for i in ids])
    return {"pairs": len(ids), "em_q": 100.0 * a.mean(), "em_paraphrase": 100.0 * b.mean(),
            "both_correct": 100.0 * (a & b).mean(), "either_correct": 100.0 * (a | b).mean()}


def bootstrap_delta(base: Sequence[float], variant: Sequence[float], n_boot: int = 2000, seed: int = 0, ci: float = 95.0) -> Optional[Dict[str, float]]:
    """Mean(variant - base) in percent with a paired percentile bootstrap interval. None for empty input."""
    d = np.asarray(variant, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    if d.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(1) * 100.0
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"delta": 100.0 * float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(d.size), "excludes_zero": bool(lo > 0 or hi < 0)}
