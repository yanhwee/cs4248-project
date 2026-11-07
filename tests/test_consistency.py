#!/usr/bin/env python3
"""Tests for the paraphrase-ensemble baseline, pair accuracy and bootstrap CIs (hermetic, no model download).

    python tests/test_consistency.py        (also collected by pytest)
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g17qa.consistency import bootstrap_delta, ensemble_predictions, pair_accuracy  # noqa: E402
from g17qa.decode import Candidate, QuestionOutput  # noqa: E402


def _c(text, score, pair, rank):
    return Candidate(text=text, char_start=0, char_end=len(text), score=score, raw_score=score, null_score=0.0, start_logit=0.0, end_logit=0.0,
                     window_index=0, tok_start=rank, tok_end=rank, rank=rank, pair_score=pair)


def _out(own, other, own_p=None, other_p=None):
    """candidates a0..; `own` = extractor scores in this view (descending), `other` = the paraphrase view's scores."""
    cands = [_c(f"a{i}", s, o, i) for i, (s, o) in enumerate(zip(own, other))]
    cands_p = [_c(f"a{i}", s, o, i) for i, (s, o) in enumerate(zip(own_p if own_p is not None else other, other_p if other_p is not None else own))]
    cands_p.sort(key=lambda c: c.score, reverse=True)
    return QuestionOutput(qid="q", n_windows=1, candidates=cands, candidates_p=cands_p, paired=True)


def test_beta_zero_is_extractor_top1():
    rng = np.random.default_rng(0)
    outs = {}
    for i in range(50):
        own = np.sort(rng.normal(size=6) * 3)[::-1]
        outs[f"q{i}"] = _out(own, rng.normal(size=6) * 3)
    pq, pp = ensemble_predictions(outs, beta=0.0)
    assert all(pq[q] == o.candidates[0].text for q, o in outs.items())
    assert all(pp[q] == o.candidates_p[0].text for q, o in outs.items())


def test_ensemble_switch_point_is_deterministic():
    # own view: a0 leads a1 by 2 nats; the paraphrase view prefers a1 over a0 by 4 nats.
    # log-odds(a1 vs a0) = -2 + beta * 4  -> a1 wins iff beta > 0.5
    o = _out([2.0, 0.0, -5.0], [0.0, 4.0, 0.0])
    outs = {"q": o}
    assert ensemble_predictions(outs, 0.25)[0]["q"] == "a0"
    assert ensemble_predictions(outs, 0.49)[0]["q"] == "a0"
    assert ensemble_predictions(outs, 0.51)[0]["q"] == "a1"
    assert ensemble_predictions(outs, 1.0)[0]["q"] == "a1"


def test_ensemble_agrees_across_views_when_evidence_is_shared():
    # both views hold the same two score vectors (swapped roles), so the fused score of a span is identical in each view
    own, other = [3.0, 1.0, 0.0], [0.5, 2.0, -1.0]  # fused scores are distinct (no ties)
    cands = [_c(f"a{i}", own[i], other[i], i) for i in range(3)]
    cands_p = [_c(f"a{i}", other[i], own[i], i) for i in range(3)]
    cands_p.sort(key=lambda c: c.score, reverse=True)
    o = QuestionOutput(qid="q", n_windows=1, candidates=cands, candidates_p=cands_p, paired=True)
    pq, pp = ensemble_predictions({"q": o}, 1.0)
    assert pq["q"] == pp["q"]


def test_missing_pair_score_keeps_extractor_top1():
    cands = [_c("a0", 2.0, None, 0), _c("a1", 1.0, None, 1)]
    o = QuestionOutput(qid="q", n_windows=1, candidates=cands, candidates_p=[], paired=False)
    pq, pp = ensemble_predictions({"q": o}, 1.0)
    assert pq["q"] == "a0" and "q" not in pp


def test_pair_accuracy():
    golds = {"1": ["x"], "2": ["x"], "3": ["x"], "4": ["x"], "5": ["x"]}
    pq = {"1": "x", "2": "x", "3": "y", "4": "y", "5": "x"}
    pp = {"1": "x", "2": "y", "3": "x", "4": "y"}  # "5" has no paired view
    r = pair_accuracy(pq, pp, golds)
    assert r["pairs"] == 4 and r["em_q"] == 50.0 and r["em_paraphrase"] == 50.0 and r["both_correct"] == 25.0 and r["either_correct"] == 75.0
    assert pair_accuracy({}, {}, {})["pairs"] == 0


def test_bootstrap_delta():
    base = np.zeros(200)
    same = bootstrap_delta(base, base)
    assert same["delta"] == 0.0 and same["ci_low"] == 0.0 and same["ci_high"] == 0.0 and not same["excludes_zero"]
    better = np.ones(200)
    b = bootstrap_delta(base, better)
    assert b["delta"] == 100.0 and b["excludes_zero"]
    rng = np.random.default_rng(3)
    x = (rng.random(2000) < 0.5).astype(float)
    y = x.copy()
    y[:40] = 1 - y[:40]  # a mix of fixes and breaks with a small known net change
    r = bootstrap_delta(x, y, n_boot=500, seed=1)
    assert r["ci_low"] <= r["delta"] <= r["ci_high"] and r["n"] == 2000
    assert bootstrap_delta([], []) is None
    assert bootstrap_delta(x, y, 300, seed=7) == bootstrap_delta(x, y, 300, seed=7)  # reproducible


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
