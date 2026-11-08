#!/usr/bin/env python3
"""Tests for consistency-trained reranking (view alignment, symmetric KL, training term, agreement). Hermetic.

    python tests/test_consistency_reranker.py        (also collected by pytest)
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g17qa.decode import Candidate, QuestionOutput  # noqa: E402
from g17qa.reranker import (  # noqa: E402
    align_views, build_items, pair_agreement, rerank_logits, symmetric_kl_aligned, train_reranker,
)

HID = 8
CPU = torch.device("cpu")


def _cand(text, score, rank, vec):
    return Candidate(text=text, char_start=0, char_end=len(text), score=score, raw_score=score, null_score=0.0, start_logit=0.0, end_logit=0.0,
                     window_index=0, tok_start=rank, tok_end=rank, rank=rank, pair_score=score, vec=vec.astype(np.float16))


def _outputs(n=120, k=5, seed=0, view_noise=1.5):
    """Each question has k spans a0..a{k-1}; both views see the same spans with independent noise on the representation,
    so an unregularised reranker can rank them differently in the two views."""
    rng = np.random.default_rng(seed)
    outs, golds = {}, {}
    for q in range(n):
        g = int(rng.integers(k))
        base = rng.normal(size=(k, 4 * HID))
        base[g, 0] += 3.0  # the gold span is identifiable from the representation
        views = []
        for _ in range(2):
            scores = np.sort(rng.normal(size=k))[::-1] * 2
            views.append([_cand(f"a{i}", float(scores[i]), i, base[i] + rng.normal(size=4 * HID) * view_noise) for i in range(k)])
        outs[f"q{q}"] = QuestionOutput(qid=f"q{q}", n_windows=1, candidates=views[0], candidates_p=views[1], paired=True)
        golds[f"q{q}"] = [f"a{g}"]
    return outs, golds


def _mean_pair_kl(model, items):
    by_q = {}
    for it, lg in zip(items, rerank_logits(model, items, CPU)):
        by_q.setdefault(it["qid"], {})[it["view"]] = (it, torch.tensor(lg))
    kls = []
    for v in by_q.values():
        al = align_views(v["q"][0], v["p"][0])
        if al is not None:
            kls.append(float(symmetric_kl_aligned(v["q"][1][al[0]], v["p"][1][al[1]])))
    return float(np.mean(kls))


def test_align_views():
    outs, golds = _outputs(n=2, k=4)
    o = outs["q0"]
    items = build_items({"q0": o}, golds)
    iq, ip = align_views(items[0], items[1])
    assert items[0]["view"] == "q" and items[1]["view"] == "p"
    assert [items[0]["keys"][i] for i in iq] == [items[1]["keys"][i] for i in ip] and len(iq) == 4
    # views listing the same spans in a different order are matched by text, not position
    o2 = QuestionOutput(qid="q0", n_windows=1, candidates=o.candidates, candidates_p=list(reversed(o.candidates_p)), paired=True)
    it2 = build_items({"q0": o2}, golds)
    iq2, ip2 = align_views(it2[0], it2[1])
    assert [it2[0]["keys"][i] for i in iq2] == [it2[1]["keys"][i] for i in ip2]
    # fewer than two shared candidates -> None
    disjoint = QuestionOutput(qid="q0", n_windows=1, candidates=o.candidates[:2], candidates_p=[_cand("zz", 1.0, 0, np.zeros(4 * HID)), _cand("a0", 0.0, 1, np.zeros(4 * HID))], paired=True)
    it3 = build_items({"q0": disjoint}, golds)
    assert align_views(it3[0], it3[1]) is None


def test_symmetric_kl_properties():
    a = torch.tensor([2.0, 0.5, -1.0], requires_grad=True)
    b = torch.tensor([0.1, 1.5, 0.0], requires_grad=True)
    assert abs(float(symmetric_kl_aligned(a, a.detach() + 5.0).detach())) < 1e-6  # identical up to a shift -> 0
    k = symmetric_kl_aligned(a, b)
    assert float(k.detach()) > 0 and abs(float(k.detach()) - float(symmetric_kl_aligned(b, a).detach())) < 1e-6
    k.backward()
    assert a.grad is not None and b.grad is not None and float(a.grad.abs().sum()) > 0
    # matches the closed form
    pa, pb = torch.softmax(a.detach(), 0), torch.softmax(b.detach(), 0)
    ref = 0.5 * ((pa * (pa / pb).log()).sum() + (pb * (pb / pa).log()).sum())
    assert abs(float(k.detach()) - float(ref)) < 1e-5


def test_pair_agreement_extractor():
    outs, golds = _outputs(n=3, k=3)
    # force the extractor top-1 to be the same span in both views for q0 and q1, different for q2
    for q, other in (("q0", "a0"), ("q1", "a0"), ("q2", "a1")):
        o = outs[q]
        o.candidates[0].text, o.candidates_p[0].text = "a0", other
        for i, c in enumerate(o.candidates[1:], 1):
            c.text = f"b{i}"
        for i, c in enumerate(o.candidates_p[1:], 1):
            c.text = f"c{i}"
    items = build_items(outs, golds)
    r = pair_agreement(None, items, CPU)
    assert r["pairs"] == 3 and abs(r["agreement_em"] - 100 * 2 / 3) < 1e-9
    assert pair_agreement(None, [it for it in items if it["view"] == "q"], CPU)["pairs"] == 0


def test_consistency_term_reduces_pair_disagreement_and_logs_metrics():
    outs, golds = _outputs()
    items = build_items(outs, golds)
    tr, dev = items[:200], items[200:]
    dev_q = [it for it in dev if it["view"] == "q"]
    torch.manual_seed(0)
    base, _ = train_reranker(tr, dev_q, golds, HID, CPU, epochs=12, use_pair_score=False, log=lambda *a: None)
    torch.manual_seed(0)
    cons, hist = train_reranker(tr, dev_q, golds, HID, CPU, epochs=12, use_pair_score=False, consistency_weight=50.0, dev_pair_items=dev, log=lambda *a: None)
    kl_base, kl_cons = _mean_pair_kl(base, tr), _mean_pair_kl(cons, tr)
    assert kl_cons < 0.5 * kl_base, (kl_base, kl_cons)  # observed: orders of magnitude lower; the factor leaves wide margin
    assert all("consistency_kl" in h and "dev_agreement_em" in h for h in hist)
    assert all(np.isfinite(h["consistency_kl"]) and h["consistency_kl"] >= 0 for h in hist)  # the logged term is a valid KL every epoch


def test_default_training_is_unchanged_by_the_new_option():
    outs, golds = _outputs(n=60)
    items = build_items(outs, golds)
    tr, dev = items[:80], [it for it in items[80:] if it["view"] == "q"]
    torch.manual_seed(1)
    a, _ = train_reranker(tr, dev, golds, HID, CPU, epochs=3, log=lambda *x: None)
    torch.manual_seed(1)
    b, _ = train_reranker(tr, dev, golds, HID, CPU, epochs=3, consistency_weight=0.0, log=lambda *x: None)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


def test_select_by_em_plus_agreement_picks_that_epoch():
    outs, golds = _outputs(n=80, seed=2)
    items = build_items(outs, golds)
    tr, dev = items[:100], items[100:]
    dev_q = [it for it in dev if it["view"] == "q"]
    from g17qa.reranker import evaluate_items
    torch.manual_seed(0)
    m, hist = train_reranker(tr, dev_q, golds, HID, CPU, epochs=8, use_pair_score=False, dev_pair_items=dev, select_by="em+agreement", log=lambda *a: None)
    best = max(hist, key=lambda h: (h["dev_exact"] + h["dev_agreement_em"], h["dev_exact"]))
    got = evaluate_items(m, dev_q, golds, CPU)["exact"]
    assert abs(got - best["dev_exact"]) < 1e-6 and abs(pair_agreement(m, dev, CPU)["agreement_em"] - best["dev_agreement_em"]) < 1e-6
    for bad in (dict(select_by="nope"), dict(select_by="em+agreement")):
        try:
            train_reranker(tr, dev_q, golds, HID, CPU, epochs=1, log=lambda *a: None, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_consistency_requires_paired_items():
    outs, golds = _outputs(n=30)
    q_only = build_items(outs, golds, use_pair_view=False)
    try:
        train_reranker(q_only, q_only, golds, HID, CPU, epochs=1, consistency_weight=1.0, log=lambda *a: None)
    except ValueError as e:
        assert "both views" in str(e)
    else:
        raise AssertionError("expected ValueError when no question has both views")


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
