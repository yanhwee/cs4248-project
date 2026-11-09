#!/usr/bin/env python3
"""Reranker ablation / fusion tests. Hermetic: synthetic candidates, no model download, runs in seconds.

    python tests/test_reranker.py        (also collected by pytest)
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g17qa.decode import Candidate, QuestionOutput  # noqa: E402
from g17qa.reranker import (  # noqa: E402
    PAIR_FEATURES, SCALAR_FEATURES, SpanReranker, analyze_changes, apply_reranker, build_items, candidate_scalars, change_counts, choose, classify_change, evaluate_items,
    load_reranker, recall_at_ks, save_reranker, scalar_feature_indices, select_index, train_reranker,
)

HID = 8
CPU = torch.device("cpu")


def _cand(text, score, rank, pair=None, rng=None):
    vec = (rng or np.random.default_rng(0)).normal(size=4 * HID).astype(np.float16)
    return Candidate(text=text, char_start=0, char_end=len(text), score=score, raw_score=score, null_score=0.0, start_logit=score / 2, end_logit=score / 2,
                     window_index=0, tok_start=rank, tok_end=rank + 1, rank=rank, pair_score=pair, vec=vec)


def _outputs(n=40, k=5, seed=0, with_pair=True):
    """n questions, k candidates 'a0'..'a{k-1}' in extractor order; the gold is 'a<g>' with g usually 0."""
    rng = np.random.default_rng(seed)
    outs, golds = {}, {}
    for q in range(n):
        g = 0 if rng.random() < 0.6 else int(rng.integers(1, k))
        scores = np.sort(rng.normal(size=k))[::-1] * 2
        cands = [_cand(f"a{i}", float(scores[i]), i, pair=float(rng.normal()) if with_pair else None, rng=rng) for i in range(k)]
        outs[f"q{q}"] = QuestionOutput(qid=f"q{q}", n_windows=1, candidates=cands, candidates_p=cands if with_pair else [], paired=with_pair)
        golds[f"q{q}"] = [f"a{g}"]
    return outs, golds


def _train(use_pair_score, **kw):
    outs, golds = _outputs()
    items = build_items(outs, golds, use_pair_view=False)
    m, hist = train_reranker(items[:30], items[30:], golds, HID, CPU, epochs=2, use_pair_score=use_pair_score, log=lambda *a: None, **kw)
    return m, items, golds, hist


def test_feature_dimensions_pair_enabled_and_disabled():
    n_full = len(SCALAR_FEATURES)
    assert scalar_feature_indices(True) == list(range(n_full))
    idx = scalar_feature_indices(False)
    assert len(idx) == n_full - len(PAIR_FEATURES) and not {SCALAR_FEATURES[i] for i in idx} & set(PAIR_FEATURES)
    on, items, golds, _ = _train(True)
    off, _, _, _ = _train(False)
    assert on.n_scalars == n_full and on.scalar_idx is None and on.use_pair_score
    assert off.n_scalars == n_full - 3 and off.sc_proj.in_features == n_full - 3 and not off.use_pair_score
    assert on.sc_proj.in_features == n_full
    assert not set(off.feature_names) & set(PAIR_FEATURES) and off.scalar_mean.shape == (n_full - 3,)
    # both variants accept the same full-width scalar tensor and agree on the output shape
    vec = torch.randn(2, 5, 4 * HID)
    sc = torch.randn(2, 5, n_full)
    assert on(vec, sc).shape == off(vec, sc).shape == (2, 5)


def test_pair_score_has_no_influence_when_disabled():
    on, _, _, _ = _train(True)
    off, _, _, _ = _train(False)
    on.eval(), off.eval()
    vec = torch.randn(3, 5, 4 * HID)
    sc = torch.randn(3, 5, len(SCALAR_FEATURES))
    sc2 = sc.clone()
    for n in PAIR_FEATURES:
        sc2[..., SCALAR_FEATURES.index(n)] += 5.0
    with torch.no_grad():
        assert torch.allclose(off(vec, sc), off(vec, sc2)), "the no-pair model must ignore the paraphrase-derived columns"
        assert not torch.allclose(on(vec, sc), on(vec, sc2)), "the default model must still use them"


def test_checkpoint_roundtrip_and_old_checkpoint_compat():
    for use_pair in (True, False):
        m, items, _, _ = _train(use_pair)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.pt")
            save_reranker(m, p, {"tag": "_x"})
            m2 = load_reranker(p, CPU)
            assert m2.use_pair_score == use_pair and m2.feature_names == m.feature_names and m2.meta["tag"] == "_x"
            assert choose(m, items, CPU) == choose(m2, items, CPU)
            if use_pair:  # a checkpoint written before the option existed has no scalar_idx / use_pair_score keys
                ck = torch.load(p, weights_only=False)
                for k in ("scalar_idx", "scalar_features_used", "use_pair_score"):
                    ck.pop(k)
                torch.save(ck, p)
                m3 = load_reranker(p, CPU)
                assert m3.use_pair_score and m3.n_scalars == len(SCALAR_FEATURES)
                assert choose(m, items, CPU) == choose(m3, items, CPU)


def test_residual_fusion_ordering_is_deterministic():
    # extractor log-odds of candidate 1 vs candidate 0: ext = [2, 0, -5] -> log p(1) - log p(0) = -2
    # reranker logits [0, 4, 0]  -> log p_rr(1) - log p_rr(0) = +4. Candidate 1 wins iff alpha * 4 > 2, i.e. alpha > 0.5
    ext = np.array([2.0, 0.0, -5.0])
    rr = np.array([0.0, 4.0, 0.0])
    assert select_index(ext, rr, "extractor")[0] == 0
    assert select_index(ext, rr, "reranker")[0] == 1
    assert select_index(ext, rr, "residual", alpha=0.25)[0] == 0
    assert select_index(ext, rr, "residual", alpha=0.49)[0] == 0
    assert select_index(ext, rr, "residual", alpha=0.51)[0] == 1
    assert select_index(ext, rr, "residual", alpha=1.0)[0] == 1
    # the normalisation is a per-question shift: adding a constant to either score set changes nothing
    assert select_index(ext + 100.0, rr - 7.0, "residual", alpha=0.51)[0] == 1
    # fused scores are log p_ext + alpha * log p_rr
    _, info = select_index(ext, rr, "residual", alpha=0.5)
    lp_e = ext - np.log(np.exp(ext).sum())
    lp_r = rr - np.log(np.exp(rr).sum())
    assert np.allclose(info["final"], lp_e + 0.5 * lp_r)


def test_alpha_zero_reproduces_extractor_ranking():
    rng = np.random.default_rng(1)
    for _ in range(200):
        k = int(rng.integers(2, 21))
        ext = np.sort(rng.normal(size=k) * 3)[::-1]
        rr = rng.normal(size=k) * 10
        assert select_index(ext, rr, "residual", alpha=0.0)[0] == 0 == select_index(ext, rr, "extractor")[0]
    # and through the model-level API
    m, items, golds, _ = _train(False)
    assert choose(m, items, CPU, mode="residual", alpha=0.0) == [0] * len(items)
    base = evaluate_items(None, items, golds, CPU)
    res = evaluate_items(m, items, golds, CPU, mode="residual", alpha=0.0)
    assert abs(base["exact"] - res["exact"]) < 1e-9 and res["fix"] == 0 and res["break"] == 0


def test_confidence_gate_and_margin():
    ext = np.array([2.0, 0.0, -5.0])
    rr = np.array([0.0, 4.0, 0.0])  # reranker prefers candidate 1 over the extractor top-1 by ~4 nats
    assert select_index(ext, rr, "reranker", override_margin=None)[0] == 1
    assert select_index(ext, rr, "reranker", override_margin=0.5)[0] == 1
    k, info = select_index(ext, rr, "reranker", override_margin=5.0)
    assert k == 0 and info["gated"]
    assert select_index(ext, rr, "residual", alpha=1.0, override_margin=3.9)[0] == 1
    assert select_index(ext, rr, "residual", alpha=1.0, override_margin=4.1)[0] == 0
    # the extractor top-1 is never gated away, and the gate is inert when the reranker agrees with the extractor
    assert select_index(ext, np.array([3.0, 0.0, 0.0]), "reranker", override_margin=100.0)[0] == 0
    # original `margin` option: candidates further than `margin` below the extractor top-1 are excluded
    assert select_index(ext, np.array([0.0, 0.0, 9.0]), "reranker", margin=None)[0] == 2
    assert select_index(ext, np.array([0.0, 0.0, 9.0]), "reranker", margin=3.0)[0] == 0


def test_fix_break_same_classification():
    assert classify_change(False, True) == "FIX"
    assert classify_change(True, False) == "BREAK"
    assert classify_change(True, True) == "SAME_CORRECT"
    assert classify_change(False, False) == "SAME_WRONG"
    labels = ["FIX", "FIX", "BREAK", "SAME_CORRECT", "SAME_CORRECT", "SAME_CORRECT", "SAME_WRONG", "SAME_WRONG"]
    c = change_counts(labels)
    assert (c["n"], c["FIX"], c["BREAK"], c["SAME_CORRECT"], c["SAME_WRONG"]) == (8, 2, 1, 3, 2)
    assert abs(c["FIX_pct"] - 25.0) < 1e-9 and abs(c["BREAK_pct"] - 12.5) < 1e-9 and c["net_em_change"] == 1
    assert change_counts([])["n"] == 0
    # evaluate_items reports the same counts as a direct comparison with the extractor top-1
    m, items, golds, _ = _train(True)
    idx = choose(m, items, CPU)
    fix = sum(1 for it, k in zip(items, idx) if not it["em"][0] and it["em"][k])
    brk = sum(1 for it, k in zip(items, idx) if it["em"][0] and not it["em"][k])
    r = evaluate_items(m, items, golds, CPU)
    assert (r["fix"], r["break"]) == (fix, brk)
    assert abs(r["exact"] - (evaluate_items(None, items, golds, CPU)["exact"] + 100.0 * (fix - brk) / len(items))) < 1e-9


def test_apply_reranker_and_change_dump():
    outs, golds = _outputs()
    m, _, _, _ = _train(True)
    ext = {q: o.pred for q, o in outs.items()}
    rp, rpp, det = apply_reranker(m, outs, CPU, return_details=True)
    assert set(rp) == set(outs) and set(rpp) == set(outs) and set(det) == set(outs)
    assert apply_reranker(m, outs, CPU)[0] == rp  # default call unchanged
    # extractor mode returns the extractor predictions; residual alpha=0 does too
    assert apply_reranker(m, outs, CPU, mode="extractor")[0] == ext
    assert apply_reranker(m, outs, CPU, mode="residual", alpha=0.0)[0] == ext
    counts, rows = analyze_changes(outs, ext, rp, golds, {q: f"question {q}" for q in outs}, det, max_examples=3)
    assert counts["n"] == len(outs) and counts["FIX"] + counts["BREAK"] + counts["SAME_CORRECT"] + counts["SAME_WRONG"] == len(outs)
    assert sum(1 for r in rows if r["type"] == "FIX") <= 3 and sum(1 for r in rows if r["type"] == "BREAK") <= 3
    for r in rows:
        assert len(r["candidates"]) == 5 and r["extractor_top1"] != r["reranker_pred"]
        assert {"extractor_logp", "reranker_logit", "reranker_logp", "final_score", "pair_score", "score"} <= set(r["candidates"][0])
        gold = golds[r["id"]]
        if r["type"] == "FIX":
            assert r["reranker_pred"] in gold and r["extractor_top1"] not in gold
        else:
            assert r["type"] == "BREAK" and r["extractor_top1"] in gold and r["reranker_pred"] not in gold
        assert any(c["is_exact_match"] for c in r["candidates"])
    counts_all, rows_all = analyze_changes(outs, ext, rp, golds, {}, det, max_examples=10**6)
    assert len(rows_all) == counts_all["FIX"] + counts_all["BREAK"]
    assert counts_all["n_prediction_changed"] >= counts_all["FIX"] + counts_all["BREAK"]


def test_recall_at_k():
    items = [{"em": np.array([False, True, False]), "f1": np.array([.2, 1., .0], dtype=np.float32)},
             {"em": np.array([True, False, False]), "f1": np.array([1., .5, .0], dtype=np.float32)},
             {"em": np.array([False, False, False]), "f1": np.array([.0, .0, .4], dtype=np.float32)}]
    r = recall_at_ks(items, (1, 2, 3))
    assert abs(r["recall_em@1"] - 100 / 3) < 1e-6 and abs(r["recall_em@2"] - 200 / 3) < 1e-6 and abs(r["recall_em@3"] - 200 / 3) < 1e-6
    assert abs(r["best_f1@3"] - 100 * (1 + 1 + .4) / 3) < 1e-4


def test_backward_compatible_defaults():
    # no new arguments: pair features on, 14 inputs, pure-reranker argmax, same as the model's raw argmax
    outs, golds = _outputs()
    items = build_items(outs, golds, use_pair_view=True)
    assert len(items) == 2 * len(outs) and items[0]["scalars"].shape[1] == 14
    m, _ = train_reranker(items[:60], items[60:], golds, HID, CPU, epochs=2, log=lambda *a: None)
    assert m.n_scalars == 14 and m.scalar_idx is None
    from g17qa.reranker import _pad_items
    vecs, scal, mask, _ = _pad_items(items[60:], CPU)
    with torch.no_grad():
        ref = m.eval()(vecs, scal).masked_fill(~mask, float("-inf")).argmax(-1).tolist()
    assert choose(m, items[60:], CPU) == ref
    ev = evaluate_items(m, items[60:], golds, CPU)
    assert {"n", "exact", "f1", "recall_em", "best_f1", "oracle_exact"} <= set(ev)
    # candidate_scalars is unchanged (14 columns, pair columns populated / flagged missing)
    cs = candidate_scalars(outs["q0"].candidates, 1)
    assert cs.shape == (5, 14) and (cs[:, SCALAR_FEATURES.index("pair_missing")] == 0).all()
    nopair = _outputs(with_pair=False)[0]["q0"].candidates
    assert (candidate_scalars(nopair, 1)[:, SCALAR_FEATURES.index("pair_missing")] == 1).all()


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
