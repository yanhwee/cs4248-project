#!/usr/bin/env python3
"""Fast correctness checks (no GPU, tiny model). Run:  python tests/test_core.py"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g17qa.data import build_features, char_span_to_token_span, load_paired_files, load_squad, make_collate_fn, split_by_article  # noqa: E402
from g17qa.losses import compute_total_loss, gather_passage_logits, masked_log_softmax, symmetric_kl, wasserstein_1d, hungarian_span_loss  # noqa: E402
from g17qa.metrics import compute_exact, compute_f1, consistency_metrics, exact_match_score, f1_score, normalize_answer, question_type  # noqa: E402
from g17qa.model import load_tokenizer, tokenizer_uses_token_type_ids  # noqa: E402

TINY = "google/bert_uncased_L-2_H-128_A-2"  # official 4.4M-parameter BERT-tiny
DEV = os.path.join(ROOT, "provided", "dev-v1.1.json")


def _official():
    spec = importlib.util.spec_from_file_location("official_eval", os.path.join(ROOT, "provided", "evaluate-v2.0.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metrics_match_official_script():
    off = _official()
    pairs = [
        ("Denver Broncos", "the Denver Broncos"), ("1842", "November 26, 1842"), ("", "x"), ("a b c", "b c d"),
        ("Saint Bernadette Soubirous", "Bernadette"), ("Levi's Stadium", "levis stadium"), ("24-10", "24–10"),
    ]
    for g, p in pairs:
        assert normalize_answer(p) == off.normalize_answer(p)
        assert compute_exact(g, p) == off.compute_exact(g, p), (g, p)
        assert abs(compute_f1(g, p) - off.compute_f1(g, p)) < 1e-9, (g, p)
    assert exact_match_score("Broncos", ["Denver Broncos", "broncos"]) == 1.0
    assert 0 < f1_score("Denver", ["Denver Broncos"]) < 1


def test_question_types():
    assert question_type("Who won Super Bowl 50?") == "who"
    assert question_type("In what year was the university founded?") == "when"
    assert question_type("How many points did Denver score?") == "how many/much"
    assert question_type("Which team won?") == "which"
    assert question_type("Where was the game played?") == "where"
    assert question_type("Name the stadium.") == "other"


def test_consistency():
    m = consistency_metrics({"a": "x", "b": "y z"}, {"a": "X", "b": "y"})
    assert m["pairs"] == 2 and m["agreement_em"] == 50.0 and 50 < m["agreement_f1"] < 100


def test_split_by_article():
    ex = load_squad(DEV)
    tr, va = split_by_article(ex, 0.1, 1)
    assert len(tr) + len(va) == len(ex) and va
    assert not ({e.title for e in tr} & {e.title for e in va})
    tr2, va2 = split_by_article(ex, 0.1, 1)
    assert [e.qid for e in va2] == [e.qid for e in va]  # deterministic


def test_char_to_token_span():
    offs = [(0, 3), (4, 7), (8, 12)]
    assert char_span_to_token_span(offs, 4, 7) == (1, 1)
    assert char_span_to_token_span(offs, 5, 12) == (1, 2)
    assert char_span_to_token_span(offs, 13, 15) == (-1, -1)


def test_features_labels_and_windows():
    tok = load_tokenizer(TINY)
    ex = load_squad(DEV, max_examples=80)
    feats, offsets = build_features(ex, tok, max_seq_len=128, doc_stride=32, with_pairs=False)
    assert all(np.all(f.token_type_ids[: f.passage_start] == 0) and np.all(f.token_type_ids[f.passage_start:] == 1) for f in feats)
    covered = 0
    multi = 0
    for e in ex:
        wins = [f for f in feats if f.qid == e.qid]
        assert wins, "every question needs at least one window"
        multi += len(wins) >= 2
        assert wins[-1].chunk_token_start + wins[-1].passage_len == len(offsets[e.qid]), "last window must reach the end"
        labelled = [f for f in wins if f.start_pos > 0]
        if labelled:
            covered += 1
            for f in labelled:
                gi = f.chunk_token_start + (f.start_pos - f.passage_start)
                gj = f.chunk_token_start + (f.end_pos - f.passage_start)
                text = e.context[offsets[e.qid][gi][0] : offsets[e.qid][gj][1]]
                assert normalize_answer(e.answers[0]) in normalize_answer(text) and len(text) <= len(e.answers[0]) + 8, (e.answers[0], text)
    assert covered >= 0.95 * len(ex), f"answers covered by at least one window: {covered}/{len(ex)}"
    assert multi >= 0.5 * len(ex), "stride 32 / len 128 should split most contexts into several windows"


def test_paired_features_share_chunk():
    tok = load_tokenizer(TINY)
    ex = load_squad(DEV, max_examples=20)
    for e in ex:
        e.paraphrase = "Rephrased: " + e.question
    feats, _ = build_features(ex, tok, max_seq_len=128, doc_stride=48, with_pairs=True)
    assert feats and all(f.has_pair for f in feats)
    for f in feats:
        a = f.input_ids[f.passage_start : f.passage_start + f.passage_len]
        b = f.input_ids_p[f.passage_start_p : f.passage_start_p + f.passage_len]
        assert np.array_equal(a, b)
    coll = make_collate_fn(tok.pad_token_id, tokenizer_uses_token_type_ids(tok))
    batch = coll(feats[:5])
    assert batch["input_ids_p"].shape[0] == 5 and batch["pair_mask"].all()


def test_paraphrase_labels_are_shifted():
    """CE(q') must be computed against the span at its position in the *paraphrase* input, which differs from the
    original input whenever the two questions have different token lengths."""
    tok = load_tokenizer(TINY)
    ex = load_squad(DEV, max_examples=40)
    for e in ex:
        e.paraphrase = "Could you please explain to me in detail " + e.question  # much longer than the original
    feats, _ = build_features(ex, tok, max_seq_len=256, doc_stride=96, with_pairs=True)
    labelled = [f for f in feats if f.start_pos > 0]
    assert labelled
    for f in labelled:
        shift = f.passage_start_p - f.passage_start
        assert shift > 0
        assert np.array_equal(f.input_ids_p[f.start_pos + shift : f.end_pos + shift + 1], f.input_ids[f.start_pos : f.end_pos + 1])
    # compute_total_loss must apply exactly that shift: put a confident peak at the shifted position in the q' view
    coll = make_collate_fn(tok.pad_token_id, tokenizer_uses_token_type_ids(tok))
    batch = coll(labelled[:6])
    B, L = batch["input_ids"].shape
    Lp = batch["input_ids_p"].shape[1]
    s_q = torch.zeros(B, L); e_q = torch.zeros(B, L)
    s_p = torch.full((B, Lp), -5.0); e_p = torch.full((B, Lp), -5.0)
    for b in range(B):
        sh = int(batch["passage_start_p"][b] - batch["passage_start"][b])
        s_p[b, int(batch["start_positions"][b]) + sh] = 20.0
        e_p[b, int(batch["end_positions"][b]) + sh] = 20.0
    comps = compute_total_loss("aug", {"start_logits": s_q, "end_logits": e_q}, batch, {"start_logits": s_p, "end_logits": e_p})
    assert comps["ce_p"].item() < 1e-3, f"ce_p should be ~0 when the peak sits at the shifted label, got {comps['ce_p'].item()}"
    # and large if the peak were at the *unshifted* position
    s_bad = torch.full((B, Lp), -5.0); e_bad = torch.full((B, Lp), -5.0)
    for b in range(B):
        s_bad[b, int(batch["start_positions"][b])] = 20.0
        e_bad[b, int(batch["end_positions"][b])] = 20.0
    comps_bad = compute_total_loss("aug", {"start_logits": s_q, "end_logits": e_q}, batch, {"start_logits": s_bad, "end_logits": e_bad})
    assert comps_bad["ce_p"].item() > 10


def test_losses():
    torch.manual_seed(0)
    B, L = 4, 40
    z = torch.randn(B, L)
    ps = torch.tensor([5, 6, 4, 7])
    pl = torch.tensor([30, 25, 33, 20])
    g, mask = gather_passage_logits(z, ps, pl)
    assert g.shape == (B, 33) and mask.sum(1).tolist() == pl.tolist()
    lp = masked_log_softmax(g, mask)
    assert torch.allclose(lp.exp().sum(1), torch.ones(B), atol=1e-4)
    assert torch.allclose(symmetric_kl(lp, lp, mask), torch.zeros(B), atol=1e-6)
    assert torch.allclose(wasserstein_1d(lp, lp, mask, pl), torch.zeros(B), atol=1e-6)
    # point masses i and j on a lattice of m: W1 = |i-j|/m
    m = 10
    a = torch.full((1, m), -30.0); a[0, 2] = 30.0
    b = torch.full((1, m), -30.0); b[0, 7] = 30.0
    mk = torch.ones(1, m, dtype=torch.bool)
    w = wasserstein_1d(torch.log_softmax(a, -1), torch.log_softmax(b, -1), mk, torch.tensor([m]))
    assert abs(w.item() - 0.5) < 1e-3, w
    kl_ab = symmetric_kl(torch.log_softmax(a, -1), torch.log_softmax(b, -1), mk)
    assert torch.isfinite(kl_ab).all()
    h = hungarian_span_loss(torch.randn(12), torch.randn(12), torch.randn(12), torch.randn(12), k=4)
    assert torch.isfinite(h)
    # every mode produces a finite, differentiable loss
    sq, eq, sp, ep = [torch.randn(B, L, requires_grad=True) for _ in range(4)]
    batch = {"start_positions": torch.tensor([0, 7, 9, 12]), "end_positions": torch.tensor([0, 8, 9, 14]), "passage_start": ps, "passage_start_p": ps + 1, "passage_len": pl, "pair_mask": torch.tensor([True, True, False, True])}
    for mode in ("ce", "aug", "kl", "ot", "hungarian", "dual"):
        comps = compute_total_loss(mode, {"start_logits": sq, "end_logits": eq}, batch, {"start_logits": sp, "end_logits": ep}, hung_topk=3)
        assert torch.isfinite(comps["total"]), mode
        comps["total"].backward()
        assert sq.grad is not None
        sq.grad = None
        if mode == "ce":
            assert "ce_p" not in comps
        else:
            assert comps["ce_p"] > 0


def test_gan_pairs():
    para, idmap = load_paired_files(os.path.join(ROOT, "data/external/gan_ng/adv_para.json"), os.path.join(ROOT, "data/external/gan_ng/adv_orig.json"))
    assert len(para) == 56 and all(p.has_pair for p in para) and len(idmap) == 56


def test_decode_and_reranker_features():
    from g17qa.decode import run_extractor
    from g17qa.model import SpanExtractor
    from g17qa.reranker import build_items, candidate_scalars, label_candidates

    tok = load_tokenizer(TINY)
    model = SpanExtractor(TINY)
    ex = load_squad(DEV, max_examples=12)
    for e in ex:
        e.paraphrase = "Could you tell me " + e.question[0].lower() + e.question[1:]
    out = run_extractor(model, tok, ex, torch.device("cpu"), batch_size=4, max_seq_len=128, doc_stride=48, top_k=10, with_pairs=True, want_vecs=True, progress=False)
    assert len(out) == len(ex)
    for e in ex:
        o = out[e.qid]
        assert o.paired and 1 <= len(o.candidates) <= 10 and o.candidates_p
        keys = [normalize_answer(c.text) for c in o.candidates]
        assert len(keys) == len(set(keys)), "candidates must be distinct"
        assert [c.score for c in o.candidates] == sorted([c.score for c in o.candidates], reverse=True)
        for c in o.candidates:
            assert e.context[c.char_start : c.char_end].strip() == c.text
            assert c.pair_score is not None and c.vec is not None and c.vec.shape == (4 * model.hidden_size,)
        sc = candidate_scalars(o.candidates, o.n_windows)
        assert sc.shape == (len(o.candidates), 14) and np.isfinite(sc).all()
        assert -1 <= label_candidates(o.candidates, e.answers) < len(o.candidates)
    items = build_items(out, {e.qid: e.answers for e in ex})
    assert len(items) == 2 * len(ex)
    # validation summary: EM/F1 plus agreement over the paired questions
    from g17qa.metrics import summarize_outputs

    s = summarize_outputs(out, {e.qid: e.answers for e in ex})
    assert s["total"] == len(ex) and s["pairs"] == len(ex) and 0.0 <= s["agreement_em"] <= 100.0 and 0.0 <= s["f1"] <= 100.0
    ex[0].paraphrase = None
    out2 = run_extractor(model, tok, ex, torch.device("cpu"), batch_size=4, max_seq_len=128, doc_stride=48, top_k=3, with_pairs=True, progress=False)
    assert summarize_outputs(out2, {e.qid: e.answers for e in ex})["pairs"] == len(ex) - 1


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
