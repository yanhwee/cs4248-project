"""Learned span reranker (proposal item 3).

The reranker chooses one of the extractor's top-k distinct candidates, so its answer is always an exact
passage span. It is a small MLP over the extractor's *own* representations:

  * hidden features  [h_start ; h_end ; mean(h_span) ; h_cls]   (4 x hidden_size, from decode.py)
  * scalar features   extractor scores, rank, span length, window count, softmax probability among the
                      candidates, and — when a paraphrase is available — the score the paraphrase view gives the
                      same span (a consistency signal), see SCALAR_FEATURES.

Training is listwise: softmax over a question's candidates, cross-entropy against the candidate that matches a
gold answer (exact match first, else the highest-F1 candidate if F1 > 0). Questions whose candidates contain no
correct span are skipped for training but still counted at evaluation time.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decode import Candidate, QuestionOutput
from .metrics import compute_f1, exact_match_score, f1_score, normalize_answer

SCALAR_FEATURES = [
    "score", "raw_score", "null_score", "start_logit", "end_logit", "gap_to_top", "prob",
    "rank", "len_tokens", "len_chars", "n_windows", "pair_score", "pair_gap", "pair_missing",
]


def candidate_scalars(cands: Sequence[Candidate], n_windows: int) -> np.ndarray:
    if not cands:
        return np.zeros((0, len(SCALAR_FEATURES)), dtype=np.float32)
    scores = np.array([c.score for c in cands], dtype=np.float64)
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    top = scores.max()
    pair = np.array([c.pair_score if c.pair_score is not None else 0.0 for c in cands], dtype=np.float64)
    pair_missing = np.array([1.0 if c.pair_score is None else 0.0 for c in cands], dtype=np.float64)
    pair_top = pair.max() if (pair_missing < 1).any() else 0.0
    rows = []
    for c, p, ps, pm in zip(cands, probs, pair, pair_missing):
        rows.append([
            c.score, c.raw_score, c.null_score, c.start_logit, c.end_logit, top - c.score, p,
            float(c.rank), float(c.tok_end - c.tok_start + 1), float(len(c.text)), float(n_windows),
            ps, pair_top - ps if pm < 1 else 0.0, pm,
        ])
    return np.asarray(rows, dtype=np.float32)


def label_candidates(cands: Sequence[Candidate], golds: Sequence[str]) -> int:
    """Index of the first candidate with EM against a gold answer; else highest-F1 candidate (F1>0); else -1."""
    gold_norm = {normalize_answer(g) for g in golds}
    for i, c in enumerate(cands):
        if normalize_answer(c.text) in gold_norm:
            return i
    best, best_f1 = -1, 0.0
    for i, c in enumerate(cands):
        f = max(compute_f1(g, c.text) for g in golds)
        if f > best_f1:
            best, best_f1 = i, f
    return best


def build_items(outputs: Dict[str, QuestionOutput], golds: Dict[str, Sequence[str]], use_pair_view: bool = True) -> List[Dict[str, Any]]:
    """One training item per (question, view). Items without any hidden vectors are skipped."""
    items = []
    for qid, o in outputs.items():
        views = [o.candidates] + ([o.candidates_p] if use_pair_view and o.paired and o.candidates_p else [])
        for cands in views:
            if not cands or cands[0].vec is None:
                continue
            items.append({
                "qid": qid,
                "vecs": np.stack([c.vec for c in cands]),
                "scalars": candidate_scalars(cands, o.n_windows),
                "target": label_candidates(cands, golds[qid]),
                "texts": [c.text for c in cands],
            })
    return items


class SpanReranker(nn.Module):
    def __init__(self, hidden_size: int, n_scalars: int, scalar_mean: np.ndarray, scalar_std: np.ndarray, proj: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_scalars = n_scalars
        self.register_buffer("scalar_mean", torch.tensor(scalar_mean, dtype=torch.float32))
        self.register_buffer("scalar_std", torch.tensor(scalar_std, dtype=torch.float32))
        self.vec_norm = nn.LayerNorm(4 * hidden_size)
        self.vec_proj = nn.Linear(4 * hidden_size, proj)
        self.sc_proj = nn.Linear(n_scalars, 32)
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(proj + 32, proj), nn.GELU(), nn.Dropout(dropout), nn.Linear(proj, 1))

    def forward(self, vecs: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """vecs [B,K,4d] float, scalars [B,K,n] -> scores [B,K]"""
        v = self.vec_proj(self.vec_norm(vecs.float()))
        s = self.sc_proj((scalars.float() - self.scalar_mean) / self.scalar_std)
        return self.mlp(torch.cat([v, s], -1)).squeeze(-1)


def _pad_items(items: Sequence[Dict[str, Any]], device: torch.device):
    K = max(len(it["texts"]) for it in items)
    d = items[0]["vecs"].shape[1]
    n = items[0]["scalars"].shape[1]
    vecs = np.zeros((len(items), K, d), dtype=np.float32)
    scal = np.zeros((len(items), K, n), dtype=np.float32)
    mask = np.zeros((len(items), K), dtype=bool)
    tgt = np.full(len(items), -1, dtype=np.int64)
    for b, it in enumerate(items):
        k = len(it["texts"])
        vecs[b, :k] = it["vecs"].astype(np.float32)
        scal[b, :k] = it["scalars"]
        mask[b, :k] = True
        tgt[b] = it["target"]
    return (torch.from_numpy(vecs).to(device), torch.from_numpy(scal).to(device), torch.from_numpy(mask).to(device), torch.from_numpy(tgt).to(device))


@torch.no_grad()
def choose(model: SpanReranker, items: Sequence[Dict[str, Any]], device: torch.device, margin: Optional[float] = None, batch_size: int = 128) -> List[int]:
    """Index chosen per item. With `margin`, only candidates within `margin` of the top extractor score compete."""
    model.eval()
    out: List[int] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        vecs, scal, mask, _ = _pad_items(chunk, device)
        scores = model(vecs, scal).masked_fill(~mask, float("-inf"))
        if margin is not None:
            ext = scal[..., 0]  # SCALAR_FEATURES[0] == "score"
            top = ext.masked_fill(~mask, float("-inf")).max(-1, keepdim=True).values
            scores = scores.masked_fill(ext < top - margin, float("-inf"))
        out.extend(scores.argmax(-1).tolist())
    return out


def evaluate_items(model: Optional[SpanReranker], items: Sequence[Dict[str, Any]], golds: Dict[str, Sequence[str]], device: torch.device, margin: Optional[float] = None) -> Dict[str, float]:
    if not items:
        return {"n": 0, "exact": 0.0, "f1": 0.0, "oracle_exact": 0.0}
    idx = choose(model, items, device, margin) if model is not None else [0] * len(items)
    em = f1 = oracle = 0.0
    for it, k in zip(items, idx):
        g = golds[it["qid"]]
        em += exact_match_score(it["texts"][k], g)
        f1 += f1_score(it["texts"][k], g)
        oracle += float(it["target"] >= 0 and exact_match_score(it["texts"][it["target"]], g) > 0)
    n = len(items)
    return {"n": n, "exact": 100.0 * em / n, "f1": 100.0 * f1 / n, "oracle_exact": 100.0 * oracle / n}


def train_reranker(
    train_items: List[Dict[str, Any]],
    dev_items: List[Dict[str, Any]],
    golds: Dict[str, Sequence[str]],
    hidden_size: int,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    proj: int = 128,
    weight_decay: float = 1e-4,
    seed: int = 42,
    log=print,
) -> Tuple[SpanReranker, List[Dict[str, Any]]]:
    if not train_items:
        raise ValueError("no reranker training items (empty training split)")
    trainable = [it for it in train_items if it["target"] >= 0]
    if not trainable:
        raise ValueError("none of the training questions has a correct span among the extractor's top-k candidates; the extractor is too weak to train a reranker")
    all_sc = np.concatenate([it["scalars"] for it in trainable], 0)
    mean, std = all_sc.mean(0), all_sc.std(0) + 1e-6
    model = SpanReranker(hidden_size, all_sc.shape[1], mean, std, proj=proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    history: List[Dict[str, Any]] = []
    best_state, best_f1 = None, -1.0
    base = evaluate_items(None, dev_items, golds, device)
    log(f"reranker: {len(trainable)} trainable items ({len(train_items) - len(trainable)} without a correct candidate); dev top-1 EM {base['exact']:.2f} F1 {base['f1']:.2f} oracle-EM {base['oracle_exact']:.2f}")
    for ep in range(1, epochs + 1):
        model.train()
        rng.shuffle(trainable)
        tot = 0.0
        nb = 0
        for i in range(0, len(trainable), batch_size):
            vecs, scal, mask, tgt = _pad_items(trainable[i : i + batch_size], device)
            scores = model(vecs, scal).masked_fill(~mask, float("-inf"))
            loss = F.cross_entropy(scores, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
            nb += 1
        dev = evaluate_items(model, dev_items, golds, device)
        history.append({"epoch": ep, "loss": tot / max(1, nb), **{f"dev_{k}": v for k, v in dev.items()}})
        log(f"  epoch {ep:2d} loss {tot / max(1, nb):.4f} | dev EM {dev['exact']:.2f} F1 {dev['f1']:.2f}")
        if dev["f1"] > best_f1:
            best_f1 = dev["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def save_reranker(model: SpanReranker, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "format": "g17qa-reranker-v1",
        "hidden_size": model.hidden_size,
        "n_scalars": model.n_scalars,
        "proj": model.vec_proj.out_features,
        "scalar_features": SCALAR_FEATURES,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_reranker(path: str, device: torch.device) -> SpanReranker:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n = ckpt["n_scalars"]
    model = SpanReranker(ckpt["hidden_size"], n, np.zeros(n), np.ones(n), proj=ckpt.get("proj", 128))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def apply_reranker(model: SpanReranker, outputs: Dict[str, QuestionOutput], device: torch.device, margin: Optional[float] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (reranked predictions for q, reranked predictions for q')."""
    preds_q: Dict[str, str] = {}
    preds_p: Dict[str, str] = {}
    for view, store in (("candidates", preds_q), ("candidates_p", preds_p)):
        items, qids = [], []
        for qid, o in outputs.items():
            cands = getattr(o, view)
            if not cands:
                if view == "candidates":
                    store[qid] = ""
                continue
            if cands[0].vec is None:
                store[qid] = cands[0].text
                continue
            items.append({"qid": qid, "vecs": np.stack([c.vec for c in cands]), "scalars": candidate_scalars(cands, o.n_windows), "target": -1, "texts": [c.text for c in cands]})
            qids.append(qid)
        if items:
            for qid, k, it in zip(qids, choose(model, items, device, margin), items):
                store[qid] = it["texts"][k]
    return preds_q, preds_p
