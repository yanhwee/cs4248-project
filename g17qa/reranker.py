"""Learned span reranker (proposal item 3).

The reranker chooses one of the extractor's top-k distinct candidates, so its answer is always an exact
passage span. It is a small MLP over the extractor's *own* representations:

  * hidden features  [h_start ; h_end ; mean(h_span) ; h_cls]   (4 x hidden_size, from decode.py)
  * scalar features   extractor scores, rank, span length, window count, softmax probability among the
                      candidates, and — when a paraphrase is available — the score the paraphrase view gives the
                      same span (a consistency signal), see SCALAR_FEATURES.

Training is listwise: softmax over a question's candidates, cross-entropy against a *target distribution*
built from per-candidate relevance (`target_mode`):

  graded (default)  exact-match candidates get relevance 1; non-EM candidates with F1 > 0 get `partial_weight` * F1
                    (< 1), so EM is always the strongest signal and partial overlaps only supervise weakly.
                    The target distribution is the relevance normalised to sum to 1.
  exact             only exact-match candidates are positives (uniform over them); questions without an EM
                    candidate are skipped.
  f1                legacy: one-hot on the first EM candidate, else on the highest-F1 candidate (F1 > 0).

Questions with no positive candidate are skipped for training but still counted at evaluation time. Because the
reranker can only choose among the top-k, evaluation reports its ceilings: `recall_em` (Recall@k EM, the best EM
any reranker could reach) and `best_f1` (best-F1@k).

Consistency training (`consistency_weight` > 0, default 0 = off): the question view and its paraphrase view of the
same question are both training items. Their candidate lists are aligned by normalised answer text; on the
candidates present in BOTH lists each view's reranker scores are log-softmaxed and a symmetric KL between the two
distributions is added to the loss, weighted by `consistency_weight`. This asks the reranker to rank the same spans
the same way for a question and its paraphrase, which targets paraphrase agreement without needing the paraphrase at
test time (it combines naturally with `use_pair_score=False`). Candidates that appear in only one list are ignored by
this term; pairs sharing fewer than two candidates are skipped.

Ablation / selection options (all default to the original behaviour):

  use_pair_score=False   drops the three paraphrase-derived scalar features (PAIR_FEATURES: pair_score, pair_gap,
                         pair_missing) from the MLP input; the model then never sees the other view's scores.
  selection modes        how the final answer is picked from the reranker's scores (see `select_index`):
                           extractor  keep the extractor's own top-1 (no reranking)
                           reranker   argmax of the reranker score (original behaviour)
                           residual   argmax of  log p_ext(i) + alpha * log p_rr(i)   (conservative fusion)
                         plus an optional confidence gate (`override_margin`): the extractor top-1 is only replaced if
                         the reranker prefers the replacement over it by more than that margin.
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


# Scalar features that are derived from the paraphrase view. `use_pair_score=False` removes all of them.
PAIR_FEATURES = ("pair_score", "pair_gap", "pair_missing")
SELECT_MODES = ("extractor", "reranker", "residual")


def scalar_feature_indices(use_pair_score: bool = True) -> List[int]:
    """Columns of `candidate_scalars` fed to the MLP."""
    return [i for i, n in enumerate(SCALAR_FEATURES) if use_pair_score or n not in PAIR_FEATURES]


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


TARGET_MODES = ("graded", "exact", "f1")


def candidate_match(cands: Sequence[Candidate], golds: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-candidate (exact-match flag, best F1 over the gold answers)."""
    gold_norm = {normalize_answer(g) for g in golds}
    em = np.array([normalize_answer(c.text) in gold_norm for c in cands], dtype=bool)
    f1 = np.array([max((compute_f1(g, c.text) for g in golds), default=0.0) for c in cands], dtype=np.float32)
    return em, f1


def label_candidates(cands: Sequence[Candidate], golds: Sequence[str]) -> int:
    """Index of the first candidate with exact match against a gold answer, else -1 (no F1 fallback)."""
    em, _ = candidate_match(cands, golds)
    return int(np.argmax(em)) if em.any() else -1


def target_distribution(em: np.ndarray, f1: np.ndarray, mode: str = "graded", partial_weight: float = 0.3) -> np.ndarray:
    """Target distribution over a question's candidates (sums to 1, or all zeros when there is no positive)."""
    if mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {mode!r}")
    rel = np.zeros(len(em), dtype=np.float64)
    if mode == "graded":
        rel = np.where(em, 1.0, np.where(f1 > 0, partial_weight * f1, 0.0))
    elif mode == "exact":
        rel = em.astype(np.float64)
    elif em.any():  # legacy "f1": first EM candidate
        rel[int(np.argmax(em))] = 1.0
    elif (f1 > 0).any():  # legacy "f1": best partial candidate
        rel[int(np.argmax(f1))] = 1.0
    tot = rel.sum()
    return (rel / tot).astype(np.float32) if tot > 0 else rel.astype(np.float32)


def build_items(outputs: Dict[str, QuestionOutput], golds: Dict[str, Sequence[str]], use_pair_view: bool = True) -> List[Dict[str, Any]]:
    """One training item per (question, view). Items without any hidden vectors are skipped."""
    items = []
    for qid, o in outputs.items():
        views = [("q", o.candidates)] + ([("p", o.candidates_p)] if use_pair_view and o.paired and o.candidates_p else [])
        for view, cands in views:
            if not cands or cands[0].vec is None:
                continue
            em, f1 = candidate_match(cands, golds[qid])
            items.append({
                "qid": qid,
                "vecs": np.stack([c.vec for c in cands]),
                "scalars": candidate_scalars(cands, o.n_windows),
                "em": em,
                "f1": f1,
                "target": int(np.argmax(em)) if em.any() else -1,  # first exact-match candidate
                "texts": [c.text for c in cands],
                "view": view,                                          # "q" question view, "p" paraphrase view
                "keys": [normalize_answer(c.text) for c in cands],     # for aligning the two views' candidates
            })
    return items


def align_views(item_q: Dict[str, Any], item_p: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Indices (into each item's candidate list) of the candidates present in both views, matched by normalised text.
    None when fewer than two candidates are shared."""
    pos_p = {k: i for i, k in enumerate(item_p["keys"])}
    pairs = [(i, pos_p[k]) for i, k in enumerate(item_q["keys"]) if k in pos_p]
    if len(pairs) < 2:
        return None
    iq, ip = zip(*pairs)
    return np.array(iq, dtype=np.int64), np.array(ip, dtype=np.int64)


def symmetric_kl_aligned(scores_q: torch.Tensor, scores_p: torch.Tensor) -> torch.Tensor:
    """0.5 * (KL(a||b) + KL(b||a)) where a, b are the log-softmaxes of two aligned 1-D score vectors (same candidates)."""
    la, lb = F.log_softmax(scores_q, dim=-1), F.log_softmax(scores_p, dim=-1)
    return 0.5 * ((la.exp() * (la - lb)).sum() + (lb.exp() * (lb - la)).sum())


def candidate_ceiling(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Ceilings of any reranker over these items: Recall@k EM and best-F1@k (both in %)."""
    n = len(items)
    if not n:
        return {"n": 0, "recall_em": 0.0, "best_f1": 0.0}
    return {
        "n": n,
        "recall_em": 100.0 * sum(float(it["em"].any()) for it in items) / n,
        "best_f1": 100.0 * sum(float(it["f1"].max()) for it in items) / n,
    }


class SpanReranker(nn.Module):
    """`scalar_idx` selects the columns of the full (len(SCALAR_FEATURES)-wide) scalar matrix the MLP reads;
    `scalar_mean`/`scalar_std` and `n_scalars` refer to those selected columns. None = all columns (original model)."""

    def __init__(self, hidden_size: int, n_scalars: int, scalar_mean: np.ndarray, scalar_std: np.ndarray, proj: int = 128, dropout: float = 0.1,
                 scalar_idx: Optional[Sequence[int]] = None):
        super().__init__()
        if scalar_idx is not None and len(scalar_idx) != n_scalars:
            raise ValueError(f"n_scalars={n_scalars} does not match len(scalar_idx)={len(scalar_idx)}")
        self.hidden_size = hidden_size
        self.n_scalars = n_scalars
        self.scalar_idx = None if scalar_idx is None else [int(i) for i in scalar_idx]
        # non-persistent: not part of the state_dict, so checkpoints written before this option still load
        self.register_buffer("_scalar_idx", torch.tensor(self.scalar_idx if self.scalar_idx is not None else [], dtype=torch.long), persistent=False)
        self.register_buffer("scalar_mean", torch.tensor(scalar_mean, dtype=torch.float32))
        self.register_buffer("scalar_std", torch.tensor(scalar_std, dtype=torch.float32))
        self.vec_norm = nn.LayerNorm(4 * hidden_size)
        self.vec_proj = nn.Linear(4 * hidden_size, proj)
        self.sc_proj = nn.Linear(n_scalars, 32)
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(proj + 32, proj), nn.GELU(), nn.Dropout(dropout), nn.Linear(proj, 1))

    @property
    def use_pair_score(self) -> bool:
        if self.scalar_idx is None:
            return True
        return all(SCALAR_FEATURES.index(n) in self.scalar_idx for n in PAIR_FEATURES)

    @property
    def feature_names(self) -> List[str]:
        return list(SCALAR_FEATURES) if self.scalar_idx is None else [SCALAR_FEATURES[i] for i in self.scalar_idx]

    def forward(self, vecs: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """vecs [B,K,4d] float, scalars [B,K,len(SCALAR_FEATURES)] (full width) -> scores [B,K]"""
        if self.scalar_idx is not None:
            scalars = scalars.index_select(-1, self._scalar_idx.to(scalars.device))
        v = self.vec_proj(self.vec_norm(vecs.float()))
        s = self.sc_proj((scalars.float() - self.scalar_mean) / self.scalar_std)
        return self.mlp(torch.cat([v, s], -1)).squeeze(-1)


def _pad_items(items: Sequence[Dict[str, Any]], device: torch.device, target_mode: str = "graded", partial_weight: float = 0.3):
    K = max(len(it["texts"]) for it in items)
    d = items[0]["vecs"].shape[1]
    n = items[0]["scalars"].shape[1]
    vecs = np.zeros((len(items), K, d), dtype=np.float32)
    scal = np.zeros((len(items), K, n), dtype=np.float32)
    mask = np.zeros((len(items), K), dtype=bool)
    tgt = np.zeros((len(items), K), dtype=np.float32)
    for b, it in enumerate(items):
        k = len(it["texts"])
        vecs[b, :k] = it["vecs"].astype(np.float32)
        scal[b, :k] = it["scalars"]
        mask[b, :k] = True
        if "em" in it:
            tgt[b, :k] = target_distribution(it["em"], it["f1"], target_mode, partial_weight)
    return (torch.from_numpy(vecs).to(device), torch.from_numpy(scal).to(device), torch.from_numpy(mask).to(device), torch.from_numpy(tgt).to(device))


def log_softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over a 1-D array (entries may be -inf)."""
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def select_index(
    ext_scores: np.ndarray,
    rr_logits: Optional[np.ndarray],
    mode: str = "reranker",
    alpha: float = 1.0,
    override_margin: Optional[float] = None,
    margin: Optional[float] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Pick one candidate. `ext_scores` are the extractor scores in extractor order (index 0 = extractor top-1),
    `rr_logits` the raw reranker outputs for the same candidates.

    Normalisation. Both score sets are turned into per-question log-probabilities over the candidate list with a
    log-softmax before they are combined. This is the sensible common scale because (i) the reranker is trained
    with a softmax cross-entropy over exactly these candidates, so its logits are only defined up to a per-question
    shift and log_softmax recovers the distribution it was trained to produce; (ii) the extractor score is
    start_logit + end_logit - null, whose differences between spans are log-odds under the extractor's own
    start/end softmaxes, so log_softmax gives its posterior over the same candidates. Both are then in nats of a
    categorical distribution over the candidates and `alpha` is the exponent of the reranker in the product
    p_final ~ p_ext * p_rr**alpha. The shift-invariance also means a constant offset (e.g. the null score) is
    irrelevant. raw (un-normalised) scores are not used because their scales differ by orders of magnitude.

      extractor : argmax log p_ext             (= index 0, the extractor top-1)
      reranker  : argmax log p_rr              (= argmax of the logits; original behaviour)
      residual  : argmax log p_ext + alpha * log p_rr;  alpha=0 reproduces the extractor ranking, alpha->inf the reranker

    `margin` (original option): candidates scoring more than `margin` below the extractor top-1 are excluded.
    `override_margin` (gate): if the selection is not the extractor top-1, it is only kept when
    log p_rr(selection) - log p_rr(extractor top-1) > override_margin; otherwise the extractor top-1 is retained.
    Returns (index, info) where info holds the normalised scores for diagnostics.
    """
    if mode not in SELECT_MODES:
        raise ValueError(f"mode must be one of {SELECT_MODES}, got {mode!r}")
    ext = np.asarray(ext_scores, dtype=np.float64)
    n = len(ext)
    allowed = np.ones(n, dtype=bool)
    if margin is not None:
        allowed = ext >= ext.max() - margin
    lp_ext = log_softmax_np(ext)
    info: Dict[str, Any] = {"ext_logp": lp_ext, "rr_logp": None, "final": lp_ext, "gated": False}
    if mode == "extractor" or rr_logits is None:
        return int(np.argmax(np.where(allowed, lp_ext, -np.inf))), info
    lp_rr = log_softmax_np(np.asarray(rr_logits, dtype=np.float64))
    final = lp_rr if mode == "reranker" else lp_ext + alpha * lp_rr
    info.update(rr_logp=lp_rr, final=final)
    k = int(np.argmax(np.where(allowed, final, -np.inf)))
    top = int(np.argmax(ext))  # extractor top-1 (index 0 for sorted candidates)
    if override_margin is not None and k != top and not (lp_rr[k] - lp_rr[top] > override_margin):
        info["gated"] = True
        k = top
    return k, info


@torch.no_grad()
def rerank_logits(model: SpanReranker, items: Sequence[Dict[str, Any]], device: torch.device, batch_size: int = 128) -> List[np.ndarray]:
    """Raw reranker logits per item (one array of length K_i each)."""
    model.eval()
    out: List[np.ndarray] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        vecs, scal, mask, _ = _pad_items(chunk, device)
        sc = model(vecs, scal).float().cpu().numpy()
        out.extend(sc[b, : len(it["texts"])] for b, it in enumerate(chunk))
    return out


def choose(model: SpanReranker, items: Sequence[Dict[str, Any]], device: torch.device, margin: Optional[float] = None, batch_size: int = 128,
           mode: str = "reranker", alpha: float = 1.0, override_margin: Optional[float] = None) -> List[int]:
    """Index chosen per item. Defaults reproduce the original behaviour (pure reranker argmax). With `margin`, only
    candidates within `margin` of the top extractor score compete; see `select_index` for mode/alpha/override_margin."""
    logits = rerank_logits(model, items, device, batch_size)
    return [select_index(it["scalars"][:, 0], lg, mode, alpha, override_margin, margin)[0] for it, lg in zip(items, logits)]  # column 0 == "score"


def classify_change(ext_correct: bool, rr_correct: bool) -> str:
    """FIX: extractor wrong -> reranker right; BREAK: extractor right -> reranker wrong; else SAME_CORRECT / SAME_WRONG."""
    if ext_correct and rr_correct:
        return "SAME_CORRECT"
    if ext_correct:
        return "BREAK"
    if rr_correct:
        return "FIX"
    return "SAME_WRONG"


def change_counts(labels: Sequence[str]) -> Dict[str, Any]:
    """Counts and percentages of FIX / BREAK / SAME_CORRECT / SAME_WRONG labels."""
    n = len(labels)
    out: Dict[str, Any] = {"n": n}
    for k in ("FIX", "BREAK", "SAME_CORRECT", "SAME_WRONG"):
        c = sum(1 for x in labels if x == k)
        out[k] = c
        out[f"{k}_pct"] = 100.0 * c / n if n else 0.0
    out["net_em_change"] = out["FIX"] - out["BREAK"]
    return out


def analyze_changes(outputs: Dict[str, QuestionOutput], ext_preds: Dict[str, str], rr_preds: Dict[str, str], golds: Dict[str, Sequence[str]],
                    questions: Dict[str, str], details: Dict[str, Dict[str, Any]], max_examples: int = 100) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """FIX / BREAK / SAME_CORRECT / SAME_WRONG counts over all questions (correct = normalised exact match against any
    gold answer, as in the official scorer) and up to `max_examples` FIX and `max_examples` BREAK rows carrying the
    question, golds, both predictions and every candidate with its extractor score, reranker score and pair_score."""
    labels: List[str] = []
    df1: List[float] = []
    rows: Dict[str, List[Dict[str, Any]]] = {"FIX": [], "BREAK": []}
    for q, o in outputs.items():
        lab = classify_change(exact_match_score(ext_preds[q], golds[q]) > 0, exact_match_score(rr_preds[q], golds[q]) > 0)
        labels.append(lab)
        if ext_preds[q] != rr_preds[q]:
            df1.append(f1_score(rr_preds[q], golds[q]) - f1_score(ext_preds[q], golds[q]))
        if lab in rows and len(rows[lab]) < max_examples and q in details:
            d = details[q]
            rows[lab].append({
                "id": q, "type": lab, "question": questions.get(q), "gold": list(golds[q]), "extractor_top1": ext_preds[q], "reranker_pred": rr_preds[q],
                "gated": bool(d["gated"]),
                "candidates": [
                    dict(c.to_json(), extractor_logp=round(float(d["ext_logp"][i]), 4), reranker_logit=round(float(d["rr_logit"][i]), 4),
                         reranker_logp=round(float(d["rr_logp"][i]), 4), final_score=round(float(d["final"][i]), 4),
                         is_exact_match=bool(exact_match_score(c.text, golds[q]) > 0))
                    for i, c in enumerate(o.candidates)
                ],
            })
    counts = change_counts(labels)
    counts["n_prediction_changed"] = len(df1)
    counts["mean_f1_change_when_changed"] = float(np.mean(df1)) if df1 else 0.0
    return counts, rows["FIX"] + rows["BREAK"]


def recall_at_ks(items: Sequence[Dict[str, Any]], ks: Sequence[int] = (1, 3, 5, 10, 20)) -> Dict[str, float]:
    """Recall@k EM and best-F1@k (%) of the extractor's ranking, for each k (candidates are in extractor order)."""
    out: Dict[str, float] = {}
    n = len(items)
    for k in ks:
        out[f"recall_em@{k}"] = 100.0 * sum(float(it["em"][:k].any()) for it in items) / n if n else 0.0
        out[f"best_f1@{k}"] = 100.0 * sum(float(it["f1"][:k].max()) if len(it["f1"][:k]) else 0.0 for it in items) / n if n else 0.0
    return out


def evaluate_items(model: Optional[SpanReranker], items: Sequence[Dict[str, Any]], golds: Dict[str, Sequence[str]], device: torch.device, margin: Optional[float] = None,
                   mode: str = "reranker", alpha: float = 1.0, override_margin: Optional[float] = None) -> Dict[str, float]:
    """Top-1 EM/F1 of the reranker (or of the extractor's own ranking when model is None) plus the ceilings."""
    if not items:
        return {"n": 0, "exact": 0.0, "f1": 0.0, "recall_em": 0.0, "best_f1": 0.0, "oracle_exact": 0.0}
    idx = choose(model, items, device, margin, mode=mode, alpha=alpha, override_margin=override_margin) if model is not None else [0] * len(items)
    em = f1 = 0.0
    labels = []
    for it, k in zip(items, idx):
        g = golds[it["qid"]]
        e = exact_match_score(it["texts"][k], g)
        em += e
        f1 += f1_score(it["texts"][k], g)
        labels.append(classify_change(bool(it["em"][0]), bool(e)))
    n = len(items)
    ceil = candidate_ceiling(items)
    cc = change_counts(labels)
    return {"n": n, "exact": 100.0 * em / n, "f1": 100.0 * f1 / n, "recall_em": ceil["recall_em"], "best_f1": ceil["best_f1"], "oracle_exact": ceil["recall_em"],
            "fix": cc["FIX"], "break": cc["BREAK"]}


def pair_agreement(model: Optional[SpanReranker], items: Sequence[Dict[str, Any]], device: torch.device, margin: Optional[float] = None,
                   mode: str = "reranker", alpha: float = 1.0, override_margin: Optional[float] = None) -> Dict[str, float]:
    """Share of questions whose chosen answer is the same (normalised) in the question and paraphrase views, over the
    questions with both views among `items` (build_items(..., use_pair_view=True)). model=None = extractor top-1."""
    idx = choose(model, items, device, margin, mode=mode, alpha=alpha, override_margin=override_margin) if model is not None else [0] * len(items)
    chosen: Dict[str, Dict[str, str]] = {}
    for it, k in zip(items, idx):
        chosen.setdefault(it["qid"], {})[it["view"]] = it["keys"][k]
    pairs = [v for v in chosen.values() if "q" in v and "p" in v]
    return {"pairs": len(pairs), "agreement_em": 100.0 * sum(v["q"] == v["p"] for v in pairs) / len(pairs) if pairs else 0.0}


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
    target_mode: str = "graded",
    partial_weight: float = 0.3,
    use_pair_score: bool = True,
    consistency_weight: float = 0.0,
    dev_pair_items: Optional[List[Dict[str, Any]]] = None,
    select_by: str = "em",
    log=print,
) -> Tuple[SpanReranker, List[Dict[str, Any]]]:
    if not train_items:
        raise ValueError("no reranker training items (empty training split)")
    if select_by not in ("em", "em+agreement"):
        raise ValueError(f"select_by must be 'em' or 'em+agreement', got {select_by!r}")
    if select_by == "em+agreement" and not dev_pair_items:
        raise ValueError("select_by='em+agreement' needs dev_pair_items (held-out questions with both views)")
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
    trainable = [it for it in train_items if target_distribution(it["em"], it["f1"], target_mode, partial_weight).sum() > 0]
    if not trainable:
        raise ValueError(f"none of the training questions has a positive candidate under target_mode={target_mode!r} among the extractor's top-k candidates; the extractor is too weak to train a reranker")
    all_sc = np.concatenate([it["scalars"] for it in trainable], 0)
    # use_pair_score=True keeps scalar_idx=None, i.e. exactly the original model; False selects the non-paraphrase columns
    sidx = None if use_pair_score else scalar_feature_indices(False)
    cols = all_sc if sidx is None else all_sc[:, sidx]
    mean, std = cols.mean(0), cols.std(0) + 1e-6
    model = SpanReranker(hidden_size, cols.shape[1], mean, std, proj=proj, scalar_idx=sidx).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    history: List[Dict[str, Any]] = []
    best_state, best_key = None, (-1.0, -1.0)
    base = evaluate_items(None, dev_items, golds, device)
    tr_ceil = candidate_ceiling(train_items)
    log(f"reranker: use_pair_score={use_pair_score} ({model.n_scalars} scalar features) target_mode={target_mode} partial_weight={partial_weight} | {len(trainable)} trainable items ({len(train_items) - len(trainable)} without a positive candidate)")
    log(f"  train ceiling: Recall@k EM {tr_ceil['recall_em']:.2f} best-F1@k {tr_ceil['best_f1']:.2f}")
    log(f"  dev top-1 EM {base['exact']:.2f} F1 {base['f1']:.2f} | dev ceiling: Recall@k EM {base['recall_em']:.2f} best-F1@k {base['best_f1']:.2f}")
    # With consistency training the batch unit is a question (1 or 2 views); otherwise the original per-item batching is kept
    units: List[Tuple[List[Dict[str, Any]], Optional[Tuple[np.ndarray, np.ndarray]]]] = []
    if consistency_weight > 0:
        by_q: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for it in trainable:
            by_q.setdefault(it["qid"], {})[it.get("view", "q")] = it
        for views in by_q.values():
            its = [views[v] for v in ("q", "p") if v in views]
            units.append((its, align_views(views["q"], views["p"]) if len(its) == 2 else None))
        n_pairs = sum(1 for _, al in units if al is not None)
        log(f"  consistency_weight={consistency_weight}: {n_pairs} aligned question/paraphrase pairs of {len(units)} questions")
        if n_pairs == 0:
            raise ValueError("consistency_weight > 0 but no training question has both views with >= 2 shared candidates (is a paraphrase cache attached?)")
    if dev_pair_items:
        base_ag = pair_agreement(None, dev_pair_items, device)
        log(f"  dev extractor agreement {base_ag['agreement_em']:.2f} over {base_ag['pairs']} pairs")
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        tot_c = 0.0
        nb = 0
        if consistency_weight > 0:
            rng.shuffle(units)
            batches, cur, n_cur = [], [], 0
            for u in units:
                cur.append(u)
                n_cur += len(u[0])
                if n_cur >= batch_size:
                    batches.append(cur)
                    cur, n_cur = [], 0
            if cur:
                batches.append(cur)
        else:
            rng.shuffle(trainable)
            batches = [None] * ((len(trainable) + batch_size - 1) // batch_size)
        for bi, bunits in enumerate(batches):
            if bunits is None:
                batch_items = trainable[bi * batch_size : (bi + 1) * batch_size]
            else:
                batch_items = [it for its, _ in bunits for it in its]
            vecs, scal, mask, tgt = _pad_items(batch_items, device, target_mode, partial_weight)
            scores = model(vecs, scal).masked_fill(~mask, float("-inf"))
            logp = torch.where(mask, F.log_softmax(scores, dim=-1), torch.zeros_like(scores))  # avoid 0 * -inf
            loss = -(tgt * logp).sum(-1).mean()
            cons = scores.new_zeros(())
            if bunits is not None:
                kls, off = [], 0
                for its, al in bunits:
                    if al is not None:
                        iq, ip = (torch.as_tensor(x, device=scores.device) for x in al)
                        kls.append(symmetric_kl_aligned(scores[off, iq], scores[off + 1, ip]))
                    off += len(its)
                if kls:
                    cons = torch.stack(kls).mean()
                    loss = loss + consistency_weight * cons
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
            tot_c += float(cons.item())
            nb += 1
        dev = evaluate_items(model, dev_items, golds, device)
        rec = {"epoch": ep, "loss": tot / max(1, nb), **{f"dev_{k}": v for k, v in dev.items()}}
        msg = f"  epoch {ep:2d} loss {tot / max(1, nb):.4f} | dev EM {dev['exact']:.2f} F1 {dev['f1']:.2f}"
        if consistency_weight > 0:
            rec["consistency_kl"] = tot_c / max(1, nb)
            msg += f" | consistency KL {tot_c / max(1, nb):.4f}"
        if dev_pair_items:
            ag = pair_agreement(model, dev_pair_items, device)
            rec["dev_agreement_em"] = ag["agreement_em"]
            msg += f" | dev agreement {ag['agreement_em']:.2f}"
        history.append(rec)
        log(msg)
        # default: the goal is EM, F1 breaks ties. 'em+agreement': held-out EM + held-out paraphrase agreement (equal weights, an
        # explicit choice), EM breaks ties, so an epoch that is clearly more consistent is not discarded for a noisy EM difference
        key = (dev["exact"], dev["f1"]) if select_by == "em" else (dev["exact"] + rec["dev_agreement_em"], dev["exact"])
        if key > best_key:
            best_key = key
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
        "scalar_features": SCALAR_FEATURES,          # all columns produced by candidate_scalars
        "scalar_idx": model.scalar_idx,              # columns the MLP reads; None = all (checkpoints before the ablation option)
        "scalar_features_used": model.feature_names,
        "use_pair_score": model.use_pair_score,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_reranker(path: str, device: torch.device) -> SpanReranker:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n = ckpt["n_scalars"]
    model = SpanReranker(ckpt["hidden_size"], n, np.zeros(n), np.ones(n), proj=ckpt.get("proj", 128), scalar_idx=ckpt.get("scalar_idx"))
    model.load_state_dict(ckpt["state_dict"])
    model.meta = {k: ckpt.get(k) for k in ("tag", "target_mode", "partial_weight", "pair_features", "extractor", "top_k")}  # provenance only
    model.meta["use_pair_score"] = model.use_pair_score
    return model.to(device).eval()


def apply_reranker(model: SpanReranker, outputs: Dict[str, QuestionOutput], device: torch.device, margin: Optional[float] = None,
                   mode: str = "reranker", alpha: float = 1.0, override_margin: Optional[float] = None, return_details: bool = False):
    """Return (reranked predictions for q, reranked predictions for q'); with `return_details` also a third dict
    {qid: {"index", "rr_logit", "rr_logp", "ext_logp", "final", "gated"}} for the q view (used for the FIX/BREAK dump).
    Defaults reproduce the original pure-reranker behaviour."""
    preds_q: Dict[str, str] = {}
    preds_p: Dict[str, str] = {}
    details: Dict[str, Dict[str, Any]] = {}
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
            for qid, it, lg in zip(qids, items, rerank_logits(model, items, device)):
                k, info = select_index(it["scalars"][:, 0], lg, mode, alpha, override_margin, margin)
                store[qid] = it["texts"][k]
                if view == "candidates":
                    details[qid] = {"index": k, "rr_logit": lg, "rr_logp": info["rr_logp"], "ext_logp": info["ext_logp"], "final": info["final"], "gated": info["gated"]}
    return (preds_q, preds_p, details) if return_details else (preds_q, preds_p)
