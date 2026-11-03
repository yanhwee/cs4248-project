"""Training objectives.

Supervised term (per window, HF convention):   L_CE = 0.5 * [CE(start) + CE(end)]
Paired objective from the proposal:            L = CE(q) + CE(q') + lambda * KL_sym(P_q, P_q')
where P are start / end distributions over the *passage tokens only* (question and special tokens excluded),
computed for the original question and its paraphrase on the same passage chunk.

Consistency regularisers (all applied to both start and end marginals and averaged):
  symmetric_kl        0.5 * [KL(P||Q) + KL(Q||P)]                      (proposal item 2)
  wasserstein_1d      (1/m) * sum_i |CDF_P(i) - CDF_Q(i)|               (proposal item 2a, 1-D optimal transport)
  hungarian_span_loss bipartite matching of top-k joint (start,end) spans (proposal item 2a, span level)

Loss modes understood by `compute_total_loss`:
  ce         CE(q)                                   plain baseline, paraphrases unused
  aug        CE(q) + CE(q')                          paraphrase augmentation only
  kl         CE(q) + CE(q') + lambda_kl * KL
  ot         CE(q) + CE(q') + lambda_ot * W1
  hungarian  CE(q) + CE(q') + mu_hung * Hungarian
  dual       CE(q) + CE(q') + lambda_ot * W1 + mu_hung * Hungarian
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

LOSS_MODES = ("ce", "aug", "kl", "ot", "hungarian", "dual")
PAIRED_MODES = ("aug", "kl", "ot", "hungarian", "dual")
NEG = -1e4


def span_cross_entropy(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    ls = F.cross_entropy(start_logits.float(), start_positions, reduction=reduction)
    le = F.cross_entropy(end_logits.float(), end_positions, reduction=reduction)
    return 0.5 * (ls + le)


def gather_passage_logits(
    logits: torch.Tensor, passage_start: torch.Tensor, passage_len: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice the passage positions out of [B, L] logits into a padded [B, M] tensor plus a validity mask."""
    B, L = logits.shape
    M = int(passage_len.max().item())
    ar = torch.arange(M, device=logits.device)
    idx = (passage_start.unsqueeze(1) + ar.unsqueeze(0)).clamp(max=L - 1)
    mask = ar.unsqueeze(0) < passage_len.unsqueeze(1)
    return logits.float().gather(1, idx), mask


def masked_log_softmax(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(x.masked_fill(~mask, NEG), dim=-1)


def symmetric_kl(logp: torch.Tensor, logq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Row-wise 0.5*[KL(P||Q)+KL(Q||P)] from log-probabilities; returns [B]."""
    p, q = logp.exp(), logq.exp()
    term = (p - q) * (logp - logq)
    return 0.5 * (term * mask).sum(-1)


def wasserstein_1d(logp: torch.Tensor, logq: torch.Tensor, mask: torch.Tensor, passage_len: torch.Tensor) -> torch.Tensor:
    """Row-wise 1-D W1 between distributions on an ordered token lattice with ground cost |i-j|/m; returns [B]."""
    p = logp.exp() * mask
    q = logq.exp() * mask
    diff = (p.cumsum(-1) - q.cumsum(-1)).abs() * mask
    return diff.sum(-1) / passage_len.clamp(min=1).float()


def _topk_joint_spans(s: torch.Tensor, e: torch.Tensor, k: int, max_span_len: int) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    m = s.shape[0]
    mat = s.unsqueeze(1) + e.unsqueeze(0)
    i = torch.arange(m, device=s.device)
    valid = (i.unsqueeze(0) >= i.unsqueeze(1)) & ((i.unsqueeze(0) - i.unsqueeze(1)) < max_span_len)
    flat = mat.masked_fill(~valid, NEG).flatten()
    k = min(k, int(valid.sum().item()))
    scores, idx = torch.topk(flat, k)
    spans = [(int(t // m), int(t % m)) for t in idx.tolist()]
    return scores, spans


def hungarian_span_loss(
    s_q: torch.Tensor, e_q: torch.Tensor, s_p: torch.Tensor, e_p: torch.Tensor, k: int = 5, max_span_len: int = 30
) -> torch.Tensor:
    """Bipartite matching between the top-k joint spans of q and q'. Cost = 0.5*position distance + 0.5*(1-IoU).
    The assignment is discrete (no gradient); the loss pushes matched candidates to agree in probability and
    down-weights probability mass on poorly matched spans."""
    from scipy.optimize import linear_sum_assignment

    m = s_q.shape[0]
    sc_q, sp_q = _topk_joint_spans(s_q, e_q, k, max_span_len)
    sc_p, sp_p = _topk_joint_spans(s_p, e_p, k, max_span_len)
    kk = min(len(sp_q), len(sp_p))
    if kk == 0:
        return s_q.new_zeros(())
    p_q = F.softmax(sc_q[:kk], 0)
    p_p = F.softmax(sc_p[:kk], 0)
    C = np.zeros((kk, kk), dtype=np.float32)
    for a in range(kk):
        ia, ja = sp_q[a]
        for b in range(kk):
            ib, jb = sp_p[b]
            d_pos = (abs(ia - ib) + abs(ja - jb)) / (2.0 * max(1, m))
            inter = max(0, min(ja, jb) - max(ia, ib) + 1)
            union = (ja - ia + 1) + (jb - ib + 1) - inter
            C[a, b] = 0.5 * d_pos + 0.5 * (1.0 - inter / max(1, union))
    rows, cols = linear_sum_assignment(C)
    c = torch.tensor(C[rows, cols], device=s_q.device)
    p_p_matched = p_p[torch.as_tensor(cols, device=s_q.device)]
    agree = 0.5 * ((1.0 - c) * (p_q - p_p_matched).abs()).sum()
    cost = (c * 0.5 * (p_q + p_p_matched)).sum()
    return agree + cost


def compute_total_loss(
    mode: str,
    out_q: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    out_p: Optional[Dict[str, torch.Tensor]] = None,
    lambda_kl: float = 1.0,
    lambda_ot: float = 0.5,
    mu_hung: float = 0.1,
    hung_topk: int = 5,
    max_answer_len: int = 30,
) -> Dict[str, torch.Tensor]:
    """Returns a dict with 'total' and the individual (unweighted) components for logging."""
    if mode not in LOSS_MODES:
        raise ValueError(f"unknown loss mode {mode!r}; choose from {LOSS_MODES}")
    sp, ep = batch["start_positions"], batch["end_positions"]
    ce_q = span_cross_entropy(out_q["start_logits"], out_q["end_logits"], sp, ep)
    comps: Dict[str, torch.Tensor] = {"ce_q": ce_q}
    total = ce_q
    if mode == "ce" or out_p is None:
        comps["total"] = total
        return comps

    pair = batch["pair_mask"]
    n_pair = int(pair.sum().item())
    zero = ce_q.new_zeros(())
    if n_pair == 0:
        comps.update({"ce_p": zero, "kl": zero, "ot": zero, "hung": zero, "total": total})
        return comps

    # CE on the paraphrase view. Both views share the passage chunk, but the labels are *sequence* positions and the
    # paraphrase's passage starts at passage_start_p (its question has a different token length), so non-null labels
    # must be shifted; the null label (0 = [CLS]) stays 0.
    shift = batch["passage_start_p"] - batch["passage_start"]
    sp_p = torch.where(sp > 0, sp + shift, sp)
    ep_p = torch.where(ep > 0, ep + shift, ep)
    ce_p_rows = span_cross_entropy(out_p["start_logits"], out_p["end_logits"], sp_p, ep_p, reduction="none")
    ce_p = (ce_p_rows * pair.float()).sum() / n_pair
    comps["ce_p"] = ce_p
    total = total + ce_p

    if mode == "aug":
        comps["total"] = total
        return comps

    # passage-token distributions for both views
    s_q, mask = gather_passage_logits(out_q["start_logits"], batch["passage_start"], batch["passage_len"])
    e_q, _ = gather_passage_logits(out_q["end_logits"], batch["passage_start"], batch["passage_len"])
    s_p, _ = gather_passage_logits(out_p["start_logits"], batch["passage_start_p"], batch["passage_len"])
    e_p, _ = gather_passage_logits(out_p["end_logits"], batch["passage_start_p"], batch["passage_len"])
    ls_q, le_q = masked_log_softmax(s_q, mask), masked_log_softmax(e_q, mask)
    ls_p, le_p = masked_log_softmax(s_p, mask), masked_log_softmax(e_p, mask)
    pf = pair.float()

    if mode == "kl":
        kl = 0.5 * (symmetric_kl(ls_q, ls_p, mask) + symmetric_kl(le_q, le_p, mask))
        kl = (kl * pf).sum() / n_pair
        comps["kl"] = kl
        total = total + lambda_kl * kl
    if mode in ("ot", "dual"):
        ot = 0.5 * (
            wasserstein_1d(ls_q, ls_p, mask, batch["passage_len"]) + wasserstein_1d(le_q, le_p, mask, batch["passage_len"])
        )
        ot = (ot * pf).sum() / n_pair
        comps["ot"] = ot
        total = total + lambda_ot * ot
    if mode in ("hungarian", "dual"):
        hs = []
        for b in range(s_q.shape[0]):
            if not bool(pair[b]):
                continue
            m = int(batch["passage_len"][b].item())
            hs.append(hungarian_span_loss(s_q[b, :m], e_q[b, :m], s_p[b, :m], e_p[b, :m], hung_topk, max_answer_len))
        hung = torch.stack(hs).mean() if hs else zero
        comps["hung"] = hung
        total = total + mu_hung * hung
    comps["total"] = total
    return comps
