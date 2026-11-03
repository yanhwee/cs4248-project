"""Inference: run the extractor over sliding windows and decode each question into its top-k *distinct*
candidate spans (proposal item 1: "keep each question's top 20 distinct spans").

Scoring. Within a window the joint span score is start_logit[i] + end_logit[j] with j >= i and
j - i < max_answer_len, restricted to passage tokens. Windows are made comparable by subtracting the window's
null score start_logit[0] + end_logit[0] (the [CLS] label used for windows that do not contain the answer),
and the best score per distinct answer text is kept.

Paired decoding. When examples carry a paraphrase, both views are run on the same chunks. Each candidate of
one view also receives the score the *other* view assigns to the same span (`pair_score`), which the reranker
can use as a consistency feature, and both views get their own top-1 prediction for the consistency metric.

Reranker features. With `want_vecs=True` every candidate carries the concatenation
[h_start ; h_end ; mean(h_span) ; h_cls] of the encoder's final hidden states (float16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SquadExample, WindowDataset, WindowFeature, build_features, make_collate_fn
from .metrics import normalize_answer
from .model import SpanExtractor, tokenizer_uses_token_type_ids


@dataclass
class Candidate:
    text: str
    char_start: int
    char_end: int
    score: float
    raw_score: float
    null_score: float
    start_logit: float
    end_logit: float
    window_index: int
    tok_start: int
    tok_end: int
    rank: int = 0
    pair_score: Optional[float] = None
    vec: Optional[np.ndarray] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "score": round(self.score, 4),
            "raw_score": round(self.raw_score, 4),
            "null_score": round(self.null_score, 4),
            "window_index": self.window_index,
            "tok_start": self.tok_start,
            "tok_end": self.tok_end,
            "rank": self.rank,
            "pair_score": None if self.pair_score is None else round(self.pair_score, 4),
        }


@dataclass
class QuestionOutput:
    qid: str
    n_windows: int
    candidates: List[Candidate]
    candidates_p: List[Candidate] = field(default_factory=list)
    paired: bool = False

    @property
    def pred(self) -> str:
        return self.candidates[0].text if self.candidates else ""

    @property
    def pred_p(self) -> str:
        return self.candidates_p[0].text if self.candidates_p else ""


# ---------------------------------------------------------------------------------------------
# Span enumeration
# ---------------------------------------------------------------------------------------------
def topn_spans(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    passage_start: int,
    passage_len: int,
    n: int,
    max_answer_len: int,
) -> List[Tuple[int, int, float]]:
    """Top-n (i_local, j_local, raw_score) over passage tokens of one window."""
    s = torch.from_numpy(start_logits[passage_start : passage_start + passage_len])
    e = torch.from_numpy(end_logits[passage_start : passage_start + passage_len])
    m = s.shape[0]
    if m == 0:
        return []
    mat = s.unsqueeze(1) + e.unsqueeze(0)
    idx = torch.arange(m)
    valid = (idx.unsqueeze(0) >= idx.unsqueeze(1)) & ((idx.unsqueeze(0) - idx.unsqueeze(1)) < max_answer_len)
    flat = mat.masked_fill(~valid, float("-inf")).flatten()
    n = min(n, int(valid.sum().item()))
    scores, order = torch.topk(flat, n)
    return [(int(t // m), int(t % m), float(v)) for t, v in zip(order.tolist(), scores.tolist())]


def _span_vec(hidden: np.ndarray, p_start: int, i: int, j: int) -> np.ndarray:
    a, b = p_start + i, p_start + j
    return np.concatenate([hidden[a], hidden[b], hidden[a : b + 1].mean(0), hidden[0]]).astype(np.float16)


def decode_candidates(
    windows: Sequence[Tuple[WindowFeature, np.ndarray, np.ndarray, Optional[np.ndarray], int]],
    other_windows: Optional[Sequence[Tuple[np.ndarray, np.ndarray, int]]],
    context: str,
    offsets: Sequence[Tuple[int, int]],
    top_k: int,
    max_answer_len: int,
) -> List[Candidate]:
    """windows: (feature, start_logits, end_logits, hidden|None, passage_start) for the view being decoded.
    other_windows: (start_logits, end_logits, passage_start) of the paired view, same order, or None."""
    best: Dict[str, Candidate] = {}
    for w_idx, (f, s_log, e_log, hid, p_start) in enumerate(windows):
        null = float(s_log[0] + e_log[0])
        other = other_windows[w_idx] if other_windows is not None else None
        other_null = float(other[0][0] + other[1][0]) if other is not None else None
        for i, j, raw in topn_spans(s_log, e_log, p_start, f.passage_len, 2 * top_k, max_answer_len):
            g_i, g_j = f.chunk_token_start + i, f.chunk_token_start + j
            cs, ce = offsets[g_i][0], offsets[g_j][1]
            text = context[cs:ce].strip()
            key = normalize_answer(text)
            if not key:
                continue
            score = raw - null
            prev = best.get(key)
            if prev is not None and prev.score >= score:
                continue
            cand = Candidate(
                text=text, char_start=cs, char_end=ce, score=score, raw_score=raw, null_score=null,
                start_logit=float(s_log[p_start + i]), end_logit=float(e_log[p_start + j]),
                window_index=f.window_index, tok_start=g_i, tok_end=g_j,
            )
            if other is not None:
                o_s, o_e, o_p = other
                cand.pair_score = float(o_s[o_p + i] + o_e[o_p + j]) - other_null
            if hid is not None:
                cand.vec = _span_vec(hid, p_start, i, j)
            best[key] = cand
    cands = sorted(best.values(), key=lambda c: c.score, reverse=True)[:top_k]
    for r, c in enumerate(cands):
        c.rank = r
    return cands


# ---------------------------------------------------------------------------------------------
# Batched extractor inference
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def run_extractor(
    model: SpanExtractor,
    tokenizer,
    examples: Sequence[SquadExample],
    device: torch.device,
    batch_size: int = 32,
    max_seq_len: int = 384,
    doc_stride: int = 128,
    max_query_len: int = 64,
    top_k: int = 20,
    max_answer_len: int = 30,
    with_pairs: bool = False,
    want_vecs: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    progress: bool = True,
    num_workers: int = 0,
) -> Dict[str, QuestionOutput]:
    """Decode every example. Windows of one question are contiguous in feature order, so questions are decoded
    in a streaming fashion and hidden states never accumulate beyond one question."""
    model.eval()
    use_tt = tokenizer_uses_token_type_ids(tokenizer)
    feats, offsets_by_qid = build_features(
        examples, tokenizer, max_seq_len=max_seq_len, doc_stride=doc_stride, max_query_len=max_query_len, with_pairs=with_pairs
    )
    ex_by_qid = {ex.qid: ex for ex in examples}
    loader = DataLoader(
        WindowDataset(feats), batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=make_collate_fn(tokenizer.pad_token_id, use_tt),
    )
    it = loader
    if progress:
        from tqdm import tqdm

        it = tqdm(loader, desc="extract", unit="batch", leave=False)

    outputs: Dict[str, QuestionOutput] = {}
    buf_q: List[Tuple[WindowFeature, np.ndarray, np.ndarray, Optional[np.ndarray], int]] = []
    buf_p: List[Tuple[np.ndarray, np.ndarray, int]] = []
    buf_hid_p: List[Optional[np.ndarray]] = []
    cur_qid: Optional[str] = None

    def flush() -> None:
        if cur_qid is None or not buf_q:
            return
        ex = ex_by_qid[cur_qid]
        paired = bool(buf_p) and ex.has_pair
        cands = decode_candidates(buf_q, buf_p if paired else None, ex.context, offsets_by_qid[cur_qid], top_k, max_answer_len)
        cands_p: List[Candidate] = []
        if paired:
            win_p = [(f, s, e, h, p) for (f, _, _, _, _), (s, e, p), h in zip(buf_q, buf_p, buf_hid_p)]
            other_q = [(s, e, p) for (_, s, e, _, p) in buf_q]
            cands_p = decode_candidates(win_p, other_q, ex.context, offsets_by_qid[cur_qid], top_k, max_answer_len)
        outputs[cur_qid] = QuestionOutput(cur_qid, len(buf_q), cands, cands_p, paired)
        buf_q.clear()
        buf_p.clear()
        buf_hid_p.clear()

    def fwd(ids, am, tt):
        kwargs = {"input_ids": ids.to(device), "attention_mask": am.to(device)}
        if tt is not None:
            kwargs["token_type_ids"] = tt.to(device)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                out = model(**kwargs, return_hidden=want_vecs)
        else:
            out = model(**kwargs, return_hidden=want_vecs)
        s = out["start_logits"].float().cpu().numpy()
        e = out["end_logits"].float().cpu().numpy()
        h = out["hidden"].to(torch.float16).cpu().numpy() if want_vecs else None
        return s, e, h

    pos = 0  # the loader is not shuffled, so rows arrive in exactly the order of `feats`
    for batch in it:
        s_q, e_q, h_q = fwd(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
        has_p = with_pairs and "input_ids_p" in batch
        if has_p:
            s_p, e_p, h_p = fwd(batch["input_ids_p"], batch["attention_mask_p"], batch.get("token_type_ids_p"))
        pm = batch["pair_mask"].tolist()
        for b, qid in enumerate(batch["qids"]):
            f = feats[pos]
            pos += 1
            if f.qid != qid or f.window_index != batch["window_index"][b]:
                raise RuntimeError("feature order mismatch in run_extractor (loader must not shuffle)")
            if qid != cur_qid:
                flush()
                cur_qid = qid
            buf_q.append((f, s_q[b], e_q[b], h_q[b] if h_q is not None else None, int(batch["passage_start"][b])))
            if has_p and pm[b]:
                buf_p.append((s_p[b], e_p[b], int(batch["passage_start_p"][b])))
                buf_hid_p.append(h_p[b] if h_p is not None else None)
    flush()
    return outputs


def predictions_from_outputs(outputs: Dict[str, QuestionOutput], paired: bool = False) -> Dict[str, str]:
    return {qid: (o.pred_p if paired else o.pred) for qid, o in outputs.items()}
