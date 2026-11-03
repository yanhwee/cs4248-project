"""SQuAD data handling.

* `load_squad` keeps every gold answer (the dev set has up to six per question).
* `split_by_article` holds out a fraction of training *articles* for validation / hyper-parameter tuning,
  so no passage is shared between the training and validation portions.
* `attach_paraphrases` joins a paraphrase cache (JSONL written by `prepare_paraphrases.py`).
* `build_features` turns examples into sliding-window model inputs. For paired training the original question
  and its paraphrase are placed on the *same* passage chunks so their start/end distributions are comparable.
  Window labelling follows the Hugging Face question-answering example: a window gets the gold span only if the
  whole answer lies inside it, otherwise the [CLS] (index 0) null label.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import read_json, read_jsonl


# ---------------------------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------------------------
@dataclass
class SquadExample:
    qid: str
    title: str
    context: str
    question: str
    answers: List[str]
    answer_starts: List[int]
    paraphrase: Optional[str] = None  # accepted paraphrase, or None if none passed the filter

    @property
    def has_pair(self) -> bool:
        return bool(self.paraphrase) and self.paraphrase.strip() != self.question.strip()


def load_squad(path: str, max_examples: Optional[int] = None) -> List[SquadExample]:
    raw = read_json(path)
    out: List[SquadExample] = []
    for article in raw.get("data", []):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            ctx = para["context"]
            for qa in para["qas"]:
                answers = [a["text"] for a in qa.get("answers", [])]
                starts = [int(a["answer_start"]) for a in qa.get("answers", [])]
                if not answers:  # SQuAD v1.1 always has answers; skip unanswerable just in case
                    continue
                out.append(SquadExample(qa["id"], title, ctx, qa["question"], answers, starts))
                if max_examples and len(out) >= max_examples:
                    return out
    return out


def gold_answers(examples: Iterable[SquadExample]) -> Dict[str, List[str]]:
    return {ex.qid: list(ex.answers) for ex in examples}


def write_squad_subset(src_path: str, qids: Iterable[str], dst_path: str) -> int:
    """Copy a SQuAD JSON file keeping only the questions in `qids` (the official scorer requires a prediction for
    every question in the data file it is given). Returns the number of questions written."""
    keep = set(qids)
    raw = read_json(src_path)
    n = 0
    articles = []
    for article in raw.get("data", []):
        paras = []
        for para in article.get("paragraphs", []):
            qas = [qa for qa in para["qas"] if qa["id"] in keep]
            if qas:
                paras.append({**para, "qas": qas})
                n += len(qas)
        if paras:
            articles.append({**article, "paragraphs": paras})
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump({"version": raw.get("version", "1.1"), "data": articles}, f, ensure_ascii=False)
    return n


def split_by_article(
    examples: List[SquadExample], val_frac: float = 0.05, seed: int = 42
) -> Tuple[List[SquadExample], List[SquadExample]]:
    """Deterministic article-level split. Returns (train, val)."""
    if val_frac <= 0:
        return list(examples), []
    titles = sorted({ex.title for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(titles)
    n_val = max(1, int(math.ceil(val_frac * len(titles))))
    val_titles = set(titles[:n_val])
    train = [ex for ex in examples if ex.title not in val_titles]
    val = [ex for ex in examples if ex.title in val_titles]
    return train, val


def load_paired_files(para_path: str, orig_path: str) -> Tuple[List[SquadExample], Dict[str, str]]:
    """Gan & Ng (2019) paraphrased test sets come as two SQuAD-format files with *different, sequential* ids
    but identical ordering, contexts and answers. Returns examples built from the paraphrased file whose
    `paraphrase` field holds the corresponding original question, plus a map para_id -> orig_id."""
    para = load_squad(para_path)
    orig = load_squad(orig_path)
    if len(para) != len(orig):
        raise ValueError(f"{para_path} and {orig_path} have different sizes ({len(para)} vs {len(orig)})")
    id_map: Dict[str, str] = {}
    for p, o in zip(para, orig):
        if p.context != o.context or p.answers != o.answers:
            raise ValueError(f"positional mismatch between {para_path} and {orig_path} at id {p.qid}/{o.qid}")
        p.paraphrase = o.question
        id_map[p.qid] = o.qid
    return para, id_map


def attach_paraphrases(examples: List[SquadExample], cache_path: str) -> Dict[str, int]:
    """Attach accepted paraphrases from a JSONL cache ({id, paraphrase, accepted, ...}). Returns counts."""
    table: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(cache_path):
        table[row["id"]] = row
    n_attached = n_missing = n_rejected = 0
    for ex in examples:
        row = table.get(ex.qid)
        if row is None:
            n_missing += 1
            continue
        if row.get("accepted", False) and row.get("paraphrase"):
            ex.paraphrase = row["paraphrase"]
            n_attached += 1
        else:
            n_rejected += 1
    return {"attached": n_attached, "rejected": n_rejected, "missing": n_missing, "total": len(examples)}


# ---------------------------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------------------------
class ContextCache:
    """Tokenise each distinct context once (contexts repeat ~5x across questions)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._cache: Dict[str, Tuple[List[int], List[Tuple[int, int]]]] = {}

    def get(self, context: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        key = hashlib.md5(context.encode("utf-8")).hexdigest()
        hit = self._cache.get(key)
        if hit is None:
            enc = self.tok(context, add_special_tokens=False, return_offsets_mapping=True)
            hit = (enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]])
            self._cache[key] = hit
        return hit


def char_span_to_token_span(offsets: Sequence[Tuple[int, int]], char_start: int, char_end: int) -> Tuple[int, int]:
    """Smallest token span covering [char_start, char_end). Returns (-1, -1) if not coverable."""
    start_tok = end_tok = -1
    for t, (a, b) in enumerate(offsets):
        if a == b:  # empty offset (special / whitespace-only tokens)
            continue
        if start_tok == -1 and b > char_start:
            start_tok = t
        if a < char_end:
            end_tok = t
    if start_tok == -1 or end_tok == -1 or start_tok > end_tok:
        return -1, -1
    return start_tok, end_tok


N_SPECIAL = 3  # [CLS] question [SEP] passage [SEP]


def special_ids(tokenizer) -> Tuple[int, int]:
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError(
            f"tokenizer {type(tokenizer).__name__} has no [CLS]/[SEP] tokens; only BERT-style and DeBERTa "
            "tokenizers ([CLS] question [SEP] passage [SEP]) are supported"
        )
    return int(cls_id), int(sep_id)


def pack_pair(cls_id: int, sep_id: int, q_ids: List[int], chunk: List[int]) -> Tuple[List[int], List[int], int]:
    """Build [CLS] q [SEP] chunk [SEP] with BERT/DeBERTa segment ids. Returns (ids, token_type_ids, passage_start)."""
    ids = [cls_id] + q_ids + [sep_id] + chunk + [sep_id]
    tt = [0] * (len(q_ids) + 2) + [1] * (len(chunk) + 1)
    return ids, tt, len(q_ids) + 2


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------
@dataclass
class WindowFeature:
    qid: str
    window_index: int
    chunk_token_start: int           # offset of this window's first passage token in the full context tokens
    input_ids: np.ndarray            # int32 [L]
    token_type_ids: np.ndarray       # int32 [L]
    passage_start: int               # index of first passage token inside input_ids
    passage_len: int
    start_pos: int                   # label (0 = null/[CLS])
    end_pos: int
    has_pair: bool = False
    input_ids_p: Optional[np.ndarray] = None
    token_type_ids_p: Optional[np.ndarray] = None
    passage_start_p: int = 0


def build_features(
    examples: Sequence[SquadExample],
    tokenizer,
    max_seq_len: int = 384,
    doc_stride: int = 128,
    max_query_len: int = 64,
    with_pairs: bool = True,
    ctx_cache: Optional[ContextCache] = None,
) -> Tuple[List[WindowFeature], Dict[str, List[Tuple[int, int]]]]:
    """Create sliding-window features. `doc_stride` is the step between window starts (as in the original
    BERT run_squad.py). Returns (features, context_offsets_by_qid) — the offsets are needed to map token
    spans back to characters at decode time."""
    ctx_cache = ctx_cache or ContextCache(tokenizer)
    cls_id, sep_id = special_ids(tokenizer)
    feats: List[WindowFeature] = []
    offsets_by_qid: Dict[str, List[Tuple[int, int]]] = {}

    for ex in examples:
        c_ids, c_offsets = ctx_cache.get(ex.context)
        offsets_by_qid[ex.qid] = c_offsets
        q_ids = tokenizer(ex.question, add_special_tokens=False, truncation=True, max_length=max_query_len)["input_ids"]
        pair = bool(with_pairs and ex.has_pair)
        qp_ids = (
            tokenizer(ex.paraphrase, add_special_tokens=False, truncation=True, max_length=max_query_len)["input_ids"]
            if pair
            else []
        )
        budget = max_seq_len - max(len(q_ids), len(qp_ids)) - N_SPECIAL
        if budget < 16:
            raise ValueError(f"max_seq_len={max_seq_len} leaves no room for the passage (question {ex.qid})")

        # gold char span (first answer) -> token span over the full context
        if ex.answers:
            a_s = ex.answer_starts[0]
            a_e = a_s + len(ex.answers[0])
            g_start, g_end = char_span_to_token_span(c_offsets, a_s, a_e)
        else:
            g_start = g_end = -1

        m = len(c_ids)
        starts = list(range(0, max(1, m - budget + 1), doc_stride))
        if starts[-1] + budget < m:
            starts.append(m - budget)  # final window always reaches the end of the context
        starts = sorted(set(max(0, s) for s in starts))

        for w, cs in enumerate(starts):
            chunk = c_ids[cs : cs + budget]
            if not chunk:
                continue
            ids, tt, p_start = pack_pair(cls_id, sep_id, q_ids, chunk)
            ce = cs + len(chunk)
            if g_start >= cs and g_end < ce:
                s_lab = p_start + (g_start - cs)
                e_lab = p_start + (g_end - cs)
            else:
                s_lab = e_lab = 0
            f = WindowFeature(
                qid=ex.qid,
                window_index=w,
                chunk_token_start=cs,
                input_ids=np.asarray(ids, dtype=np.int32),
                token_type_ids=np.asarray(tt, dtype=np.int32),
                passage_start=p_start,
                passage_len=len(chunk),
                start_pos=s_lab,
                end_pos=e_lab,
            )
            if pair:
                ids_p, tt_p, p_start_p = pack_pair(cls_id, sep_id, qp_ids, chunk)
                f.has_pair = True
                f.input_ids_p = np.asarray(ids_p, dtype=np.int32)
                f.token_type_ids_p = np.asarray(tt_p, dtype=np.int32)
                f.passage_start_p = p_start_p
            feats.append(f)
    return feats, offsets_by_qid


class WindowDataset(Dataset):
    def __init__(self, features: List[WindowFeature]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> WindowFeature:
        return self.features[i]


def _pad(arrs: List[np.ndarray], pad_value: int) -> torch.Tensor:
    L = max(len(a) for a in arrs)
    out = np.full((len(arrs), L), pad_value, dtype=np.int64)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    return torch.from_numpy(out)


def make_collate_fn(pad_token_id: int, use_token_type_ids: bool):
    """Dynamic-padding collate. Paired inputs are padded independently; `pair_mask` marks rows with a pair."""

    def collate(batch: List[WindowFeature]) -> Dict[str, Any]:
        ids = _pad([f.input_ids for f in batch], pad_token_id)
        out: Dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": _pad([np.ones(len(f.input_ids), dtype=np.int32) for f in batch], 0),
            "passage_start": torch.tensor([f.passage_start for f in batch], dtype=torch.long),
            "passage_len": torch.tensor([f.passage_len for f in batch], dtype=torch.long),
            "start_positions": torch.tensor([f.start_pos for f in batch], dtype=torch.long),
            "end_positions": torch.tensor([f.end_pos for f in batch], dtype=torch.long),
            "pair_mask": torch.tensor([f.has_pair for f in batch], dtype=torch.bool),
            "qids": [f.qid for f in batch],
            "window_index": [f.window_index for f in batch],
            "chunk_token_start": [f.chunk_token_start for f in batch],
        }
        if use_token_type_ids:
            out["token_type_ids"] = _pad([f.token_type_ids for f in batch], 0)
        if any(f.has_pair for f in batch):
            pids = [f.input_ids_p if f.has_pair else f.input_ids for f in batch]
            ids_p = _pad(pids, pad_token_id)
            out["input_ids_p"] = ids_p
            out["attention_mask_p"] = _pad([np.ones(len(a), dtype=np.int32) for a in pids], 0)
            out["passage_start_p"] = torch.tensor(
                [f.passage_start_p if f.has_pair else f.passage_start for f in batch], dtype=torch.long
            )
            if use_token_type_ids:
                out["token_type_ids_p"] = _pad(
                    [f.token_type_ids_p if f.has_pair else f.token_type_ids for f in batch], 0
                )
        return out

    return collate


def group_features_by_qid(features: Sequence[WindowFeature]) -> Dict[str, List[WindowFeature]]:
    groups: Dict[str, List[WindowFeature]] = {}
    for f in features:
        groups.setdefault(f.qid, []).append(f)
    return groups
