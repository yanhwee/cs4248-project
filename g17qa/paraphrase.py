"""Question paraphrasing for consistency training.

Primary method (proposal item 2): neural back-translation EN -> DE -> EN with the open MarianMT models
`Helsinki-NLP/opus-mt-en-de` and `Helsinki-NLP/opus-mt-de-en`, batched, followed by a sentence-embedding
similarity filter (`sentence-transformers/all-MiniLM-L6-v2`): a pair is accepted when it is not identical to
the original after normalisation and its cosine similarity is at least `threshold`.

A rule-based `HeuristicParaphraser` is kept only for fast smoke tests (`--method heuristic`).

`generate_paraphrase_cache` writes one JSONL row per question and is resumable, because back-translating the
full training set on CPU takes a while.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .data import SquadExample
from .metrics import normalize_answer
from .utils import append_jsonl, batched, read_jsonl

MARIAN_EN_DE = "Helsinki-NLP/opus-mt-en-de"
MARIAN_DE_EN = "Helsinki-NLP/opus-mt-de-en"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HeuristicParaphraser:
    """Regex rewrites of interrogatives (smoke-test fallback only)."""

    RULES = [
        (r"^To whom (was|is|did) (.*)\?$", r"Whom \1 \2 to?"),
        (r"^(In what year|During what year|At what time) (did|was|is|were) (.*)\?$", r"When \2 \3?"),
        (r"^When (did|was|is|were) (.*)\?$", r"In what year \1 \2?"),
        (r"^Where (was|did|is|were) (.*)\?$", r"In what place \1 \2?"),
        (r"^How many (.*)\?$", r"What is the number of \1?"),
        (r"^How much (.*)\?$", r"What amount of \1?"),
        (r"^Who (was|is|were) (.*)\?$", r"Which person \1 \2?"),
        (r"^Who (.*)\?$", r"Which person \1?"),
        (r"^What (is|was) (.*) (called|named)\?$", r"By what name \1 \2 known?"),
        (r"^What (is|was|are|were) (.*)\?$", r"Can you say what \2 \1?"),
        (r"^What (did|does|do) (.*)\?$", r"Which thing \1 \2?"),
        (r"^Why (did|was|is|were|do|does) (.*)\?$", r"For what reason \1 \2?"),
        (r"^Which (.*)\?$", r"What particular \1?"),
        (r"^How (did|does|do|can|is|was) (.*)\?$", r"In what way \1 \2?"),
    ]
    SUBS = [
        (r"\bWhen\b", "At what time"), (r"\bWhere\b", "In what place"), (r"\bWho\b", "Which person"),
        (r"\bWhat\b", "Which"), (r"\bWhy\b", "For what reason"), (r"\bHow\b", "In what way"),
        (r"\bfounded\b", "established"), (r"\bcalled\b", "named"), (r"\blocated\b", "situated"),
    ]

    def paraphrase(self, question: str) -> str:
        q = question.strip()
        for pat, repl in self.RULES:
            if re.search(pat, q, flags=re.IGNORECASE):
                t = re.sub(pat, repl, q, flags=re.IGNORECASE).strip()
                if t.lower() != q.lower():
                    return t[0].upper() + t[1:]
        for pat, repl in self.SUBS:
            if re.search(pat, q, flags=re.IGNORECASE):
                t = re.sub(pat, repl, q, count=1, flags=re.IGNORECASE).strip()
                if t.lower() != q.lower():
                    return t[0].upper() + t[1:]
        return q

    def batch(self, questions: Sequence[str]) -> List[str]:
        return [self.paraphrase(q) for q in questions]


class MarianBackTranslator:
    def __init__(
        self,
        device: torch.device,
        src_model: str = MARIAN_EN_DE,
        tgt_model: str = MARIAN_DE_EN,
        batch_size: int = 32,
        num_beams: int = 4,
        max_length: int = 64,
        retry_with_sampling: bool = True,
    ):
        from transformers import MarianMTModel, MarianTokenizer

        self.device = device
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_length = max_length
        self.retry_with_sampling = retry_with_sampling
        self.tok_fwd = MarianTokenizer.from_pretrained(src_model)
        self.mt_fwd = MarianMTModel.from_pretrained(src_model).to(device).eval()
        self.tok_bwd = MarianTokenizer.from_pretrained(tgt_model)
        self.mt_bwd = MarianMTModel.from_pretrained(tgt_model).to(device).eval()

    @torch.no_grad()
    def _translate(self, texts: Sequence[str], tok, model, sample: bool = False) -> List[str]:
        out: List[Optional[str]] = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length bucketing
        for chunk in batched(order, self.batch_size):
            enc = tok([texts[i] for i in chunk], return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            gen_kwargs = dict(max_length=self.max_length)
            if sample:
                gen_kwargs.update(do_sample=True, top_k=50, top_p=0.95, temperature=1.0, num_beams=1)
            else:
                gen_kwargs.update(num_beams=self.num_beams)
            gen = model.generate(**enc, **gen_kwargs)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            for i, t in zip(chunk, dec):
                out[i] = t.strip()
        return [o or "" for o in out]

    def back_translate(self, questions: Sequence[str]) -> List[str]:
        de = self._translate(questions, self.tok_fwd, self.mt_fwd)
        en = self._translate(de, self.tok_bwd, self.mt_bwd)
        if self.retry_with_sampling:
            same = [i for i, (q, p) in enumerate(zip(questions, en)) if normalize_answer(q) == normalize_answer(p)]
            if same:
                de2 = self._translate([questions[i] for i in same], self.tok_fwd, self.mt_fwd, sample=True)
                en2 = self._translate(de2, self.tok_bwd, self.mt_bwd, sample=True)
                for i, p in zip(same, en2):
                    if p:
                        en[i] = p
        return en


class SimilarityFilter:
    def __init__(self, device: torch.device, model_name: str = SBERT_MODEL, batch_size: int = 128):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=str(device))
        self.batch_size = batch_size

    @torch.no_grad()
    def similarities(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray:
        ea = self.model.encode(list(a), batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True)
        eb = self.model.encode(list(b), batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return (ea * eb).sum(-1)


def accept_pair(question: str, paraphrase: str, similarity: float, threshold: float) -> bool:
    if not paraphrase or not paraphrase.strip():
        return False
    if normalize_answer(question) == normalize_answer(paraphrase):
        return False
    return float(similarity) >= threshold


def existing_cache_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    return {row["id"] for row in read_jsonl(path)}


def generate_paraphrase_cache(
    examples: Sequence[SquadExample],
    out_path: str,
    device: torch.device,
    method: str = "marian",
    threshold: float = 0.85,
    chunk_size: int = 512,
    batch_size: int = 32,
    resume: bool = True,
    log=print,
) -> Dict[str, int]:
    """Generate paraphrases for `examples` and append rows to `out_path` (JSONL). Resumable."""
    done = existing_cache_ids(out_path) if resume else set()
    todo = [ex for ex in examples if ex.qid not in done]
    log(f"paraphrase cache {out_path}: {len(done)} already done, {len(todo)} to generate (method={method})")
    if not todo:
        return {"generated": 0, "accepted": 0}

    if method == "marian":
        generator = MarianBackTranslator(device, batch_size=batch_size)
        gen_fn = generator.back_translate
    elif method == "heuristic":
        generator = HeuristicParaphraser()
        gen_fn = generator.batch
    else:
        raise ValueError("method must be 'marian' or 'heuristic'")
    sim_filter = SimilarityFilter(device)

    n_gen = n_acc = 0
    for chunk in batched(list(todo), chunk_size):
        qs = [ex.question for ex in chunk]
        ps = gen_fn(qs)
        sims = sim_filter.similarities(qs, ps)
        rows = []
        for ex, p, s in zip(chunk, ps, sims):
            ok = accept_pair(ex.question, p, float(s), threshold)
            rows.append({"id": ex.qid, "question": ex.question, "paraphrase": p, "similarity": round(float(s), 4), "accepted": ok, "method": method})
            n_acc += int(ok)
        append_jsonl(rows, out_path)
        n_gen += len(rows)
        log(f"  generated {n_gen}/{len(todo)}  accepted so far {n_acc} ({100.0 * n_acc / max(1, n_gen):.1f}%)")
    return {"generated": n_gen, "accepted": n_acc}


def load_paraphrase_table(path: str) -> Dict[str, Dict]:
    return {row["id"]: row for row in read_jsonl(path)}
