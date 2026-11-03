#!/usr/bin/env python3
"""Generate and cache question paraphrases (proposal item 2).

Back-translates every question EN -> DE -> EN with MarianMT, scores (question, paraphrase) similarity with a
sentence-transformer and marks pairs as accepted when similarity >= threshold and the texts differ. Output is a
JSONL file consumed by train.py (--paraphrase_cache) and evaluate.py. The run is resumable.

Examples
  # full training set (GPU recommended; on CPU this takes a few hours but can be interrupted and resumed)
  python prepare_paraphrases.py --input provided/train-v1.1.json --output data/paraphrases/train.jsonl

  # dev set, needed for the consistency metric at evaluation time
  python prepare_paraphrases.py --input provided/dev-v1.1.json --output data/paraphrases/dev.jsonl

  # quick smoke test with the rule-based fallback
  python prepare_paraphrases.py --input provided/train-v1.1.json --output data/paraphrases/smoke.jsonl \
      --method heuristic --max_examples 300
"""

from __future__ import annotations

import argparse
import time

from g17qa.data import load_squad
from g17qa.paraphrase import generate_paraphrase_cache
from g17qa.utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="SQuAD-format JSON file")
    p.add_argument("--output", required=True, help="JSONL cache to write/append")
    p.add_argument("--method", choices=["marian", "heuristic"], default="marian")
    p.add_argument("--threshold", type=float, default=0.85, help="minimum cosine similarity to accept a pair")
    p.add_argument("--batch_size", type=int, default=32, help="translation batch size")
    p.add_argument("--chunk_size", type=int, default=512, help="questions per flush to disk")
    p.add_argument("--max_examples", type=int, default=0, help="0 = all")
    p.add_argument("--device", default="auto")
    p.add_argument("--no_resume", action="store_true", help="ignore rows already present in --output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    t0 = time.time()
    examples = load_squad(args.input, max_examples=args.max_examples or None)
    print(f"loaded {len(examples)} questions from {args.input}; device={device}")
    stats = generate_paraphrase_cache(
        examples, args.output, device, method=args.method, threshold=args.threshold,
        chunk_size=args.chunk_size, batch_size=args.batch_size, resume=not args.no_resume,
    )
    print(f"done in {time.time() - t0:.0f}s: generated {stats['generated']}, accepted {stats['accepted']} -> {args.output}")


if __name__ == "__main__":
    main()
