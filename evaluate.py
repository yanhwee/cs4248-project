#!/usr/bin/env python3
"""
CS4248 Group G17: SQuAD v1.1 Dev Set Evaluation Pipeline
Evaluates fine-tuned checkpoints on the held-out development set (data/dev-v1.1.json)
computing Exact Match (EM), F1 Score, Paraphrase Consistency Agreement, and Token Displacement.

Usage Examples:
  # Quick 100-sample dev set evaluation:
  python evaluate.py --weights weights/best_model.pt --samples 100

  # Full dev set evaluation on GPU:
  python evaluate.py --weights weights/model_dual.pt --samples 0 --fp16
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import logging as tf_logging
from transformers.utils.logging import disable_progress_bar

tf_logging.set_verbosity_error()
disable_progress_bar()

# Reusable components from core codebase
from app import (
    DEVICE,
    MarianParaphraser,
    ParaphraseConsistentQAExtractor,
    compute_exact_match,
    compute_f1,
    load_squad_data,
    predict_answer_pair_detailed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SQuAD v1.1 Checkpoints on Dev Set (CS4248 Group G17)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best_model.pt",
        help="Path to trained PyTorch model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dev-v1.1.json"),
        help="Path to SQuAD dev-v1.1.json file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google-bert/bert-base-uncased",
        help="Base model architecture",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of dev samples to evaluate (0 = all dev records)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="runs/evaluation_results.json",
        help="Path to save evaluation summary metrics",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable CUDA FP16 inference",
    )
    return parser.parse_args()


def load_checkpoint_model(weights_path: str, model_name: str, device: torch.device):
    model = ParaphraseConsistentQAExtractor(model_name=model_name).to(device)
    if os.path.exists(weights_path):
        print(f"Loading checkpoint weights from: {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            print(f"Warning: Unexpected checkpoint format, using initialized model.")
    else:
        print(f"Warning: Weights file {weights_path} not found. Running with base pretrained weights.")
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("CS4248 SQuAD v1.1 Dev Set Evaluation & Paraphrase Consistency Engine")
    print(f"Device:       {DEVICE}")
    print(f"Weights:      {args.weights}")
    print(f"Dev Set:      {args.data_file}")
    print(f"Max Samples:  {'All' if args.samples == 0 else args.samples}")
    print("=" * 72)

    if not os.path.exists(args.data_file):
        print(f"ERROR: Dev dataset file not found at: {args.data_file}")
        sys.exit(1)

    # 1. Load dev records
    max_rec = None if args.samples == 0 else args.samples
    print(f"Loading dev records from {args.data_file}...")
    dev_records = load_squad_data(args.data_file, max_records=max_rec)
    print(f"Loaded {len(dev_records)} evaluation instances.")

    # 2. Back-translate queries to obtain paraphrases
    print("Generating/verifying evaluation paraphrases in memory...")
    paraphraser = MarianParaphraser(device=DEVICE)
    questions = [rec["question"] for rec in dev_records]
    paraphrases = paraphraser.batch_back_translate(questions)

    # 3. Initialize Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = load_checkpoint_model(args.weights, args.model_name, DEVICE)

    # 4. Evaluation Loop
    em_scores: List[float] = []
    f1_scores: List[float] = []
    consistency_matches: List[bool] = []
    displacements: List[int] = []

    print("\nRunning inference and evaluation across dev records...")
    start_time = time.time()

    for idx, (rec, qp) in enumerate(tqdm(zip(dev_records, paraphrases), total=len(dev_records))):
        context = rec["context"]
        q = rec["question"]
        gold_ans = rec.get("answer_text", "")

        try:
            with torch.amp.autocast("cuda", enabled=args.fp16 and torch.cuda.is_available()):
                res = predict_answer_pair_detailed(
                    model=model,
                    tokenizer=tokenizer,
                    context=context,
                    question=q,
                    paraphrase=qp,
                    device=DEVICE,
                )
        except Exception as e:
            continue

        pred_q = res["ans_q"]
        pred_qp = res["ans_qp"]
        is_cons = res["is_consistent"]

        # EM and F1 evaluated on original question against gold standard
        em_val = compute_exact_match(pred_q, gold_ans)
        f1_val = compute_f1(pred_q, gold_ans)

        # Token displacement between original and paraphrased boundary predictions
        span_q = res.get("detail_q", {}).get("span", (0, 0))
        span_qp = res.get("detail_qp", {}).get("span", (0, 0))
        disp = abs(span_q[0] - span_qp[0]) + abs(span_q[1] - span_qp[1])

        em_scores.append(em_val)
        f1_scores.append(f1_val)
        consistency_matches.append(is_cons)
        displacements.append(disp)

    elapsed = time.time() - start_time
    total_eval = len(em_scores)

    if total_eval == 0:
        print("ERROR: No evaluation instances completed successfully.")
        return

    mean_em = 100.0 * np.mean(em_scores)
    mean_f1 = 100.0 * np.mean(f1_scores)
    agreement = 100.0 * np.mean(consistency_matches)
    mean_disp = float(np.mean(displacements))

    print("\n" + "=" * 72)
    print("CS4248 DEV SET EVALUATION BENCHMARK RESULTS")
    print(f"Total Evaluated Samples: {total_eval} | Time: {elapsed:.1f}s ({total_eval/elapsed:.1f} samples/s)")
    print(f"Exact Match (EM):        {mean_em:.2f}%")
    print(f"F1 Score:                {mean_f1:.2f}%")
    print(f"Paraphrase Agreement:    {agreement:.2f}%")
    print(f"Mean Token Displacement: {mean_disp:.2f} tokens")
    print("=" * 72)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    summary = {
        "weights": args.weights,
        "dev_samples": total_eval,
        "exact_match": round(mean_em, 2),
        "f1": round(mean_f1, 2),
        "consistency_agreement": round(agreement, 2),
        "mean_token_displacement": round(mean_disp, 2),
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved evaluation metrics to: {args.output_file}")


if __name__ == "__main__":
    main()
