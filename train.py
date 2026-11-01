#!/usr/bin/env python3
"""
CS4248 Group G17: Paraphrase-Consistent Extractive Question Answering
Standalone Cluster Training Pipeline & CLI Engine

This script runs headless training on local workstations or cluster GPU nodes.
It imports the core model architecture and loss functions from app.py, executes
the optimization loop with CUDA FP16 AMP, and flushes real-time telemetry to disk.

Usage Examples:
  # Quick 50-sample local smoke test:
  python train.py --samples 50 --epochs 1 --batch_size 4

  # Full cluster training run with Proposed Dual Regularization:
  python train.py --loss_type dual --epochs 2 --batch_size 16 --fp16

  # Baseline run for ablation study:
  python train.py --loss_type ce_only --epochs 2 --batch_size 16 --fp16
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import logging as tf_logging
from transformers.utils.logging import disable_progress_bar

# Suppress Hugging Face verbosity
tf_logging.set_verbosity_error()
disable_progress_bar()

# Reusable imports from core academic codebase
from app import (
    DEVICE,
    MarianParaphraser,
    PairedQAFeatureDataset,
    ParaphraseConsistentQAExtractor,
    compute_1d_wasserstein_loss,
    compute_exact_match,
    compute_f1,
    compute_hungarian_loss_single_sample,
    compute_supervised_ce_loss,
    compute_symmetric_kl_loss,
    load_squad_data,
    paired_collate_fn,
    prepare_paired_sliding_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Paraphrase-Consistent Extractive QA Models (CS4248 Group G17)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model architecture
    parser.add_argument(
        "--model",
        type=str,
        default="google-bert/bert-base-uncased",
        help="Pretrained HuggingFace encoder backbone",
    )
    # Loss formulation
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["dual", "kl", "ot", "hungarian", "ce_only"],
        default="dual",
        help="Ablation loss objective: dual | kl | ot | hungarian | ce_only",
    )
    # Hyperparameters
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of training samples (0 = full SQuAD dataset)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per device",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Encoder backbone learning rate",
    )
    parser.add_argument(
        "--lambda_ot",
        type=float,
        default=0.5,
        help="Weight multiplier for 1D Optimal Transport loss",
    )
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=0.5,
        help="Weight multiplier for Symmetric KL Divergence loss",
    )
    parser.add_argument(
        "--mu_hung",
        type=float,
        default=0.1,
        help="Weight multiplier for Hungarian bipartite span matching loss",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top spans matched via Hungarian algorithm",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256,
        help="Maximum token sequence length per sliding window",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=96,
        help="Sliding window token stride overlap",
    )
    # Optimization & Hardware
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable CUDA Automatic Mixed Precision (AMP FP16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    # Logging & Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="Path to directory containing train-v1.1.json and dev-v1.1.json",
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Directory to save step telemetry JSON logs",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="weights",
        help="Directory to save model checkpoint weights",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Interval of steps for console printing and disk flush",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.runs_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    print("=" * 72)
    print("CS4248 SQuAD Paraphrase-Consistent Extractive QA Training Engine")
    print(f"Device:       {DEVICE}")
    print(f"Model:        {args.model}")
    print(f"Loss Type:    {args.loss_type}")
    print(f"Epochs:       {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Weights:      lambda_ot={args.lambda_ot}, lambda_kl={args.lambda_kl}, mu_hung={args.mu_hung}")
    print(f"AMP FP16:     {args.fp16 and torch.cuda.is_available()}")
    print("=" * 72)

    # 1. Dataset Loading
    train_path = os.path.join(args.data_dir, "train-v1.1.json")
    if not os.path.exists(train_path):
        print(f"ERROR: Dataset not found at {train_path}")
        sys.exit(1)

    print(f"Loading raw SQuAD records from {train_path}...")
    max_rec = None if args.samples == 0 else args.samples
    raw_train = load_squad_data(train_path, max_records=max_rec)
    print(f"Loaded {len(raw_train)} training records.")

    # 2. In-Memory Paraphrase Pairing
    print("Generating/verifying question paraphrase pairs in memory...")
    paraphraser = MarianParaphraser(device=DEVICE)
    questions = [item["question"] for item in raw_train]
    paraphrases = paraphraser.batch_back_translate(questions)
    paired_records = [
        {**rec, "paraphrase": p} for rec, p in zip(raw_train, paraphrases)
    ]
    print(f"Successfully paired {len(paired_records)} queries with back-translations.")

    # 3. Tokenization & Sliding Windows
    print(f"Tokenizing with {args.model} (seq_len={args.max_seq_len}, stride={args.doc_stride})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    features = prepare_paired_sliding_windows(
        paired_records,
        tokenizer,
        max_seq_len=args.max_seq_len,
        doc_stride=args.doc_stride,
    )
    dataset = PairedQAFeatureDataset(features)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=paired_collate_fn,
    )

    total_batches = len(dataloader)
    total_steps = total_batches * args.epochs
    print(f"Features ready: {len(features)} sliding windows across {total_batches} batches/epoch.")

    # 4. Model & Optimizer Initialization
    print(f"Initializing ParaphraseConsistentQAExtractor with {args.model}...")
    model = ParaphraseConsistentQAExtractor(model_name=args.model).to(DEVICE)

    backbone_params = [p for _, p in model.encoder.named_parameters()]
    head_params = [p for _, p in model.qa_outputs.named_parameters()]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": head_params, "lr": args.lr * 3.3, "weight_decay": 0.0},
        ]
    )

    use_cuda_fp16 = args.fp16 and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_fp16)

    # Telemetry tracking files
    history_file_specific = os.path.join(args.runs_dir, f"history_{args.loss_type}.json")
    history_file_latest = os.path.join(args.runs_dir, "history.json")

    training_logs: List[Dict[str, Any]] = []
    start_time = time.time()
    global_step = 0

    model.train()
    print("\nStarting training loop...")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        for step, batch in enumerate(dataloader):
            global_step += 1
            optimizer.zero_grad()

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=use_cuda_fp16):
                out_q = model(
                    input_ids=batch["input_ids_q"].to(DEVICE),
                    attention_mask=batch["attn_q"].to(DEVICE),
                    pass_mask=batch["pass_mask_q"].to(DEVICE),
                )
                out_qp = model(
                    input_ids=batch["input_ids_qp"].to(DEVICE),
                    attention_mask=batch["attn_qp"].to(DEVICE),
                    pass_mask=batch["pass_mask_qp"].to(DEVICE),
                )

                loss_ce = compute_supervised_ce_loss(
                    out_q["z_start"],
                    out_q["z_end"],
                    batch["s_pos_q"].to(DEVICE),
                    batch["e_pos_q"].to(DEVICE),
                )

                b_size = batch["input_ids_q"].shape[0]
                batch_ot = torch.tensor(0.0, device=DEVICE)
                batch_kl = torch.tensor(0.0, device=DEVICE)
                batch_hung = torch.tensor(0.0, device=DEVICE)

                for b in range(b_size):
                    m_len = batch["pass_lens"][b]
                    p_mask_q = batch["pass_mask_q"][b] == 1
                    p_mask_qp = batch["pass_mask_qp"][b] == 1

                    s_q = out_q["z_start"][b, p_mask_q][:m_len]
                    e_q = out_q["z_end"][b, p_mask_q][:m_len]
                    s_qp = out_qp["z_start"][b, p_mask_qp][:m_len]
                    e_qp = out_qp["z_end"][b, p_mask_qp][:m_len]

                    if s_q.shape[0] == m_len and s_qp.shape[0] == m_len and m_len > 1:
                        p_start_q = F.softmax(s_q, dim=0).unsqueeze(0)
                        p_start_qp = F.softmax(s_qp, dim=0).unsqueeze(0)
                        p_end_q = F.softmax(e_q, dim=0).unsqueeze(0)
                        p_end_qp = F.softmax(e_qp, dim=0).unsqueeze(0)

                        if "ot" in args.loss_type or "dual" in args.loss_type:
                            ot_s = compute_1d_wasserstein_loss(p_start_q, p_start_qp)
                            ot_e = compute_1d_wasserstein_loss(p_end_q, p_end_qp)
                            batch_ot = batch_ot + 0.5 * (ot_s + ot_e)

                        if "kl" in args.loss_type:
                            kl_s = compute_symmetric_kl_loss(p_start_q, p_start_qp)
                            kl_e = compute_symmetric_kl_loss(p_end_q, p_end_qp)
                            batch_kl = batch_kl + 0.5 * (kl_s + kl_e)

                        if "hungarian" in args.loss_type or "dual" in args.loss_type:
                            h_loss = compute_hungarian_loss_single_sample(
                                s_q, e_q, s_qp, e_qp, k=args.top_k, max_span_len=30
                            )
                            batch_hung = batch_hung + h_loss

                l_ot = batch_ot / float(max(1, b_size))
                l_kl = batch_kl / float(max(1, b_size))
                l_hung = batch_hung / float(max(1, b_size))

                # Composite loss selection
                if args.loss_type == "ce_only":
                    loss_tot = loss_ce
                elif args.loss_type == "kl":
                    loss_tot = loss_ce + args.lambda_kl * l_kl
                elif args.loss_type == "ot":
                    loss_tot = loss_ce + args.lambda_ot * l_ot
                elif args.loss_type == "hungarian":
                    loss_tot = loss_ce + args.mu_hung * l_hung
                else:  # dual (1D OT + 2D Hungarian)
                    loss_tot = loss_ce + args.lambda_ot * l_ot + args.mu_hung * l_hung

            scaler.scale(loss_tot).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            entry = {
                "step": global_step,
                "epoch": epoch + 1,
                "loss_total": round(loss_tot.item(), 4),
                "loss_ce": round(loss_ce.item(), 4),
                "loss_ot": round(l_ot.item(), 4),
                "loss_kl": round(l_kl.item(), 4),
                "loss_hung": round(l_hung.item(), 4),
                "elapsed_sec": round(time.time() - start_time, 2),
            }
            training_logs.append(entry)

            # Periodic console output & incremental disk flush
            if global_step % args.log_interval == 0 or step == total_batches - 1:
                elapsed_cur = time.time() - start_time
                steps_per_sec = global_step / max(0.1, elapsed_cur)
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{total_batches}] "
                    f"({global_step}/{total_steps}) | "
                    f"Total: {loss_tot.item():.4f} | CE: {loss_ce.item():.4f} | "
                    f"OT: {l_ot.item():.4f} | KL: {l_kl.item():.4f} | Hung: {l_hung.item():.4f} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

                # Incremental flush so external monitors / notebooks can watch live
                try:
                    with open(history_file_specific, "w", encoding="utf-8") as f:
                        json.dump(training_logs, f, indent=2)
                    with open(history_file_latest, "w", encoding="utf-8") as f:
                        json.dump(training_logs, f, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to flush telemetry to disk: {e}")

        epoch_dur = time.time() - epoch_start
        print(f"--- Epoch {epoch+1}/{args.epochs} completed in {epoch_dur:.1f}s ---")

    total_duration = time.time() - start_time
    print("\n" + "=" * 72)
    print(f"Training completed successfully in {total_duration:.1f}s!")

    # 5. Save Final Checkpoint
    checkpoint_specific = os.path.join(args.weights_dir, f"model_{args.loss_type}.pt")
    checkpoint_best = os.path.join(args.weights_dir, "best_model.pt")

    ckpt_payload = {
        "model_state_dict": model.state_dict(),
        "model_name": args.model,
        "loss_type": args.loss_type,
        "hyperparameters": vars(args),
        "elapsed_seconds": total_duration,
        "final_step": global_step,
        "training_logs": training_logs,
    }

    print(f"Saving model checkpoint to {checkpoint_specific} and {checkpoint_best}...")
    torch.save(ckpt_payload, checkpoint_specific)
    torch.save(ckpt_payload, checkpoint_best)

    # Final telemetry dump
    with open(history_file_specific, "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2)
    with open(history_file_latest, "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2)

    print(f"All logs saved to {history_file_specific}")
    print("=" * 72)


if __name__ == "__main__":
    main()
