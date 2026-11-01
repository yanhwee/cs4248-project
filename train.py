import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="SQuAD Consistency Training Pipeline")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Extractive QA Paraphrase Consistency: Cluster Training Pipeline
    ### CS4248 Group G17 • Large-Scale SQuAD v1.1 Training Engine

    This notebook serves a dual purpose:
    1. **Interactive Marimo Dashboard**: Configure training runs, monitor loss telemetry live, test subsets locally, and inspect cluster hardware.
    2. **Headless Cluster CLI (`python train.py`)**: Submittable directly to the NUS SoC Slurm Cluster (`xlogin.comp.nus.edu.sg`) via `sbatch`, reading command-line arguments and running headless with PyTorch CUDA FP16 acceleration.

    > **Architecture Note**: Core model classes, sliding window feature generators, and consistency loss functions are directly imported from [`app.py`](file:///Users/yanhwee/Downloads/CS4248%20Student%20Project/cs4248-project-1/app.py) via Marimo's native module reusability.
    """)
    return


@app.cell
def _():
    import argparse
    import json
    import os
    import sys
    import time
    from typing import Any, Dict, List, Optional, Tuple

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer

    # Import core reusable architectures and loss functions from app.py
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

    DATA_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    )
    train_path = os.path.join(DATA_DIR, "train-v1.1.json")
    dev_path = os.path.join(DATA_DIR, "dev-v1.1.json")

    return (
        DATA_DIR,
        DEVICE,
        DataLoader,
        Dict,
        F,
        List,
        MarianParaphraser,
        Optional,
        PairedQAFeatureDataset,
        ParaphraseConsistentQAExtractor,
        Tuple,
        argparse,
        compute_1d_wasserstein_loss,
        compute_exact_match,
        compute_f1,
        compute_hungarian_loss_single_sample,
        compute_supervised_ce_loss,
        compute_symmetric_kl_loss,
        dev_path,
        json,
        load_squad_data,
        nn,
        np,
        os,
        paired_collate_fn,
        prepare_paired_sliding_windows,
        sys,
        time,
        torch,
        tqdm,
        train_path,
        AutoTokenizer,
    )


@app.cell
def _(DEVICE, mo, torch):
    # Cluster Hardware Diagnostic Cards
    cuda_available = torch.cuda.is_available()
    device_name = (
        torch.cuda.get_device_name(0)
        if cuda_available
        else ("Apple Silicon MPS" if DEVICE.type == "mps" else "CPU")
    )
    vram_gb = (
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        if cuda_available
        else ("Unified System RAM" if DEVICE.type == "mps" else "Host RAM")
    )
    amp_supported = cuda_available and torch.cuda.is_bf16_supported()

    hw_cards = mo.hstack(
        [
            mo.stat(
                value=device_name,
                label="Compute Accelerator",
                caption=f"PyTorch Backend: {DEVICE.type.upper()}",
                bordered=True,
            ),
            mo.stat(
                value=vram_gb,
                label="Accelerator Memory",
                caption="Total Accessible VRAM",
                bordered=True,
            ),
            mo.stat(
                value="BF16 / FP16" if amp_supported or cuda_available else "FP32 Only",
                label="Mixed Precision (AMP)",
                caption="Tensor Core Acceleration",
                bordered=True,
            ),
            mo.stat(
                value="Available" if cuda_available else "Local Machine",
                label="SoC Cluster Mode",
                caption="Slurm Batch Ready",
                bordered=True,
            ),
        ],
        justify="space-between",
        widths=[1, 1, 1, 1],
    )

    mo.vstack([
        mo.md("### 🖥️ Hardware & Accelerator Diagnostics"),
        hw_cards,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. Mathematical Objective & Consistency Regularization

    The model is optimized under the composite multi-task objective:
    $$\min_{\theta} \mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{CE}}(\theta) + \lambda_{\text{OT}} \mathcal{L}_{W_1}(\theta) + \mu_{\text{Hung}} \mathcal{L}_{\text{Hungarian}}(\theta)$$

    where:
    1. **Supervised Cross-Entropy**:
       $$\mathcal{L}_{\text{CE}} = - \log P_{\text{start}}(a^*_{\text{start}} \mid q, c) - \log P_{\text{end}}(a^*_{\text{end}} \mid q, c)$$
    2. **1D Optimal Transport ($W_1$ Wasserstein Distance)**:
       $$\mathcal{L}_{W_1} = \frac{1}{2} \int_0^1 |F_{S, q}(t) - F_{S, q'}(t)| dt + \frac{1}{2} \int_0^1 |F_{E, q}(t) - F_{E, q'}(t)| dt$$
    3. **2D Hungarian Bipartite Span Matching**:
       $$\mathcal{L}_{\text{Hungarian}} = \min_{\pi \in \Pi_K} \sum_{i=1}^K C(i, \pi(i))$$
       penalizing semantic span divergences across question paraphrases $q$ and $q'$.
    """)
    return


@app.cell
def _(mo, sys):
    # Reactive Controls for Interactive Training & Slurm Generation
    cli = mo.cli_args()

    if "-h" in sys.argv or "--help" in sys.argv:
        print("""
SQuAD Extractive QA Paraphrase Consistency Training Pipeline

Usage: python train.py [OPTIONS]

Options:
  --model TEXT        Model backbone [default: google-bert/bert-base-uncased]
  --loss_type TEXT    Loss strategy: dual | kl | ot | hungarian | ce_only [default: dual]
  --samples INT       Training samples (0 for full 87.5k SQuAD) [default: 100]
  --batch_size INT    Batch size per device [default: 8]
  --epochs INT        Number of epochs [default: 2]
  --lr FLOAT          Backbone learning rate [default: 3e-5]
  --lambda_ot FLOAT   1D Optimal Transport loss weight [default: 0.5]
  --lambda_kl FLOAT   Symmetric KL divergence loss weight [default: 0.5]
  --mu_hung FLOAT     Hungarian span bipartite matching loss weight [default: 0.1]
  --top_k INT         Top-K spans to match [default: 3]
  --help, -h          Show this message and exit
""")
        sys.exit(0)

    model_sel = mo.ui.dropdown(
        options=[
            "google-bert/bert-base-uncased",
            "microsoft/deberta-v3-small",
            "microsoft/deberta-v3-base",
        ],
        value=cli.get("model") or "google-bert/bert-base-uncased",
        label="Model Backbone",
    )
    loss_type_sel = mo.ui.dropdown(
        options=["dual", "kl", "ot", "hungarian", "ce_only"],
        value=cli.get("loss_type") or "dual",
        label="Loss Strategy",
    )
    samples_in = mo.ui.number(
        start=0,
        stop=87599,
        step=50,
        value=int(cli.get("samples") or 100),
        label="Training Samples (0 = Full 87.5k SQuAD)",
    )
    batch_in = mo.ui.number(
        start=1,
        stop=64,
        step=1,
        value=int(cli.get("batch_size") or 8),
        label="Batch Size per Device",
    )
    epochs_in = mo.ui.number(
        start=1,
        stop=10,
        step=1,
        value=int(cli.get("epochs") or 2),
        label="Training Epochs",
    )
    lr_in = mo.ui.number(
        start=1e-6,
        stop=1e-3,
        step=5e-6,
        value=float(cli.get("lr") or 3e-5),
        label="Backbone Learning Rate",
    )
    lambda_ot_in = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.05,
        value=float(cli.get("lambda_ot") or 0.5),
        label=r"1D OT Weight ($\lambda_{\text{OT}}$)",
    )
    lambda_kl_in = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.05,
        value=float(cli.get("lambda_kl") or 0.5),
        label=r"Symmetric KL Weight ($\lambda_{\text{KL}}$)",
    )
    mu_hung_in = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.05,
        value=float(cli.get("mu_hung") or 0.1),
        label=r"Hungarian Span Weight ($\mu_{\text{Hung}}$)",
    )
    topk_in = mo.ui.number(
        start=1,
        stop=10,
        step=1,
        value=int(cli.get("top_k") or 3),
        label="Top-K Bipartite Spans ($K$)",
    )
    fp16_switch = mo.ui.checkbox(
        value=True,
        label="Enable CUDA Mixed Precision (AMP FP16)",
    )
    run_btn = mo.ui.run_button(label="🚀 Start Training Pipeline")

    control_form = mo.vstack([
        mo.md("### ⚙️ Training Hyperparameter Configuration"),
        mo.hstack([model_sel, loss_type_sel, samples_in], widths=[2, 1, 1]),
        mo.hstack([epochs_in, batch_in, lr_in, fp16_switch], widths=[1, 1, 1, 1]),
        mo.hstack([lambda_ot_in, lambda_kl_in, mu_hung_in, topk_in], widths=[1, 1, 1, 1]),
        mo.hstack([run_btn]),
    ])
    return (
        batch_in,
        cli,
        control_form,
        epochs_in,
        fp16_switch,
        lambda_kl_in,
        lambda_ot_in,
        loss_type_sel,
        lr_in,
        model_sel,
        mu_hung_in,
        run_btn,
        samples_in,
        topk_in,
    )


@app.cell
def _(control_form):
    control_form
    return


@app.cell
def _(
    batch_in,
    epochs_in,
    lambda_kl_in,
    lambda_ot_in,
    loss_type_sel,
    lr_in,
    mo,
    model_sel,
    mu_hung_in,
    samples_in,
):
    # Slurm Submission Script Generator for SoC Cluster
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=cs4248_consistency
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm_%j.log
#SBATCH --error=runs/slurm_%j.err

echo "=== Running CS4248 SQuAD Consistency Training on SoC Cluster ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# 1. Load Cluster CUDA Modules
module load cuda/12.2

# 2. Activate Python Environment
source .venv/bin/activate

# 3. Execute Headless Training Script
python train.py \\
    --model "{model_sel.value}" \\
    --loss_type "{loss_type_sel.value}" \\
    --samples {int(samples_in.value)} \\
    --epochs {int(epochs_in.value)} \\
    --batch_size {int(batch_in.value)} \\
    --lr {float(lr_in.value)} \\
    --lambda_ot {float(lambda_ot_in.value)} \\
    --lambda_kl {float(lambda_kl_in.value)} \\
    --mu_hung {float(mu_hung_in.value)} \\
    --fp16

echo "=== Training Complete. Weights saved to weights/ and logs to runs/ ==="
"""

    slurm_callout = mo.accordion(
        {
            "📋 NUS SoC Slurm Batch Submission Script (`job.sbatch`)": mo.vstack([
                mo.md(
                    "You can copy this script directly into `job.sbatch` on `xlogin.comp.nus.edu.sg` and submit with `sbatch job.sbatch`:"
                ),
                mo.ui.code_editor(
                    value=sbatch_script,
                    language="shell",
                    disabled=True,
                ),
            ])
        }
    )
    slurm_callout
    return (sbatch_script,)


@app.cell
def _(
    AutoTokenizer,
    DATA_DIR,
    DEVICE,
    DataLoader,
    MarianParaphraser,
    PairedQAFeatureDataset,
    ParaphraseConsistentQAExtractor,
    batch_in,
    cli,
    compute_1d_wasserstein_loss,
    compute_hungarian_loss_single_sample,
    compute_supervised_ce_loss,
    compute_symmetric_kl_loss,
    dev_path,
    epochs_in,
    fp16_switch,
    json,
    lambda_kl_in,
    lambda_ot_in,
    load_squad_data,
    loss_type_sel,
    lr_in,
    mo,
    model_sel,
    mu_hung_in,
    np,
    os,
    paired_collate_fn,
    prepare_paired_sliding_windows,
    run_btn,
    samples_in,
    time,
    topk_in,
    torch,
    tqdm,
    train_path,
):
    # Unified Training Execution (Interactive UI + Headless Script Mode)
    app_mode = mo.app_meta().mode
    is_script_run = app_mode == "script" or len(cli) > 0
    should_run = run_btn.value or is_script_run

    cluster_model = None
    cluster_logs = []

    if should_run:
        loss_type = str(cli.get("loss_type") or loss_type_sel.value).lower()
        num_s = int(samples_in.value)
        epochs = int(epochs_in.value)
        batch_size = int(batch_in.value)
        lr = float(lr_in.value)
        lambda_ot = float(lambda_ot_in.value)
        lambda_kl = float(lambda_kl_in.value)
        mu_hung = float(mu_hung_in.value)
        top_k = int(topk_in.value)
        model_name = model_sel.value
        use_fp16 = fp16_switch.value and torch.cuda.is_available()

        os.makedirs("weights", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        # 1. Load Dataset
        raw_train = load_squad_data(
            train_path, max_records=None if num_s == 0 else num_s
        )
        total_samples = len(raw_train)

        # 2. In-Memory Paraphrase Pairing
        paraphraser = MarianParaphraser(device=DEVICE)
        questions = [item["question"] for item in raw_train]
        paraphrases = paraphraser.batch_back_translate(questions)
        paired_records = [
            {**rec, "paraphrase": p} for rec, p in zip(raw_train, paraphrases)
        ]

        # 3. Tokenization & Sliding Windows
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        features = prepare_paired_sliding_windows(
            paired_records, tokenizer, max_seq_len=256, doc_stride=96
        )
        dataset = PairedQAFeatureDataset(features)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=paired_collate_fn
        )

        total_batches = len(dataloader)
        total_steps = total_batches * epochs

        # 4. Model & Optimizer Initialization
        cluster_model = ParaphraseConsistentQAExtractor(model_name=model_name).to(DEVICE)
        backbone_p = [p for _, p in cluster_model.encoder.named_parameters()]
        head_p = [p for _, p in cluster_model.qa_outputs.named_parameters()]

        optimizer = torch.optim.AdamW([
            {"params": backbone_p, "lr": lr, "weight_decay": 0.01},
            {"params": head_p, "lr": lr * 3.3, "weight_decay": 0.0},
        ])
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

        cluster_model.train()
        cluster_logs = []

        def live_ui_update(logs):
            lat = logs[-1]
            cards = mo.hstack([
                mo.stat(value=f"{lat['loss_total']:.4f}", label="Total Loss", caption=f"Strategy: {loss_type}", bordered=True),
                mo.stat(value=f"{lat['loss_ce']:.4f}", label="CE Loss", caption="Cross Entropy", bordered=True),
                mo.stat(value=f"{lat['loss_ot']:.4f}", label="1D OT", caption=r"$W_1$ Metric", bordered=True),
                mo.stat(value=f"{lat['loss_kl']:.4f}", label="Sym KL", caption=r"$\mathcal{D}_{\text{KL}}^{\text{sym}}$", bordered=True),
                mo.stat(value=f"{lat['loss_hung']:.4f}", label="Hungarian", caption="Top-K Bipartite", bordered=True),
            ], justify="space-between", widths=[1, 1, 1, 1, 1])
            status_text = mo.md(
                f"⚡ **Step {lat['step']} / {total_steps}** (Epoch {lat['epoch']}/{epochs}) — "
                f"Loss: `{lat['loss_total']:.4f}` | CE: `{lat['loss_ce']:.4f}` | "
                f"OT: `{lat['loss_ot']:.4f}` | KL: `{lat['loss_kl']:.4f}` | Hung: `{lat['loss_hung']:.4f}`"
            )
            mo.output.replace(mo.vstack([mo.md("### 📊 Active Cluster Training Telemetry"), cards, status_text]))

        start_time = time.time()

        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=use_fp16):
                    out_q = cluster_model(
                        input_ids=batch["input_ids_q"].to(DEVICE),
                        attention_mask=batch["attn_q"].to(DEVICE),
                        pass_mask=batch["pass_mask_q"].to(DEVICE),
                    )
                    out_qp = cluster_model(
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

                            if "ot" in loss_type or "dual" in loss_type:
                                ot_s = compute_1d_wasserstein_loss(p_start_q, p_start_qp)
                                ot_e = compute_1d_wasserstein_loss(p_end_q, p_end_qp)
                                batch_ot = batch_ot + 0.5 * (ot_s + ot_e)

                            if "kl" in loss_type:
                                kl_s = compute_symmetric_kl_loss(p_start_q, p_start_qp)
                                kl_e = compute_symmetric_kl_loss(p_end_q, p_end_qp)
                                batch_kl = batch_kl + 0.5 * (kl_s + kl_e)

                            if "hungarian" in loss_type or "dual" in loss_type:
                                h_loss = compute_hungarian_loss_single_sample(
                                    s_q, e_q, s_qp, e_qp, k=top_k, max_span_len=30
                                )
                                batch_hung = batch_hung + h_loss

                    l_ot = batch_ot / float(max(1, b_size))
                    l_kl = batch_kl / float(max(1, b_size))
                    l_hung = batch_hung / float(max(1, b_size))

                    if "kl" in loss_type:
                        loss_tot = loss_ce + lambda_kl * l_kl
                    elif "ce_only" in loss_type:
                        loss_tot = loss_ce
                    elif "hungarian" in loss_type and "dual" not in loss_type:
                        loss_tot = loss_ce + mu_hung * l_hung
                    elif "ot" in loss_type and "dual" not in loss_type:
                        loss_tot = loss_ce + lambda_ot * l_ot
                    else:  # dual (OT + Hungarian)
                        loss_tot = loss_ce + lambda_ot * l_ot + mu_hung * l_hung

                scaler.scale(loss_tot).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(cluster_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                global_step = epoch * total_batches + step + 1
                entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss_total": round(loss_tot.item(), 4),
                    "loss_ce": round(loss_ce.item(), 4),
                    "loss_ot": round(l_ot.item(), 4),
                    "loss_kl": round(l_kl.item(), 4),
                    "loss_hung": round(l_hung.item(), 4),
                }
                cluster_logs.append(entry)

                if is_script_run:
                    if step % 20 == 0 or step == total_batches - 1:
                        print(
                            f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{total_batches}] | "
                            f"Loss: {loss_tot.item():.4f} (CE: {loss_ce.item():.4f}, OT: {l_ot.item():.4f}, KL: {l_kl.item():.4f}, Hung: {l_hung.item():.4f})"
                        )
                else:
                    if step % 2 == 0 or step == total_batches - 1:
                        live_ui_update(cluster_logs)

        elapsed = time.time() - start_time

        # 5. Save Checkpoints & Telemetry
        save_path = "weights/best_model.pt"
        torch.save(
            {
                "model_state_dict": cluster_model.state_dict(),
                "model_name": model_name,
                "elapsed_seconds": elapsed,
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "loss_type": loss_type,
                    "lambda_ot": lambda_ot,
                    "lambda_kl": lambda_kl,
                    "mu_hung": mu_hung,
                    "top_k": top_k,
                },
                "training_logs": cluster_logs,
            },
            save_path,
        )

        with open("runs/history.json", "w", encoding="utf-8") as f:
            json.dump(cluster_logs, f, indent=2)

        if not is_script_run and cluster_logs:
            final_table = mo.ui.table(
                data=cluster_logs,
                pagination=True,
                page_size=10,
                label="Completed Training History",
            )
            mo.output.replace(
                mo.vstack([
                    mo.md(f"### 🏁 Training Completed in {elapsed:.1f}s!"),
                    mo.md(f"Checkpoint saved to `{save_path}` and telemetry to `runs/history.json`."),
                    final_table,
                ])
            )

    return (cluster_logs, cluster_model)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Checkpoint Inspection & Validation Link
    Once training completes, the model weights stored in `weights/best_model.pt` and loss history in `runs/history.json` can be loaded into the main [`app.py`](file:///Users/yanhwee/Downloads/CS4248%20Student%20Project/cs4248-project-1/app.py) notebook to produce publication-grade evaluation curves and interactive QA inference playground testing.
    """)
    return


if __name__ == "__main__":
    app.run()
