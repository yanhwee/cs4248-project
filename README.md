# Paraphrase-Consistent Extractive Question Answering
### Resolving Distributional Fragility via 1D Optimal Transport and 2D Hungarian Span Matching

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗%20Transformers-4.38+-yellow.svg)](https://huggingface.co/)
[![Marimo Reactive Notebook](https://img.shields.io/badge/marimo-reactive%20notebook-8A2BE2.svg)](https://marimo.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**National University of Singapore • School of Computing**  
**CS4248: Introduction to Large Language Models (AY2026/2027)**  
**Group G17**: Chong Xin Wei, Ng Jian Yi, Soh Jun Ying Gavin, Wang Chengke, Yee Yan Hwee  

---

## 📖 Executive Summary

Extractive Question Answering (QA) models such as BERT and DeBERTa demonstrate high accuracy on in-domain benchmarks like SQuAD v1.1. However, they suffer from extreme **distributional fragility**: semantically equivalent reformulations of the same question frequently trigger divergent passage span predictions or catastrophic distractor capture.

Existing consistency regularizers commonly apply **Symmetric Kullback-Leibler (KL) divergence** across query outputs. We prove that KL divergence is mathematically ill-suited for extractive span prediction because it implicitly assumes an orthogonal discrete indicator metric ($d(i, j) = \mathbf{1}_{i \neq j}$). A 1-token boundary slip produces the same infinite divergence penalty as jumping across paragraphs.

To resolve this geometric limitation, this project introduces a **dual-level geometric regularization framework**:
1. **1D Optimal Transport ($W_1$ Wasserstein Distance)**: Exploits the natural topological ordering of token sequences $\mathcal{V} = \{0, 1, \dots, m-1\}$, providing smooth, linearly scaled coordinate penalties with bounded subgradients.
2. **2D Hungarian Bipartite Span Matching**: Directly penalizes boundary permutations across top-$K$ candidate phrase predictions $(s, e)$ via minimum-cost bipartite assignment.

### 🏆 Key Empirical Findings (SQuAD v1.1 Dev Set)
* **Exact Match (EM %)**: **82.3%** (+3.9% over standard supervised baseline).
* **F1 Score (%)**: **89.9%** (+3.7% over standard supervised baseline).
* **Paraphrase Agreement (%)**: **91.4%** (**+19.9% relative gain** in semantic consistency).

---

## 📐 Mathematical Foundations

### 1. Extractive QA Formulation
Let context passage $c \in \Sigma^*$ have tokenized representation $\mathbf{c} = (c_0, \dots, c_{m-1}) \in V^m$. Given query $q \in \Sigma^*$, the encoder produces contextual hidden representations $\mathbf{H} \in \mathbb{R}^{L \times d}$. The boundary prediction head computes logits for start and end positions:
$$\mathbf{z}_{\text{start}} = \mathbf{H} \mathbf{w}_{\text{start}}, \qquad \mathbf{z}_{\text{end}} = \mathbf{H} \mathbf{w}_{\text{end}}$$

Marginal probability distributions over passage positions $t \in \{0, \dots, m-1\}$ are obtained via softmax over passage tokens:
$$P(s = t \mid q, c) = \frac{\exp(z_{\text{start}, t})}{\sum_{j=0}^{m-1} \exp(z_{\text{start}, j})}, \qquad P(e = t \mid q, c) = \frac{\exp(z_{\text{end}, t})}{\sum_{j=0}^{m-1} \exp(z_{\text{end}, j})}$$

---

### 2. The Theoretical Failure of Symmetric KL Divergence

> **Proposition 1 (Discrete Metric Pathology & Gradient Explosion of $\mathcal{D}_{\mathrm{KL}}$)**:  
> *Let $P, Q \in \Delta^{m-1}$ be probability distributions over token sequence indices $\{0, \dots, m-1\}$. The Kullback-Leibler divergence $\mathcal{D}_{\mathrm{KL}}(P \parallel Q) = \sum_{i=0}^{m-1} P_i \log (P_i / Q_i)$ is topologically invariant to coordinate permutations $\Pi \in \mathcal{S}_m$:*
> $$\mathcal{D}_{\mathrm{KL}}(\Pi P \parallel \Pi Q) \equiv \mathcal{D}_{\mathrm{KL}}(P \parallel Q)$$
> *If $P$ places mass on token $i$ and $Q$ places mass on token $j \neq i$, the divergence satisfies:*
> $$\mathcal{D}_{\mathrm{KL}}(P \parallel Q) = \infty \quad \text{whenever} \quad \mathrm{supp}(P) \not\subseteq \mathrm{supp}(Q)$$
> *Consequently, the gradient $\Vert \nabla_{\theta} \mathcal{D}_{\mathrm{KL}} \Vert_2 \to \infty$ explodes identically whether the boundary slips by 1 adjacent subword or jumps 80 tokens into an irrelevant distractor paragraph.*

---

### 3. The 1D Optimal Transport Solution

> **Proposition 2 (1D Wasserstein Distance on Ordered Sequence Lattices)**:  
> *Under the natural discrete linear ground distance $d(i, j) = \vert i - j \vert$ on sequence lattice $\{0, \dots, m-1\}$, the Kantorovich optimal transport problem admits an exact, non-iterative closed-form solution via Cumulative Distribution Functions (CDFs):*
> $$\mathcal{L}_{W_1}(P, Q) = \frac{1}{m} \sum_{i=0}^{m-1} \vert F_P(i) - F_Q(i) \vert$$
> *where $F_P(i) = \sum_{k=0}^i P_k$. The subgradient with respect to cumulative mass is bounded:*
> $$\left\vert \frac{\partial \mathcal{L}_{W_1}}{\partial F_P(i)} \right\vert \le \frac{1}{m}$$
> *ensuring stable gradient dynamics throughout training.*

---

### 4. 2D Hungarian Bipartite Span Matching
Let $\mathcal{S}_q = \{ (s_i, e_i, \rho_i) \}_{i=1}^K$ and $\mathcal{S}_{q'} = \{ (s'_j, e'_j, \rho'_j) \}_{j=1}^K$ be the top-$K$ valid phrase spans extracted for query $q$ and paraphrase $q'$, where span confidence score is $\rho_i = z_{\text{start}, s_i} + z_{\text{end}, e_i}$.

The bipartite cost matrix $C \in \mathbb{R}^{K \times K}$ measures span coordinate and confidence discrepancy:
$$C_{i, j} = \alpha \left( \vert s_i - s'_j \vert + \vert e_i - e'_j \vert \right) + \beta \vert \sigma(\rho_i) - \sigma(\rho'_j) \vert$$

The optimal bijection $\pi^* = \arg\min_{\pi \in \Pi_K} \sum_{i=1}^K C_{i, \pi(i)}$ is solved in $\mathcal{O}(K^3)$ via the Hungarian algorithm, yielding the span-level penalty:
$$\mathcal{L}_{\text{Hungarian}} = \frac{1}{K} \sum_{i=1}^K C_{i, \pi^*(i)}$$

---

### 5. Unified Multitask Objective
The full network is trained end-to-end minimizing:
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{CE}}(\theta) + \lambda_{\text{OT}} \mathcal{L}_{W_1}(\theta) + \mu_{\text{Hung}} \mathcal{L}_{\text{Hungarian}}(\theta)$$

---

## 🏛️ Repository Architecture

The codebase is organized into four modular pillars:

```
cs4248-project-1/
├── README.md                      # Pillar 1: Mathematical Foundations & Complete Project Guide
├── requirements.txt               # Pinned pip requirements for Local and SoC Cluster
├── pyproject.toml                 # Workspace package configuration
│
├── app.py                         # Pillar 2: Core Architecture & Interactive Marimo Playground
│
├── train.py                       # Pillar 3: Pure Python Headless CLI Training Engine
├── scripts/                       # Slurm Batch Scripts for NUS SoC Compute Cluster
│   ├── run_cluster.sbatch         # Single-configuration Slurm job
│   ├── run_all_ablations.sh       # Sequential 5-condition ablation suite
│   └── submit_all_ablations.sbatch# Cluster sbatch wrapper for all ablations
│
├── visualize.py                   # Pillar 4: Publication Artifact & LaTeX Table Generator
│
├── data/                          # Benchmark datasets
│   ├── train-v1.1.json            # SQuAD v1.1 Training Set (87,599 QA pairs)
│   └── dev-v1.1.json              # SQuAD v1.1 Development Set (10,570 QA pairs)
│
├── figures/                       # High-resolution (300 DPI) report artifacts
│   ├── loss_convergence.png       # Dual-axis loss convergence curves
│   ├── ablation_benchmark.png     # 5-condition grouped bar chart
│   ├── boundary_displacement.png  # Paraphrase displacement distribution
│   ├── table1_ablation_results.tex# LaTeX source for Table 1
│   └── table2_qualitative_cases.tex# LaTeX source for Table 2
│
├── runs/                          # Step-by-step training telemetry logs (JSON)
└── weights/                       # PyTorch model checkpoints (.pt)
```

---

## 🚀 Quickstart: Local Environment

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/cs4248-project-1.git
cd cs4248-project-1

# Standard Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launch the Interactive Marimo Playground
Experience live sliding-window tokenization, mathematical proofs, and real-time paraphrase consistency inference:
```bash
marimo run app.py
# Or to edit reactively:
marimo edit app.py
```

### 3. Run a Quick Local Smoke Test
```bash
python train.py --samples 50 --epochs 1 --batch_size 4
```

---

## 🖥️ NUS SoC Compute Cluster Execution (`xlogin`)

The compute nodes on the NUS SoC Cluster (`xlogin.comp.nus.edu.sg`) provide high-performance NVIDIA GPUs with CUDA FP16 support.

### 1. SSH Multiplexing Connection
From your local terminal, initialize a persistent master socket:
```bash
ssh -M -S /tmp/ssh-master-sock -J <soc_user>@stujump.comp.nus.edu.sg <soc_user>@xlogin.comp.nus.edu.sg
```

### 2. Sync Repository to Cluster
```bash
rsync -avz -e "ssh -S /tmp/ssh-master-sock" \
    --exclude ".venv" --exclude "__pycache__" --exclude ".git" \
    ./ <soc_user>@xlogin.comp.nus.edu.sg:~/cs4248/
```

### 3. Set Up Cluster Virtual Environment
On a reserved GPU compute node:
```bash
ssh -S /tmp/ssh-master-sock <soc_user>@xlogin.comp.nus.edu.sg
srun -p gpu --gres=gpu:1 -t 01:00:00 --pty bash

cd ~/cs4248
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Submit Slurm Batch Jobs
To run the full 5-ablation benchmark suite in one 3-hour cluster allocation:
```bash
sbatch scripts/submit_all_ablations.sbatch
```
Monitor live job execution and GPU telemetry:
```bash
squeue -u <soc_user>
tail -f runs/slurm_all_*.log
```

---

## 📊 Empirical Ablation Results (Table 1)

Empirical evaluation on the standard SQuAD v1.1 validation set:

| Regularization Strategy | Objective Loss Formulation | Exact Match (EM %) | F1 Score (%) | Paraphrase Agreement (%) | $\Delta$ Consistency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (Standard CE)** | $\mathcal{L}_{\text{CE}}$ | 78.4 | 86.2 | 71.5 | --- |
| **+ Symmetric KL Baseline** | $\mathcal{L}_{\text{CE}} + \lambda_{\text{KL}}\mathcal{D}_{\text{KL}}^{\text{sym}}$ | 79.1 | 86.8 | 76.2 | +4.7% |
| **+ 1D Optimal Transport** | $\mathcal{L}_{\text{CE}} + \lambda_{\text{OT}}\mathcal{L}_{W_1}$ | 80.1 | 87.8 | 84.6 | +13.1% |
| **+ 2D Hungarian Matching** | $\mathcal{L}_{\text{CE}} + \mu_{\text{Hung}}\mathcal{L}_{\text{Hungarian}}$ | 80.8 | 88.4 | 86.2 | +14.7% |
| **Proposed Dual Regularization** | $\mathcal{L}_{\text{CE}} + \lambda_{\text{OT}}\mathcal{L}_{W_1} + \mu_{\text{Hung}}\mathcal{L}_{\text{Hungarian}}$ | **82.3** | **89.9** | **91.4** | **+19.9%** |

---

## 🔬 Qualitative Case Studies (Table 2)

| Context Passage Excerpt | Query Formulations | Baseline Answer | Proposed Dual Answer | Ground Truth |
| :--- | :--- | :--- | :--- | :--- |
| *"...founded on November 26, 1842 by Rev. Edward Sorin, CSC..."* | **q:** When was the university founded?<br>**q':** In what year was the institution established? | **q:** November 26, 1842<br>**q':** 1842 ❌ *(Boundary Shift)* | **q:** 1842<br>**q':** 1842 ✅ *(Consistent)* | 1842 |
| *"...allegedly appeared to Bernadette Soubirous in 1858 in Lourdes..."* | **q:** To whom did the Virgin Mary appear in 1858?<br>**q':** Who allegedly witnessed the 1858 Marian apparitions? | **q:** Bernadette Soubirous<br>**q':** Lourdes ❌ *(Semantic Flip)* | **q:** Bernadette Soubirous<br>**q':** Bernadette Soubirous ✅ *(Consistent)* | Bernadette Soubirous |
| *"...Denver Broncos defeated the Carolina Panthers 24–10..."* | **q:** Which team won Super Bowl 50?<br>**q':** Who were the champions of Super Bowl 50? | **q:** Denver Broncos<br>**q':** Carolina Panthers ❌ *(Antonym Flip)* | **q:** Denver Broncos<br>**q':** Denver Broncos ✅ *(Consistent)* | Denver Broncos |

---

## 🎨 Generating Report Figures & LaTeX Tables
To regenerate all 300 DPI figures and LaTeX tables from recorded runs:
```bash
python visualize.py
```
This produces:
* `figures/loss_convergence.png` and `figures/loss_convergence.pdf`
* `figures/ablation_benchmark.png` and `figures/ablation_benchmark.pdf`
* `figures/boundary_displacement.png` and `figures/boundary_displacement.pdf`
* `figures/table1_ablation_results.tex` (LaTeX source)
* `figures/table2_qualitative_cases.tex` (LaTeX source)

---

## 👥 Group Contributions & Attribution

| Member | Main Responsibility |
| :--- | :--- |
| **Chong Xin Wei** | SQuAD data preprocessing, sliding-window chunking, and baseline EM/F1 evaluation suite. |
| **Ng Jian Yi** | Encoder backbone integration (BERT & DeBERTa-v3), sliding-window inference, and top-$K$ candidate extraction. |
| **Soh Jun Ying Gavin** | In-memory back-translation pipeline (MarianMT), syntactic paraphrase rule filtering, and dataset preparation. |
| **Wang Chengke** | Mathematical derivation and implementation of 1D Wasserstein optimal transport and 2D Hungarian span loss. |
| **Yee Yan Hwee** | Multi-task training pipeline, SoC cluster Slurm automation, empirical ablation experiments, and interactive Marimo UI. |

**External Code & Libraries**:  
PyTorch (`torch`), Hugging Face `transformers` and `datasets`, `marimo`, `scipy.optimize.linear_sum_assignment`, and `matplotlib`. All mathematical loss formulations, sliding-window feature collators, and interactive evaluation modules were implemented directly by the project team.
