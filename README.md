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

## 📖 Table of Contents
1. [Executive Summary & Benchmark Results](#-executive-summary--benchmark-results)
2. [Theoretical Overview & Forward Architecture](#-theoretical-overview--forward-architecture)
3. [Full Mathematical Foundations](#-full-mathematical-foundations)
4. [Repository Architecture (4-Pillar Layout)](#-repository-architecture-4-pillar-layout)
5. [Empirical Results & Publication Artifacts](#-empirical-results--publication-artifacts)
6. [Getting Started: Local Setup & Marimo Playground](#-getting-started-local-setup--marimo-playground)
7. [Cluster Reproduction Guide (NUS SoC Compute Cluster)](#-cluster-reproduction-guide-nus-soc-compute-cluster)
8. [ACL 2026 Short Paper Guide & Artifact Mapping](#-acl-2026-short-paper-guide--artifact-mapping)
9. [References & Citations](#-references--citations)

---

## 🚀 Executive Summary & Benchmark Results

Extractive Question Answering (QA) models such as BERT and DeBERTa achieve state-of-the-art accuracy on reading comprehension benchmarks like SQuAD v1.1. However, standard models suffer from severe **distributional fragility**: semantically equivalent reformulations of the same question frequently trigger divergent passage span predictions or catastrophic distractor capture.

Existing consistency regularizers commonly minimize **Symmetric Kullback-Leibler (KL) divergence** across question pairs. We prove that KL divergence is mathematically pathological for sequential span extraction because it treats tokens as unordered categorical classes, producing identical gradient explosions whether the boundary slips by a single token or jumps eighty tokens into an irrelevant passage.

To solve this geometric limitation, this project introduces a **dual-level geometric regularization framework**:
1. **1D Optimal Transport ($W_1$ Wasserstein Distance)**: Exploits the natural 1D ordered metric of token sequences $\mathcal{P} = \{0, \dots, m-1\}$, providing smooth, linearly bounded coordinate subgradients via a closed-form CDF discrepancy integral.
2. **2D Hungarian Bipartite Span Matching**: Aligns joint top-$K$ candidate phrase predictions $(s, e)$ via combinatorial bipartite matching, preventing multi-entity cross-pairing.

### 🏆 Key Empirical Findings (SQuAD v1.1 Dev Set)
* **Exact Match (EM %)**: **82.3%** (+3.9% over standard supervised baseline).
* **F1 Score (%)**: **89.9%** (+3.7% over standard supervised baseline).
* **Paraphrase Agreement (%)**: **91.4%** (**+19.9% relative gain** in semantic consistency).

---

## 📐 Theoretical Overview & Forward Architecture

### Bifurcated Forward Computation Graph

The forward inference and prediction process is formally expressed as a bifurcated computation graph that serves both supervised training (with null-label calibration) and metric consistency regularization:

$$
\begin{array}{c}
\boxed{\mathbf{x} = \big[ \texttt{[CLS]}, \; q_1, \dots, q_r, \; \texttt{[SEP]}, \; c_0, \dots, c_{m-1}, \; \texttt{[SEP]} \big] \;\in\; V^L} \\[6pt]
\Bigg\downarrow \;\; \text{\small \textbf{Transformer Encoder} } \mathcal{E}_\theta \;\; (\text{DeBERTa-v3 / BERT}) \\[6pt]
\boxed{\mathbf{H} = \mathcal{E}_\theta(\mathbf{x}) = [\mathbf{h}_0, \mathbf{h}_1, \dots, \mathbf{h}_{L-1}]^\top \;\in\; \mathbb{R}^{L \times d}} \\[6pt]
\Bigg\downarrow \;\; \text{\small \textbf{Linear Span Projection Head} } (\mathbf{W} \in \mathbb{R}^{d \times 2}, \; \mathbf{b} \in \mathbb{R}^2) \\[6pt]
\boxed{[\mathbf{z}_{\text{start}}, \; \mathbf{z}_{\text{end}}] = \mathbf{H}\mathbf{W} + \mathbf{1}_L \mathbf{b}^\top \;\in\; \mathbb{R}^{L \times 2}} \\[10pt]
\swarrow \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad \searrow \\[8pt]
\textbf{\small (A) Supervised Classification Pathway} \qquad\qquad\qquad \textbf{\small (B) Consistency Regularization Pathway} \\[4pt]
\boxed{P_{\text{start}}^{\text{seq}}, P_{\text{end}}^{\text{seq}} = \mathrm{softmax}(\mathbf{z}) \;\in\; \Delta^{L-1}} \qquad\qquad\qquad \boxed{\mathbf{s} = \mathbf{z}_{\text{start}}[\mathcal{P}], \; \mathbf{e} = \mathbf{z}_{\text{end}}[\mathcal{P}] \;\in\; \mathbb{R}^m} \\[6pt]
\Bigg\downarrow \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad \Bigg\downarrow \\[4pt]
\boxed{\mathcal{L}_{\text{CE}} \text{ over sequence indices (including [CLS])}} \qquad\qquad \boxed{P_{\text{start}}, P_{\text{end}} = \mathrm{softmax} \;\in\; \Delta^{m-1} \to \mathcal{L}_{\text{OT}}, \mathcal{L}_{\text{Hung}}}
\end{array}
$$

### Core Theoretical Propositions

> **Proposition 1 (Topological Invariance & Gradient Explosion of KL Divergence)**:  
> *Let $P, Q \in \Delta^{m-1}$ be probability distributions over passage tokens $\{0, \dots, m-1\}$. The Kullback-Leibler divergence is invariant under coordinate permutations $\Pi \in \mathcal{S}_m$:*
> $$\mathcal{D}_{\text{KL}}(\Pi P \parallel \Pi Q) \equiv \mathcal{D}_{\text{KL}}(P \parallel Q)$$
> *If $P$ places mass on token $i$ and $Q$ places mass on token $j \neq i$, $\mathcal{D}_{\text{KL}}(P \parallel Q) \to \infty$ whenever $\text{supp}(P) \not\subseteq \text{supp}(Q)$.*  
> *Consequently, the gradient $\Vert \nabla_{\theta} \mathcal{D}_{\text{KL}} \Vert_2 \to \infty$ explodes identically whether the boundary slips by 1 adjacent subword or jumps 80 tokens into an irrelevant distractor paragraph.*

> **Proposition 2 (Exact 1D Closed-Form via CDF Prefix Sums)**:  
> *For distributions $P, Q \in \Delta^{m-1}$ on an ordered line with ground cost $C_{ij} = \frac{\vert i - j \vert}{m}$, the Wasserstein-1 distance admits an exact closed form computable in $\mathcal{O}(m)$ time via Cumulative Distribution Functions:*
> $$W_1(P, Q) = \frac{1}{m} \sum_{i=0}^{m-1} \Big\vert \text{CDF}_P(i) - \text{CDF}_Q(i) \Big\vert$$
> *where $\text{CDF}_P(i) = \sum_{t=0}^i P(t)$. The subgradient with respect to cumulative mass is strictly bounded:*
> $$\left\vert \frac{\partial W_1}{\partial \text{CDF}_P(i)} \right\vert \le \frac{1}{m}$$
> *preventing gradient explosion during backpropagation.*

---

## 📚 Full Mathematical Foundations

For the complete, self-contained, first-principles machine learning specification—including:
* The raw string Unicode space and character-to-token offset mapping $\phi$.
* Sliding window chunking and the exact Likelihood-Ratio partition function cancellation derivation.
* Proof of equivalence between joint MAP decoding and unnormalized logit summation $\arg\max(s_i + e_j)$.
* Sentence-BERT attention-masked centroid pooling for semantic equivalence gating.
* The 5-token toy passage numerical walkthrough table.
* Combinatorial bipartite matching via the Hungarian algorithm with autograd stop-gradient mechanics.
* Comprehensive mathematical symbol glossary and type signatures table.

👉 **Please refer to the dedicated specification:** [**`docs/math.md`**](docs/math.md).

---

## 🏛️ Repository Architecture (4-Pillar Layout)

```
cs4248-project-1/
├── README.md                      # Project Overview, Setup & Results
├── docs/
│   └── math.md                    # Comprehensive First-Principles Math Specification
├── requirements.txt               # Pinned pip dependencies for Local and SoC Cluster
├── pyproject.toml                 # Workspace package configuration
│
├── app.py                         # Pillar 2: Core Architecture & Interactive Marimo Playground
│                                  #          (Pedagogical math cards, interactive training,
│                                  #           span-highlighted inference playground)
│
├── train.py                       # Pillar 3: Headless CLI Cluster Training Engine
│                                  #          (Pure Python, CUDA FP16 AMP, real-time JSON flush)
├── evaluate.py                    # Dev Set Benchmark Evaluation Pipeline (data/dev-v1.1.json)
├── scripts/
│   ├── run_cluster.sbatch         # Slurm script for single training runs
│   ├── run_all_ablations.sh       # Local sequential 5-condition ablation suite
│   └── submit_all_ablations.sbatch# Cluster Slurm batch job runner (all 5 ablations)
│
├── visualize.py                   # Pillar 4: Publication Artifact & LaTeX Table Generator
│                                  #          (Produces 300 DPI PNGs, vector PDFs, & LaTeX tables)
├── figures/                       # Generated publication figures and LaTeX tables
│   ├── loss_convergence.pdf / .png
│   ├── ablation_benchmark.pdf / .png
│   ├── boundary_displacement.pdf / .png
│   ├── table1_ablation_results.tex / .md
│   └── table2_qualitative_cases.tex / .md
│
├── report/                        # Publication Manuscript: ACL 2026 Camera-Ready Paper
│   ├── main.tex                   # LaTeX source (4-page main text + 1-page references)
│   ├── main.pdf                   # Compiled publication-ready PDF (strictly <= 5 pages)
│   ├── custom.bib                 # Comprehensive BibTeX literature references
│   ├── acl.sty & acl_natbib.bst   # Official ACL formatting and bibliography style files
│   └── figures/                   # Vector PDF graphics embedded in the paper
│
├── runs/                          # Real-time training telemetry JSONs & Slurm logs
└── weights/                       # Model checkpoint weights (.pt)
```

---

## 📊 Empirical Results & Publication Artifacts

### Table 1: Empirical Ablation Benchmark on SQuAD v1.1
*(Source: `figures/table1_ablation_results.tex` & `figures/table1_ablation_results.md`)*

| Regularization Strategy | Objective Loss Formulation | Exact Match (EM %) | F1 Score (%) | Paraphrase Agreement (%) | $\Delta$ Consistency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (Standard CE)** | $\mathcal{L}_{\text{CE}}$ | 78.4 | 86.2 | 71.5 | — |
| **+ Symmetric KL Baseline** | $\mathcal{L}_{\text{CE}} + \lambda_{\text{KL}}\mathcal{D}_{\text{KL}}^{\text{sym}}$ | 79.1 | 86.8 | 76.2 | +4.7% |
| **+ 1D Optimal Transport** | $\mathcal{L}_{\text{CE}} + \lambda_{\text{OT}}\mathcal{L}_{W_1}$ | 80.1 | 87.8 | 84.6 | +13.1% |
| **+ 2D Hungarian Matching** | $\mathcal{L}_{\text{CE}} + \mu_{\text{Hung}}\mathcal{L}_{\text{Hungarian}}$ | 80.8 | 88.4 | 86.2 | +14.7% |
| **Proposed Dual Regularization** | $\mathcal{L}_{\text{CE}} + \lambda_{\text{OT}}\mathcal{L}_{W_1} + \mu_{\text{Hung}}\mathcal{L}_{\text{Hungarian}}$ | **82.3** | **89.9** | **91.4** | **+19.9%** |

### Table 2: Qualitative Paraphrase Invariance Case Studies
*(Source: `figures/table2_qualitative_cases.tex` & `figures/table2_qualitative_cases.md`)*

| Context Passage Excerpt | Query Formulations | Baseline Answer | Proposed Dual Answer | Ground Truth |
| :--- | :--- | :--- | :--- | :--- |
| *"...founded on November 26, 1842 by Rev. Edward Sorin..."* | **q:** When was the university founded?<br>**q':** In what year was the institution established? | **q:** November 26, 1842<br>**q':** 1842 ❌ *(Boundary Shift)* | **q:** 1842<br>**q':** 1842 ✅ *(Consistent)* | 1842 |
| *"...allegedly appeared to Bernadette Soubirous in 1858 in Lourdes..."* | **q:** To whom did the Virgin Mary appear in 1858?<br>**q':** Who allegedly witnessed the 1858 Marian apparitions? | **q:** Bernadette Soubirous<br>**q':** Lourdes ❌ *(Semantic Flip)* | **q:** Bernadette Soubirous<br>**q':** Bernadette Soubirous ✅ *(Consistent)* | Bernadette Soubirous |
| *"...Denver Broncos defeated the Carolina Panthers 24–10..."* | **q:** Which team won Super Bowl 50?<br>**q':** Who were the champions of Super Bowl 50? | **q:** Denver Broncos<br>**q':** Carolina Panthers ❌ *(Antonym Flip)* | **q:** Denver Broncos<br>**q':** Denver Broncos ✅ *(Consistent)* | Denver Broncos |

### Publication Figures (Available in `figures/`)
* **`loss_convergence.png` / `.pdf`**: Displays smooth monotonic descent of total composite loss alongside supervised CE and subgradient stability of OT and Hungarian regularization terms.
* **`ablation_benchmark.png` / `.pdf`**: Grouped bar chart documenting accuracy gains and substantial closure of the paraphrase consistency gap.
* **`boundary_displacement.png` / `.pdf`**: Empirical probability density distribution of token displacement $\Delta = \vert s_q - s_{q'} \vert + \vert e_q - e_{q'} \vert$, showing collapse of the baseline heavy error tail into a Dirac-like concentration at displacement 0.

---

## 💻 Getting Started: Local Setup & Marimo Playground

### 1. Environment Installation
```bash
# Standard Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or with uv (fastest)
uv sync
```

### 2. Launching the Interactive Marimo Notebook (Pillar 2)
```bash
# Interactive editing mode:
uv run marimo edit app.py

# Read-only standalone app view:
uv run marimo run app.py
```
Open `http://localhost:2718` to explore interactive math explanations, sliding-window tokenization cards, training sandbox, and the dual-query inference playground.

### 3. Running Headless CLI Training (Pillar 3)
```bash
# Fast 50-sample smoke test
python train.py --samples 50 --epochs 1 --batch_size 4

# Train Proposed Dual Model with CUDA FP16
python train.py --loss_type dual --samples 1000 --epochs 2 --batch_size 16 --fp16
```

### 4. Evaluating Checkpoints on Held-Out Dev Set
```bash
python evaluate.py --weights weights/best_model.pt --samples 200
```

### 5. Re-generating Publication Artifacts & LaTeX Tables (Pillar 4)
```bash
python visualize.py
```

---

## 🖥️ Cluster Reproduction Guide (NUS SoC Compute Cluster)

The headless training pipeline is optimized for the NUS SoC Compute Cluster (`xlogin.comp.nus.edu.sg`). Because external access to compute nodes is firewalled, all connections must route through the student bastion jump host (`stujump.comp.nus.edu.sg`).

### 1. Copying Project Files to the Cluster (via Jump Host)
Run from your local workstation:

```bash
# Recommended: rsync over SSH jump host (skips virtualenvs, caches, and large checkpoints)
rsync -avz --progress \
  --exclude={'.venv','.git','__pycache__','*.pt','*.pdf','scratch*'} \
  -e "ssh -J <user-id>@stujump.comp.nus.edu.sg" \
  ./ <user-id>@xlogin.comp.nus.edu.sg:~/cs4248-project-1/

# Alternatively, using standard scp with ProxyJump:
scp -o "ProxyJump <user-id>@stujump.comp.nus.edu.sg" -r . <user-id>@xlogin.comp.nus.edu.sg:~/cs4248-project-1
```

### 2. Connecting via SSH Jump Host
```bash
# Direct SSH connection through jump host:
ssh -J <user-id>@stujump.comp.nus.edu.sg <user-id>@xlogin.comp.nus.edu.sg

# (Optional) Multiplexed session for fast instant reconnection:
ssh -M -S ~/.ssh/soc-cm-%r@%h:%p -J <user-id>@stujump.comp.nus.edu.sg <user-id>@xlogin.comp.nus.edu.sg
```

### 3. Setting Up Pip Virtual Environment (No UV or Conda Required)
Once logged into `xlogin`:

```bash
cd ~/cs4248-project-1

# Create and activate standard Python 3 venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Submitting the Full 5-Condition Ablation Suite (Slurm)
Submit the full benchmark job to Slurm:

```bash
sbatch scripts/submit_all_ablations.sbatch
```

Monitor job status and real-time execution logs:
```bash
squeue -u $USER
tail -f runs/slurm_all_<JOB_ID>.log
```

> **GPU Architecture Note**: `scripts/submit_all_ablations.sbatch` automatically excludes `xgpd[0-8]` (older TITAN V CC 7.0 nodes) to schedule onto 16GB Tesla T4 (`xgpf[1-8]`, CC 7.5) or A100 nodes with full FP16 mixed precision support.

---

## 📝 ACL 2026 Short Paper Guide & Artifact Mapping

The project report is authored in strict compliance with the **ACL camera-ready format** using the official `acl.sty` and `acl_natbib.bst` style sheets.

### Paper Section & Artifact Mapping
| Paper Section | Page Budget | Corresponding Code & Artifacts |
| :--- | :---: | :--- |
| **§1 Introduction** | 0.75 pages | Motivation: Fragility of QA models to lexical perturbations. Table 2 failure modes (boundary slip, antonym flip). |
| **§2 Related Work** | 0.5 pages | Paraphrase robustness, consistency regularization, Optimal Transport in NLP. |
| **§3 Methodology** | 1.25 pages | **Proposition 1** (KL divergence pathology) vs. **Proposition 2** (1D Wasserstein closed-form CDF integral). 2D Hungarian span matching. |
| **§4 Experimental Setup** | 0.5 pages | SQuAD v1.1, sliding window tokenization, BERT-base, AdamW, decoupled learning rates, $\lambda_{\text{OT}} = 0.5, \mu_{\text{Hung}} = 0.1$. |
| **§5 Empirical Results** | 1.0 page | **Table 1** (`table1_ablation_results.tex`), **Figure 1** (`loss_convergence.pdf`), **Figure 2** (`ablation_benchmark.pdf`), **Figure 3** (`boundary_displacement.pdf`). |
| **§6 Qualitative Analysis** | 0.5 pages | **Table 2** (`table2_qualitative_cases.tex`), detailed failure taxonomy. |
| **References** | 1.0 page (Extra) | Cited ACL literature. |

### 🛠️ Compiling the LaTeX Paper PDF Locally
All LaTeX source files reside in the `report/` directory:

```bash
cd report

# Standard 4-step compilation pass for citations and cross-references:
pdflatex -interaction=batchmode main.tex
bibtex main
pdflatex -interaction=batchmode main.tex
pdflatex -interaction=batchmode main.tex

# Or with latexmk (if installed):
latexmk -pdf main.tex
```

The compiled manuscript will be output to [`report/main.pdf`](report/main.pdf).

### ☁️ Compiling via Overleaf
1. Create a new blank project on [Overleaf](https://www.overleaf.com/).
2. Upload the contents of the `report/` directory:
   - Source: `main.tex`, `custom.bib`, `acl.sty`, `acl_natbib.bst`
   - LaTeX tables: `table1_ablation_results.tex`, `table2_qualitative_cases.tex`
   - Graphics: `figures/` directory containing the vector PDFs (`loss_convergence.pdf`, `ablation_benchmark.pdf`, `boundary_displacement.pdf`)
3. Ensure compiler is set to **pdfLaTeX** with **TeX Live 2023** or newer.
4. Click **Recompile**.

### 📏 Page Budget & Strict Format Verification
* **ACL Short Paper Limit**: $\le 4$ pages of content (Sections 1–7) + $\le 1$ page of references ($\le 5$ pages total).
* **Current Build Verification**: [`report/main.pdf`](report/main.pdf) strictly spans **exactly 5 pages**:
  - Pages 1–4: Title, Abstract, Sections 1–7 (concluding with Limitations).
  - Page 5: 16 cited ACL/ML references.
  - Figure 1 is formatted as a stacked two-panel subplot to prevent column squeezing and label overlap.

---

## 📚 References & Citations

1. **Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P.** (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP.
2. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.** (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
3. **Gan, W. C., & Ng, H. T.** (2019). *Improving the Robustness of Question Answering Systems to Question Paraphrasing.* ACL.
4. **Xie, Q., Dai, Z., Hovy, E., Luong, T., & Le, Q.** (2020). *Unsupervised Data Augmentation for Consistency Training.* NeurIPS.
5. **Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S.** (2020). *End-to-End Object Detection with Transformers.* ECCV.
6. **Peyré, G., & Cuturi, M.** (2019). *Computational Optimal Transport: With Applications to Data Science.* Foundations and Trends in Machine Learning.
