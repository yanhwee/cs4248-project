# Mathematical Foundations of Extractive Question Answering & Paraphrase Consistency
## A Self-Contained, First-Principles Machine Learning Specification

---

## Table of Contents
1. [Problem Formulation: Extractive QA from First Principles](#1-problem-formulation-extractive-qa-from-first-principles)
2. [Document Chunking & Sliding Window Mechanics](#2-document-chunking--sliding-window-mechanics)
3. [Model Architecture & The Forward Computation Graph](#3-model-architecture--the-forward-computation-graph)
4. [Standard Supervised Learning & Cross-Entropy Loss](#4-standard-supervised-learning--cross-entropy-loss)
5. [The Fragility of Lexical Overlap & Consistency Regularization](#5-the-fragility-of-lexical-overlap--consistency-regularization)
6. [Token-Level Consistency Regularization: Symmetric KL vs. 1D Optimal Transport](#6-token-level-consistency-regularization-symmetric-kl-vs-1d-optimal-transport)
7. [From 1D Marginal Tokens to 2D Phrases: Span-Level Hungarian Matching](#7-from-1d-marginal-tokens-to-2D-phrases-span-level-hungarian-matching)
8. [The Unified Optimization Objective & Second-Stage Span Selection](#8-the-unified-optimization-objective--second-stage-span-selection)
9. [Appendix: Comprehensive Symbol Glossary & Type Signatures](#9-appendix-comprehensive-symbol-glossary--type-signatures)

---

## 1. Problem Formulation: Extractive QA from First Principles

### 1.1. The Raw Text Space
Let $\Sigma$ denote the finite set of Unicode characters (the text alphabet). The space of all finite-length strings is denoted by the Kleene closure $\Sigma^*$. 

An extractive question-answering dataset (such as SQuAD v1.1) is formalised as a collection of $N$ ground-truth instances:
$$\mathcal{D} = \big\{ (q_n, c_n, a^*_n) \big\}_{n=1}^N$$

From a statistical learning perspective, our goal is to approximate a hypothesis function:
$$f_\theta : \mathcal{X} \to \mathcal{Y}$$
where:
* **Input Domain $\mathcal{X} = \Sigma^* \times \Sigma^*$**: The cartesian product of question strings $q$ and context passage strings $c$.
* **Structured Output Space $\mathcal{Y}_c \subset \mathbb{N}^2$**: Unlike unconstrained generation where $\mathcal{Y} = \Sigma^*$, extractive QA exploits the inductive bias that the answer is always an exact excerpt of $c$. Thus, $\mathcal{Y}_c$ is the finite discrete set of valid start-end character intervals:
  $$\mathcal{Y}_c = \big\{ (a_{\text{start}}, a_{\text{end}}) \in \mathbb{N}^2 \;\big\vert\; 0 \le a_{\text{start}} \le a_{\text{end}} < \vert c \vert \big\}$$

For every training triplet $(q, c, a^*)$:
1. $q \in \Sigma^*$ is the **Question string** of length $\vert q \vert \in \mathbb{N}$.
2. $c \in \Sigma^*$ is the **Context Passage string** of length $\vert c \vert \in \mathbb{N}$ containing the evidence.
3. $a^* \in \Sigma^*$ is the gold-standard **Answer text substring**. In an extractive paradigm, $a^*$ is strictly a contiguous substring of passage $c$, defined by integer character offsets:
   $$a^* = c[a_{\text{start}} : a_{\text{end}} + 1]$$
   where $0 \le a_{\text{start}} \le a_{\text{end}} < \vert c \vert$.

> [!NOTE]
> **Extractive vs. Generative Formulation**: The model does *not* generate text autoregressively token-by-token. Its objective is strictly classification over candidate boundaries: identifying the exact boundary indices $(a_{\text{start}}, a_{\text{end}})$ directly within the context string $c$. Standard empirical risk minimization tunes parameters $\theta$ to maximize the likelihood of the true boundary pair under function $f_\theta$.

---

### 1.2. Subword Tokenization & The Offset Mapping ($\mathcal{T}_V$)
Continuous neural networks cannot directly process character strings. Instead, a deterministic subword tokenizer $\mathcal{T}_V$ parameterized by a finite vocabulary $V = \{0, 1, \dots, \vert V \vert - 1\}$ converts character strings into discrete sequences of integer token IDs.

1. **Tokenization Mappings**:
   $$q \xrightarrow{\mathcal{T}_V} \mathbf{q} = (q_1, \dots, q_{r}) \in V^r, \qquad c \xrightarrow{\mathcal{T}_V} \mathbf{c} = (c_0, \dots, c_{m-1}) \in V^m$$
   where $r \in \mathbb{N}$ and $m \in \mathbb{N}$ are the token sequence lengths of the question and passage, respectively.

2. **The Character-to-Token Offset Map $\phi$**:
   To project character-level answers to token indices, the tokenizer maintains an explicit offset map $\phi$:
   $$\phi: \{0, \dots, m-1\} \to \mathbb{N} \times \mathbb{N}$$
   $$\phi(t) = (\alpha_t, \beta_t)$$
   where $\alpha_t \in \mathbb{N}$ is the inclusive start character offset and $\beta_t \in \mathbb{N}$ is the exclusive end character offset of token $c_t$ in string $c$, satisfying $\text{substring}(c_t) = c[\alpha_t : \beta_t]$.

3. **Projecting Character Ground Truth to Token Targets**:
   Using offset mapping $\phi$, the raw character ground truth $[a_{\text{start}}, a_{\text{end}}]$ is projected into discrete **passage token target indices** $(s^*, e^*) \in \{0, \dots, m-1\}^2$:
   $$s^* = \min \big\{ t \in \{0, \dots, m-1\} \;\big\vert\; \beta_t > a_{\text{start}} \big\}$$
   $$e^* = \max \big\{ t \in \{0, \dots, m-1\} \;\big\vert\; \alpha_t \le a_{\text{end}} \big\}$$

This guarantees that the token sequence $(c_{s^*}, \dots, c_{e^*})$ minimally and fully covers the original character answer string $a^*$.

---

## 2. Document Chunking & Sliding Window Mechanics

### 2.1. The Sequence Length Bottleneck
Transformer self-attention exhibits $\mathcal{O}(L^2)$ time and memory complexity with respect to total sequence length $L$. Pre-trained language models (such as BERT and DeBERTa-v3) enforce a fixed maximum sequence capacity, typically $L = 384$ or $L = 512$ tokens.

However, real-world passages often contain 800 to 1,500 subwords ($m \gg L$). If we simply truncated passages at token $L$, all answers appearing beyond the cutoff would be irrevocably lost. To retain full document recall without violating memory boundaries, we partition lengthy passages using a **sliding window with stride**.

---

### 2.2. The Sliding Window Formulation
Let $L \in \mathbb{N}$ be the model's total token budget (e.g., $L = 384$), and let $r = \vert\mathbf{q}\vert$ be the token length of query $q$. The budget allocated to the passage within any single model input is:
$$L_{\text{passage}} = L - r - 3$$
where the constant $3$ accounts for the structural delimiter tokens: $\texttt{[CLS]}$, $\texttt{[SEP]}$, and $\texttt{[SEP]}$.

Let $S \in \mathbb{N}$ denote the **sliding window stride** (e.g., $S = 128$). The full passage token sequence $\mathbf{c} = (c_0, \dots, c_{m-1})$ is partitioned into $W$ overlapping chunks:

$$W = 1 + \max\left(0, \; \left\lceil \frac{m - L_{\text{passage}}}{S} \right\rceil\right)$$

where chunk $w \in \{0, 1, \dots, W-1\}$ spans:
$$\mathbf{c}^{(w)} = \big( c_{w \cdot S}, \; c_{w \cdot S + 1}, \; \dots, \; c_{\min(m, \, w \cdot S + L_{\text{passage}}) - 1} \big) \in V^{m_w}$$

```
Passage Tokens: [------------------ m tokens ------------------]
Window 0:       [====== L_passage ======]
Window 1:                 [====== L_passage ======]   <-- Stride S shift
Window 2:                           [====== L_passage ======]
```

> [!TIP]
> **Why Overlap with Stride $S$?**: If windows were partitioned without overlap ($S = L_{\text{passage}}$), an answer span that crosses the boundary between window $w$ and window $w+1$ would be split in half, preventing either window from capturing the full answer. Setting $S \ll L_{\text{passage}}$ (e.g., $128$ vs $300$) guarantees that any answer of length up to $(L_{\text{passage}} - S)$ tokens is fully contained within at least one window.

---

### 2.3. Window-Local Ground-Truth Assignment
Each window $w$ constitutes an independent training feature $\mathbf{x}^{(w)} \in V^L$. Because the global answer span $(s^*, e^*)$ may or may not fall inside window $w$, target labels must be assigned conditionally:

Let $[t_{\text{start}}^{(w)}, t_{\text{end}}^{(w)}] = [w \cdot S, \; w \cdot S + m_w - 1]$ be the global token range covered by window $w$.
The local target indices $(s^{(w)}, e^{(w)}) \in \{0, \dots, L-1\}^2$ are determined by:

$$(s^{(w)}, e^{(w)}) = \begin{cases}
\big( s^* - w \cdot S + (r + 2), \; e^* - w \cdot S + (r + 2) \big) & \text{if } t_{\text{start}}^{(w)} \le s^* \le e^* \le t_{\text{end}}^{(w)} \\
(0, 0) & \text{otherwise (answer not in window)}
\end{cases}$$

Here, index $0$ refers to the $\texttt{[CLS]}$ token at the very start of the sequence. During training, directing probability mass to index $0$ serves as the formal signal that **the question is unanswerable from the context provided in this particular window**.

---

### 2.4. Multi-Window Inference & Relative Margin Aggregation

At inference time, each window $w \in \{0, \dots, W-1\}$ produces independent unnormalized boundary logit vectors $\mathbf{z}_{\text{start}}^{(w)}, \mathbf{z}_{\text{end}}^{(w)} \in \mathbb{R}^L$. Because overlapping windows process shared passage regions with stride $S$, multiple windows often propose candidate answer spans covering the same underlying tokens. 

A central question arises: **how should we compare and aggregate predictions across independent windows?**

#### 2.4.1. Why Naive Logit Averaging Fails
A simple heuristic might be to average raw boundary logits $(z_{\text{start}}^{(w)}[i] + z_{\text{end}}^{(w)}[j])$ across overlapping windows. In practice, this naive arithmetic mean breaks down due to three mathematical and structural reasons:
1. **Window-Edge Truncation Dilutes Confidence**:
   Suppose the true answer resides at global passage tokens $[240, 250]$. Window 0 (covering tokens $[0, 256]$) clips the passage immediately after the answer, starving the Transformer of downstream syntactic and causal context; its model outputs low or negative logits. In contrast, Window 1 (covering tokens $[128, 384]$) sees the answer centered with full bilateral context, producing high confidence ($+12.0$). An arithmetic mean penalizes the valid prediction by averaging in the truncated window's confusion.
2. **Arbitrary DC Baseline Shifts Between Forward Passes**:
   Each window undergoes an independent forward pass. Due to variations in token representations, LayerNorm activations, and background sequence lengths, raw logits undergo arbitrary additive constant shifts across windows (i.e., Softmax is invariant to shifts $\mathbf{z} \to \mathbf{z} + c$). Window A might center logits around $+20$ with its unanswerable score at $+19$ (a net margin of $+1$), while Window B centers logits around $+2$ with its unanswerable score at $-8$ (a net margin of $+10$). Averaging raw logits erroneously favors Window A despite Window B possessing far higher relative confidence.
3. **Disjunctive Evidence Logic ($\text{OR}$ vs. $\text{AND}$)**:
   In multi-window reading comprehension, an answer is valid if evidence is present in Window 0 **OR** Window 1 **OR** Window 2. In fuzzy and probabilistic logic, the disjunctive aggregation of independent hypotheses corresponds to the supremum ($\bigvee \equiv \max$), whereas an arithmetic average acts as a conjunctive agreement requirement that penalizes windows that only partially overlap the evidence.

#### 2.4.2. First-Principles Derivation: Log-Odds and Partition Cancellation
To place all windows on a mathematically unified and calibrated scale, we compare candidate spans against the **null hypothesis** (the state where a window contains no answer and directs its probability mass to index $0$, corresponding to $\texttt{[CLS]}$).

Under standard Softmax normalization over sequence positions $0, \dots, L-1$:
$$P_{\text{start}}(i \mid w) = \frac{\exp(z_{\text{start}}^{(w)}[i])}{\sum_{l=0}^{L-1} \exp(z_{\text{start}}^{(w)}[l])} = \frac{\exp(z_{\text{start}}^{(w)}[i])}{Z_s^{(w)}}, \qquad P_{\text{end}}(j \mid w) = \frac{\exp(z_{\text{end}}^{(w)}[j])}{\sum_{k=0}^{L-1} \exp(z_{\text{end}}^{(w)}[k])} = \frac{\exp(z_{\text{end}}^{(w)}[j])}{Z_e^{(w)}}$$

Here, the denominators $Z_s^{(w)} \triangleq \sum_{l=0}^{L-1} \exp(z_{\text{start}}^{(w)}[l])$ and $Z_e^{(w)} \triangleq \sum_{k=0}^{L-1} \exp(z_{\text{end}}^{(w)}[k])$ represent the **normalization constants** (or **partition functions**) for start and end boundaries in window $w$.

Under boundary conditional independence, the joint probability assigned by window $w$ to a candidate span $(i, j)$ is:
$$P(\text{span}=(i, j) \mid w) = P_{\text{start}}(i \mid w) \cdot P_{\text{end}}(j \mid w) = \frac{\exp(z_{\text{start}}^{(w)}[i] + z_{\text{end}}^{(w)}[j])}{Z_s^{(w)} \cdot Z_e^{(w)}}$$

Similarly, the probability that window $w$ considers the question unanswerable (null prediction at index 0) is:
$$P(\text{null} \mid w) = P_{\text{start}}(0 \mid w) \cdot P_{\text{end}}(0 \mid w) = \frac{\exp(z_{\text{start}}^{(w)}[0] + z_{\text{end}}^{(w)}[0])}{Z_s^{(w)} \cdot Z_e^{(w)}}$$

Evaluating the **Likelihood Ratio** between the candidate answer and the null hypothesis reveals that the window-specific partition constants $Z_s^{(w)}$ and $Z_e^{(w)}$ **cancel out identically**:
$$\frac{P(\text{span}=(i, j) \mid w)}{P(\text{null} \mid w)} = \frac{\frac{\exp(z_{\text{start}}^{(w)}[i] + z_{\text{end}}^{(w)}[j])}{Z_s^{(w)} \cdot Z_e^{(w)}}}{\frac{\exp(z_{\text{start}}^{(w)}[0] + z_{\text{end}}^{(w)}[0])}{Z_s^{(w)} \cdot Z_e^{(w)}}} = \frac{\exp(z_{\text{start}}^{(w)}[i] + z_{\text{end}}^{(w)}[j])}{\exp(z_{\text{start}}^{(w)}[0] + z_{\text{end}}^{(w)}[0])}$$

Taking the natural logarithm yields the exact **Relative Margin (Log-Likelihood Ratio / Log-Odds)**:
$$\ln \left( \frac{P(\text{span}=(i, j) \mid w)}{P(\text{null} \mid w)} \right) = \big(z_{\text{start}}^{(w)}[i] + z_{\text{end}}^{(w)}[j]\big) - \big(z_{\text{start}}^{(w)}[0] + z_{\text{end}}^{(w)}[0]\big) = \text{score}^{(w)}(i, j) - \text{score}_{\text{null}}^{(w)}$$

* If $\text{score}^{(w)}(i, j) - \text{score}_{\text{null}}^{(w)} > 0$, the candidate span is strictly more probable than having no answer in window $w$.
* Subtracting $\text{score}_{\text{null}}^{(w)}$ anchors every window to a universal baseline ($0 \equiv \text{null}$), eliminating arbitrary inter-window DC offsets.

#### 2.4.3. The Complete Aggregation Pipeline
Combining multi-window candidate spans into a final global prediction follows a four-step aggregation procedure:

1. **Local Scoring**: In each window $w$, compute candidate scores and the null baseline:
   $$\text{score}^{(w)}(i, j) = z_{\text{start}}^{(w)}[i] + z_{\text{end}}^{(w)}[j], \qquad \text{score}_{\text{null}}^{(w)} = z_{\text{start}}^{(w)}[0] + z_{\text{end}}^{(w)}[0]$$
   for all local indices satisfying $r+2 \le i \le j < r+2+m_w$ and $(j - i + 1) \le L_{\max}$.

2. **Global Coordinate Projection**: Map local indices $(i, j)$ to absolute passage token indices:
   $$i_{\text{global}} = i - (r + 2) + (w \cdot S), \qquad j_{\text{global}} = j - (r + 2) + (w \cdot S)$$

3. **Relative Margin Deduplication**: For each unique global span $(i_{\text{global}}, j_{\text{global}})$, take the maximum relative margin across all overlapping windows $\mathcal{W}(i_{\text{global}}, j_{\text{global}})$ that contain it:
   $$\text{Score}_{\text{final}}(i_{\text{global}}, j_{\text{global}}) = \max_{w \in \mathcal{W}(i_{\text{global}}, j_{\text{global}})} \Big( \text{score}^{(w)}(i, j) - \text{score}_{\text{null}}^{(w)} \Big)$$

4. **Optimal Span Selection**: The globally predicted answer is the span maximizing the aggregated relative score:
   $$(i^*, j^*) = \arg\max_{i_{\text{global}}, j_{\text{global}}} \text{Score}_{\text{final}}(i_{\text{global}}, j_{\text{global}})$$

---

## 3. Model Architecture & The Forward Computation Graph

### 3.1. Unified Input Packing
The query tokens $\mathbf{q} \in V^r$ and passage chunk tokens $\mathbf{c} \in V^m$ are packed into a single linear sequence $\mathbf{x} \in V^L$:
$$\mathbf{x} = \big[ \texttt{[CLS]}, \; q_1, \dots, q_r, \; \texttt{[SEP]}, \; c_0, \dots, c_{m-1}, \; \texttt{[SEP]}, \; \texttt{[PAD]}, \dots \big]$$

To isolate the passage tokens during loss calculation and optimal transport, we construct a **Boolean Passage Mask** $\mathbf{m}_{\text{pass}} \in \{0, 1\}^L$:
$$m_{\text{pass}}[t] = \begin{cases} 1 & \text{if token } t \text{ is an answerable context token } c_i \\ 0 & \text{if token } t \in \{ \texttt{[CLS]}, \mathbf{q}, \texttt{[SEP]}, \texttt{[PAD]} \} \end{cases}$$

Let $\mathcal{P} = \{ t \in \{0, \dots, L-1\} \mid m_{\text{pass}}[t] = 1 \}$ denote the ordered sequence indices corresponding to the $m$ passage tokens.

---

### 3.2. Forward Computational Pipeline

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

1. **Transformer Encoder ($\mathcal{E}_\theta$)**:
   The encoder $\mathcal{E}_\theta : V^L \to \mathbb{R}^{L \times d}$ maps discrete token IDs into contextualized vectors:
   $$\mathbf{H} = \mathcal{E}_\theta(\mathbf{x}) = [\mathbf{h}_0, \mathbf{h}_1, \dots, \mathbf{h}_{L-1}]^\top \in \mathbb{R}^{L \times d}$$
   where $d \in \mathbb{N}$ is the hidden dimension ($d = 768$ for base models). Every vector $\mathbf{h}_t \in \mathbb{R}^d$ encodes token $t$ conditioned bidirectionally on all tokens in the sequence through multi-head self-attention.

2. **Linear Span Prediction Head**:
   A linear layer parameterized by weight matrix $\mathbf{W} = [\mathbf{w}_{\text{start}}, \mathbf{w}_{\text{end}}] \in \mathbb{R}^{d \times 2}$ and bias $\mathbf{b} = [b_{\text{start}}, b_{\text{end}}]^\top \in \mathbb{R}^2$ scores boundary positions across the entire sequence:
   $$\mathbf{z}_{\text{start}} = \mathbf{H} \mathbf{w}_{\text{start}} + b_{\text{start}} \mathbf{1}_L \in \mathbb{R}^L$$
   $$\mathbf{z}_{\text{end}} = \mathbf{H} \mathbf{w}_{\text{end}} + b_{\text{end}} \mathbf{1}_L \in \mathbb{R}^L$$

3. **Bifurcated Logit Normalization**:
   * **Full-Sequence Pathway (Supervised Training & Null Calibration)**: Softmax is evaluated across the sequence length $L$ (with padding tokens masked to $-\infty$). This pathway trains boundary positions and explicitly supervises index $0$ (`[CLS]`) on unanswerable windows to calibrate $\text{score}_{\text{null}}$:
     $$P_{\text{start}}^{\text{seq}} = \mathrm{softmax}(\mathbf{z}_{\text{start}}) \in \Delta^{L-1}, \qquad P_{\text{end}}^{\text{seq}} = \mathrm{softmax}(\mathbf{z}_{\text{end}}) \in \Delta^{L-1}$$
   * **Passage-Only Simplex Pathway (Metric Consistency)**: Slicing logits via index set $\mathcal{P} = \{t \mid m_{\text{pass}}[t] = 1\}$ isolates the $m$ passage tokens:
     $$\mathbf{s} = \mathbf{z}_{\text{start}}[\mathcal{P}] \in \mathbb{R}^m, \qquad \mathbf{e} = \mathbf{z}_{\text{end}}[\mathcal{P}] \in \mathbb{R}^m$$
     Applying Softmax over these $m$ passage logits projects them onto the passage probability simplex $\Delta^{m-1} = \{ \mathbf{p} \in \mathbb{R}^m \mid p_i \ge 0, \sum_{i=0}^{m-1} p_i = 1 \}$:
     $$P_{\text{start}}(i) = \frac{\exp(s_i)}{\sum_{l=0}^{m-1} \exp(s_l)}, \qquad P_{\text{end}}(j) = \frac{\exp(e_j)}{\sum_{l=0}^{m-1} \exp(e_l)}$$
     This guarantees that distribution support is strictly confined to the 1D physical passage tokens, enabling geometrically sound Optimal Transport and Hungarian candidate matching.

---

## 4. Standard Supervised Learning & Cross-Entropy Loss

### 4.1. First-Principles Derivation via Maximum Likelihood Estimation (MLE)
In statistical machine learning, training the model parameters $\theta$ corresponds to maximizing the conditional log-likelihood of the observed training instances.

For window $w$ with window-local ground-truth target indices $(s^{(w)}, e^{(w)}) \in \{0, \dots, L-1\}^2$ assigned via Section 2.3:
* If the answer falls within the window, $(s^{(w)}, e^{(w)}) = (s^* - wS + r + 2, \; e^* - wS + r + 2)$.
* If the answer does not fall within the window, $(s^{(w)}, e^{(w)}) = (0, 0)$ (the $\texttt{[CLS]}$ null token).

We model the joint conditional probability of start and end boundary positions under conditional independence given $\mathbf{H} = \mathcal{E}_\theta(\mathbf{x})$:
$$P(S = s^{(w)}, E = e^{(w)} \mid \mathbf{x}) = P_{\text{start}}^{\text{seq}}(s^{(w)}) \cdot P_{\text{end}}^{\text{seq}}(e^{(w)})$$

Maximizing likelihood is mathematically equivalent to minimizing Negative Log-Likelihood (NLL), which equates to sequence categorical cross-entropy:
$$\mathcal{L}_{\text{CE}}(q) = -\frac{1}{2} \Big[ \log P_{\text{start}}^{\text{seq}}(s^{(w)}) + \log P_{\text{end}}^{\text{seq}}(e^{(w)}) \Big]$$

> [!NOTE]
> **Why Sequence-Level Softmax Matters for $\mathcal{L}_{\text{CE}}$**: Computing $\mathcal{L}_{\text{CE}}$ over the full sequence (including index $0$) ensures that sliding windows devoid of the gold answer actively supervise $\texttt{[CLS]}$ as the unanswerable state. This trains the baseline logit parameters $(z_{\text{start}}[0], z_{\text{end}}[0])$ required for the log-odds margin cancellation derived in Section 2.4.2. In contrast, consistency losses ($\mathcal{L}_{\text{OT}}, \mathcal{L}_{\text{Hungarian}}$) are evaluated on the passage simplex $\Delta^{m-1}$ using paired answer-bearing contexts.

---

### 4.2. Test-Time Decoding: Why Maximizing $(s_i + e_j)$ is Identical to Maximizing Joint Probability
At inference time, the model must select a valid contiguous span interval $(i^*, j^*)$ satisfying:
$$0 \le i \le j < m \qquad \text{and} \qquad (j - i + 1) \le L_{\max}$$
where $L_{\max} \in \mathbb{N}$ is the maximum permissible answer token length (typically $L_{\max} = 30$).

Under Maximum A Posteriori (MAP) decoding, we seek the span maximizing the joint posterior probability:
$$(i^*, j^*) = \arg\max_{\substack{0 \le i \le j < m \\ j - i + 1 \le L_{\max}}} P(S = i, E = j \mid \mathbf{x})$$

Substituting the independent softmax probability definitions:
$$P(S = i, E = j \mid \mathbf{x}) = P_{\text{start}}(i) \cdot P_{\text{end}}(j) = \left( \frac{\exp(s_i)}{\sum_{l=0}^{m-1} \exp(s_l)} \right) \cdot \left( \frac{\exp(e_j)}{\sum_{k=0}^{m-1} \exp(e_k)} \right)$$

Let $Z_s = \sum_{l=0}^{m-1} \exp(s_l)$ and $Z_e = \sum_{k=0}^{m-1} \exp(e_k)$ denote the respective Softmax normalization partition functions. Crucially:
* $Z_s$ and $Z_e$ depend solely on the full passage logit vectors $\mathbf{s}, \mathbf{e} \in \mathbb{R}^m$.
* For any candidate span $(i, j)$ evaluated within the passage, $Z_s$ and $Z_e$ are strictly constant invariants.

Taking the natural logarithm (which is a strictly monotonically increasing function on $\mathbb{R}_{>0}$ and therefore preserves the $\arg\max$ argument exactly):
$$\begin{aligned}
\arg\max_{i, j} P(S = i, E = j \mid \mathbf{x}) &= \arg\max_{i, j} \log \Big( P(S = i, E = j \mid \mathbf{x}) \Big) \\
&= \arg\max_{i, j} \log \left( \frac{\exp(s_i) \cdot \exp(e_j)}{Z_s \cdot Z_e} \right) \\
&= \arg\max_{i, j} \Big( \log \exp(s_i) + \log \exp(e_j) - \log Z_s - \log Z_e \Big) \\
&= \arg\max_{i, j} \Big( s_i + e_j - \underbrace{(\log Z_s + \log Z_e)}_{\text{constant with respect to } (i, j)} \Big) \\
&= \arg\max_{i, j} \big( s_i + e_j \big)
\end{aligned}$$

> [!TIP]
> **Computational Efficiency at Inference**: Because of this algebraic identity, the model never needs to compute exponentiations or partition sums over tokens during evaluation. Joint decoding reduces to finding the maximum sum of unnormalized logits $(s_i + e_j)$ over the upper-triangular band where $0 \le j - i < L_{\max}$, executing in $\mathcal{O}(m \cdot L_{\max})$ time with minimal FLOPs.

---

## 5. The Fragility of Lexical Overlap & Consistency Regularization

### 5.1. The Failure Mode: Shortcut Learning
Standard Empirical Risk Minimization ($\min_\theta \mathbb{E}[\mathcal{L}_{\text{CE}}]$) frequently converges to **superficial lexical matching**:
* If question $q$ contains the keyword `"established"`, the self-attention heads learn to locate the exact word `"established"` in passage $c$ and extract the immediately adjacent noun phrase.
* When the query is rephrased as $q'$ (*"When was the university founded?"*), the lexical anchor disappears.
* Despite the context passage $c$ containing identical semantic facts, the predicted distributions $P(q')$ shift erratically away from the true span.

### 5.2. Semantic Invariance & Paraphrase Generation
By first principles, if question $q$ and paraphrase $q'$ share identical semantic intent ($\text{Sem}(q) \equiv \text{Sem}(q')$), the true posterior distribution over passage answers must be invariant:
$$P(\text{answer} \mid q, c) \equiv P(\text{answer} \mid q', c)$$

To generate high-quality paraphrases offline without human annotation:
1. **Back-Translation via Pivot Language $\mathcal{Z}$**:
   We translate question $q$ into an intermediate pivot language $\mathcal{Z}$ and back to English using open neural machine translation models:
   $$q \xrightarrow{\text{NMT}_{EN \to \mathcal{Z}}} q_{\mathcal{Z}} \xrightarrow{\text{NMT}_{\mathcal{Z} \to EN}} q'$$

   * **Why German (DE) as the Primary Pivot?**:
     * **Syntactic Restructuring**: German features Verb-Second (V2) main-clause order, Subject-Object-Verb (SOV) subordinate clauses, and productive compounding. Translating into and out of German forces the model to dismantle and reconstruct the English grammatical parse tree, producing genuine syntactic paraphrases rather than trivial word substitutions.
     * **Corpus & Model Maturity**: High-resource European parallel corpora (WMT, Europarl) ensure high-fidelity MarianMT models (`Helsinki-NLP/opus-mt-en-de` and `de-en`) with negligible hallucination.
     * **Trade-Offs with Other Languages**:
       * *Romance Languages (French, Spanish)*: Closer syntactic alignment to English (SVO), often yielding minor lexical synonym replacements rather than structural shifts.
       * *Distant Languages (Chinese, Japanese, Arabic)*: High syntactic variance, but higher risk of semantic drift or unfaithful question inversion (e.g., flipping "who" into "what").
       * *Multi-Hop Pivot Chains (e.g., $EN \to DE \to FR \to EN$)*: Incurs compounding translation errors. German provides the optimal Pareto trade-off between structural diversity and semantic preservation.

2. **Semantic Verification Filter ($\mathbf{e}_q$)**:
   * **Semantic Drift in Automated Augmentation**:
     While neural machine translation produces fluent paraphrases, automated translation models occasionally introduce subtle *semantic drift* or interrogative inversion—for example, mutating *"When was Abraham Lincoln born?"* into *"Where was Abraham Lincoln born?"*. If such an unfaithful pair were passed into consistency training, the objective would erroneously penalize the model for correctly predicting distinct answer categories (a date vs. a location), corrupting the representation space. Consequently, every candidate pair $(q, q')$ must pass an automated semantic equivalence gate before entering consistency regularization.
   * **Sequence Encoding & Attention-Masked Centroid Pooling**:
     To verify semantic equivalence, questions $q$ and $q'$ are encoded using a pre-trained sentence encoder $\mathcal{E}_{\text{sbert}}$ (specifically `sentence-transformers/all-MiniLM-L6-v2`, fine-tuned via Siamese metric learning).
     
     Because natural language queries vary in token count, batched tensor execution pads sequences to a common capacity $L_q$ by appending dummy $\texttt{[PAD]}$ tokens (token ID $0$). To instruct the Transformer's attention mechanism to ignore these positions, the batch is paired with a binary **attention mask** $\mathbf{m}_{\text{attn}} \in \{0, 1\}^{L_q}$:
     $$m_{\text{attn}}[t] = \begin{cases} 1 & \text{if token } t \text{ is an informative word token} \\ 0 & \text{if token } t \text{ is a dummy padding token } \texttt{[PAD]} \end{cases}$$
     
     The encoder maps the input tokens into contextualized hidden vectors $\mathbf{H}_q = [\mathbf{h}_{q, 1}, \dots, \mathbf{h}_{q, L_q}] \in \mathbb{R}^{L_q \times d_{\text{sbert}}}$ ($d_{\text{sbert}} = 384$). To collapse this variable-length sequence into a single fixed-dimensional sentence embedding without allowing arbitrary padding values to corrupt the representation, we compute **attention-masked mean pooling**:
     $$\mathbf{e}_q = \frac{\sum_{t=1}^{L_q} \mathbf{h}_{q, t} \cdot m_{\text{attn}}[t]}{\max\left( \sum_{t=1}^{L_q} m_{\text{attn}}[t], \, \epsilon \right)} \;\in\; \mathbb{R}^{d_{\text{sbert}}}$$
     Multiplying by $m_{\text{attn}}[t]$ zeros out hidden vectors corresponding to $\texttt{[PAD]}$, while dividing by $\sum m_{\text{attn}}[t]$ normalizes strictly by the number of true words, yielding the true semantic centroid of question $q$.
   * **Cosine Similarity Gating**:
     Both sentence embeddings are projected onto the unit hypersphere:
     $$\hat{\mathbf{e}}_q = \frac{\mathbf{e}_q}{\|\mathbf{e}_q\|_2}, \qquad \hat{\mathbf{e}}_{q'} = \frac{\mathbf{e}_{q'}}{\|\mathbf{e}_{q'}\|_2}$$
     Their semantic equivalence is measured via their inner product (cosine similarity):
     $$\text{sim}(q, q') = \hat{\mathbf{e}}_q^\top \hat{\mathbf{e}}_{q'} \ge \tau \quad (\tau = 0.85)$$
     Candidate pairs satisfying $\text{sim}(q, q') \ge 0.85$ are admitted to paired consistency regularization, whereas pairs exhibiting semantic drift are discarded.

### 5.3. Why Data Augmentation Alone Is Insufficient: Equivalence of $\mathcal{L}_{\text{Aug}}$ and $\mathcal{L}_{\text{supervised}}$
Prior work (*Gan & Ng, 2019*) introduces paraphrased questions as independent training samples, which is formalized as:
$$\mathcal{L}_{\text{Aug}} = \frac{1}{2}\big(\mathcal{L}_{\text{CE}}(q) + \mathcal{L}_{\text{CE}}(q')\big)$$

In our paired consistency framework, each training instance is a paired tuple $(q, q', c, a^*)$. Thus, our supervised loss component is **identically equal to data augmentation**:
$$\mathcal{L}_{\text{supervised}} \equiv \mathcal{L}_{\text{Aug}} = \frac{1}{2}\big(\mathcal{L}_{\text{CE}}(q) + \mathcal{L}_{\text{CE}}(q')\big)$$

While $\mathcal{L}_{\text{Aug}}$ exposes the encoder to more phrasing diversity, **it places zero penalty on prediction disagreement**. The model remains entirely free to predict span $A$ for question $q$ and span $B$ for question $q'$. Setting consistency weights to zero ($\lambda_{\text{OT}} = 0, \mu_{\text{Hung}} = 0$) collapses our unified objective identically back to data augmentation $\mathcal{L}_{\text{Aug}}$.

To enforce true robustness, training must explicitly penalize distributional divergence between the pair:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}} + \mathcal{L}_{\text{consistency}}\big(P(q), P(q')\big)$$

---

## 6. Token-Level Consistency Regularization: Symmetric KL vs. 1D Optimal Transport

Feeding both $q$ and $q'$ with passage $c$ through the shared encoder yields two discrete probability distributions over the $m$ passage tokens for both boundary heads ($t \in \{\text{start}, \text{end}\}$):
$$P_t = P_t^{(q)} \in \Delta^{m-1}, \qquad P'_t = P_t^{(q')} \in \Delta^{m-1}$$

How should we define the discrepancy $\mathcal{D}(P_t, P'_t)$ between these two distributions? We formulate and compare our **primary consistency regularizer (Symmetric KL Divergence)** against a **metric-aware ablation (1D Wasserstein Optimal Transport)**.

---

### 6.1. Primary Objective: Symmetric KL Divergence ($\mathcal{D}_{\text{KL}}^{\text{sym}}$)
Following consistency training principles (*Xie et al., 2020*), our primary consistency term penalizes divergence between the passage-token start and end distributions of $q$ and $q'$ using symmetric Kullback-Leibler (KL) divergence (`--loss_mode kl`):
$$\mathcal{D}_{\text{KL}}^{\text{sym}} = \frac{1}{4} \sum_{t \in \{\text{start}, \text{end}\}} \Big[ \mathrm{KL}(P_t \parallel P'_t) + \mathrm{KL}(P'_t \parallel P_t) \Big]$$
where:
$$\mathrm{KL}(P \parallel Q) = \sum_{i=0}^{m-1} P(i) \log \frac{P(i)}{Q(i)}$$

#### Why Symmetric KL Works Strongly under Joint Supervision:
When trained jointly with supervised cross-entropy on both views ($\mathcal{L}_{\text{CE}}(q) + \mathcal{L}_{\text{CE}}(q')$), both $P_t$ and $P'_t$ are anchored toward the true gold boundary $(s^*, e^*)$. Symmetric KL directly penalizes log-probability disagreement at every token index, strongly encouraging the two phrasings to concentrate probability mass on the exact same start and end tokens (critical for Exact Match).

#### Theoretical Motivation for a Metric-Space Alternative:
From a purely geometric perspective, KL divergence treats passage token indices $\{0, 1, \dots, m-1\}$ as an **unordered discrete set of categorical classes**, implicitly assuming the discrete indicator ground metric:
$$d_{\text{KL}}(i, j) = \begin{cases} 0 & \text{if } i = j \\ \infty & \text{if } i \neq j \end{cases}$$

Consider an unanchored prediction for an answer starting at token index 10 (`"Virgin Mary"`):
* **Case 1 (1-token boundary shift)**: $P$ peaks at token 10 (`"Virgin"`), while $Q$ peaks at token 11 (`"Mary"`).
* **Case 2 (80-token distractor jump)**: $P$ peaks at token 10, while $Q$ peaks at token 90 (an unrelated paragraph).

$$
\begin{array}{lcccc|c|c}
\hline
\textbf{Sequence Position } t & \mathbf{10} & \mathbf{11} & \dots & \mathbf{90} & \text{Metric Distance } \vert i - j \vert & \text{KL Divergence } \mathcal{D}_{\text{KL}} \\
\hline
\textbf{Context Token } c_t & \text{\small [Virgin]} & \text{\small [Mary]} & \dots & \text{\small [Distractor]} & \text{Physical Separation} & \text{Categorical Metric} \\
\hline\hline
\textbf{Case 1 (1-Token Slip)} & P(10)=1 & Q(11)=1 & \dots & 0 & \vert 10 - 11 \vert = \mathbf{1} & \mathcal{D}_{\text{KL}} \to \boldsymbol{\infty} \\
\textbf{Case 2 (Distractor Jump)} & P(10)=1 & 0 & \dots & Q(90)=1 & \vert 10 - 90 \vert = \mathbf{80} & \mathcal{D}_{\text{KL}} \to \boldsymbol{\infty} \\
\hline
\end{array}
$$

> [!NOTE]
> **Motivating the Optimal Transport Ablation**: Because KL divergence is position-agnostic ($d_{\text{KL}}(10,11) = d_{\text{KL}}(10,90)$), we also derive and evaluate a 1D Optimal Transport ($W_1$) consistency loss (`--loss_mode ot`) that incorporates physical token distance $\vert i - j \vert / m$. Empirically on SQuAD v1.1 (see [docs/RESULTS.md](RESULTS.md) §10), symmetric KL ($\lambda=1$) achieves higher paraphrase agreement than untuned 1D OT (`87.9%` vs. `86.3%` on BERT-base), because exact token-index alignment is what Exact Match rewards once both views are simultaneously supervised by $\mathcal{L}_{\text{CE}}(q) + \mathcal{L}_{\text{CE}}(q')$.

---

### 6.2. Metric-Space Ablation: 1D Optimal Transport ($W_1$ Distance)

Tokens in a passage reside on an **ordered 1D metric space** $(\{0, \dots, m-1\}, d)$ endowed with the physical coordinate ground metric $C_{ij} = \frac{\vert i - j \vert}{m}$.

#### 6.2.1. Kantorovich Formulation
The **Wasserstein-1 ($W_1$) distance** (Earth Mover's Distance) computes the minimum cost required to transport the probability mass of distribution $P$ into distribution $Q$:
$$W_1(P, Q) = \min_{\Pi \in \mathcal{U}(P, Q)} \sum_{i=0}^{m-1} \sum_{j=0}^{m-1} \Pi_{ij} \, C_{ij}$$

where:
1. $C_{ij} = \frac{\vert i - j \vert}{m} \in [0, 1]$ is the normalized **ground cost matrix**.
2. $\mathcal{U}(P, Q) = \big\{ \Pi \in \mathbb{R}_{+}^{m \times m} \;\big\vert\; \sum_{j=0}^{m-1} \Pi_{ij} = P(i), \; \sum_{i=0}^{m-1} \Pi_{ij} = Q(j) \big\}$ is the convex polytope of valid joint transport plans.

---

### 6.3. The 1D Monge-Kantorovich Closed Form via CDF Discrepancy

While multi-dimensional Optimal Transport requires iterative approximation algorithms (such as Sinkhorn iterations), on a **one-dimensional ordered line**, optimal transport has an **exact, non-iterative closed-form solution** computable in $\mathcal{O}(m)$ time via Cumulative Distribution Functions (CDFs).

Let the CDFs of $P$ and $Q$ be defined by the prefix sums:
$$\text{CDF}_P(i) = \sum_{t=0}^i P(t), \qquad \text{CDF}_Q(i) = \sum_{t=0}^i Q(t) \quad \text{for } i \in \{0, \dots, m-1\}$$

> [!IMPORTANT]
> **Theorem (1D Earth Mover's Distance as $L_1$ CDF Discrepancy)**:
> For any two probability distributions $P, Q \in \Delta^{m-1}$ defined on an ordered line with ground cost $C_{ij} = \frac{\vert i - j \vert}{m}$, the exact Wasserstein-1 distance is:
> 
> $$W_1(P, Q) = \frac{1}{m} \sum_{i=0}^{m-1} \Big\vert \text{CDF}_P(i) - \text{CDF}_Q(i) \Big\vert$$

#### Pedagogical Derivation & Proof Intuition:
Consider the passage coordinate axis divided into $m$ intervals by boundaries $0, 1, \dots, m-1$.
1. At any coordinate threshold $i$, the total probability mass located at or to the left of $i$ is $\text{CDF}_P(i)$ under distribution $P$, and $\text{CDF}_Q(i)$ under distribution $Q$.
2. Because probability mass cannot disappear, the **net probability mass that must physically cross boundary $i$** to transform $P$ into $Q$ is exactly:
   $$\text{Flux}(i) = \Big\vert \text{CDF}_P(i) - \text{CDF}_Q(i) \Big\vert$$
3. Each step across boundary $i$ incurs a unit cost of $\frac{1}{m}$.
4. By the Monge monotonicity principle in 1D, transport paths never cross in an optimal assignment. Thus, summing the net cross-boundary flux across all $m-1$ boundaries yields the exact minimum transport cost.

---

### 6.4. Numerical Walkthrough: 5-Token Toy Passage
To illustrate the stark geometric difference between KL and 1D OT, consider a passage of $m = 5$ tokens:

Let distribution $P$ predict token 1 with 100% confidence:
$$P = [0.0, \; 1.0, \; 0.0, \; 0.0, \; 0.0]$$

Compare two hypothetical paraphrased distributions:
* **$Q_{\text{near}}$ (1-token shift to index 2)**: $Q_{\text{near}} = [0.0, \; 0.0, \; 1.0, \; 0.0, \; 0.0]$
* **$Q_{\text{far}}$ (3-token shift to index 4)**: $Q_{\text{far}} = [0.0, \; 0.0, \; 0.0, \; 0.0, \; 1.0]$

#### Computation Table:

| Index $i$ | $P(i)$ | $Q_{\text{near}}(i)$ | $Q_{\text{far}}(i)$ | $\text{CDF}_P(i)$ | $\text{CDF}_{Q_{\text{near}}}(i)$ | $\text{CDF}_{Q_{\text{far}}}(i)$ | $\vert \Delta\text{CDF}_{\text{near}} \vert$ | $\vert \Delta\text{CDF}_{\text{far}} \vert$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | **1.0** | **1.0** |
| 2 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | **0.0** | **1.0** |
| 3 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | **0.0** | **1.0** |
| 4 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| **Sum** | | | | | | | **1.0** | **3.0** |

#### Comparison of Loss Values:
1. **Symmetric KL Divergence**:
   $$\mathcal{D}_{\text{KL}}(P \parallel Q_{\text{near}}) = \infty, \qquad \mathcal{D}_{\text{KL}}(P \parallel Q_{\text{far}}) = \infty$$
   *Both cases produce catastrophic numerical overflow.*
2. **1D Optimal Transport ($W_1 = \frac{1}{m} \sum \vert\Delta\text{CDF}\vert$)**:
   $$W_1(P, Q_{\text{near}}) = \frac{1.0}{5} = \mathbf{0.20}$$
   $$W_1(P, Q_{\text{far}}) = \frac{3.0}{5} = \mathbf{0.60}$$

**Conclusion**: The 1D Optimal Transport loss scales **linearly with physical token distance** ($0.20 \to 0.60$), providing smooth, proportional subgradients that encourage convergence without exploding over minor boundary ambiguities.

The total token-level consistency loss is:
$$\mathcal{L}_{\text{OT}} = \frac{1}{2} \Big[ W_1\big(P_{\text{start}}^{(q)}, P_{\text{start}}^{(q')}\big) + W_1\big(P_{\text{end}}^{(q)}, P_{\text{end}}^{(q')}\big) \Big]$$

---

## 7. From 1D Marginal Tokens to 2D Phrases: Span-Level Hungarian Matching

### 7.1. The Limitation of Marginal Distributions
While 1D Optimal Transport successfully aligns the continuous token probability clouds, an extractive answer is **not** an independent collection of tokens—it is a **2D contiguous interval** $(i, j)$ with $0 \le i \le j < m$.

Because start and end heads are uncoupled:
* $P_{\text{start}}$ cannot condition on the end position.
* $P_{\text{end}}$ cannot condition on the start position.

In passages containing multiple recurring entities (e.g., two dates or two names), the model could produce valid marginal distributions for $q$ and $q'$, yet cross-pair the start of entity #1 with the end of entity #2, creating a nonsensical run-on span. To enforce consistency at the **phrase level**, we must operate directly on the joint candidate spans.

---

### 7.2. Candidate Span Extraction & Scoring
For any candidate span $(i, j)$ satisfying $0 \le i \le j < m$ and length $(j - i + 1) \le L_{\max}$, its unnormalized span score is:
$$s(i, j) = s_i + e_j = \mathbf{z}_{\text{start}}[i] + \mathbf{z}_{\text{end}}[j]$$

We extract the top-$k$ highest-scoring distinct spans for question $q$ and paraphrase $q'$:
$$S_q = \big\{ \text{span}_a = (i_a, j_a) \big\}_{a=1}^k \quad \text{with scores } \mathbf{s}^{(q)} \in \mathbb{R}^k$$
$$S_{q'} = \big\{ \text{span}_b = (i'_b, j'_b) \big\}_{b=1}^k \quad \text{with scores } \mathbf{s}^{(q')} \in \mathbb{R}^k$$

Applying Softmax over these $k$ candidate scores yields normalized span probabilities:
$$\mathbf{p}^{(q)} = \text{Softmax}(\mathbf{s}^{(q)}) \in \Delta^{k-1}, \qquad \mathbf{p}^{(q')} = \text{Softmax}(\mathbf{s}^{(q')}) \in \Delta^{k-1}$$

---

### 7.3. The Permutation Problem & Bipartite Assignment

The candidates in sets $S_q$ and $S_{q'}$ do not naturally appear in identical rank order:
* Candidate $a=1$ in $S_q$ might be `"Sir Isaac Newton"`.
* Candidate $b=1$ in $S_{q'}$ might be `"Isaac Newton"`, with `"Sir Isaac Newton"` ranked at position $b=2$.

Direct element-wise comparison ($a=1$ vs. $b=1$) is invalid because candidate rankings naturally invert between phrasings. We discover the **optimal 1-to-1 bipartite assignment** $\pi^*$ by solving an optimal assignment problem over the permutation space:

$$
\begin{array}{ccc}
S_q \text{ \textbf{(Original Question Candidates)}} & & S_{q'} \text{ \textbf{(Paraphrased Question Candidates)}} \\[8pt]
\boxed{a=1: \text{``Isaac Newton''}} & \xrightarrow[\text{cost } C_{12} \approx 0.05]{\text{Optimal Bipartite Match } \pi^*(1)=2} & \boxed{b=2: \text{``Isaac Newton''}} \\[14pt]
\boxed{a=2: \text{``Sir Isaac Newton''}} & \xrightarrow[\text{cost } C_{21} \approx 0.04]{\text{Optimal Bipartite Match } \pi^*(2)=1} & \boxed{b=1: \text{``Sir Isaac Newton''}} \\[14pt]
\boxed{a=3: \text{``English polymath''}} & \xrightarrow[\text{cost } C_{33} \approx 0.12]{\text{Optimal Bipartite Match } \pi^*(3)=3} & \boxed{b=3: \text{``English physicist''}}
\end{array}
$$

#### 7.3.1. Defining the Pairwise Span Distance Matrix $\mathbf{C} \in [0, 1]^{k \times k}$
For every pair of candidate spans $a \in S_q$ and $b \in S_{q'}$, we compute a composite distance $C_{ab}$ combining spatial proximity and token overlap:

1. **Normalized Boundary Distance**:
   $$D_{\text{pos}}(a, b) = \frac{\vert i_a - i'_b \vert + \vert j_a - j'_b \vert}{2m} \in [0, 1]$$
2. **Token Intersection-over-Union (IoU) Cost**:
   $$D_{\text{IoU}}(a, b) = 1 - \frac{\vert \text{span}_a \cap \text{span}_b \vert}{\vert \text{span}_a \cup \text{span}_b \vert} \in [0, 1]$$
   where:
   $$\vert \text{span}_a \cap \text{span}_b \vert = \max\big(0, \; \min(j_a, j'_b) - \max(i_a, i'_b) + 1\big)$$
   $$\vert \text{span}_a \cup \text{span}_b \vert = (j_a - i_a + 1) + (j'_b - i'_b + 1) - \vert \text{span}_a \cap \text{span}_b \vert$$

The combined pairwise matching cost is:
$$C_{ab} = \frac{1}{2} D_{\text{pos}}(a, b) + \frac{1}{2} D_{\text{IoU}}(a, b)$$

> [!NOTE]
> **Why Coordinate Distance & Token IoU Suffice (Omitting Sentence Transformer Encodings)**:
> In contrast to open-domain retrieval where candidate answers originate from different documents and require dense semantic embeddings (e.g., SBERT), candidate spans $\text{span}_a$ and $\text{span}_b$ here are extracted from the **exact same underlying context passage** $c$. Consequently, semantic identity is completely captured by physical coordinate proximity and token overlap: spans at identical token offsets have identical lexical content, while overlapping spans share identical subwords. Combining $D_{\text{pos}}$ and $D_{\text{IoU}}$ provides a deterministic, exact distance metric computable in $\mathcal{O}(1)$ without incurring extra neural forward passes or embedding noise over short phrases during training.

#### 7.3.2. Solving the Optimal Assignment via the Hungarian Algorithm
Let $\mathcal{S}_k$ denote the set of all $k!$ permutations of $\{1, \dots, k\}$. The optimal bijection $\pi^* : \{1, \dots, k\} \to \{1, \dots, k\}$ minimizes the total matching cost:
$$\pi^* = \arg\min_{\pi \in \mathcal{S}_k} \sum_{a=1}^k C_{a, \pi(a)}$$

This classical combinatorial assignment problem is solved globally in $\mathcal{O}(k^3)$ time using the **Hungarian (Kuhn-Munkres) Algorithm** (`scipy.optimize.linear_sum_assignment`). For typical values of $k = 10$ or $k = 20$, computation takes less than $1$ millisecond on CPU.

---

### 7.4. The Hungarian Span Consistency Loss
Once optimal assignment $\pi^*$ is solved, candidate $a \in S_q$ is matched to candidate $\pi^*(a) \in S_{q'}$. We define the match confidence weight:
$$w_a = 1 - C_{a, \pi^*(a)} \in [0, 1]$$

The span consistency loss penalizes probability discrepancies between matched spans:
$$\mathcal{L}_{\text{Hungarian}} = \frac{1}{2} \sum_{a=1}^k w_a \cdot \Big\vert p_a^{(q)} - p_{\pi^*(a)}^{(q')} \Big\vert + \sum_{a=1}^k C_{a, \pi^*(a)} \cdot \left( \frac{p_a^{(q)} + p_{\pi^*(a)}^{(q')}}{2} \right)$$

* **Term 1 (Agreement Penalty with Total Variation Scaling)**: For well-matched spans ($w_a \approx 1$), penalizes differences between candidate probabilities $p_a^{(q)}$ and $p_{\pi^*(a)}^{(q')}$. Because $\mathbf{p}^{(q)}, \mathbf{p}^{(q')} \in \Delta^{k-1}$ are normalized probability distributions ($\sum_{a=1}^k p_a = 1$), scaling by $\frac{1}{2}$ corresponds to the exact Total Variation (TV) distance metric bounded in $[0, 1]$, preventing the loss from vanishing as $k$ increases.
* **Term 2 (Outlier Penalty)**: If a candidate has no plausible counterpart in the paraphrased question ($C_{a, \pi^*(a)} \approx 1$), penalizes the model for placing high probability mass on this unaligned candidate (bounded in $[0, 1]$).

During backpropagation, gradients flow through the continuous softmax probabilities $\mathbf{p}^{(q)}$ and $\mathbf{p}^{(q')}$, treating the discrete matching $\pi^*$ as a fixed assignment.

---

## 8. The Unified Optimization Objective & Second-Stage Span Selection

### 8.1. Extractor Training Objectives (`g17qa/losses.py`)

Our training pipeline supports five nested loss regimes over paired windows $(q, q', c^{(w)}, a^*)$:

1. **Plain Cross-Entropy Baseline (`--loss_mode ce`)**:
   $$\mathcal{L}_{\text{CE-only}} = \mathcal{L}_{\text{CE}}(q)$$
2. **Paraphrase Data Augmentation Only (`--loss_mode aug`, *Gan & Ng, 2019*)**:
   $$\mathcal{L}_{\text{aug}} = \mathcal{L}_{\text{CE}}(q) + \mathcal{L}_{\text{CE}}(q')$$
3. **Proposed Symmetric-KL Consistency Training (`--loss_mode kl`, $\lambda = 1.0$)**:
   $$\boxed{\mathcal{L}_{\text{KL}} = \mathcal{L}_{\text{CE}}(q) + \mathcal{L}_{\text{CE}}(q') + \lambda \, \mathcal{D}_{\text{KL}}^{\text{sym}}}$$
4. **1D Optimal Transport Ablation (`--loss_mode ot`, $\lambda_{\text{OT}} = 0.5$)**:
   $$\mathcal{L}_{\text{OT-mode}} = \mathcal{L}_{\text{CE}}(q) + \mathcal{L}_{\text{CE}}(q') + \lambda_{\text{OT}} \, \mathcal{L}_{\text{OT}}$$
5. **Dual OT + Hungarian Span-Matching Ablation (`--loss_mode dual`, $\lambda_{\text{OT}} = 0.5, \mu_{\text{Hung}} = 0.1$)**:
   $$\mathcal{L}_{\text{dual}} = \mathcal{L}_{\text{CE}}(q) + \mathcal{L}_{\text{CE}}(q') + \lambda_{\text{OT}} \, \mathcal{L}_{\text{OT}} + \mu_{\text{Hung}} \, \mathcal{L}_{\text{Hungarian}}$$

| Objective Term | Input Signatures | Target Domain | Pedagogical Function |
| :--- | :--- | :--- | :--- |
| **$\mathcal{L}_{\text{CE}}(q) + \mathcal{L}_{\text{CE}}(q')$** | Logits $\mathbf{z}$, Gold $(s^*, e^*)$ | Discrete Index Labels | Supervises both phrasings against gold-standard human annotations. |
| **$\mathcal{D}_{\text{KL}}^{\text{sym}}$ (Proposed)** | Passage $P_t^{(q)}, P_t^{(q')}$ | Simplex $\Delta^{m-1}$ | Aligns token-level start/end distributions across question and paraphrase. |
| **$\mathcal{L}_{\text{OT}}$ (Ablation)** | Continuous $P_{\text{start}}, P_{\text{end}}$ | 1D Metric Simplex $\Delta^{m-1}$ | Penalizes probability mass transport proportional to token distance $\vert i - j \vert / m$. |
| **$\mathcal{L}_{\text{Hungarian}}$ (Ablation)** | Joint Top-$k$ Candidate Spans | Permutation Space $\mathcal{S}_k$ | Penalizes probability differences between bipartite-matched top-$k$ spans. |

---

### 8.2. Optimization Mechanics: Autograd Backward & Discrete Assignment Stop-Gradient
Can all five objectives be optimized directly with standard PyTorch `total_loss.backward()`? **Yes.**
* **Continuous Operations ($\mathcal{L}_{\text{CE}}, \mathcal{D}_{\text{KL}}^{\text{sym}}, \mathcal{L}_{\text{OT}}$)**: Cross-entropy and symmetric KL are smooth over log-softmax probabilities; 1D Wasserstein uses `torch.cumsum` and `torch.abs` with well-defined piecewise subgradients $\partial \vert u \vert / \partial u = \text{sign}(u)$.
* **Discrete Permutation as a Stop-Gradient Operator ($\mathcal{L}_{\text{Hungarian}}$)**: The Hungarian algorithm solves $\pi^* = \arg\min_{\pi \in \mathcal{S}_k} \sum_{a=1}^k C_{a, \pi(a)}$ via SciPy on CPU and acts as a stop-gradient index permutation $\text{sg}[\pi^*]$ (analogous to DETR), while gradients flow through the continuous softmax probabilities $\mathbf{p}^{(q)}$ and $\mathbf{p}^{(q')}$.
* **AdamW with Linear Warmup**: Parameters are partitioned into weight-decay ($\lambda_{\text{wd}} = 0.01$) and no-decay (`bias`, `LayerNorm`) groups with a $10\%$ linear warmup followed by linear decay ($\eta = 3 \times 10^{-5}$ for base models, $\eta = 1 \times 10^{-5}$ with gradient accumulation $8 \times 2$ for `DeBERTa-v3-large`).

---

### 8.3. Second-Stage Span Selection: Learned Reranker & Paraphrase Ensemble (`g17qa/reranker.py`, `g17qa/consistency.py`)

Given the extractor's top-$k$ ($k=20$) distinct candidate spans $\{(i_a, j_a)\}_{a=1}^k$ for question $q$, we also formulate two second-stage mechanisms:

1. **Learned MLP Span Reranker (`g17qa/reranker.py`)**:
   * **Feature Representation**: Each candidate span $a = (i_a, j_a)$ in window $w$ is represented by concatenating four $d$-dimensional hidden vectors from the extractor's final encoder layer $\mathbf{H} \in \mathbb{R}^{L \times d}$ with $14$ scalar features $\mathbf{f}_a \in \mathbb{R}^{14}$ (null-calibrated score, raw logit sum, null score, start/end logits, start/end probabilities, token length, normalized rank, window index, gap to top-1, and—when enabled—three paraphrase-view features: `pair_score`, `pair_gap`, `pair_missing`):
     $$\mathbf{v}_a = \Big[ \mathbf{h}_{i_a} \;\Vert\; \mathbf{h}_{j_a} \;\Vert\; \frac{1}{j_a - i_a + 1}\sum_{t=i_a}^{j_a} \mathbf{h}_t \;\Vert\; \mathbf{h}_{\texttt{[CLS]}} \Big] \in \mathbb{R}^{4d}$$
   * **Graded Listwise Cross-Entropy**: An MLP $g_\phi(\mathbf{v}_a, \mathbf{f}_a) \to r_a \in \mathbb{R}$ produces candidate logits $\mathbf{r} \in \mathbb{R}^k$. Against graded relevance targets $y_a = 1$ if $\text{EM}(a, a^*) = 1$ else $0.3 \cdot \mathrm{F1}(a, a^*)$, normalized to target distribution $\tilde{y}_a = \exp(y_a / \tau) / \sum_b \exp(y_b / \tau)$, the reranker minimizes listwise cross-entropy on the held-out training articles (plus an optional symmetric-KL consistency term $w \cdot \mathcal{D}_{\text{KL}}^{\text{sym}}(P_\phi(\cdot\mid q), P_\phi(\cdot\mid q'))$ over aligned candidates).
   * **Inference Rules**: Pure reranker ($\arg\max_a r_a$), log-probability residual blending ($\arg\max_a [\log p_{\text{ext}}(a\mid q) + \alpha \log p_\phi(a\mid q)]$), and confidence gating (keeping the extractor's top-1 span unless the reranker's margin exceeds $\gamma$ nats).

2. **Training-Free Paraphrase Ensemble (`g17qa/consistency.py`)**:
   * Averages the extractor's normalized log-probabilities across the original question $q$ and paraphrase $q'$ without training a second model:
     $$a^*_{\text{ens}} = \arg\max_{a \in S_q \cup S_{q'}} \Big[ \log p_{\text{ext}}(a \mid q) + \beta \log p_{\text{ext}}(a \mid q') \Big] \qquad (\beta = 1.0)$$
   * As shown in [docs/RESULTS.md](RESULTS.md) §9b, this learning-free ensemble achieves higher agreement (`99.3%`, both-views-correct `89.3%`) than the learned reranker (`94.1%`, both-views-correct `87.4%`), while both require a test-time paraphrase—motivating our choice of the **extractor alone** (`DeBERTa-v3-large` + symmetric KL) as the proposed single-pass system.

---

## 9. Appendix: Comprehensive Symbol Glossary & Type Signatures

For reference, the table below catalogs all formal mathematical symbols, domains, and type signatures used throughout this specification:

| Symbol | Mathematical Domain / Type | Type Signature | Conceptual Meaning |
| :--- | :--- | :--- | :--- |
| $\Sigma$ | Alphabet | Finite set | Alphabet of Unicode characters. |
| $\Sigma^*$ | String Space | $\bigcup_{k=0}^\infty \Sigma^k$ | Set of all finite character strings. |
| $q, q'$ | $\Sigma^*$ | String | Original question string $q$ and paraphrased question $q'$. |
| $c$ | $\Sigma^*$ | String | Raw context passage string. |
| $a^*$ | $\Sigma^*$ | String | Ground-truth answer substring ($a^* = c[a_{\text{start}}:a_{\text{end}}+1]$). |
| $a_{\text{start}}, a_{\text{end}}$ | $\mathbb{N}$ | Offset Index | Inclusive start and end character offsets within string $c$. |
| $V$ | $\{0, \dots, \vert V \vert - 1\}$ | Integer Set | Vocabulary of subword token IDs. |
| $\mathcal{T}_V$ | $\Sigma^* \to V^*$ | Tokenizer | Mapping from text string to integer token sequence. |
| $\phi$ | $\{0, \dots, m-1\} \to \mathbb{N}^2$ | Index Map | Offset map: token index $t \mapsto (\alpha_t, \beta_t)$ character offsets. |
| $(s^*, e^*)$ | $\{0, \dots, m-1\}^2$ | Token Index Pair | Ground-truth answer start and end token indices in passage $c$. |
| $L$ | $\mathbb{N}$ (e.g., $384$) | Scalar Length | Maximum sequence capacity packed into the Transformer. |
| $m$ | $\mathbb{N}$ ($m < L$) | Scalar Length | Number of subword tokens in context passage $c$ within a window. |
| $S$ | $\mathbb{N}$ (e.g., $128$) | Scalar Stride | Step size used to slide overlapping windows across passage $c$. |
| $\mathbf{m}_{\text{pass}}$ | $\{0, 1\}^L$ | Boolean Vector | Binary mask identifying active passage tokens within sequence. |
| $\mathcal{P}$ | $\{0, \dots, L-1\}$ | Index Subset | Ordered set of sequence indices corresponding to passage tokens. |
| $\mathcal{E}_\theta$ | $V^L \to \mathbb{R}^{L \times d}$ | Neural Network | Pre-trained Transformer encoder parameterized by weights $\theta$. |
| $\mathbf{H}$ | $\mathbb{R}^{L \times d}$ | Matrix | Contextualized token representations of hidden dimension $d$. |
| $\mathbf{W}, \mathbf{b}$ | $\mathbb{R}^{d \times 2}, \; \mathbb{R}^2$ | Tensors | Weights and biases of the linear span prediction head. |
| $\mathbf{z}_{\text{start}}, \mathbf{z}_{\text{end}}$ | $\mathbb{R}^L$ | Logit Vectors | Unnormalized boundary logits over the entire input sequence. |
| $\mathbf{s}, \mathbf{e}$ | $\mathbb{R}^m$ | Logit Vectors | Passage-only boundary logit vectors ($\mathbf{s} = \mathbf{z}_{\text{start}}[\mathcal{P}]$). |
| $\Delta^{m-1}$ | $\{ \mathbf{p} \in \mathbb{R}_+^m \mid \sum_i p_i = 1 \}$ | Simplex | Valid probability distribution space over $m$ passage tokens. |
| $P, Q$ | $\Delta^{m-1}$ | Vector | Discrete probability distributions over passage tokens. |
| $\text{CDF}_P$ | $\{0, \dots, m-1\} \to [0, 1]$ | Monotonic Vector | Cumulative distribution function ($\text{CDF}_P(i) = \sum_{t=0}^i P(t)$). |
| $C_{ij}$ | $[0, 1]^{m \times m}$ | Ground Metric | Normalized token distance matrix ($C_{ij} = \frac{\vert i - j \vert}{m}$). |
| $W_1(P, Q)$ | $\mathbb{R}_{\ge 0}$ | Metric Distance | Exact 1D Wasserstein-1 (Earth Mover's) Optimal Transport distance. |
| $S_q, S_{q'}$ | $\{ (i, j) \in \mathbb{N}^2 \}^k$ | Candidate Sets | Top-$k$ extracted candidate spans for $q$ and $q'$. |
| $\mathbf{C}$ | $[0, 1]^{k \times k}$ | Distance Matrix | Pairwise bipartite cost matrix between candidates ($C_{ab}$). |
| $\pi^*$ | Permutation of $\{1, \dots, k\}$ | Bijection | Optimal 1-to-1 candidate assignment via Hungarian algorithm. |
| $\mathcal{L}_{\text{CE}}$ | $\mathbb{R}_{\ge 0}$ | Scalar Loss | Supervised start/end cross-entropy loss. |
| $\mathcal{L}_{\text{OT}}$ | $\mathbb{R}_{\ge 0}$ | Scalar Loss | Token-level 1D Wasserstein Optimal Transport consistency loss. |
| $\mathcal{L}_{\text{Hungarian}}$ | $\mathbb{R}_{\ge 0}$ | Scalar Loss | Span-level Hungarian candidate score alignment loss. |
| $\lambda_{\text{OT}}, \mu_{\text{Hung}}$ | $\mathbb{R}_{\ge 0}$ | Hyperparameters | Weighting factors balancing OT and Hungarian consistency losses. |
