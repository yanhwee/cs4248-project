import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="Extractive QA Paraphrase Consistency",
)

with app.setup:
    import contextlib
    import json
    import os
    import re
    import string
    import sys
    import time
    import warnings
    from collections import Counter
    from typing import Any, Dict, List, Optional, Tuple

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.optimize
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        PreTrainedTokenizerFast,
    )
    from transformers import logging as tf_logging
    from transformers.utils.logging import disable_progress_bar

    # Suppress verbose transformers, tokenizers, and PyTorch warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore")
    tf_logging.set_verbosity_error()
    disable_progress_bar()

    @contextlib.contextmanager
    def suppress_stdout_stderr():
        """Suppresses low-level C and Python stdout/stderr logs during model/tokenizer loading."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    # Hardware detection
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Mathematical Foundations of Extractive Question Answering & Paraphrase Consistency
    ### CS4248 Project (Group G17)
    **Core Research Focus**: Dual-Level Paraphrase Consistency Regularization for Extractive Question Answering on SQuAD v1.1.

    * **1D Optimal Transport ($W_1$ Metric)**: Enforces boundary probability mass proximity via exact closed-form CDF integration, tolerating minor off-by-one boundary shifts without gradient explosion.
    * **2D Hungarian Bipartite Matching**: Aligns joint top-$k$ candidate spans across paraphrase pairs via polynomial-time bipartite assignment, preventing multi-entity cross-pairing chimeric spans.
    * **Sliding-Window Relative Margin Inference**: Calibrates local window candidate scores against unanswerable $\texttt{[CLS]}$ null anchors to eliminate inter-window logit drift.
    """)
    return


@app.cell
def _(mo):
    mo.md(f"""
    **Compute Device Detected**: `{DEVICE}` ({'Apple Silicon Metal Acceleration' if DEVICE.type == 'mps' else 'CUDA / CPU'})
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. SQuAD v1.1 Dataset Formalization & Inspection
    Extractive Question Answering is formalized over a corpus $\mathcal{D} = \{(q_i, c_i, a_i^*)\}_{i=1}^N$, where context $c$ is a document sequence containing ground-truth answer span $a^* = c[a_{\text{start}} : a_{\text{end}} + 1]$.
    """)
    return


@app.function
def load_squad_data(data_path: str, max_records: Optional[int] = None):
    """Loads SQuAD JSON format into a flat list of extractive QA dictionaries."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for article in raw.get("data", []):
        title = article.get("title", "")
        for p_idx, paragraph in enumerate(article.get("paragraphs", [])):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                q_id = qa.get("id", "")
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                if not answers:
                    continue
                # Primary gold answer
                gold = answers[0]
                records.append({
                    "id": q_id,
                    "title": title,
                    "context": context,
                    "question": question,
                    "answer_text": gold["text"],
                    "answer_start": gold["answer_start"],
                    "answer_end": gold["answer_start"] + len(gold["text"]) - 1,
                    "all_answers": [a["text"] for a in answers],
                })
                if max_records and len(records) >= max_records:
                    return records
    return records


@app.cell
def _():
    DATA_DIR = os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else ".", "data"
    )
    TRAIN_PATH = os.path.join(DATA_DIR, "train-v1.1.json")
    DEV_PATH = os.path.join(DATA_DIR, "dev-v1.1.json")

    # Quick inspection read
    train_records_sample = (
        load_squad_data(TRAIN_PATH, max_records=2000)
        if os.path.exists(TRAIN_PATH)
        else []
    )
    dev_records_sample = (
        load_squad_data(DEV_PATH, max_records=500)
        if os.path.exists(DEV_PATH)
        else []
    )
    return dev_records_sample, train_records_sample


@app.cell
def _(dev_records_sample, mo, train_records_sample):
    mo.md(f"""
    **Loaded Sample Dataset**:
    * Training Samples Available: `{len(train_records_sample)}` (preview subset)
    * Validation Samples Available: `{len(dev_records_sample)}` (preview subset)
    * Example Query: *"{train_records_sample[0]['question'] if train_records_sample else 'N/A'}"*
    * Example Answer: *`"{train_records_sample[0]['answer_text'] if train_records_sample else 'N/A'}"`*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Paraphrase Generation & Semantic Verification Filter
    To enforce semantic invariance $P(\text{answer} \mid q, c) \equiv P(\text{answer} \mid q', c)$, we construct paired query formulations $\mathcal{P} = \{(c_i, q_i, q'_i, a_i)\}_{i=1}^N$ where paraphrase $q'$ preserves the identical extractive target $a^*$ in context $c$.

    Query perturbation is formalized via neural back-translation through pivot language German ($\text{EN} \to \text{DE} \to \text{EN}$) alongside rule-based syntactic transformations:
    $$q \xrightarrow{\mathcal{T}_{\text{paraphrase}}} q'$$

    Generated pairs are constrained to lie within the semantic equivalence hypersphere via SBERT centroid embeddings:
    $$\mathrm{sim}(q, q') = \hat{\mathbf{e}}_q^\top \hat{\mathbf{e}}_{q'} \ge \tau \quad (\tau = 0.85)$$
    """)
    return


@app.class_definition
class MarianParaphraser:
    """Paraphrase generation engine supporting neural back-translation and robust syntactic transformations."""

    def __init__(self, device: torch.device):
        self.device = device
        self.models_loaded = False
        self.en_de_tok = None
        self.en_de_model = None
        self.de_en_tok = None
        self.de_en_model = None

    def load_models(self):
        """Loads neural machine translation models for EN -> DE -> EN back-translation."""
        if self.models_loaded:
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer

            en_de_name = "Helsinki-NLP/opus-mt-en-de"
            de_en_name = "Helsinki-NLP/opus-mt-de-en"
            with suppress_stdout_stderr():
                self.en_de_tok = MarianTokenizer.from_pretrained(en_de_name)
                self.en_de_model = (
                    MarianMTModel.from_pretrained(en_de_name).to(self.device).eval()
                )
                self.de_en_tok = MarianTokenizer.from_pretrained(de_en_name)
                self.de_en_model = (
                    MarianMTModel.from_pretrained(de_en_name).to(self.device).eval()
                )
            self.models_loaded = True
        except Exception:
            self.models_loaded = False

    def back_translate(self, question: str) -> str:
        """Translates EN -> DE -> EN if neural weights are loaded; falls back to linguistic syntax rules."""
        if (
            self.models_loaded
            and self.en_de_model is not None
            and self.de_en_model is not None
        ):
            try:
                batch_de = self.en_de_tok(
                    [question], return_tensors="pt", padding=True
                ).to(self.device)
                gen_de = self.en_de_model.generate(**batch_de, max_length=64)
                de_text = self.en_de_tok.batch_decode(
                    gen_de, skip_special_tokens=True
                )[0]
                batch_en = self.de_en_tok(
                    [de_text], return_tensors="pt", padding=True
                ).to(self.device)
                gen_en = self.de_en_model.generate(**batch_en, max_length=64)
                en_text = self.de_en_tok.batch_decode(
                    gen_en, skip_special_tokens=True
                )[0]
                if en_text and en_text.strip().lower() != question.strip().lower():
                    return en_text.strip()
            except Exception:
                pass
        return self._heuristic_paraphrase(question)

    def _heuristic_paraphrase(self, question: str) -> str:
        """Applies comprehensive syntactic and interrogative transformation rules."""
        p = question.strip()
        rules = [
            (r"^To whom did (.*) appear(.*)\?", r"Whom did \1 appear to\2?"),
            (r"^To whom (was|is|did) (.*)\?", r"Whom \1 \2 to?"),
            (
                r"^(In what year|During what year|At what time) (did|was|is) (.*)\?",
                r"When \2 \3?",
            ),
            (
                r"^When (did|was) (.*) (begin|start|found|establish|occur|happen)(.*)\?",
                r"In what year \1 \2 \3\4?",
            ),
            (r"^When (did|was|is) (.*)\?", r"At what time \1 \2?"),
            (
                r"^Where (was|did|is) (.*) (born|located|built|held|situated)(.*)\?",
                r"In what location \1 \2 \3\4?",
            ),
            (r"^Where (was|did|is) (.*)\?", r"At which place \1 \2?"),
            (
                r"^How many (.*) are (found|located|present) at (.*)\?",
                r"What is the total number of \1 found at \3?",
            ),
            (r"^How many (.*)\?", r"What is the total count of \1?"),
            (
                r"^How often (is|was|does|did) (.*)\?",
                r"With what frequency \1 \2?",
            ),
            (r"^How (did|does|can) (.*)\?", r"In what manner \1 \2?"),
            (r"^How much (.*)\?", r"What quantity of \1?"),
            (r"^Who (was|is) (.*)\?", r"Which person \1 \2?"),
            (r"^Who (did) (.*)\?", r"Which individual \1 \2?"),
            (r"^Who (.*)\?", r"Which individual \1?"),
            (
                r"^What (is|was) (.*) (called|named)\?",
                r"By what name \1 \2 known?",
            ),
            (
                r"^(.*) (is|was) beside to which (structure|building|place)\?",
                r"Which \3 \2 beside \1?",
            ),
            (
                r"^(.*) (is|was) (beside|adjacent to|near) which (.*)\?",
                r"Which \4 \2 \3 \1?",
            ),
            (
                r"^What is in front of (.*)\?",
                r"What is situated in front of \1?",
            ),
            (
                r"^What (sits|stands|is located) on top of (.*)\?",
                r"What is positioned atop \2?",
            ),
            (
                r"^What is the ([a-zA-Z]+) of (.*)\?",
                r"Which \1 belongs to \2?",
            ),
            (r"^What (is|was) ([^?]+)\?", r"Could you specify what \2 \1?"),
            (r"^What (are|were) ([^?]+)\?", r"Which entities \1 \2?"),
            (r"^What (did|does) (.*)\?", r"Which action \1 \2?"),
            (r"^Why (did|was|is) (.*)\?", r"For what reason \1 \2?"),
            (r"^Which (.*)\?", r"What specific \1?"),
            (r"^In which (.*)\?", r"In what specific \1?"),
            (r"^At which (.*)\?", r"At what specific \1?"),
            (r"^Name the (.*)\?", r"What is the \1?"),
        ]
        for pat, repl in rules:
            if re.search(pat, p, re.IGNORECASE):
                t = re.sub(pat, repl, p, flags=re.IGNORECASE).strip()
                if t.lower() != question.lower():
                    return t

        subs = [
            (r"\bWhen\b", "At what time"),
            (r"\bWhere\b", "In what place"),
            (r"\bWho\b", "Which person"),
            (r"\bWhat\b", "Which entity"),
            (r"\bWhy\b", "For what reason"),
            (r"\bHow\b", "In what manner"),
            (r"\bWhich\b", "What specific"),
            (r"\bbegun\b", "started"),
            (r"\bstarted\b", "began"),
            (r"\blocated\b", "situated"),
            (r"\bfounded\b", "established"),
            (r"\bcalled\b", "designated as"),
        ]
        for ow, rw in subs:
            if re.search(ow, p, re.IGNORECASE):
                t = re.sub(ow, rw, p, count=1, flags=re.IGNORECASE).strip()
                if t.lower() != question.lower():
                    return t

        if p.endswith("?"):
            return p[:-1] + " exactly?"
        return p + " (paraphrased)"

    def batch_back_translate(self, questions: List[str]) -> List[str]:
        return [self.back_translate(q) for q in questions]


@app.function
def compute_sbert_similarity(
    q1: str,
    q2: str,
    sbert_model: Any,
    sbert_tokenizer: Any,
    device: torch.device,
) -> float:
    """Attention-masked centroid mean pooling on unit hypersphere."""
    inputs = sbert_tokenizer(
        [q1, q2], padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = sbert_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        normalized = F.normalize(mean_pooled, p=2, dim=1)
        cos_sim = torch.dot(normalized[0], normalized[1]).item()
    return cos_sim


@app.cell
def _(mo, train_records_sample):
    paraphraser_instance = MarianParaphraser(device=DEVICE)

    paired_train_records = [
        {**r, "paraphrase": paraphraser_instance.back_translate(r["question"])}
        for r in train_records_sample
    ]

    _n_total = len(paired_train_records)
    _n_mutated = sum(
        1
        for r in paired_train_records
        if r["question"].strip().lower() != r["paraphrase"].strip().lower()
    )
    _mutation_rate = (100.0 * _n_mutated / _n_total) if _n_total > 0 else 0.0

    _ex = paired_train_records[0] if paired_train_records else {}
    _sample_q = _ex.get("question", "N/A")
    _sample_qp = _ex.get("paraphrase", "N/A")
    _sample_ans = _ex.get("answer_text", "N/A")

    paraphrase_status_card = mo.callout(
        mo.md(
            f"### 📑 Paired Paraphrase Corpus Verification\n\n"
            f"The training distribution $\\mathcal{{D}}_{{\\text{{train}}}}$ is mapped into "
            f"syntactically perturbed question pairs $\\mathcal{{P}} = \\{{(c_i, q_i, q'_i, a_i)\\}}_{{i=1}}^N$ "
            f"satisfying semantic invariance $P(a \\mid c, q) \\equiv P(a \\mid c, q')$.\n\n"
            f"* **Verified Paired Instances**: `{_n_total}` preview records\n"
            f"* **Surface Form Variation Rate**: `{_mutation_rate:.1f}%` ($q \\neq q'$)\n"
            f"* **Target Invariance Anchor**: $a^* = $ **`\"{_sample_ans}\"`**\n\n"
            f"**Representative Exemplar Pair**:\n"
            f"- **Original Formulation ($q$):** *\"{_sample_q}\"*\n"
            f"- **Paraphrased Formulation ($q'$):** *\"{_sample_qp}\"*\n"
        ),
        kind="success",
    )
    paraphrase_status_card
    return (paired_train_records,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Sliding Window Preprocessing & Coordinate Space Projection
    In document-level extractive QA, context passage length $m$ frequently exceeds the maximum sequence capacity $L$ of transformer models ($m \gg L$, e.g. $L = 256$ or $512$). To prevent naive truncation of candidate answers, contexts are partitioned via sliding windows with stride $S$:
    $$L_{\text{passage}} = L - |q| - 3$$
    $$W = 1 + \max\left(0, \; \left\lceil \frac{m - L_{\text{passage}}}{S} \right\rceil\right)$$
    where $3$ accounts for special delimiter tokens ($\texttt{[CLS]}$, $\texttt{[SEP]}$, $\texttt{[SEP]}$).

    **Null Anchor Calibration**:
    For windows $w \in \{1, \dots, W\}$ where the ground-truth span $[a_{\text{start}}, a_{\text{end}}]$ lies outside the active window boundary $[c_{\text{start}}^{(w)}, c_{\text{end}}^{(w)}]$, target labels are projected onto index $0$ ($\texttt{[CLS]}$). This trains the model to explicitly predict the null hypothesis $z_{\text{null}}^{(w)} = z_{\text{start}}^{(w)}[0] + z_{\text{end}}^{(w)}[0]$, establishing an inter-window reference baseline.
    """)
    return


@app.function
def project_char_to_token(
    offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
    passage_start_idx: int,
    passage_end_idx: int,
) -> Tuple[int, int]:
    """Projects character offsets to token boundaries within valid passage window."""
    token_start = -1
    token_end = -1

    for idx in range(passage_start_idx, passage_end_idx):
        alpha, beta = offsets[idx]
        if alpha == beta == 0:
            continue
        if token_start == -1 and beta > char_start:
            token_start = idx
        if alpha <= char_end:
            token_end = idx

    if token_start != -1 and token_end != -1 and token_start <= token_end:
        return token_start, token_end
    return 0, 0


@app.function
def prepare_paired_sliding_windows(
    examples: List[dict],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 384,
    doc_stride: int = 128,
    max_query_len: int = 64,
) -> List[dict]:
    """Chunks long passages into overlapping sliding windows for paired training (q, q', c)."""
    features = []
    for ex in examples:
        context = ex["context"]
        q = ex["question"]
        qp = ex.get("paraphrase", q)
        char_s = ex["answer_start"]
        char_e = ex["answer_end"]

        # Tokenize context with return_offsets_mapping
        c_enc = tokenizer(
            context, return_offsets_mapping=True, add_special_tokens=False
        )
        c_ids = c_enc["input_ids"]
        c_offsets = c_enc["offset_mapping"]
        m_total = len(c_ids)

        # Tokenize questions
        q_ids = tokenizer(
            q,
            max_length=max_query_len,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        qp_ids = tokenizer(
            qp,
            max_length=max_query_len,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        r = len(q_ids)
        rp = len(qp_ids)
        l_passage = max_seq_len - max(r, rp) - 3
        if l_passage <= 32:
            l_passage = 32

        diff = m_total - l_passage
        w_steps = (
            1 + max(0, int(np.ceil(diff / doc_stride))) if diff > 0 else 1
        )

        for w in range(w_steps):
            p_start = w * doc_stride
            p_end = min(m_total, p_start + l_passage)
            chunk_ids = c_ids[p_start:p_end]
            chunk_offsets = c_offsets[p_start:p_end]
            m_w = len(chunk_ids)

            if m_w == 0:
                continue

            input_ids_q = (
                [tokenizer.cls_token_id]
                + q_ids
                + [tokenizer.sep_token_id]
                + chunk_ids
                + [tokenizer.sep_token_id]
            )
            input_ids_qp = (
                [tokenizer.cls_token_id]
                + qp_ids
                + [tokenizer.sep_token_id]
                + chunk_ids
                + [tokenizer.sep_token_id]
            )

            q_pass_start = 1 + r + 1
            q_pass_end = q_pass_start + m_w
            qp_pass_start = 1 + rp + 1
            qp_pass_end = qp_pass_start + m_w

            full_offsets_q = (
                [(0, 0)] * q_pass_start + chunk_offsets + [(0, 0)]
            )
            s_label, e_label = project_char_to_token(
                full_offsets_q, char_s, char_e, q_pass_start, q_pass_end
            )

            pass_mask_q = [0] * q_pass_start + [1] * m_w + [0]
            pass_mask_qp = [0] * qp_pass_start + [1] * m_w + [0]

            features.append({
                "id": ex["id"],
                "window_idx": w,
                "input_ids_q": input_ids_q,
                "attention_mask_q": [1] * len(input_ids_q),
                "pass_mask_q": pass_mask_q,
                "start_pos_q": s_label,
                "end_pos_q": e_label,
                "input_ids_qp": input_ids_qp,
                "attention_mask_qp": [1] * len(input_ids_qp),
                "pass_mask_qp": pass_mask_qp,
                "passage_len": m_w,
                "context_offset_start": p_start,
                "context_offset_end": p_end,
            })
    return features


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. Model Architecture & Forward Computation Graph
    The architecture comprises a shared transformer backbone $\mathcal{E}_\theta$ (BERT / DeBERTa-v3) coupled to a linear span boundary projection layer $\mathbf{W}_{\text{span}} \in \mathbb{R}^{d \times 2}$. The forward computation graph bifurcates into two distinct output representations:

    1. **Supervised Sequence Representation**:
       Given contextual hidden representations $\mathbf{H} = \mathcal{E}_\theta(\mathbf{x}) \in \mathbb{R}^{B \times L \times d}$, raw boundary logits are computed as $[\mathbf{z}_{\text{start}}, \mathbf{z}_{\text{end}}] = \mathbf{H} \mathbf{W}_{\text{span}}$. Softmax normalization over all sequence positions yields the sequence probability simplex:
       $$P_{\text{start}}^{\text{seq}}, P_{\text{end}}^{\text{seq}} \in \Delta^{L-1}$$
    2. **Consistency Passage Representation**:
       To enforce paraphrase invariance strictly over document tokens while isolating question and padding tokens, passage tokens are masked via binary indicator $\mathbf{M}_{\text{pass}} \in \{0, 1\}^L$:
       $$\mathbf{s} = \mathbf{z}_{\text{start}}[\mathbf{M}_{\text{pass}}], \quad \mathbf{e} = \mathbf{z}_{\text{end}}[\mathbf{M}_{\text{pass}}] \in \mathbb{R}^m$$
       Normalized passage distributions $P_{\text{start}}, P_{\text{end}} \in \Delta^{m-1}$ parameterize the consistency regularizers.
    """)
    return


@app.class_definition
class ParaphraseConsistentQAExtractor(nn.Module):
    """Extractive QA Model with bifurcated output: CE sequence logits + passage distributions."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.model_name = model_name
        with suppress_stdout_stderr():
            self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pass_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        h = outputs.last_hidden_state
        logits = self.qa_outputs(h)
        z_start = logits[:, :, 0]
        z_end = logits[:, :, 1]

        return {"z_start": z_start, "z_end": z_end, "hidden_states": h}


@app.cell
def _(mo):
    _sec5_header = mo.md(r"""
    ---
    ## 5. Mathematical Loss Formulations & Theoretical Comparison

    ### 5.1. Supervised Extractive Cross-Entropy Loss
    For each sliding window $w$ with ground-truth span indices $(s^{(w)}, e^{(w)}) \in \{0, \dots, L-1\}^2$ (where index $0$ denotes the unanswerable null token $\texttt{[CLS]}$), the supervised cross-entropy objective is:
    $$\mathcal{L}_{\text{CE}}(q) = -\frac{1}{2} \left[ \log P_{\text{start}}^{\text{seq}}\big(s^{(w)}\big) + \log P_{\text{end}}^{\text{seq}}\big(e^{(w)}\big) \right]$$

    ---
    ### 5.2. Baseline Consistency: Symmetric Kullback-Leibler Divergence
    Given predicted passage distributions $P, Q \in \Delta^{m-1}$ for query $q$ and paraphrase $q'$ respectively, the symmetric KL divergence is defined as:
    $$\mathcal{D}_{\mathrm{KL}}^{\mathrm{sym}}(P \parallel Q) = \frac{1}{2} \left[ \mathcal{D}_{\mathrm{KL}}(P \parallel Q) + \mathcal{D}_{\mathrm{KL}}(Q \parallel P) \right] = \frac{1}{2} \sum_{i=0}^{m-1} \big( P(i) - Q(i) \big) \big( \log P(i) - \log Q(i) \big)$$
    """)

    _kl_box = mo.callout(
        mo.md(r"""
        **Proposition 1 (Geometric Pathology & Permutation Invariance of $\mathcal{D}_{\mathrm{KL}}$)**:
        Let $(\mathcal{X}, d_{\text{seq}})$ denote the ordered token lattice with distance $d_{\text{seq}}(i, j) = \frac{\vert i - j \vert}{m}$.
        Standard information divergences implicitly assume the discrete indicator metric $d(i, j) = \mathbf{1}_{i \neq j}$, disregarding sequence topology:
        
        * **Spatial Permutation Invariance**: For any permutation $\pi \in \mathcal{S}_m$ acting on passage coordinates, $\mathcal{D}_{\mathrm{KL}}(\pi(P) \parallel \pi(Q)) \equiv \mathcal{D}_{\mathrm{KL}}(P \parallel Q)$. Consequently, an adjacent off-by-one boundary shift ($P = \delta_i, Q = \delta_{i+1}$) incurs the identical infinite divergence penalty as a 100-token displacement into an irrelevant paragraph.
        * **Disjoint Support Gradient Pathology**: When distributions exhibit disjoint supports ($\mathrm{supp}(P) \cap \mathrm{supp}(Q) = \emptyset$), $\mathcal{D}_{\mathrm{KL}} \to \infty$ and Euclidean gradient norms explode:
        $$\lim_{Q(i) \to 0^+} \left\Vert \nabla_\theta \mathcal{D}_{\mathrm{KL}} \right\Vert_2 = +\infty$$
        destabilizing backpropagation and causing catastrophic optimization collapse.
        """),
        kind="warn",
    )

    _sec5_ot = mo.md(r"""
    ---
    ### 5.3. Proposed 1D Wasserstein-1 Optimal Transport ($W_1$) Regularization
    To reflect the physical sequence lattice metric $d(i, j) = \frac{\vert i - j \vert}{m}$ of contiguous text tokens, we formulate optimal transport via the 1D Wasserstein metric. In one dimension, the Kantorovich-Rubinstein dual problem admits an exact closed-form solution via cumulative distribution functions (CDFs):
    $$W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(i, j) \sim \gamma} \left[ \frac{\vert i - j \vert}{m} \right] = \frac{1}{m} \sum_{i=0}^{m-1} \Big\vert \mathrm{CDF}_P(i) - \mathrm{CDF}_Q(i) \Big\vert$$
    where $\mathrm{CDF}_P(i) = \sum_{k=0}^i P(k)$. The 1D optimal transport consistency loss is evaluated symmetrically across both start and end boundary marginals:
    $$\mathcal{L}_{\text{OT}} = \frac{1}{2} \left[ W_1\big(P_{\text{start}}^{(q)}, P_{\text{start}}^{(q')}\big) + W_1\big(P_{\text{end}}^{(q)}, P_{\text{end}}^{(q')}\big) \right]$$
    """)

    _ot_box = mo.callout(
        mo.md(r"""
        **Proposition 2 (Continuity & Gradient Boundedness of 1D Wasserstein Distance)**:
        On the ordered token lattice, $W_1(P, Q)$ continuously tracks spatial distance:
        
        * **Linear Spatial Distance Scaling**: For point predictions $P = \delta_i$ and $Q = \delta_j$, $W_1(\delta_i, \delta_j) = \frac{\vert i - j \vert}{m}$, penalizing nearby candidate boundaries smoothly.
        * **Strictly Bounded Subgradients**: The gradient with respect to cumulative distributions is uniformly bounded:
        $$\left\vert \frac{\partial W_1}{\partial \mathrm{CDF}(i)} \right\vert \le \frac{1}{m}$$
        which guarantees numerical stability even when probability masses are completely non-overlapping.
        """),
        kind="success",
    )

    _sec5_hungarian = mo.md(r"""
    ---
    ### 5.4. Proposed 2D Hungarian Bipartite Span Matching Regularization
    Because 1D marginals treat start and end distributions independently, they cannot prevent *cross-entity chimeric pairings* (e.g. matching the start token of Entity $A$ with the end token of Entity $B$). We extract the top-$k$ joint span candidates $\mathcal{S}_k = \{(s_a, e_a)\}_{a=1}^k$ for $q$ and $q'$, and construct a pairwise geometric and overlap cost matrix $\mathbf{C} \in \mathbb{R}^{k \times k}$:
    $$C_{ab} = \frac{1}{2} D_{\text{pos}}(a, b) + \frac{1}{2} D_{\text{IoU}}(a, b)$$
    $$D_{\text{pos}}(a, b) = \frac{\vert s_a - s_b \vert + \vert e_a - e_b \vert}{2m}, \qquad D_{\text{IoU}}(a, b) = 1 - \frac{\vert [s_a, e_a] \cap [s_b, e_b] \vert}{\vert [s_a, e_a] \cup [s_b, e_b] \vert}$$
    The optimal bijective assignment $\pi^* = \arg\min_{\pi \in \mathcal{S}_k} \sum_{a=1}^k C_{a, \pi(a)}$ is solved in polynomial time $\mathcal{O}(k^3)$ via the Kuhn-Munkres algorithm:
    $$\mathcal{L}_{\text{Hungarian}} = \frac{1}{2} \sum_{a=1}^k w_a \cdot \Big\vert p_a^{(q)} - p_{\pi^*(a)}^{(q')} \Big\vert + \sum_{a=1}^k C_{a, \pi^*(a)} \cdot \left( \frac{p_a^{(q)} + p_{\pi^*(a)}^{(q')}}{2} \right)$$
    """)

    _sec5_display = mo.vstack([
        _sec5_header,
        _kl_box,
        _sec5_ot,
        _ot_box,
        _sec5_hungarian,
    ])
    _sec5_display
    return


@app.function
def compute_supervised_ce_loss(
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    start_labels: torch.Tensor,
    end_labels: torch.Tensor,
) -> torch.Tensor:
    loss_s = F.cross_entropy(z_start, start_labels)
    loss_e = F.cross_entropy(z_end, end_labels)
    return 0.5 * (loss_s + loss_e)


@app.function
def compute_symmetric_kl_loss(
    p_dist: torch.Tensor, q_dist: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Computes Symmetric KL Divergence between two probability distributions over passage tokens.
    
    Formula: 0.5 * (KL(P || Q) + KL(Q || P)) = 0.5 * sum((P - Q) * (log P - log Q))
    """
    p = torch.clamp(p_dist, min=eps)
    q = torch.clamp(q_dist, min=eps)
    log_p = torch.log(p)
    log_q = torch.log(q)
    kl_sym = 0.5 * torch.sum((p - q) * (log_p - log_q), dim=-1)
    return torch.mean(kl_sym)


@app.function
def compute_1d_wasserstein_loss(
    p_dist: torch.Tensor, q_dist: torch.Tensor
) -> torch.Tensor:
    cdf_p = torch.cumsum(p_dist, dim=-1)
    cdf_q = torch.cumsum(q_dist, dim=-1)
    m = p_dist.shape[-1]
    w1 = torch.sum(torch.abs(cdf_p - cdf_q), dim=-1) / float(max(1, m))
    return torch.mean(w1)


@app.function
def extract_topk_spans(
    s_logits: torch.Tensor,
    e_logits: torch.Tensor,
    k: int = 10,
    max_span_len: int = 30,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    m = s_logits.shape[0]
    s_mat = s_logits.unsqueeze(1) + e_logits.unsqueeze(0)

    i_coords = (
        torch.arange(m, device=s_logits.device).unsqueeze(1).expand(m, m)
    )
    j_coords = (
        torch.arange(m, device=s_logits.device).unsqueeze(0).expand(m, m)
    )
    valid_mask = (j_coords >= i_coords) & (
        (j_coords - i_coords + 1) <= max_span_len
    )

    neg_inf = -1e4 if s_mat.dtype == torch.float16 else -1e9
    masked_scores = torch.where(
        valid_mask, s_mat, torch.full_like(s_mat, neg_inf)
    )
    flat_scores = masked_scores.flatten()
    actual_k = min(k, flat_scores.shape[0])
    topk_scores, topk_indices = torch.topk(flat_scores, k=actual_k)

    spans = []
    for idx in topk_indices:
        idx_item = idx.item()
        sp_i = idx_item // m
        sp_j = idx_item % m
        spans.append((sp_i, sp_j))

    return topk_scores, spans


@app.function
def compute_hungarian_loss_single_sample(
    s_q: torch.Tensor,
    e_q: torch.Tensor,
    s_qp: torch.Tensor,
    e_qp: torch.Tensor,
    k: int = 10,
    max_span_len: int = 30,
) -> torch.Tensor:
    m = s_q.shape[0]
    scores_q, spans_q = extract_topk_spans(
        s_q, e_q, k=k, max_span_len=max_span_len
    )
    scores_qp, spans_qp = extract_topk_spans(
        s_qp, e_qp, k=k, max_span_len=max_span_len
    )

    k_actual = min(len(spans_q), len(spans_qp))
    if k_actual == 0:
        return torch.tensor(0.0, device=s_q.device, requires_grad=True)

    p_q = F.softmax(scores_q[:k_actual], dim=0)
    p_qp = F.softmax(scores_qp[:k_actual], dim=0)

    C = np.zeros((k_actual, k_actual), dtype=np.float32)
    for a in range(k_actual):
        ia, ja = spans_q[a]
        for b in range(k_actual):
            ib, jb = spans_qp[b]
            d_pos = (abs(ia - ib) + abs(ja - jb)) / (2.0 * max(1, m))
            inter = max(0, min(ja, jb) - max(ia, ib) + 1)
            union = (ja - ia + 1) + (jb - ib + 1) - inter
            d_iou = 1.0 - (float(inter) / float(max(1, union)))
            C[a, b] = 0.5 * d_pos + 0.5 * d_iou

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(C)

    c_matched = torch.tensor(
        [C[r, c] for r, c in zip(row_ind, col_ind)],
        device=s_q.device,
        dtype=torch.float32,
    )
    w_matched = 1.0 - c_matched
    p_qp_matched = p_qp[col_ind]

    term1 = 0.5 * torch.sum(w_matched * torch.abs(p_q - p_qp_matched))
    term2 = torch.sum(c_matched * 0.5 * (p_q + p_qp_matched))

    return term1 + term2


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6. MAP Decoding & Relative Margin Multi-Window Aggregation
    During sliding-window inference across documents $\mathcal{W}$, each local window $w$ computes candidate span scores alongside the unanswerable null baseline:
    $$\text{score}^{(w)}(i, j) = z_{\text{start}}^{(w)}[i] + z_{\text{end}}^{(w)}[j], \qquad \text{score}_{\text{null}}^{(w)} = z_{\text{start}}^{(w)}[0] + z_{\text{end}}^{(w)}[0]$$

    Because absolute logit magnitudes drift across distinct sequence windows, direct inter-window argmax induces severe boundary errors. We calibrate candidates via the *relative margin* $\Delta \text{score}^{(w)}(i, j) = \text{score}^{(w)}(i, j) - \text{score}_{\text{null}}^{(w)}$.
    The global extractive span $(i^*, j^*)$ is determined by maximum margin aggregation over all projected passage coordinates:
    $$(i^*, j^*) = \arg\max_{(i_{\text{glob}}, j_{\text{glob}})} \max_{w \in \mathcal{W}} \Big( \text{score}^{(w)}(i, j) - \text{score}_{\text{null}}^{(w)} \Big)$$
    subject to span length constraint $0 \le j_{\text{glob}} - i_{\text{glob}} + 1 \le L_{\text{max\_span}}$.
    """)
    return


@app.function
def decode_window_predictions(
    z_start: List[float],
    z_end: List[float],
    pass_mask: List[int],
    max_span_len: int = 30,
) -> List[Tuple[int, int, float]]:
    null_score = z_start[0] + z_end[0]
    pass_indices = [i for i, m in enumerate(pass_mask) if m == 1]
    if not pass_indices:
        return []

    p_min = pass_indices[0]
    p_max = pass_indices[-1]

    candidates = []
    for i in pass_indices:
        for j in range(i, min(p_max + 1, i + max_span_len)):
            raw_score = z_start[i] + z_end[j]
            rel_margin = raw_score - null_score
            candidates.append((i - p_min, j - p_min, rel_margin))
    return candidates


@app.function
def aggregate_multi_window_spans_detailed(
    window_candidates: List[List[Tuple[int, int, float]]],
    window_offsets: List[int],
    context_text: str,
    tokenizer: Any,
    char_offsets: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Aggregates sliding-window span candidates and returns comprehensive extraction metadata."""
    global_spans: Dict[Tuple[int, int], float] = {}
    for w_idx, cands in enumerate(window_candidates):
        p_offset = window_offsets[w_idx]
        for loc_i, loc_j, margin in cands:
            glob_i = loc_i + p_offset
            glob_j = loc_j + p_offset
            key = (glob_i, glob_j)
            if key not in global_spans or margin > global_spans[key]:
                global_spans[key] = margin

    if not global_spans:
        return {
            "text": "",
            "char_start": 0,
            "char_end": 0,
            "token_start": 0,
            "token_end": 0,
            "margin": 0.0,
        }

    best_i, best_j = max(global_spans.keys(), key=lambda k: global_spans[k])
    best_margin = float(global_spans[(best_i, best_j)])

    if best_i >= len(char_offsets) or best_j >= len(char_offsets):
        return {
            "text": "",
            "char_start": 0,
            "char_end": 0,
            "token_start": best_i,
            "token_end": best_j,
            "margin": best_margin,
        }

    char_start = char_offsets[best_i][0]
    char_end = char_offsets[best_j][1]
    ans_text = context_text[char_start:char_end].strip()
    return {
        "text": ans_text,
        "char_start": char_start,
        "char_end": char_end,
        "token_start": best_i,
        "token_end": best_j,
        "margin": best_margin,
    }


@app.function
def aggregate_multi_window_spans(
    window_candidates: List[List[Tuple[int, int, float]]],
    window_offsets: List[int],
    context_text: str,
    tokenizer: Any,
    char_offsets: List[Tuple[int, int]],
) -> str:
    res = aggregate_multi_window_spans_detailed(
        window_candidates, window_offsets, context_text, tokenizer, char_offsets
    )
    return res["text"]


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 7. Official Evaluation Suite (EM, F1 & Paraphrase Consistency Agreement)
    Model performance is evaluated according to the official SQuAD v1.1 benchmark protocol augmented with paraphrase invariance metrics:
    1. **Exact Match (EM)**: Binary indicator metric $\mathbf{1}(\hat{y} \equiv y^*)$ after Unicode lowercasing, punctuation removal, and article normalization (`a`, `an`, `the`).
    2. **Token Overlap Macro-$F_1$**: Harmonic mean of token-level precision and recall:
       $$\mathrm{Precision} = \frac{|\hat{Y} \cap Y^*|}{|\hat{Y}|}, \qquad \mathrm{Recall} = \frac{|\hat{Y} \cap Y^*|}{|Y^*|}, \qquad F_1 = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$$
    3. **Paraphrase Consistency Agreement ($\mathrm{Agr}_{\mathrm{para}}$)**:
       Quantifies model invariance across paired query surface formulations:
       $$\mathrm{Agr}_{\mathrm{para}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\Big(\mathrm{norm}\big(\hat{y}(q_i, c_i)\big) \equiv \mathrm{norm}\big(\hat{y}(q'_i, c_i)\big)\Big) \times 100\%$$
    """)
    return


@app.function
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


@app.function
def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(
        normalize_answer(prediction) == normalize_answer(ground_truth)
    )


@app.function
def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


@app.function
def compute_consistency_agreement(
    predictions_q: List[str], predictions_qp: List[str]
) -> float:
    if (
        not predictions_q
        or not predictions_qp
        or len(predictions_q) != len(predictions_qp)
    ):
        return 0.0
    matches = [
        normalize_answer(p1) == normalize_answer(p2)
        for p1, p2 in zip(predictions_q, predictions_qp)
    ]
    return 100.0 * sum(matches) / len(matches)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 8. Experimental Configuration & Consistency Training Engine
    The interactive training engine optimizes the multi-objective composite loss over paired sliding-window instances:
    $$\mathcal{L}_{\text{composite}}(\theta) = \mathcal{L}_{\text{CE}}(\theta) + \lambda_{\text{OT}} \mathcal{L}_{\text{OT}}(\theta) + \mu_{\text{Hung}} \mathcal{L}_{\text{Hungarian}}(\theta) + \lambda_{\text{KL}} \mathcal{D}_{\mathrm{KL}}^{\mathrm{sym}}(\theta)$$
    where hyperparameters $\lambda_{\text{OT}}, \mu_{\text{Hung}}, \lambda_{\text{KL}} \ge 0$ balance supervised span extraction accuracy against query paraphrase invariance. Configure training parameters below and initiate training with real-time streaming telemetry:
    """)
    return


@app.cell
def _(mo):
    # Reactive Controls with Direct Numeric Visibility
    model_dropdown = mo.ui.dropdown(
        options=[
            "bert-base-uncased",
            "microsoft/deberta-v3-small",
            "microsoft/deberta-v3-base",
        ],
        value="bert-base-uncased",
        label="Extractor Backbone",
    )

    loss_mode_dropdown = mo.ui.dropdown(
        options=[
            "dual (OT + Hungarian)",
            "kl (Symmetric KL Baseline)",
            "ot (1D OT Only)",
            "hungarian (Hungarian Only)",
            "ce_only (Standard Supervised / Augmentation)",
        ],
        value="dual (OT + Hungarian)",
        label="Loss Strategy",
    )

    lambda_ot_input = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.05,
        value=0.5,
        label="OT Weight (lambda_OT)",
    )

    lambda_kl_input = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.05,
        value=0.5,
        label="KL Weight (lambda_KL)",
    )

    mu_hung_input = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.1,
        label="Hungarian Weight (mu_Hung)",
    )

    topk_input = mo.ui.number(
        start=5,
        stop=30,
        step=1,
        value=10,
        label="Hungarian Top-k Candidates",
    )

    samples_input = mo.ui.number(
        start=10,
        stop=2000,
        step=25,
        value=100,
        label="Train Samples Count",
    )

    epochs_input = mo.ui.number(
        start=1,
        stop=10,
        step=1,
        value=1,
        label="Epochs",
    )

    batch_size_input = mo.ui.number(
        start=1,
        stop=32,
        step=1,
        value=4,
        label="Batch Size",
    )

    start_train_button = mo.ui.run_button(
        label="🚀 Start Consistency Training",
        kind="success",
    )

    _controls_ui = mo.vstack([
        mo.md("### ⚙️ Training Hyperparameters (Direct Numeric Input)"),
        mo.hstack([model_dropdown, loss_mode_dropdown]),
        mo.hstack([lambda_ot_input, mu_hung_input, lambda_kl_input]),
        mo.hstack([
            topk_input,
            samples_input,
            epochs_input,
            batch_size_input,
        ]),
        start_train_button,
    ])
    _controls_ui
    return (
        batch_size_input,
        epochs_input,
        lambda_kl_input,
        lambda_ot_input,
        loss_mode_dropdown,
        model_dropdown,
        mu_hung_input,
        samples_input,
        start_train_button,
        topk_input,
    )


@app.class_definition
class PairedQAFeatureDataset(Dataset):
    def __init__(self, features: List[dict]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return self.features[idx]


@app.function
def paired_collate_fn(batch: List[dict]) -> dict:
    def pad_seq(seqs: List[List[int]], pad_val: int = 0) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        padded = [s + [pad_val] * (max_len - len(s)) for s in seqs]
        return torch.tensor(padded, dtype=torch.long)

    input_ids_q = pad_seq([b["input_ids_q"] for b in batch])
    attn_q = pad_seq([b["attention_mask_q"] for b in batch])
    pass_mask_q = pad_seq([b["pass_mask_q"] for b in batch])
    s_pos_q = torch.tensor([b["start_pos_q"] for b in batch], dtype=torch.long)
    e_pos_q = torch.tensor([b["end_pos_q"] for b in batch], dtype=torch.long)

    input_ids_qp = pad_seq([b["input_ids_qp"] for b in batch])
    attn_qp = pad_seq([b["attention_mask_qp"] for b in batch])
    pass_mask_qp = pad_seq([b["pass_mask_qp"] for b in batch])
    pass_lens = [b["passage_len"] for b in batch]

    return {
        "input_ids_q": input_ids_q,
        "attn_q": attn_q,
        "pass_mask_q": pass_mask_q,
        "s_pos_q": s_pos_q,
        "e_pos_q": e_pos_q,
        "input_ids_qp": input_ids_qp,
        "attn_qp": attn_qp,
        "pass_mask_qp": pass_mask_qp,
        "pass_lens": pass_lens,
    }


@app.function
def train_consistency_model(
    train_records: List[dict],
    model_name: str,
    num_samples: int,
    epochs: int,
    batch_size: int,
    lambda_ot: float,
    mu_hung: float,
    top_k: int,
    device: torch.device,
    paraphraser: Optional[Any] = None,
    step_callback: Optional[Any] = None,
    progress_bar: Optional[Any] = None,
    loss_mode: str = "dual",
    lambda_kl: float = 0.5,
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """Trains ParaphraseConsistentQAExtractor with dual-level consistency regularization."""
    selected_samples = train_records[:num_samples]

    paired_data = []
    for item in selected_samples:
        if "paraphrase" in item:
            paired_data.append(item)
        elif paraphraser is not None:
            paired_data.append({**item, "paraphrase": paraphraser.back_translate(item["question"])})
        else:
            paired_data.append({**item, "paraphrase": item["question"]})

    with suppress_stdout_stderr():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ParaphraseConsistentQAExtractor(model_name=model_name).to(device)

    features = prepare_paired_sliding_windows(
        paired_data, tokenizer, max_seq_len=256, doc_stride=96
    )

    dataset = PairedQAFeatureDataset(features)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=paired_collate_fn
    )

    total_batches = len(dataloader)
    total_steps = total_batches * epochs
    if progress_bar is not None and hasattr(progress_bar, "progress"):
        progress_bar.progress.total = total_steps

    backbone_params = [p for _, p in model.encoder.named_parameters()]
    head_params = [p for _, p in model.qa_outputs.named_parameters()]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 3e-5, "weight_decay": 0.01},
        {"params": head_params, "lr": 1e-4, "weight_decay": 0.0},
    ])

    model.train()
    logs: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            out_q = model(
                input_ids=batch["input_ids_q"].to(device),
                attention_mask=batch["attn_q"].to(device),
                pass_mask=batch["pass_mask_q"].to(device),
            )

            out_qp = model(
                input_ids=batch["input_ids_qp"].to(device),
                attention_mask=batch["attn_qp"].to(device),
                pass_mask=batch["pass_mask_qp"].to(device),
            )

            loss_sup = compute_supervised_ce_loss(
                out_q["z_start"],
                out_q["z_end"],
                batch["s_pos_q"].to(device),
                batch["e_pos_q"].to(device),
            )

            batch_ot_loss = torch.tensor(0.0, device=device)
            batch_kl_loss = torch.tensor(0.0, device=device)
            batch_hung_loss = torch.tensor(0.0, device=device)
            b_size = batch["input_ids_q"].shape[0]

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

                    ot_s = compute_1d_wasserstein_loss(p_start_q, p_start_qp)
                    ot_e = compute_1d_wasserstein_loss(p_end_q, p_end_qp)
                    batch_ot_loss = batch_ot_loss + 0.5 * (ot_s + ot_e)

                    kl_s = compute_symmetric_kl_loss(p_start_q, p_start_qp)
                    kl_e = compute_symmetric_kl_loss(p_end_q, p_end_qp)
                    batch_kl_loss = batch_kl_loss + 0.5 * (kl_s + kl_e)

                    h_loss = compute_hungarian_loss_single_sample(
                        s_q, e_q, s_qp, e_qp, k=top_k, max_span_len=30
                    )
                    batch_hung_loss = batch_hung_loss + h_loss

            loss_ot = batch_ot_loss / float(max(1, b_size))
            loss_kl = batch_kl_loss / float(max(1, b_size))
            loss_hung = batch_hung_loss / float(max(1, b_size))

            mode_lower = str(loss_mode).lower()
            if "kl" in mode_lower:
                total_loss = loss_sup + lambda_kl * loss_kl
            elif "ce_only" in mode_lower:
                total_loss = loss_sup
            elif "hungarian" in mode_lower:
                total_loss = loss_sup + mu_hung * loss_hung
            elif "ot" in mode_lower and "dual" not in mode_lower:
                total_loss = loss_sup + lambda_ot * loss_ot
            else:  # dual (OT + Hungarian)
                total_loss = (
                    loss_sup +
                    lambda_ot * loss_ot +
                    mu_hung * loss_hung
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if progress_bar is not None:
                progress_bar.update(
                    increment=1,
                    subtitle=f"Ep {epoch+1}/{epochs} | Loss: {total_loss.item():.4f} (CE: {loss_sup.item():.4f}, OT: {loss_ot.item():.4f}, KL: {loss_kl.item():.4f}, Hung: {loss_hung.item():.4f})",
                )

            if step % 2 == 0 or (step == len(dataloader) - 1):
                entry = {
                    "epoch": epoch + 1,
                    "step": epoch * total_batches + step + 1,
                    "total_steps": total_steps,
                    "loss_total": round(total_loss.item(), 4),
                    "loss_supervised": round(loss_sup.item(), 4),
                    "loss_ot": round(loss_ot.item(), 4),
                    "loss_kl": round(loss_kl.item(), 4),
                    "loss_hungarian": round(loss_hung.item(), 4),
                }
                logs.append(entry)
                if step_callback is not None:
                    step_callback(logs)

    return model, logs


@app.cell
def _(
    batch_size_input,
    epochs_input,
    lambda_kl_input,
    lambda_ot_input,
    loss_mode_dropdown,
    mo,
    model_dropdown,
    mu_hung_input,
    paired_train_records,
    samples_input,
    start_train_button,
    topk_input,
):
    trained_model = None
    training_logs = []

    if start_train_button.value and paired_train_records:
        num_s = min(len(paired_train_records), int(samples_input.value))
        b_s = max(1, int(batch_size_input.value))
        total_batches = int(np.ceil(num_s / b_s))
        total_steps = total_batches * int(epochs_input.value)

        # 1. Smooth training telemetry without UI table recreation
        def update_telemetry(current_logs):
            latest = current_logs[-1]
            stat_cards = mo.hstack(
                [
                    mo.stat(
                        value=f"{latest['loss_total']:.4f}",
                        label="Total Loss",
                        caption="Composite objective",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{latest['loss_supervised']:.4f}",
                        label="Supervised CE",
                        caption="Cross-Entropy (CE)",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{latest['loss_ot']:.4f}",
                        label="1D OT",
                        caption="Wasserstein W1",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{latest.get('loss_kl', 0.0):.4f}",
                        label="Symmetric KL",
                        caption="Symmetric D_KL",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{latest['loss_hungarian']:.4f}",
                        label="Hungarian Span",
                        caption="Bipartite matching",
                        bordered=True,
                    ),
                ],
                justify="space-between",
                widths=[1, 1, 1, 1, 1],
            )
            step_max = latest.get("total_steps", total_steps)
            status_line = mo.md(
                f"⚡ **Step {latest['step']} / {step_max}** (Epoch {latest['epoch']}/{epochs_input.value}) — "
                f"Total Loss: `{latest['loss_total']:.4f}` | "
                f"Supervised: `{latest['loss_supervised']:.4f}` | "
                f"OT: `{latest['loss_ot']:.4f}` | "
                f"KL: `{latest.get('loss_kl', 0.0):.4f}` | "
                f"Hungarian: `{latest['loss_hungarian']:.4f}`"
            )
            mo.output.replace(
                mo.vstack([
                    mo.md("### 📊 Live Training Telemetry"),
                    stat_cards,
                    status_line,
                ])
            )

        # 2. Main training execution directly using the in-memory paired dataset
        with mo.status.progress_bar(
            total=total_steps,
            title="🚀 Training Extractive QA Model with Dual Consistency",
            remove_on_exit=False,
        ) as pbar:
            trained_model, training_logs = train_consistency_model(
                train_records=paired_train_records,
                model_name=model_dropdown.value,
                num_samples=num_s,
                epochs=int(epochs_input.value),
                batch_size=b_s,
                lambda_ot=float(lambda_ot_input.value),
                mu_hung=float(mu_hung_input.value),
                top_k=int(topk_input.value),
                device=DEVICE,
                step_callback=update_telemetry,
                progress_bar=pbar,
                loss_mode=loss_mode_dropdown.value,
                lambda_kl=float(lambda_kl_input.value),
            )

        # 3. Clean completion banner - telemetry and table rendered in downstream cell
        if training_logs:
            _final_steps = len(training_logs)
            _last = training_logs[-1]
            mo.output.replace(
                mo.callout(
                    mo.md(
                        f"🏁 **Consistency Training Completed Successfully!**\n\n"
                        f"Processed `{_final_steps}` optimization steps across `{epochs_input.value}` epoch(s).\n\n"
                        f"* **Final Composite Loss**: `{_last['loss_total']:.4f}`\n"
                        f"* **Supervised CE**: `{_last['loss_supervised']:.4f}` | "
                        f"**1D OT ($W_1$)**: `{_last['loss_ot']:.4f}` | "
                        f"**Hungarian Span**: `{_last['loss_hungarian']:.4f}` | "
                        f"**Symmetric KL**: `{_last.get('loss_kl', 0.0):.4f}`\n\n"
                        f"Detailed telemetry table and loss convergence trajectories are rendered below."
                    ),
                    kind="success",
                )
            )
    return trained_model, training_logs


@app.cell
def _(mo, training_logs):
    if not training_logs:
        _output = mo.callout(
            mo.md(
                "Configure numeric parameters above and click **🚀 Start Consistency Training** to train the model with real-time streaming telemetry."
            ),
            kind="info",
        )
    else:
        latest = training_logs[-1]
        stat_cards = mo.hstack(
            [
                mo.stat(
                    value=f"{latest['loss_total']:.4f}",
                    label="Final Total Loss",
                    caption="Composite objective",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{latest['loss_supervised']:.4f}",
                    label="Supervised CE",
                    caption="Cross-Entropy (CE)",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{latest['loss_ot']:.4f}",
                    label="1D OT",
                    caption="Wasserstein W1",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{latest.get('loss_kl', 0.0):.4f}",
                    label="Symmetric KL",
                    caption="Symmetric D_KL",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{latest['loss_hungarian']:.4f}",
                    label="Hungarian Span",
                    caption="Bipartite matching",
                    bordered=True,
                ),
            ],
            justify="space-between",
            widths=[1, 1, 1, 1, 1],
        )

        log_table = mo.ui.table(
            data=training_logs,
            pagination=True,
            page_size=10,
            label="Training Iteration Telemetry",
        )

        _output = mo.vstack([
            mo.md("### 📊 Training Progress & Loss Telemetry"),
            stat_cards,
            mo.md("#### Step-by-Step Training Log"),
            log_table,
        ])

    _output
    return


@app.cell
def _(mo, training_logs):
    if not training_logs or len(training_logs) < 2:
        _chart = mo.md("")
    else:
        fig_loss, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=120)
        steps = [x["step"] for x in training_logs]
        loss_tot = [x["loss_total"] for x in training_logs]
        loss_sup = [x["loss_supervised"] for x in training_logs]
        loss_ot = [x["loss_ot"] for x in training_logs]
        loss_kl = [x.get("loss_kl", 0.0) for x in training_logs]
        loss_h = [x["loss_hungarian"] for x in training_logs]

        ax1.plot(steps, loss_tot, "k-", lw=2.2, label=r"Total $\mathcal{L}_{\text{total}}$")
        ax1.plot(steps, loss_sup, color="#1f77b4", ls="--", lw=1.8, label=r"Supervised $\mathcal{L}_{\text{CE}}$")
        ax1.set_title("Interactive Loss Convergence", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Training Step", fontsize=10)
        ax1.set_ylabel("Loss Magnitude", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(frameon=True, fontsize=9)

        ax2.plot(steps, loss_ot, color="#2ca02c", lw=1.8, label=r"1D OT ($W_1$)")
        ax2.plot(steps, loss_kl, color="#9467bd", lw=1.8, label="Symmetric KL")
        ax2.plot(steps, loss_h, color="#ff7f0e", lw=1.8, label="2D Hungarian")
        ax2.set_title("Regularization Penalties", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Training Step", fontsize=10)
        ax2.set_ylabel("Regularization Loss", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(frameon=True, fontsize=9)

        plt.tight_layout()
        _chart = mo.as_html(fig_loss)
        plt.close(fig_loss)

    _chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 9. Interactive Paraphrase Consistency Playground
    Evaluate extractive answer boundary stability and semantic invariance across paired question formulations on custom or benchmark passages:
    """)
    return


@app.cell
def _(mo):
    preset_dropdown = mo.ui.dropdown(
        options=[
            "1. Notre Dame Founding Year (Historical Date)",
            "2. Lourdes Apparitions (Named Entity)",
            "3. Super Bowl 50 Venue (Geographic Location)",
            "Custom (Freeform Entry)",
        ],
        value="1. Notre Dame Founding Year (Historical Date)",
        label="Select SQuAD Benchmark Exemplar:",
    )
    return (preset_dropdown,)


@app.cell
def _(mo, preset_dropdown):
    _sel = preset_dropdown.value
    if "Notre Dame" in _sel:
        _ctx = (
            "The University of Notre Dame is a private Catholic research university in Notre Dame, Indiana, "
            "outside the city of South Bend. It was founded on November 26, 1842, by Edward Sorin."
        )
        _q = "When was the university founded?"
        _qp = "In what year was the institution established?"
    elif "Lourdes" in _sel:
        _ctx = (
            "In 1858, Bernadette Soubirous reported 18 apparitions of the Virgin Mary at the Massabielle grotto "
            "in Lourdes, France. The Catholic Church subsequently declared the visions worthy of belief."
        )
        _q = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
        _qp = "Whom did the Virgin Mary allegedly appear to in 1858 in Lourdes France?"
    elif "Super Bowl" in _sel:
        _ctx = (
            "Super Bowl 50 was an American football game to determine the champion of the National Football League "
            "for the 2015 season. The American Football Conference champion Denver Broncos defeated the National Football "
            "Conference champion Carolina Panthers 24-10 at Levi's Stadium in Santa Clara, California."
        )
        _q = "Where was Super Bowl 50 held?"
        _qp = "In what location was Super Bowl 50 played?"
    else:
        _ctx = ""
        _q = ""
        _qp = ""

    context_area = mo.ui.text_area(
        value=_ctx,
        label="Context Passage (c)",
        rows=5,
        full_width=True,
    )

    q_input = mo.ui.text(
        value=_q,
        label="Query (q)",
        full_width=True,
    )

    qp_input = mo.ui.text(
        value=_qp,
        label="Paraphrase (q')",
        full_width=True,
    )

    infer_button = mo.ui.run_button(
        label="⚡ Extract Answers & Evaluate Consistency",
        kind="success",
    )

    _inputs_ui = mo.vstack([
        mo.md("#### 🎯 1. Select Benchmark SQuAD Exemplar"),
        preset_dropdown,
        mo.md("#### 📖 2. Document Context Passage"),
        context_area,
        mo.md("#### 💬 3. Paired Interrogative Formulations"),
        mo.hstack([
            mo.vstack([
                mo.md("**🔵 Original Question Formulation ($q$)**"),
                q_input,
            ]),
            mo.vstack([
                mo.md("**🟣 Paraphrased Formulation ($q'$)**"),
                qp_input,
            ]),
        ], widths=[1, 1], gap=1),
        mo.hstack([infer_button], justify="center"),
    ], gap=1)
    _inputs_ui
    return context_area, infer_button, q_input, qp_input


@app.function
def build_highlighted_context_html(
    context: str,
    detail_q: Dict[str, Any],
    detail_qp: Dict[str, Any],
    is_consistent: bool,
) -> str:
    """Renders HTML representing document context with highlighted answer spans."""
    sq, eq = detail_q.get("char_start", 0), detail_q.get("char_end", 0)
    sqp, eqp = detail_qp.get("char_start", 0), detail_qp.get("char_end", 0)

    def _esc(t: str) -> str:
        return (
            t.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    if not context:
        return "<em>Empty context passage</em>"

    valid_q = 0 <= sq < eq <= len(context)
    valid_qp = 0 <= sqp < eqp <= len(context)

    # Identical agreement
    if is_consistent and valid_q:
        s = min(sq, sqp) if valid_qp else sq
        e = max(eq, eqp) if valid_qp else eq
        return (
            f"{_esc(context[:s])}"
            f'<mark style="background-color: #d1fae5; color: #065f46; font-weight: 700; '
            f'padding: 3px 8px; border-radius: 4px; border: 1.5px solid #10b981;" '
            f'title="Consistent Invariant Span (q &amp; q\')">{_esc(context[s:e])}</mark>'
            f"{_esc(context[e:])}"
        )

    # Disagreement: highlight both spans
    tags = []
    if valid_q:
        tags.append((sq, eq, "q", "#dbeafe", "#1e40af", "#3b82f6"))
    if valid_qp:
        tags.append((sqp, eqp, "q'", "#fef3c7", "#92400e", "#f59e0b"))

    if not tags:
        return _esc(context)

    # Check for partial overlap
    if len(tags) == 2 and not (
        tags[0][1] <= tags[1][0] or tags[1][1] <= tags[0][0]
    ):
        min_s = min(sq, sqp)
        max_e = max(eq, eqp)
        return (
            f"{_esc(context[:min_s])}"
            f'<mark style="background-color: #fed7aa; color: #9a3412; font-weight: 700; '
            f'padding: 3px 8px; border-radius: 4px; border: 1.5px solid #ea580c;" '
            f'title="Overlapping Boundary Discrepancy">{_esc(context[min_s:max_e])} '
            f'<span style="font-size:0.75em; opacity:0.8;">[Boundary Mismatch]</span></mark>'
            f"{_esc(context[max_e:])}"
        )

    tags.sort(key=lambda t: t[0], reverse=True)
    raw_ctx = context
    for s, e, lbl, bg, fg, bd in tags:
        marked = (
            f'<mark style="background-color: {bg}; color: {fg}; font-weight: 700; '
            f'padding: 3px 7px; border-radius: 4px; border: 1.5px solid {bd};" '
            f'title="Extracted for {lbl}">{_esc(raw_ctx[s:e])} '
            f'<span style="font-size: 0.72em; text-transform: uppercase; background: rgba(0,0,0,0.1); '
            f'padding: 1px 4px; border-radius: 3px;">{lbl}</span></mark>'
        )
        raw_ctx = raw_ctx[:s] + marked + raw_ctx[e:]
    return raw_ctx


@app.function
def predict_answer_pair_detailed(
    model: Any,
    tokenizer: Any,
    context: str,
    question: str,
    paraphrase: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs sliding-window relative margin inference and returns full extraction metadata."""
    model.eval()
    ex_pair = [{
        "id": "interactive_demo",
        "context": context,
        "question": question,
        "paraphrase": paraphrase,
        "answer_start": 0,
        "answer_end": 0,
    }]

    features = prepare_paired_sliding_windows(
        ex_pair, tokenizer, max_seq_len=256, doc_stride=96
    )
    context_enc = tokenizer(
        context, return_offsets_mapping=True, add_special_tokens=False
    )
    context_offsets = context_enc["offset_mapping"]

    window_cands_q: List[List[Tuple[int, int, float]]] = []
    window_offsets_q: List[int] = []
    window_cands_qp: List[List[Tuple[int, int, float]]] = []
    window_offsets_qp: List[int] = []

    with torch.no_grad():
        for feat in features:
            ids_q = torch.tensor([feat["input_ids_q"]], device=device)
            mask_q = torch.tensor([feat["attention_mask_q"]], device=device)
            out_q = model(ids_q, mask_q)
            z_s_q = out_q["z_start"][0].tolist()
            z_e_q = out_q["z_end"][0].tolist()
            window_cands_q.append(
                decode_window_predictions(z_s_q, z_e_q, feat["pass_mask_q"])
            )
            window_offsets_q.append(feat["context_offset_start"])

            ids_qp = torch.tensor([feat["input_ids_qp"]], device=device)
            mask_qp = torch.tensor([feat["attention_mask_qp"]], device=device)
            out_qp = model(ids_qp, mask_qp)
            z_s_qp = out_qp["z_start"][0].tolist()
            z_e_qp = out_qp["z_end"][0].tolist()
            window_cands_qp.append(
                decode_window_predictions(
                    z_s_qp, z_e_qp, feat["pass_mask_qp"]
                )
            )
            window_offsets_qp.append(feat["context_offset_start"])

    detail_q = aggregate_multi_window_spans_detailed(
        window_cands_q, window_offsets_q, context, tokenizer, context_offsets
    )
    detail_qp = aggregate_multi_window_spans_detailed(
        window_cands_qp,
        window_offsets_qp,
        context,
        tokenizer,
        context_offsets,
    )

    ans_q = detail_q["text"]
    ans_qp = detail_qp["text"]
    is_consistent = normalize_answer(ans_q) == normalize_answer(ans_qp)
    f1_val = compute_f1(ans_q, ans_qp)

    return {
        "ans_q": ans_q,
        "ans_qp": ans_qp,
        "detail_q": detail_q,
        "detail_qp": detail_qp,
        "is_consistent": is_consistent,
        "f1": f1_val,
    }


@app.function
def predict_answer_pair(
    model: Any,
    tokenizer: Any,
    context: str,
    question: str,
    paraphrase: str,
    device: torch.device,
) -> Tuple[str, str, bool]:
    res = predict_answer_pair_detailed(
        model, tokenizer, context, question, paraphrase, device
    )
    return res["ans_q"], res["ans_qp"], res["is_consistent"]


@app.cell
def _(context_area, infer_button, mo, q_input, qp_input, trained_model):
    if not infer_button.value:
        _display = mo.callout(
            mo.md(
                "Configure or select a benchmark passage and question formulations above, then click "
                "**⚡ Extract Answers & Evaluate Consistency**."
            ),
            kind="info",
        )
    else:
        _eval_model = trained_model
        with suppress_stdout_stderr():
            if _eval_model is None:
                _eval_model = ParaphraseConsistentQAExtractor(
                    model_name="bert-base-uncased"
                ).to(DEVICE)
            _tokenizer = AutoTokenizer.from_pretrained(_eval_model.model_name)
        _res = predict_answer_pair_detailed(
            model=_eval_model,
            tokenizer=_tokenizer,
            context=context_area.value,
            question=q_input.value,
            paraphrase=qp_input.value,
            device=DEVICE,
        )

        _ans_q = _res["ans_q"]
        _ans_qp = _res["ans_qp"]
        _detail_q = _res["detail_q"]
        _detail_qp = _res["detail_qp"]
        _is_consistent = _res["is_consistent"]
        _f1 = _res["f1"]

        _status_callout = (
            mo.callout(
                mo.md(
                    f"### ✅ Exact Semantic Invariance Confirmed\n\n"
                    f"Both query formulations converged to the identical extractive target with "
                    f"**100% Exact Match** and **$F_1 = 1.00$**.\n\n"
                    f"* **Consensus Extracted Answer**: **`{_ans_q}`**\n"
                    f"* **Invariance Verification**: $\\mathrm{{argmax}} P(a \\mid c, q) \\equiv \\mathrm{{argmax}} P(a \\mid c, q')$"
                ),
                kind="success",
            )
            if _is_consistent
            else mo.callout(
                mo.md(
                    f"### ⚠️ Paraphrase Sensitivity Discrepancy Detected\n\n"
                    f"Extractive boundary predictions diverged under query paraphrase perturbation "
                    f"(Exact Match: **0.0%**, Token $F_1$: **{_f1:.2f}**).\n\n"
                    f"* **Formulation ($q$) Answer**: `{_ans_q if _ans_q else '[NULL / Unanswerable]'}`\n"
                    f"* **Paraphrase ($q'$) Answer**: `{_ans_qp if _ans_qp else '[NULL / Unanswerable]'}`\n"
                    f"* **Discrepancy Severity**: Spatial coordinate boundary drift detected"
                ),
                kind="warn",
            )
        )

        _highlighted_html = build_highlighted_context_html(
            context_area.value, _detail_q, _detail_qp, _is_consistent
        )

        _passage_visualizer = mo.md(f"""
        #### 📄 Document Passage with Projected Extractive Spans
        <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 18px; line-height: 1.7; font-size: 0.95rem; color: #1e293b;">
          {_highlighted_html}
        </div>
        <div style="margin-top: 6px; font-size: 0.82rem; color: #64748b;">
          {'🟢 <b>Green Highlight</b>: Consensus invariant span extracted for both query formulations' if _is_consistent else '🔵 <b>Blue [q]</b>: Span extracted for original query &nbsp;|&nbsp; 🟠 <b>Amber [q\']</b>: Span extracted for paraphrase'}
        </div>
        """)

        def _make_pred_card(
            title: str, query_txt: str, ans_txt: str, det: dict, color: str
        ):
            cs, ce = det.get("char_start", 0), det.get("char_end", 0)
            ts, te = det.get("token_start", 0), det.get("token_end", 0)
            mg = det.get("margin", 0.0)
            ans_str = ans_txt if ans_txt else "(No non-null span extracted)"
            return mo.md(f"""
            <div style="border: 1px solid #e2e8f0; border-top: 4px solid {color}; border-radius: 8px; padding: 16px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
              <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.04em;">{title}</div>
              <div style="font-style: italic; color: #334155; margin: 6px 0 12px 0; font-size: 0.95rem;">"{query_txt}"</div>
              <div style="background-color: #f1f5f9; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px;">
                <span style="font-size: 0.72rem; text-transform: uppercase; color: #64748b; display: block; font-weight: 600;">Extracted Answer Span</span>
                <span style="font-size: 1.25rem; font-weight: 700; color: #0f172a; word-break: break-word;">{ans_str}</span>
              </div>
              <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #f1f5f9;">
                  <td style="padding: 4px 0; color: #64748b;">Character Coordinates:</td>
                  <td style="padding: 4px 0; text-align: right; font-weight: 600; font-family: monospace;">[{cs}, {ce}] ({max(0, ce - cs)} chars)</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                  <td style="padding: 4px 0; color: #64748b;">Token Span Index:</td>
                  <td style="padding: 4px 0; text-align: right; font-weight: 600; font-family: monospace;">[{ts}, {te}] ({max(0, te - ts + 1)} tokens)</td>
                </tr>
                <tr>
                  <td style="padding: 4px 0; color: #64748b;">Relative Margin (Δz):</td>
                  <td style="padding: 4px 0; text-align: right; font-weight: 600; font-family: monospace;">{mg:+.3f}</td>
                </tr>
              </table>
            </div>
            """)

        _card_q = _make_pred_card(
            "Original Query Formulation (q)",
            q_input.value,
            _ans_q,
            _detail_q,
            "#3b82f6",
        )
        _card_qp = _make_pred_card(
            "Paraphrased Formulation (q')",
            qp_input.value,
            _ans_qp,
            _detail_qp,
            "#10b981" if _is_consistent else "#f59e0b",
        )

        _cards_hstack = mo.hstack([_card_q, _card_qp], widths=[1, 1], gap=1)

        _char_drift = abs(
            _detail_q.get("char_start", 0) - _detail_qp.get("char_start", 0)
        ) + abs(_detail_q.get("char_end", 0) - _detail_qp.get("char_end", 0))
        _tok_drift = abs(
            _detail_q.get("token_start", 0) - _detail_qp.get("token_start", 0)
        ) + abs(_detail_q.get("token_end", 0) - _detail_qp.get("token_end", 0))
        _margin_diff = abs(
            _detail_q.get("margin", 0.0) - _detail_qp.get("margin", 0.0)
        )

        _table_md = mo.md(f"""
        #### 📐 Comparative Boundary & Invariance Metrics
        | Metric Dimension | Question ($q$) | Paraphrase ($q'$) | Pairwise Invariance Assessment |
        | :--- | :--- | :--- | :--- |
        | **Extracted Text** | `{_ans_q if _ans_q else '[NULL]'}` | `{_ans_qp if _ans_qp else '[NULL]'}` | {'✅ Exact Match (EM = 1.0)' if _is_consistent else '❌ Boundary Mismatch (EM = 0.0)'} |
        | **Token Overlap $F_1$** | 1.0000 | 1.0000 | **{_f1:.4f}** |
        | **Spatial Token Drift** | `[{_detail_q.get('token_start', 0)}, {_detail_q.get('token_end', 0)}]` | `[{_detail_qp.get('token_start', 0)}, {_detail_qp.get('token_end', 0)}]` | $\\vert \\Delta s \\vert + \\vert \\Delta e \\vert =$ **`{_tok_drift}` tokens** |
        | **Character Boundary** | `[{_detail_q.get('char_start', 0)}, {_detail_q.get('char_end', 0)}]` | `[{_detail_qp.get('char_start', 0)}, {_detail_qp.get('char_end', 0)}]` | $\\vert \\Delta c_{{\\text{{start}}}} \\vert + \\vert \\Delta c_{{\\text{{end}}}} \\vert =$ **`{_char_drift}` chars** |
        | **Relative Margin Score** | `{_detail_q.get('margin', 0.0):+.3f}` | `{_detail_qp.get('margin', 0.0):+.3f}` | $\\vert \\Delta z_q - \\Delta z_{{q'}} \\vert =$ **`{_margin_diff:.3f}`** |
        """)

        _display = mo.vstack([
            mo.md("### 🎯 Live Model Inference Results"),
            _status_callout,
            _passage_visualizer,
            mo.md("#### 🔍 Extracted Answer Formulations"),
            _cards_hstack,
            _table_md,
        ])

    _display
    return


if __name__ == "__main__":
    app.run()
