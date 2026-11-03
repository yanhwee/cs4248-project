# G17 Project Status Review and Gap Analysis

> **Superseded.** This is the snapshot taken *before* the code changes of 3 November 2026. The changes that address
> the gaps listed here are recorded in `docs/CHANGES.md`; the README describes the current pipeline.

*Snapshot: 3 November 2026. Reference proposal: `docs/CS4248 Project.pdf`. Final deliverables fall in Week 13 of the proposal timeline.*

**Purpose.** This document lists what the repository currently contains, how it maps onto each component of our proposal, where it diverges from the project requirements, and what is still open. It is meant as a shared checklist so we can split the remaining work. A lot of reusable scaffolding is already in place.

---

## 1. Summary

- The repo has a working end-to-end scaffold for the **extractor half** of the proposal: SQuAD loading, sliding-window preprocessing, a BERT-base span model, the symmetric-KL consistency loss (plus the optimal-transport and Hungarian variants from item 2a), a training CLI, an evaluation CLI, Slurm scripts, figure generation, and an ACL-format report skeleton.
- The **reranker half** of the proposal (top-20 distinct candidates per question, learned span reranker) does not exist yet.
- The **paraphrase pipeline** is only partly wired: MarianMT back-translation and sentence-transformer filtering are coded but never activated, so training currently uses regex-rewritten questions.
- The consistency objective is **CE(q) + λ·KL**; the proposal's **CE(q′)** term is missing, which also means the "augmentation without the loss" ablation cannot be run yet.
- **No validation split** (5% of training articles) exists; λ cannot be tuned as proposed, and the dev set is the only evaluation data.
- The **robustness evaluation** (Gan & Ng paraphrased test sets, Adversarial SQuAD AddSent / AddOneSent) is not present, and the official `evaluate-v2.0.py` is not wired in.
- **All reported numbers are placeholders** written directly into `visualize.py`; no full training run or full dev-set evaluation has been done.
- The **report narrative** is currently built around "KL is pathological, OT + Hungarian fixes it", whereas the proposal makes symmetric KL the primary consistency loss and OT a sub-item. The write-up needs re-centring on consistency training + reranker, with OT as an ablation.
- The code only imports on **Python 3.12+**; the two environment files disagree.
- Slides and the 15-minute video have not been started.

---

## 2. What is already in place

| Area | Status | Location |
| :--- | :--- | :--- |
| SQuAD JSON loader | Done | `app.py` `load_squad_data` |
| Sliding-window tokenisation with char-to-token offsets, null-anchor labels for windows without the answer | Done | `app.py` `prepare_paired_sliding_windows`, `project_char_to_token` |
| Paired (q, q′) dataset and collate function | Done | `app.py` `PairedQAFeatureDataset`, `paired_collate_fn` |
| BERT-base encoder + linear start/end head; backbone swappable via `AutoModel` | Done | `app.py` `ParaphraseConsistentQAExtractor` |
| Losses: CE, **symmetric KL** (proposal item 2), **1D Wasserstein OT** and **Hungarian span matching** (proposal item 2a); selectable with `--loss_type` | Done | `app.py` lines ~678–872; `train.py` |
| Null-margin multi-window decoding with 30-token span cap (best span only) | Done | `app.py` `decode_window_predictions`, `aggregate_multi_window_spans_detailed` |
| Top-k span extraction inside one window (used by the Hungarian loss) | Done | `app.py` `extract_topk_spans` |
| EM / F1 reimplementation, paraphrase-agreement (consistency) metric, token displacement | Done, see §4 | `app.py`, `evaluate.py` |
| MarianMT EN→DE→EN back-translation class | Coded, not activated | `app.py` `MarianParaphraser` |
| Sentence-embedding cosine similarity | Coded, not called | `app.py` `compute_sbert_similarity` |
| Training CLI with AMP, gradient clipping, JSON telemetry, checkpointing | Done | `train.py` |
| Evaluation CLI (EM, F1, agreement, displacement) | Done, see §4 | `evaluate.py` |
| Slurm scripts for the SoC cluster | Done | `scripts/` |
| Figure and LaTeX table generator | Done, see §5 | `visualize.py` |
| ACL-format report, 4 + 1 pages, Times 11pt; bib already cites Gan & Ng 2019, Xie et al. 2020, Devlin et al. 2019, He et al. 2021 | Compiles | `report/` |
| Maths writeup | Done | `docs/math.md` |
| Interactive marimo notebook (training sandbox, paraphrase playground) | Done | `app.py` |
| Official scorer and data files | Present | `provided/` |
| Compliance: no closed-source LLMs or API calls | OK | — |

Evidence of one short real run exists: Figure 1 in the compiled PDF shows about 160 optimisation steps with cross-entropy going from ~5.6 to ~2.3. That is a smoke test, not a trained model.

---

## 3. Mapping to the proposal

### 3.1 Extractor (proposal item 1)

| Proposal | Repo status | Gap |
| :--- | :--- | :--- |
| Fine-tune BERT-base with span head and sliding windows (baseline) | Model and preprocessing done; no full-dataset checkpoint or dev score yet | Run full training; score with official script |
| Fine-tune DeBERTa-v3 extractor | Only a dropdown option in the notebook; never run | Train and evaluate; check fp16 stability with DeBERTa-v3 (bf16 or fp32 may be safer) |
| Keep each question's **top-20 distinct spans** | Inference returns only the single best span; `extract_topk_spans` works per window with k = 3 (training) and no text-level de-duplication | Add cross-window top-k candidate generation, de-duplicated by answer text, exported per question for the reranker |

### 3.2 Paraphrase-consistency training (proposal item 2)

| Proposal | Repo status | Gap |
| :--- | :--- | :--- |
| Back-translation EN→DE→EN with open MarianMT | `MarianParaphraser.load_models()` is never called, so `back_translate` always falls through to the regex rules in `_heuristic_paraphrase` | Call `load_models()`; batch the translation (currently one question per call); generate once and cache to disk |
| Keep pairs with high embedding similarity (sentence-transformers) | `compute_sbert_similarity` exists but is never used; no threshold, no filtering step; `sentence-transformers` missing from `requirements.txt` | Build the filter step and the paraphrase dataset file |
| Loss = **CE(q) + CE(q′) + λ·KL** over passage-token start/end distributions | Symmetric KL implemented and selectable (`--loss_type kl`); **CE(q′) is not computed** (`train.py` line ~294 and `app.py` line ~1299 call CE on `out_q` only); the q′ forward pass is discarded under `ce_only` | Add CE(q′); add an "augmentation only" mode (CE(q) + CE(q′), λ = 0) |
| 2a. Optimal-transport loss (Hungarian) | Done: 1D Wasserstein and Hungarian top-k matching, modes `ot`, `hungarian`, `dual` | Keep as ablation |
| 2b. t-SNE visualisation | Not implemented | Add: embed (q, q′) pairs or span representations, plot before/after consistency training |
| Tune λ on **5% of training articles** held out; dev used only for final test | No article-level split, no validation loop; the dev set is the only evaluation set | Add an article-level 95/5 split in the loader; evaluate on the 5% split during training; select λ there |

### 3.3 Learned span reranker (proposal item 3)

| Proposal | Repo status | Gap |
| :--- | :--- | :--- |
| Reranker picks one of the numbered top-k candidates using the extractor's own output representations; output is always an exact passage span | Not implemented; no candidate export to feed it | Design and implement (e.g. a small head over pooled span representations, or an open-weight LLM scoring the numbered candidates); train on the 5% split or on training-set candidates; add a with/without-reranker ablation |

### 3.4 Evaluation protocol

| Proposal | Repo status | Gap |
| :--- | :--- | :--- |
| EM, F1 via `evaluate-v2.0.py` | `evaluate.py` uses its own EM/F1; no predictions JSON is written | Export `{id: answer}` and run `python provided/evaluate-v2.0.py provided/dev-v1.1.json preds.json` |
| Consistency = share of paraphrase pairs with the same answer | Implemented (normalised string match) | OK |
| Gan & Ng (2019) paraphrased test sets: 1,062 non-adversarial + 56 adversarial questions from dev-v1.1 | Not in repo; no loader | Fetch from the authors' release, `github.com/nusnlp/paraphrasing-squad`; add a loader and an evaluation pass |
| Adversarial SQuAD (Jia & Liang 2017): AddSent 3,560, AddOneSent 1,787 | Not in repo; no loader | Available on the Hugging Face Hub as `squad_adversarial` (configs `AddSent`, `AddOneSent`), or from `github.com/robinjia/adversarial-squad`; add a loader and an evaluation pass |
| Compare against published BERT and DeBERTa-v3 results | Not in the report | Add a comparison row/table with cited numbers |
| Ablate augmentation with vs. without the consistency loss | Ablation grid in `scripts/` is `ce_only / kl / ot / hungarian / dual`; "augmentation only" is absent (blocked by the missing CE(q′)); with/without reranker also absent | Extend the grid once CE(q′) and the reranker exist |
| Error analysis | Not done; Table 2 is placeholder text | Produce from real predictions on dev, paraphrased, and adversarial sets |

### 3.5 Timeline check

| Proposal week | Planned | Repo status |
| :--- | :--- | :--- |
| 8 | Preprocessing, BERT baseline | Preprocessing done; baseline not trained at scale |
| 9 | DeBERTa extractor, top-k generation, paraphrase pipeline | DeBERTa untrained; top-k not exported; paraphrase pipeline not activated |
| 10 | Consistency loss, train consistency-aware extractor | Loss coded (minus CE(q′)); not trained at scale |
| 11 | Reranker; tune λ | Not started; no validation split |
| 12 | Ablations, paraphrase/adversarial tests, error analysis | Not started; test sets not in repo |
| 13 | Final experiments, report, slides, video, code docs | Report skeleton exists; slides and video not started |

---

## 4. Alignment with the project requirements

Source: `docs/cs4248_project_requirements (1).pdf` and `docs/cs4248_projects_2627s1.pdf`.

| Requirement | Current state | Gap / action |
| :--- | :--- | :--- |
| Evaluate with the official `evaluate-v2.0.py` on `dev-v1.1.json` | Custom EM/F1; compares against the **first gold answer only** (`all_answers` is loaded but unused); default `--samples 100`; failed examples are skipped silently | Wire in the official script on the full dev set. 5,625 of the 10,570 dev questions have more than one distinct gold answer, so first-answer-only scoring understates EM/F1 |
| Results reflect actual work done and results achieved | All numbers are placeholders (see §5) | Replace with real runs before submission |
| ACL short paper: 4 pages + 1 page references, Times 11pt, official style files | `main.tex` uses `acl.sty`, `times`, 11pt; currently exactly 5 pages | Format OK; content changes below will need re-balancing |
| State clearly which parts are our own contribution vs. external code | Not present | Add a short paragraph (proposal already lists the split: we write paraphrase filtering, consistency loss, candidate generation, reranker, evaluation code) |
| Cite relevant prior work and compare to it | 16 works cited, including Gan & Ng and Xie et al. | Add Jia & Liang 2017; add published BERT / DeBERTa-v3 comparison |
| No closed-source LLMs or API calls | None used | OK; the reranker must stay open-weight |
| Submit source code, external code, data, run instructions | README has instructions, but the two environment files disagree (see §7) | Align the environment spec; agree on data packaging |
| 15-minute recorded presentation, every member speaks; slides | Not started | Schedule recording |
| Author names on the report | Author block is a single placeholder email | Add all five names |

---

## 5. Results, tables, and figures

Everything below currently has placeholder content and must be regenerated from real runs.

| Artifact | Where the values come from today |
| :--- | :--- |
| Table 1 (EM / F1 / Agreement for five conditions) | Literal lists at `visualize.py` lines ~144–146 and ~229–239; copied into `report/main.tex`, `figures/*.md/.tex`, and `README.md` |
| Figure 2 (boundary-displacement histogram) | Sampled from `np.random.exponential` / `np.random.geometric` with a fixed seed at `visualize.py` lines ~206–215 |
| Figure 1 (loss convergence) | Reads `runs/history_dual.json` if present, otherwise an analytic exponential curve; the compiled PDF's version is from the ~160-step smoke run |
| Table 2 (qualitative cases) | Hand-written question / answer pairs, not model outputs |
| "Δ Consistency", "28.5% of pairs drift", "+19.9%" etc. | Derived from the placeholder numbers above |
| README "Key Empirical Findings" | Same placeholder numbers |

Suggested change: remove every hard-coded fallback from `visualize.py` so it fails loudly when `runs/` is empty, and have it read evaluation JSONs for table values. Add the new tables the proposal needs: paraphrased-set and adversarial-set results, with/without reranker, published-model comparison.

---

## 6. Report text vs. proposal and vs. code

**Narrative alignment with the proposal**

| Report currently says | Proposal says | Adjustment |
| :--- | :--- | :--- |
| Symmetric KL is "fundamentally ill-suited"; OT + Hungarian is the proposed method | Symmetric KL is the primary consistency loss; OT is sub-item 2a | Re-centre: consistency training (KL) + top-k + reranker as the contribution; present OT/Hungarian as an ablation of the consistency term |
| "We prove" KL has infinite penalties and exploding gradients | — | With softmax outputs and the `eps = 1e-8` clamp in `compute_symmetric_kl_loss`, KL is always finite. The defensible point is that KL ignores token distance (permutation invariance). Soften or drop |
| No reranker, no adversarial or paraphrased test sets, no validation protocol, no published-model comparison | All four are core to the proposal | Add sections / rows once results exist |

**Text vs. implementation**

| Report states | Code currently does |
| :--- | :--- |
| MarianMT EN→DE→EN back-translation | Regex rewrites (`load_models()` never called) |
| Sentence-BERT filtering, cos ≥ 0.85, edit distance ≥ 3 | No filtering |
| Loss = CE + λ_OT·OT + μ_Hung·Hung (no CE(q′)) | Matches code, but not the proposal's CE(q) + CE(q′) + λ·KL |
| μ_Hung = 0.3 | 0.1 |
| Stride S = 128 | 96 |
| Linear warmup ratio 0.1 | No scheduler or warmup |
| Top-k with k ≤ 5 | k = 3 in `train.py`, 10 in the notebook |
| Trained on all 87,599 examples; evaluated on all 10,570 dev questions | Cluster job trains on 1,000 samples; `evaluate.py` defaults to 100 |
| Figure 2 caption: Δs = \|s_q − s_q′\| | Axis and code use start + end displacement |
| Figure 1 caption: penalties "stabilize smoothly" | The OT penalty rises throughout the plotted run |

---

## 7. Code and environment gaps

**Blockers (quick to fix)**

- `app.py` line ~1965 has a backslash inside an f-string expression. This is a `SyntaxError` on Python ≤ 3.11, and because `train.py` and `evaluate.py` import `app.py`, they fail too. One-line fix.
- `pyproject.toml` requires Python ≥ 3.15 and torch ≥ 2.15; `requirements.txt` says Python ≥ 3.10 and torch ≥ 2.1. Pick one spec that works on the SoC cluster. `sentence-transformers` must be in whichever file is used. The package name in `pyproject.toml` is `cs4284-project-1`.

**Training**

- No learning-rate schedule or warmup; no evaluation during training; `best_model.pt` is simply the last checkpoint.
- Paraphrases are regenerated on every run (fast with regex, hours once MarianMT is enabled); generate once and cache.
- OT and Hungarian losses run in a per-sample Python loop with a scipy call each, and every step does two forward passes. Full-dataset runs will be slow; the 3.5-hour Slurm limit in `scripts/submit_all_ablations.sbatch` will not cover the ablation grid.
- `project_char_to_token` labels a window as containing the answer even when only part of the answer lies inside it.

**Evaluation**

- First-gold-answer-only scoring; default 100 samples; `except Exception: continue` drops failures from the denominator; no predictions JSON.
- No loaders for the paraphrased or adversarial test sets.

**Housekeeping**

- `report/figures/*.pdf` and `figures/*.pdf` differ for two of three figures.
- `datasets` and `accelerate` are listed but unused (the proposal cites the HF Datasets QA example as a preprocessing reference; `squad_adversarial` loading would use it).
- All core code lives inside the marimo notebook, so marimo is a dependency for cluster training. Fine for now; consider moving the core into a plain module later.
- README shows an MIT badge with no LICENSE file and describes the paper as "ACL 2026 camera-ready".

---

## 8. Open work and suggested order

| # | Task | Depends on |
| :-: | :--- | :--- |
| 1 | Fix the f-string and the environment spec; add `sentence-transformers` | — |
| 2 | Activate MarianMT back-translation (batched), add the similarity filter, generate and cache the paraphrase dataset | 1 |
| 3 | Add CE(q′) to the objective and an "augmentation only" mode | — |
| 4 | Add the article-level 5% validation split and a validation pass inside `train.py` | — |
| 5 | Add predictions export + official-scorer call; score against all gold answers; default to the full dev set | — |
| 6 | Remove placeholder numbers and random fallbacks from `visualize.py` | — |
| 7 | Train BERT-base baseline on the full training set; score with the official script | 1, 5 |
| 8 | Train consistency-aware BERT (KL) and DeBERTa-v3 variants; tune λ on the 5% split | 2, 3, 4, 7 |
| 9 | Export top-20 distinct candidates per question; implement and train the span reranker; with/without-reranker ablation | 8 |
| 10 | Add loaders for Gan & Ng paraphrased sets and Adversarial SQuAD; run robustness evaluation; published-model comparison | 5, 8 |
| 11 | Ablation grid (augmentation-only / +KL / +OT / +Hungarian / ±reranker), t-SNE figure, error analysis | 8, 9, 10 |
| 12 | Regenerate figures and tables from `runs/`; rewrite report (abstract, §3 framing, §4–§7); add names and the contributions statement; keep to 4 + 1 pages | 11 |
| 13 | Slides; 15-minute video with all five members; README run instructions; package code + data | 12 |

Rough GPU budget: a full-dataset BERT-base run with two forward passes per step is on the order of 1–2 hours per epoch on a T4 before the Python-loop overhead. DeBERTa-v3-base is roughly 1.5–2× slower. Plan the number of conditions against the cluster time we can actually get; if time is short, prioritise baseline, +KL, +KL+reranker, and the robustness evaluation over the OT/Hungarian variants.

---

## Appendix: quick reference

**Official scoring**
```bash
python provided/evaluate-v2.0.py provided/dev-v1.1.json runs/preds_<condition>.json
```

**External evaluation sets**
- Gan & Ng (2019) paraphrased SQuAD: `https://github.com/nusnlp/paraphrasing-squad`
- Adversarial SQuAD (Jia & Liang 2017): Hugging Face dataset `squad_adversarial`, configs `AddSent` (3,560) and `AddOneSent` (1,787); original release at `https://github.com/robinjia/adversarial-squad`

**Files with placeholder numbers to clear**
`visualize.py`, `figures/table1_ablation_results.{md,tex}`, `figures/table2_qualitative_cases.{md,tex}`, `report/table1_ablation_results.tex`, `report/table2_qualitative_cases.tex`, `report/main.tex` (abstract, Table 1, Table 2, §5, §6, §7), `README.md` (Executive Summary, Table 1, Table 2).

**Report sections that need rewriting once results exist**
Abstract; §1 and §3 framing (KL as the primary consistency loss, reranker as the second contribution); §4 Experimental Setup (paraphrase pipeline, validation split, hyperparameters, data sizes, external test sets); §5 Results; §6 Analysis; §7 Conclusion; Figure 1 and Figure 2 captions.
