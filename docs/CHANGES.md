# Change notes — 3 November 2026

Everything added, modified or removed since the group took over the repository, grouped by the gap it closes.
The gaps themselves are listed in `docs/status_review_2026-11-03.md` (the pre-change snapshot).

## Summary

The training/evaluation code was moved out of the marimo notebook into a plain package (`g17qa/`) and completed so
that the pipeline now implements the proposal end to end: real MarianMT paraphrases with a similarity filter, the
full objective CE(q) + CE(q′) + λ·KL with an augmentation-only ablation, an article-level validation split, top-20
candidate decoding, a learned span reranker, official-scorer evaluation on the dev set plus the paraphrased and
adversarial test sets, question-type and disagreement analysis, and figures/tables generated only from real runs.
All placeholder numbers, random "results" figures and the PDF built from them were removed. A tiny-model smoke test
runs the whole chain on a laptop CPU in a few minutes.

## New files

| Path | Purpose |
| :-- | :-- |
| `g17qa/__init__.py` | package marker / module map |
| `g17qa/utils.py` | seeding, device selection (`cuda` > `mps` > `cpu`), AMP dtype resolution, JSON/JSONL helpers |
| `g17qa/metrics.py` | line-for-line port of `evaluate-v2.0.py` normalisation/EM/F1 with max over **all** gold answers; consistency metrics; question-type buckets; `run_official_scorer()` that shells out to the real script |
| `g17qa/data.py` | SQuAD loader keeping every gold answer; deterministic **article-level 5 % validation split**; paraphrase attachment; sliding-window features with HF-style labelling (answer must lie fully inside the window); paired (q, q′) features on identical chunks; Gan & Ng positional pairing loader |
| `g17qa/paraphrase.py` | **MarianMT EN→DE→EN back-translation** (batched, length-bucketed, sampled retry when the output equals the input), **sentence-transformer similarity filter** (≥ 0.85, non-identical), resumable JSONL cache; rule-based fallback kept for smoke tests only |
| `g17qa/model.py` | `SpanExtractor` (AutoModel + linear head) for BERT and DeBERTa-v3; checkpoint save/load that records the training config |
| `g17qa/losses.py` | vectorised passage-token distributions; **CE(q) + CE(q′) + λ·symmetric-KL** (proposal objective); 1-D Wasserstein and Hungarian span-matching variants; loss modes `ce / aug / kl / ot / hungarian / dual` |
| `g17qa/decode.py` | streaming multi-window inference; **top-k distinct candidate spans** (null-margin scoring, de-duplicated by normalised text); paired decoding that gives every candidate the other view's score; reranker hidden-state features |
| `g17qa/reranker.py` | **learned span reranker**: MLP over `[h_start; h_end; mean(h_span); h_cls]` + scalar features, list-wise cross-entropy, early stopping, margin-restricted reranking |
| `prepare_paraphrases.py` | CLI to build/resume the paraphrase caches for any SQuAD-format file |
| `train_reranker.py` | trains the reranker on the held-out training articles recorded in the extractor checkpoint (never on dev) |
| `tsne_pairs.py` | t-SNE of question/paraphrase `[CLS]` representations + mean pair distance (proposal item 2b) |
| `download_eval_sets.py` | fetches Gan & Ng (2019) `dev/adv_{para,orig}.json` and Adversarial SQuAD AddSent/AddOneSent into `data/external/` |
| `tests/test_core.py` | 10 checks: metrics identical to the official script, split determinism, window labels recover the answer text, paired windows share chunks, every loss mode finite and differentiable, Gan & Ng pairing, candidate decoding/reranker features |
| `scripts/smoke_test_cpu.sh` | end-to-end CPU run with `google/bert_uncased_L-2_H-128_A-2` |
| `scripts/run_pipeline.sh` | sequential full pipeline (paraphrases → train modes → reranker → all evaluations → figures), idempotent |
| `scripts/benchmark_cpu.py` | measures step times on CPU/MPS and extrapolates to full SQuAD (see "Local CPU feasibility" below) |
| `report/tables/` (generated), `report/figures/` (generated) | written by `visualize.py` |
| `docs/status_review_2026-11-03.md`, `docs/CHANGES.md` | this review and these notes |

## Rewritten files

| Path | What changed |
| :-- | :-- |
| `train.py` | now imports from `g17qa`; validation split; `--loss_mode` with CE(q′) and augmentation-only mode; AdamW with no-decay groups, **linear warm-up schedule**, grad accumulation, AMP (`--precision`); validation EM/F1 per epoch or every N steps; **best checkpoint by validation F1** (`best.pt`) plus `last.pt`; config recorded in the checkpoint; telemetry `history.json` / `val_history.json` / `train_summary.json` |
| `evaluate.py` | evaluation sets `dev / gan_dev / gan_adv / addsent / addonesent / custom`; paraphrases from cache or generated on the fly; **official scorer** on top-1 and reranked predictions (and on the original-question view for the Gan & Ng sets); consistency metrics; per-question-type EM/F1; candidate dump; disagreement dump; `results.json` |
| `visualize.py` | reads only `runs/` artefacts; loss curves, dev ablation bars, robustness lines, LaTeX + Markdown tables; **no hard-coded numbers or random fallbacks** (exits non-zero when there is nothing to plot) |
| `report/main.tex` | re-centred on the proposal (KL consistency as the primary objective, OT/Hungarian as ablation, reranker, robustness sets, validation protocol); all placeholder numbers removed; tables/figures are `\input` from generated files; `\TBD{}` markers where results/discussion must be filled; author names added; required **"Contributions and external code"** section added |
| `report/custom.bib` | added Jia & Liang 2017, He et al. 2023 (DeBERTaV3), Wang et al. 2018 (answer re-ranking) |
| `report/README.md` | build instructions for the new table/figure flow |
| `README.md` | describes the actual pipeline, commands, outputs, cluster usage, CPU feasibility, contributions statement |
| `requirements.txt` / `pyproject.toml` | one consistent spec: Python ≥ 3.10, adds `sentence-transformers`, `tiktoken`, `protobuf`, `sacremoses`, `scikit-learn`; marimo optional; package name fixed |
| `.gitignore` | ignores data, weights and the large run dumps but keeps the small JSON result files |
| `scripts/job.sbatch`, `scripts/submit_all_ablations.sh`, `scripts/setup_env_cluster.sh` (replace the three old Slurm scripts) | rewritten to follow `docs/SoC_Cluster_Guide.pdf` and the course `job_template.sbatch`: Conda env activated from `~/miniconda3`, `gpu-long` partition, `--gpus=h100-47:1`, `--cpus-per-task=8`, `--mem-per-cpu=20000`, logs in `logs/slurm/<job>-<id>.out`, pipeline invoked with forwarded arguments; environment setup is an interactive script for a compute node (`salloc` + `srun --pty bash`), never xlogin; one GPU job per (backbone, objective) with a dependency on the paraphrase-cache job; `STAGES` lets a job run only part of the pipeline |
| `app.py` (notebook) | fixed the Python 3.12-only f-string that broke `import app` on 3.10/3.11; added a status banner marking it a legacy demo whose numbers must not be quoted |
| `docs/math.md` | status note explaining that KL is now the primary objective and OT the ablation |

## Removed files

| Path | Reason |
| :-- | :-- |
| `figures/*.png|pdf|md|tex`, `report/figures/*.pdf`, `report/table1_*.tex`, `report/table2_*.tex`, `report/main.pdf` | generated from hard-coded placeholder numbers; regenerated from real runs by `visualize.py` |
| `scripts/run_all_ablations.sh`, `scripts/run_cluster.sbatch`, `scripts/setup_env.sbatch`, `scripts/submit_all_ablations.sbatch` | replaced by `scripts/run_pipeline.sh`, `scripts/job.sbatch`, `scripts/setup_env_cluster.sh`, `scripts/submit_all_ablations.sh` |
| `uv.lock` | pinned Python 3.15; stale after the `pyproject.toml` change (`uv lock` regenerates it) |

## Library-version pitfalls handled in the code

- transformers ≥ 5 no longer provides `build_inputs_with_special_tokens`; the `[CLS] q [SEP] c [SEP]` layout and segment ids are built explicitly (`g17qa/data.py: pack_pair`), valid for BERT and DeBERTa tokenizers.
- transformers ≥ 5 needs `tiktoken` and `protobuf` to load the DeBERTa-v3 sentencepiece tokenizer; both are in `requirements.txt`.
- transformers ≥ 5 loads `microsoft/deberta-v3-base` in fp16 (the dtype stored in the checkpoint); `SpanExtractor` casts the encoder to fp32 so the span head and AMP behave as expected (`g17qa/model.py`). Verified with a CPU forward/backward + decode check on DeBERTa-v3-base.
- Python ≤ 3.11 rejects backslashes inside f-string expressions; the one in the notebook was fixed, and the new code avoids the construct.

## Behavioural differences worth knowing

- Evaluation numbers are now the official scorer's output; previously EM/F1 compared against the first gold answer only (5,625 of 10,570 dev questions have several), so new numbers will be higher than the old custom metric would have given.
- Training uses two forward passes per step in paired modes; `--loss_mode ce` skips the paraphrase pass entirely.
- The reranker is trained on the validation articles; its early-stopping set is 20 % of those articles, so the dev set stays untouched end to end.
- `prepare_paraphrases.py` is resumable and the cache is shared by all runs; generate it once per file.

## Local CPU feasibility

Measured on an Apple M2 Pro (12 cores, 16 GB) with `bert-base-uncased`, L = 384, stride 128 (`scripts/benchmark_cpu.py`):

| Task | CPU only | Apple-silicon GPU (MPS) |
| :-- | :-- | :-- |
| `ce` training, one epoch over full SQuAD | 8.0 s/step (batch 16) → ≈ 12 h | 0.44 s/step (batch 8) → ≈ 1.4 h |
| `kl` training (2 forwards/step), one epoch | 26 s/step → ≈ 40 h | 1.07 s/step → ≈ 3.3 h |
| Dev inference (×2 with paraphrases) | ≈ 6.5 min | ≈ 2 min |
| MarianMT back-translation, full training set | 11 q/s → ≈ 2.2 h | 23 q/s → ≈ 1.1 h (≈ 88 % of a 128-question dev sample passed the 0.85 filter; the training-set rate is printed by `prepare_paraphrases.py`) |

Verdict: CPU-only is enough for paraphrase caches, evaluation, the reranker and figures, but not for fine-tuning the
extractors; the Mac's GPU can do single BERT-base runs overnight (batch 8, `--grad_accum 2`; batch 16 is out of
memory in paired modes); the experiment grid should run on the SoC cluster.

## Verification performed

- `python tests/test_core.py`: 11/11 pass (including a test that the paraphrase-view labels are shifted correctly, added after the review below).
- `bash scripts/smoke_test_cpu.sh`: paraphrase cache → KL training with validation → reranker → official-scorer evaluation on dev, Gan & Ng and AddOneSent with consistency, all completed, exit 0 (tiny model, numbers meaningless).
- DeBERTa-v3-base: features, a paired KL training step and top-k decoding with reranker features verified on CPU.
- `python -m py_compile`, `pyflakes` (clean) on every script; `bash -n` on every shell script; `app.py` parses on Python 3.11.
- `visualize.py` and `tsne_pairs.py` exercised on the smoke run (artefacts deleted afterwards).
- Not verified: CUDA/bf16 paths and Slurm submission (no GPU or cluster access from this machine); LaTeX compilation of `report/main.tex` (no TeX installation here; compile on Overleaf and check the 4 + 1 page limit).

## Independent review (separate reviewer, 3 November 2026) and what was done about it

The rebuilt repository was handed to an independent reviewer with the proposal, the course requirements and the change notes. Its findings and the resulting fixes:

| Severity | Finding | Fix |
| :-- | :-- | :-- |
| Critical | CE(q′) was computed against span labels positioned for the original question's input; the paraphrase input's passage starts at a different offset whenever the two questions have different token lengths, so every paired mode trained the paraphrase view on a shifted span. | `g17qa/losses.py` now shifts non-null labels by `passage_start_p − passage_start`; `tests/test_core.py::test_paraphrase_labels_are_shifted` checks both the feature layout and the loss. Smoke run after the fix: CE(q′) tracks CE(q). |
| Major | Figures/tables generated from the tiny-model smoke run sat in `report/` and `figures/`, and `visualize.py` would have included the `smoke` run in the real tables. | Generated files deleted; `visualize.py` skips `smoke*` runs by default (`--exclude`, `--runs`). |
| Major | Change notes referred to a review section that did not exist yet. | This section. |
| Major | Consistency on the adversarial sets was not produced by default, although the proposal asks for agreement there. | `evaluate.py` now generates/caches back-translated paraphrases for every set by default (`--paraphrase_method none` to skip); the pipeline caches them in the paraphrase stage. |
| Major | A published DeBERTa-v3-base SQuAD v1.1 number may not exist (the paper reports SQuAD 2.0 for base models). | Report text now cites BERT-base v1.1 and DeBERTa-v3-base SQuAD 2.0 with a caveat, and uses the plain-CE DeBERTa run as the v1.1 reference. |
| Minor | `results.json` overwrote the reranker path with the reranker metrics. | Key renamed to `reranker_path`. |
| Minor | During-training validation used the first 2,000 held-out questions (a few articles). | Deterministic shuffle across the held-out articles before capping. |
| Minor | `.gitignore` dropped `runs/*/history.json`, which the loss figure needs. | Kept. |
| Minor | Slurm output directory had to exist before submission; setup time limit tight; bf16 silently fell back to fp32 on GPUs without bf16. | Job script creates `logs/slurm`, `runs`, `weights` itself; setup is interactive; `amp_dtype_for` warns loudly on fallback. |
| Minor | README omitted copying the t-SNE figures into `report/figures/` and did not say that evaluating `dev` back-translates 10,570 questions on first use. | Both documented. |
| Minor | Paraphrase acceptance rate was quoted from a 128-question dev sample. | Labelled as an estimate. |
| Minor | Proposal wording "open-weight LLM reranker" vs the implemented learned MLP reranker. | Report states the choice and the reason explicitly; see "Direction notes" below. |
| Minor | Unused imports. | Removed; pyflakes clean. |

Reviewed and found correct by the reviewer (no action): KL / Wasserstein / Hungarian implementations and masking; window labelling and paired chunks; article-level split; `write_squad_subset`; DeBERTa offset handling; streaming decode and null-margin scoring; reranker training/leakage; byte-for-byte metric port; optimiser/scheduler/GradScaler order; MarianMT batching and filtering; Gan & Ng pairing; no closed-source models or APIs.

## Addendum (3 Nov 2026, later): validation agreement and the λ selection rule

Problem: λ trades accuracy for consistency, and validation measured accuracy only, so selecting λ by held-out F1 would always favour the smallest value and switch the method off. Changes:

- `train.py`: the held-out questions get their cached paraphrases in **every** mode (also the `ce` baseline), validation decodes both views and records paraphrase agreement next to EM/F1 (`val_history.json`, `train_summary.json`: `best_val_agreement_em`, `best_val_pairs`). Checkpoint selection within a run stays by F1.
- `g17qa/metrics.py`: `summarize_outputs()` (unit-tested) computes EM/F1 and agreement from decoded outputs.
- `visualize.py`: "Val Agr." column in the ablation table; new `lambda_sweep.pdf` (held-out F1 and agreement against λ) and `table_lambda.tex|md`.
- Report protocol: λ is the largest value whose held-out F1 is within a stated tolerance of the best; `tables/table_lambda.tex` is included in the experiments section.
- README: the `ce` command now passes the paraphrase cache (validation only) and the selection rule is documented.
- Verified with the unit tests and the tiny-model smoke runs (`kl` λ = 1 and λ = 2, and `ce` with the cache).

## Addendum: cluster setup, CUDA build

First run on the SoC cluster (node `xcnc28`, Nov 2026): all packages and models installed and 11/11 tests passed, but the default PyPI PyTorch wheel is built for CUDA 13.2 while the node driver is CUDA 12.9, so `torch.cuda.is_available()` was `False`. Fix: `scripts/setup_env_cluster.sh` now installs PyTorch from the CUDA 12 index (`TORCH_CUDA=cu128`, override e.g. `cu126`) before the other requirements, and exits with an error if the GPU is still not visible. Manual fix on an already-created env: `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`.

## Addendum: live logs on the cluster

Python block-buffers `print` when stdout is a file, so progress lines reached `logs/slurm/*.out` in bursts and jobs looked stuck. `scripts/job.sbatch` now exports `PYTHONUNBUFFERED=1`. A liveness check that works regardless: `wc -l data/paraphrases/train.jsonl` (grows every 512 questions) during the paraphrase stage, and `runs/<run>/history.json` (rewritten every 20 steps) during training. First real submission on the cluster: 3 Nov 2026, jobs 1068559–1068565 on H100 NVL MIG 3g.47gb slices.

## Results snapshot (cluster runs of 3–4 Nov 2026, official scorer, full dev set)

All nine runs completed on H100 NVL MIG slices (15–33 min of training each; 10,564 steps = 2 epochs over the 95 % training split). Numbers below are copied from `runs/<run>/eval_*/results.json`; regenerate tables with `python visualize.py --best bert_kl_lam1.0`.

| run | dev EM | dev F1 | dev agreement | +reranker EM / F1 / agreement | AddSent F1 | AddOneSent F1 | G&N para F1 | G&N agreement |
| :-- | --: | --: | --: | :-- | --: | --: | --: | --: |
| bert_ce (baseline) | 81.1 | 88.5 | 83.3 | 81.4 / 88.3 / 86.3 | 64.9 | 73.9 | 85.6 | 89.1 |
| bert_aug | 81.0 | 88.2 | 86.5 | 80.6 / 87.5 / 93.4 | 65.9 | 73.8 | 85.7 | 91.4 |
| bert_kl λ=0.5 | 81.1 | 88.5 | 87.6 | 81.3 / 88.3 / 89.1 | 65.8 | 74.1 | 86.0 | 90.8 |
| **bert_kl λ=1.0** (selected) | 81.4 | 88.6 | 87.9 | 80.0 / 87.7 / 92.9 | 66.1 | 74.5 | 86.1 | 91.0 |
| bert_kl λ=2.0 | 81.3 | 88.4 | 88.8 | 80.4 / 87.8 / 93.7 | 65.7 | 73.8 | 86.9 | 91.9 |
| bert_ot | 80.9 | 88.2 | 86.3 | 78.4 / 86.8 / 91.3 | 65.8 | 73.9 | 85.6 | 90.3 |
| bert_dual | 81.1 | 88.3 | 86.7 | 81.1 / 88.1 / 90.0 | 66.0 | 74.2 | 85.7 | 90.0 |
| deb_ce | 88.3 | 93.9 | 88.0 | 87.4 / 93.3 / 89.7 | 85.1 | 88.7 | 91.3 | 91.8 |
| deb_kl λ=1.0 | 88.3 | 93.7 | 91.8 | 88.0 / 93.7 / 92.6 | 84.2 | 87.9 | 91.0 | 93.7 |
| deb_large_ce (4 Nov, batch 8×2, lr 1e-5) | 90.3 | 95.5 | 89.4 | 90.2 / 95.4 / 93.4 | 86.0 | 89.3 | 93.8 | 92.3 |
| deb_large_kl λ=1.0 (4 Nov, batch 8×2, lr 1e-5) | 90.0 | 95.4 | 92.7 | 90.0 / 95.3 / 95.1 | 87.0 | 90.2 | 93.8 | 95.0 |

The two DeBERTa-v3-large rows were added on 4 Nov 2026 (jobs 1072440–1072441; 43 / 108 min of training, peak GPU memory 16.4 / 25.3 GB); their rerankers use the graded targets merged from PR #2, unlike the nine earlier rows. Full analysis in `docs/RESULTS.md` §6–§8.

λ selection (held-out articles, rule: largest λ with F1 within 0.5 of the best): val F1 82.5 / 82.1 / 81.7 and agreement 89.5 / 90.6 / 91.0 for λ = 0.5 / 1 / 2 → **λ = 1.0**.

Reading: the baseline reproduces the published BERT-base result (80.8 / 88.5). Consistency training raises paraphrase agreement by 4.6 points over the baseline and 1.4 over augmentation-only at no accuracy cost; DeBERTa shows the same pattern (+3.8 agreement). Gains on the adversarial sets are small (+1 F1 on AddSent, +0.6 on AddOneSent) but agreement there rises by 6 points. The reranker raises agreement strongly (dev 87.9 → 92.9 for the selected run) but costs about 1 F1 on dev; it is clearly beneficial on the 56-question adversarial paraphrases of Gan & Ng (F1 63.6 → 75.6) and neutral on AddSent. OT/Hungarian variants do not beat KL.

## Results report

`docs/RESULTS.md` was regenerated on 4 Nov 2026 with the two DeBERTa-v3-large runs and then compacted at the group's request: prose cut by about a fifth, one or two sentences of reading per table, a backbone comparison table added to §6, one-line caveats; all numbers remain computed from `runs/` by the generator. Independently fact-checked after each regeneration (0 numeric mismatches; prose corrections applied). It is the full results analysis (12 sections: setup, dev results, augmentation vs. consistency, λ selection, backbone, robustness, reranker, OT ablation, question types, caveats, provenance). It is generated by `scripts/make_results_report.py` from `runs/` and every number traces to a run file (`docs/results_digest.json`); an independent fact-check on 4 Nov 2026 recomputed every table cell and prose number from the raw files and found no mismatches. Regenerate after new runs with `python scripts/make_results_report.py --verified <date-of-last-check>` (omit `--verified` until re-checked).

## Addendum (4 Nov 2026): merged PR #2 — reranker graded targets and Recall@k ceilings

A teammate's branch `reranker-graded-targets` (one commit, PR #2 against `ck`) was merged into `ck`. What it changes:

- `g17qa/reranker.py`: the listwise target is now a *distribution* over a question's candidates, selected by `--target_mode` (`graded`, the new default: exact-match candidates get relevance 1, non-EM candidates with F1 > 0 get `--partial_weight` (0.3) × F1, normalised to sum to 1; `exact`: uniform over EM candidates only; `f1`: the previous one-hot rule, first EM candidate else the highest-F1 candidate). The loss is soft-target cross-entropy. The best epoch is now chosen by held-out **EM** (F1 breaks ties) instead of F1. New helpers `candidate_match`, `target_distribution`, `candidate_ceiling`; `evaluate_items` additionally returns `recall_em` (Recall@k EM) and `best_f1` (best-F1@k), and keeps `oracle_exact` as an alias of `recall_em`, so `scripts/make_results_report.py` still reads it.
- `train_reranker.py`: new flags `--target_mode`, `--partial_weight`; `reranker_summary.json` gains `target_mode`, `partial_weight`, `ceiling_train`, `ceiling_dev`.
- `evaluate.py`: `results.json` gains `extractor.candidate_ceiling` (Recall@k EM and best-F1@k of the extractor's top-k on the evaluated set).
- `README.md`: the reranker step documents the new flags.

Merge conflict: only `evaluate.py`, where the PR's branch point predates commit 3d62dc6 (the `full_paired_view` fallback for identical Gan & Ng pairs). Resolved by keeping the `full_paired_view(...)` call and appending the PR's ceiling block after it; nothing from either side was dropped.

Verification after the merge: `python -m py_compile` on the three Python files, `python tests/test_core.py` (11/11 pass) and the end-to-end `bash scripts/smoke_test_cpu.sh` (all six stages, exit 0; `runs/smoke/reranker_summary.json` and `eval_*/results.json` contain the new keys). The PR author had not been able to run any of these.

Consequences the group should know:

1. The reranker numbers in `runs/*/reranker_summary.json`, the results snapshot above, `docs/RESULTS.md` §8 and the method paragraph of `report/main.tex` were produced by the *previous* rule (`--target_mode f1` targets, best epoch by F1). With the new default, `train_reranker.py` will produce a different reranker. To refresh the artefacts, re-run the reranker stage for each run on the cluster (cheap: it reuses `weights/<run>/best.pt`; e.g. `STAGES=reranker,eval` with `scripts/run_pipeline.sh`), then `python visualize.py` and `python scripts/make_results_report.py`, and update the sentence in `report/main.tex` that says the target is "the candidate that exactly matches a gold answer (or the highest-F1 candidate)".
2. There is no flag that restores the old *epoch selection by F1*, so `--target_mode f1` reproduces the old targets but not necessarily the old checkpoint choice.
3. In `evaluate.py` the ceiling is divided by the number of questions in the set while `n` counts only questions with at least one candidate; the two coincide in practice (every question yields candidates), so Recall@k is a recall over all questions.

## Addendum (4 Nov 2026): DeBERTa-v3-large runs

Set-up for fine-tuning `microsoft/deberta-v3-large` (24 layers, hidden 1024, ~435 M parameters; the base model has 12 layers, hidden 768, ~184 M) with the same pipeline, as runs `deb_large_ce` and `deb_large_kl` (λ = 1).

- `g17qa/model.py`: `SpanExtractor(..., gradient_checkpointing=True)` enables activation checkpointing in the encoder (`gradient_checkpointing_enable`, non-reentrant). Off by default; checkpoints saved with it load unchanged.
- `train.py`: new flag `--grad_checkpointing`; `train_summary.json` gains `peak_gpu_mem_gb` (`torch.cuda.max_memory_allocated`, `null` off-GPU) and the final log line prints it, so the memory headroom of a run is on record.
- `scripts/run_pipeline.sh`: new variables `GRAD_ACCUM` (default 1) and `LR` (default 3e-5) passed to `train.py`; defaults reproduce the existing runs exactly.
- `scripts/submit_deberta_large.sh` (new): submits one Slurm job per mode through `scripts/job.sbatch` with `BATCH=8 GRAD_ACCUM=2` (effective batch 16, identical to the base runs so the comparison is clean), `LR=1e-5`, `PRECISION=bf16`, `EPOCHS=2`, `--time=10:00:00`, `STAGES="train reranker eval"`. It reuses the paraphrase caches and refuses to submit if they are missing. `GRAD_CKPT=1` adds `--grad_checkpointing`; `GPU=<type>` picks another GPU type; `MODES="ce"` runs a subset.
- `scripts/make_results_report.py`: labels for `deb_large_ce` / `deb_large_kl` (the report already lists unknown runs under their run names; `visualize.py` groups by the model recorded in `config.json`, so it needs no change).

Why these settings. The learning rate 3e-5 of the base runs is above the range normally used for large DeBERTa encoders (1e-5 or lower) and frequently diverges; 1e-5 with the same 10 % warm-up is the conventional choice. The per-step batch is halved because the paired modes put both q and q′ windows through the encoder (16 pairs = 32 sequences of 384 tokens), which at 24 layers × hidden 1024 is estimated at ~45 GB of activations plus ~7 GB of weights, gradients and Adam state — too close to the 47 GB MIG slice; at batch 8 the estimate is ~25 GB for kl and ~15 GB for ce. These are estimates, not measurements: if a job stops with CUDA out of memory, resubmit it with `GRAD_CKPT=1` (about 30 % slower), or with `GPU=<type>` if `gpufree` lists a GPU with more memory. Expected time on one h100-47 slice, scaled ×3.5 from the base runs (17 min ce, 33 min kl): ~1 h ce, ~2 h kl, plus ~15 min for the reranker and the five evaluation sets; the 10 h limit leaves room for a slower slice. Each run writes two fp32 checkpoints of ~1.7 GB (`best.pt`, `last.pt`).

Verified locally: `python tests/test_core.py` (11/11); a tiny-BERT CPU run of `train.py` with `--grad_accum 2 --grad_checkpointing` (32 batches → 16 optimiser steps, config and summary record both settings); `bash -n` on both shell scripts. Not verified: GPU memory and time on the cluster (no GPU locally). Local CPU/MPS training of the large model is not feasible (the base BERT already needs ~12 h per epoch on CPU; the 16 GB MPS device cannot hold the large model's optimiser state).

## Direction notes (decisions the group should confirm)

- **Reranker form (decided 3 Nov 2026).** The system is fixed as two versions: *extractor only* (top-1 span) and *extractor + learned span reranker* (the small MLP over the extractor's own representations, trained only on the held-out training articles). No LLM-based reranker is pursued. Every `evaluate.py` run and every generated table reports both versions, so the reranker's contribution is always visible rather than assumed. The report's method section states this explicitly.
- **Paraphrase filter quality.** Back-translation plus a 0.85 similarity threshold admits some meaning-changing pairs (e.g. "won" → "got"); this is a known limitation of the proposed pipeline and is stated in the report's Limitations.

## What the group still has to do

1. Run the real experiments on the cluster (`bash scripts/submit_all_ablations.sbatch`), including a small λ sweep.
2. `python visualize.py --best <run>` and the two `tsne_pairs.py` calls; fill every `\TBD{}` in `report/main.tex`; compile and check the 4 + 1 page limit.
3. Pick the qualitative examples from `runs/<run>/eval_dev/disagreements.jsonl`.
4. Slides and the 15-minute video (not covered here).

## Addendum (4 Nov 2026): reranker ablations — `pair_score` switch, residual fusion, confidence gate, FIX/BREAK analysis

Branch `reranker-ablations` (from `ck` at 063394d). Only the reranker and its evaluation were touched; the extractor, paraphrase pipeline, losses, splits and existing checkpoints/artefacts are unchanged. All defaults reproduce the previous behaviour.

| File | Change |
| :-- | :-- |
| `g17qa/reranker.py` | `PAIR_FEATURES`, `scalar_feature_indices`, `SpanReranker(scalar_idx=…)` (MLP input = selected columns; non-persistent index buffer so old checkpoints load); `train_reranker(use_pair_score=…)`; checkpoints record `scalar_idx`, `scalar_features_used`, `use_pair_score`; `select_index` (modes `extractor` / `reranker` / `residual`, `alpha`, `override_margin`, the old `margin`); `rerank_logits`; `choose` / `evaluate_items` / `apply_reranker` take the new kwargs (defaults = old behaviour; `apply_reranker(return_details=True)` also returns per-candidate scores); `classify_change`, `change_counts`, `analyze_changes` (FIX / BREAK / SAME_CORRECT / SAME_WRONG), `recall_at_ks` |
| `train_reranker.py` | `--use_pair_score` / `--no_pair_score`, `--tag` (output names `reranker<tag>.pt`, `reranker_summary<tag>.json`); summary records `use_pair_score`, `scalar_features_used`, `tag` and a `heldout_selection_grid` (EM / F1 / FIX / BREAK for extractor, reranker, residual alpha 0.25/0.5/1.0 and gate margins, on the held-out articles, so alpha can be chosen without the dev set) |
| `evaluate.py` | `--select_mode`, `--alpha`, `--override_margin`, `--tag` (outputs to `runs/<run>/rerank_variants/<tag>/eval_<set>/`), `--n_change_examples`; `results.json` gains `reranker_variant`, `extractor.recall_at_k`, `reranker.delta` (dEM, dF1, dAgreement), `reranker.changes` (FIX/BREAK counts and %); new `rerank_changes.jsonl`; removed a duplicated `load_reranker` line left by the PR #2 merge |
| `scripts/run_reranker_variants.sh`, `job_reranker_variants.sbatch`, `submit_reranker_variants.sh`, `summarize_reranker_variants.py` | variants A–E on existing extractors, one Slurm job per extractor, and a generated Markdown table per run/eval set |
| `tests/test_reranker.py` | 10 hermetic tests (synthetic candidates, no model download) |

Residual fusion: `final_i = log p_ext(i) + alpha * log p_rr(i)`, where both are per-question log-softmaxes over the candidate list (the reranker is trained with a softmax over exactly these candidates; extractor score differences are log-odds under its start/end softmaxes). `alpha = 0` is exactly the extractor ranking.
