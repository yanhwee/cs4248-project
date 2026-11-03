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

## Direction notes (decisions the group should confirm)

- **Reranker form (decided 3 Nov 2026).** The system is fixed as two versions: *extractor only* (top-1 span) and *extractor + learned span reranker* (the small MLP over the extractor's own representations, trained only on the held-out training articles). No LLM-based reranker is pursued. Every `evaluate.py` run and every generated table reports both versions, so the reranker's contribution is always visible rather than assumed. The report's method section states this explicitly.
- **Paraphrase filter quality.** Back-translation plus a 0.85 similarity threshold admits some meaning-changing pairs (e.g. "won" → "got"); this is a known limitation of the proposed pipeline and is stated in the report's Limitations.

## What the group still has to do

1. Run the real experiments on the cluster (`bash scripts/submit_all_ablations.sbatch`), including a small λ sweep.
2. `python visualize.py --best <run>` and the two `tsne_pairs.py` calls; fill every `\TBD{}` in `report/main.tex`; compile and check the 4 + 1 page limit.
3. Pick the qualitative examples from `runs/<run>/eval_dev/disagreements.jsonl`.
4. Slides and the 15-minute video (not covered here).
