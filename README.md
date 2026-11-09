# Paraphrase-Consistent Extractive Question Answering on SQuAD v1.1

**CS4248 Introduction to Large Language Models (AY2026/27 S1), National University of Singapore — Group G17**  
Chong Xin Wei, Ng Jian Yi, Soh Jun Ying Gavin, Wang Chengke, Yee Yan Hwee

Topic: Extractive question answering on SQuAD v1.1. Full results analysis: [docs/RESULTS.md](docs/RESULTS.md). Mathematical specification: [docs/math.md](docs/math.md). Final report: [report/main.pdf](report/main.pdf).

## Proposed system

**One DeBERTa-v3-large extractor fine-tuned with paraphrase augmentation and a symmetric-KL consistency loss, used on its own (no second stage).** Each training question is back-translated (EN→DE→EN, MarianMT) and kept when its sentence-embedding cosine similarity to the original is ≥ 0.85 (68,439 of the 83,484 training questions remaining after the 5 % article hold-out). Question and paraphrase are trained on the same passage window with:

```text
loss = CE(q) + CE(q′) + λ · KL_sym(P_q, P_q′)    (λ = 1)
```

over the start/end distributions on passage tokens. At test time the system reads only the question and returns the top-scoring span. On SQuAD v1.1 dev it reaches **90.0 EM / 95.4 F1** with **92.7 %** agreement between a question and its paraphrase (and **95.0 %** agreement on Gan & Ng's human paraphrases, up from 92.3 % at unchanged 93.8 F1). Every reported number is generated from the committed `runs/` artefacts.

### Design choices and evidence

- **Backbone (`DeBERTa-v3-large`).** Plain cross-entropy fine-tuning gives 81.1 / 88.5 (`bert-base-uncased`), 88.3 / 93.9 (`deberta-v3-base`), and 90.3 / 95.5 (`deberta-v3-large`) EM / F1; no objective change moves accuracy by more than 0.3 points.
- **Objective (paraphrase augmentation + symmetric KL).** The full recipe raises paraphrase agreement on every backbone (+4.6 on BERT-base, +3.8 on DeBERTa-v3-base, +3.3 on DeBERTa-v3-large) while keeping EM and F1 within 0.3 points of plain CE in single-seed runs. On BERT-base, where augmentation-only (`aug`) was also run, paraphrase augmentation accounts for +3.2 points and the KL penalty adds a further +1.4; on the two DeBERTa backbones the gain reflects the combined recipe. 1-D Wasserstein (`ot`) and Hungarian span-matching (`dual`) variants do not improve over augmentation alone and are reported as negative ablations.
- **λ = 1, selected on held-out training articles.** 5 % of training articles (23 articles, 4,115 questions) are held out for checkpoint selection, λ tuning (`λ ∈ {0.5, 1, 2}` on BERT-base using the rule "largest λ within 0.5 F1 of the best"), and reranker training. SQuAD dev and external test sets are used only for final evaluation.
- **Extractor alone (why the second stage was dropped).** Although the top-20 candidates contain an exact match for 99.5 % of dev questions, a learned MLP span reranker changes dev EM by only +0.2 to +0.3 points (95 % bootstrap CIs include zero), and its consistency gain relies on test-time paraphrase features that a training-free paraphrase ensemble exceeds (+6.6 vs +1.4 agreement).

### Main results

Official `evaluate-v2.0.py` on SQuAD v1.1 dev (10,570 questions; agreement over the 8,622 pairs with an accepted back-translated paraphrase), Gan & Ng (2019) human paraphrases (1,062 dev and 56 adversarial questions), and Adversarial SQuAD AddSent / AddOneSent (Jia & Liang, 2017):

| System (single seed) | Dev EM | Dev F1 | Dev agreement | G&N orig F1 | G&N para F1 | G&N agreement | AddSent F1 | AddOneSent F1 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| BERT-base, CE (baseline) | 81.1 | 88.5 | 83.3 | 88.4 | 85.6 | 89.1 | 64.9 | 73.9 |
| BERT-base, CE + aug. | 81.0 | 88.2 | 86.5 | 87.4 | 85.7 | 91.4 | 65.9 | 73.8 |
| BERT-base, CE + aug. + KL (λ = 1) | 81.4 | 88.6 | 87.9 | 87.5 | 86.1 | 91.0 | 66.1 | 74.5 |
| DeBERTa-v3-base, CE | 88.3 | 93.9 | 88.0 | 93.5 | 91.3 | 91.8 | 85.1 | 88.7 |
| DeBERTa-v3-base, CE + aug. + KL (λ = 1) | 88.3 | 93.7 | 91.8 | 92.4 | 91.0 | 93.7 | 84.2 | 87.9 |
| DeBERTa-v3-large, CE | 90.3 | 95.5 | 89.4 | 94.9 | 93.8 | 92.3 | 86.0 | 89.3 |
| **DeBERTa-v3-large, CE + aug. + KL (λ = 1) — proposed** | **90.0** | **95.4** | **92.7** | **94.9** | **93.8** | **95.0** | **87.0** | **90.2** |

Second-stage variants on top of the proposed `deb_large_kl` extractor (dev; full breakdown in [docs/RESULTS.md](docs/RESULTS.md) §9–9b):

| Variant on `deb_large_kl` (dev) | EM | F1 | Agreement | Both views correct | Needs paraphrase at test time |
| :-- | --: | --: | --: | --: | :-- |
| Extractor alone (**proposed**) | 90.0 | 95.4 | 92.7 | 86.6 | no |
| + Learned span reranker (pair features) | 90.2 | 95.4 | 94.1 | 87.4 | yes |
| + Learned span reranker (no pair features) | 90.3 | 95.5 | 93.1 | 87.0 | no |
| + Paraphrase ensemble (no learning) | 89.6 | 95.1 | 99.3 | 89.3 | yes |

---

## Setup & Quick Verification

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install -r requirements.txt
# Place SQuAD v1.1 files from Canvas at provided/train-v1.1.json and provided/dev-v1.1.json
python scripts/download_eval_sets.py                     # Gan & Ng sets + Adversarial SQuAD -> data/external/

# Run unit tests (< 1 min) and CPU smoke test (~5 min with BERT-tiny)
python tests/test_core.py && python tests/test_reranker.py && python tests/test_consistency.py && python tests/test_consistency_reranker.py
bash scripts/smoke_test_cpu.sh
```

---

## Running & Reproducing Experiments

### 1. Single-GPU reproduction (proposed system & ablations)

```bash
# 1. Build resumable paraphrase caches (MarianMT EN->DE->EN + MiniLM similarity >= 0.85)
python scripts/prepare_paraphrases.py --input provided/train-v1.1.json --output data/paraphrases/train.jsonl
python scripts/prepare_paraphrases.py --input provided/dev-v1.1.json   --output data/paraphrases/dev.jsonl

# 2. Train the proposed DeBERTa-v3-large extractor (effective batch 8x2 = 16, lr 1e-5, 2 epochs, bf16)
#    Add --grad_checkpointing or use --batch_size 4 --grad_accum 4 on 24 GB GPUs.
python scripts/train.py --run_name deb_large_kl --model microsoft/deberta-v3-large --loss_mode kl --lambda_kl 1.0 \
  --paraphrase_cache data/paraphrases/train.jsonl --batch_size 8 --grad_accum 2 --lr 1e-5 --precision bf16

# 3. Official-scorer evaluation across all five test sets
for S in dev gan_dev gan_adv addsent addonesent; do
  python scripts/evaluate.py --weights weights/deb_large_kl/best.pt --eval_set $S --precision bf16
done

# Optional: train & evaluate the second-stage span reranker on held-out training articles
python scripts/train_reranker.py --weights weights/deb_large_kl/best.pt --paraphrase_cache data/paraphrases/train.jsonl --precision bf16
python scripts/evaluate.py --weights weights/deb_large_kl/best.pt --eval_set dev --reranker weights/deb_large_kl/reranker.pt --precision bf16

# Or run the full sequential pipeline in one resumable command:
MODEL=microsoft/deberta-v3-large RUN_PREFIX=deb_large MODES="ce kl" BATCH=8 GRAD_ACCUM=2 LR=1e-5 PRECISION=bf16 bash scripts/run_pipeline.sh
MODEL=bert-base-uncased RUN_PREFIX=bert MODES="ce aug kl ot dual" KL_LAMBDAS="0.5 1.0 2.0" PRECISION=bf16 bash scripts/run_pipeline.sh
MODEL=microsoft/deberta-v3-base RUN_PREFIX=deb MODES="ce kl" PRECISION=bf16 bash scripts/run_pipeline.sh
```

Loss modes (`--loss_mode`): `ce` (plain extractor), `aug` (`CE(q) + CE(q′)`), `kl` (`CE(q) + CE(q′) + λ·KL_sym`), and `ot` / `hungarian` / `dual` (1-D Wasserstein and/or Hungarian span-matching ablations).

### 2. NUS SoC compute cluster (Slurm)

```bash
# 1. One-off environment setup on a compute node (never on xlogin)
salloc -t 120 && srun --pty bash
bash scripts/setup_env_cluster.sh
exit; exit

# 2. Submit main experiment jobs from the repo root on xlogin
bash scripts/submit_all_ablations.sh          # paraphrase job -> BERT (ce, aug, kl sweep, ot, dual) + DeBERTa-base (ce, kl)
bash scripts/submit_deberta_large.sh          # DeBERTa-v3-large (ce, kl)

# 3. Optional reranker & ensemble ablation jobs on existing checkpoints
sbatch --job-name=g17_rr --export=ALL,RUNS="bert_kl_lam1.0 deb_kl deb_large_kl" scripts/job_reranker_variants.sbatch
sbatch --job-name=g17_cb --time=03:00:00 --export=ALL,RUNS=deb_large_kl,PIPELINE=scripts/run_consistency_baseline.sh scripts/job_reranker_variants.sbatch
sbatch --job-name=g17_cr --time=04:00:00 --export=ALL,RUNS=deb_large_kl,PIPELINE=scripts/run_consistency_reranker.sh scripts/job_reranker_variants.sbatch
```

### 3. Regenerating tables, figures, and the report

All tables and figures are built directly from the committed JSON artefacts under `runs/`:

```bash
python scripts/make_results_report.py             # runs/ -> docs/RESULTS.md + docs/results_digest.json
python scripts/make_report_tables.py              # docs/results_digest.json -> report/tables/rep_main.tex, rep_second.tex
python scripts/visualize.py --best bert_kl_lam1.0 # runs/ -> docs/figures/*.pdf|png|md
cd report && tectonic main.tex                    # compiles report/main.pdf (or pdflatex + bibtex)
```

---

## Repository layout

```text
g17qa/          core package (data, paraphrase, model, losses, decode, reranker, consistency, metrics, utils)
scripts/        CLI entrypoints (prepare_paraphrases.py, train.py, train_reranker.py, evaluate.py, visualize.py,
                tsne_pairs.py, download_eval_sets.py), pipeline runners (run_pipeline.sh, smoke_test_cpu.sh,
                run_reranker_variants.sh, run_consistency_baseline.sh, run_consistency_reranker.sh), Slurm jobs
                (job.sbatch, job_reranker_variants.sbatch, submit_all_ablations.sh, submit_deberta_large.sh,
                setup_env_cluster.sh), and report generators (make_results_report.py, make_report_tables.py,
                bootstrap_main.py, summarize_reranker_variants.py)
tests/          unit & integration tests (test_core.py, test_reranker.py, test_consistency.py, test_consistency_reranker.py)
runs/           committed JSON telemetry and official-scorer outputs for all 11 cluster runs & ablations
report/         ACL short paper (main.tex, compiled main.pdf, custom.bib, generated tables/rep_*.tex)
docs/           RESULTS.md (full experimental report), math.md (mathematical specification), results_digest.json,
                bootstrap_main.json, and figures/ (generated plots .pdf/.png and Markdown summary tables .md)
provided/       official SQuAD scorer evaluate-v2.0.py (train-v1.1.json and dev-v1.1.json placed here locally)
```

## Contributions and external code

**Written by the group:** paraphrase generation and similarity-filtering pipeline, sliding-window paired feature builder, symmetric-KL consistency objective and its OT/Hungarian variants, top-k candidate decoding, learned span reranker and paraphrase ensemble, article-level hold-out protocol, and all evaluation, bootstrap, analysis, and plotting code (`g17qa/`, `*.py`, `scripts/`).  
**External:** PyTorch, Hugging Face Transformers (pre-trained BERT, DeBERTa-v3, and MarianMT weights), sentence-transformers (MiniLM), scikit-learn, SciPy, matplotlib, the official SQuAD scorer (`evaluate-v2.0.py`), and the public test sets of Gan & Ng (2019) and Jia & Liang (2017). No closed-source models or APIs are used.
