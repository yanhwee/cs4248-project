# Paraphrase-Consistent Extractive QA with a Learned Span Reranker

**CS4248 Introduction to Large Language Models (AY2026/27 S1), National University of Singapore — Group G17**
Chong Xin Wei, Ng Jian Yi, Soh Jun Ying Gavin, Wang Chengke, Yee Yan Hwee

Topic: Extractive question answering on SQuAD v1.1. Results: `docs/RESULTS.md`.

## Proposed system

**One DeBERTa-v3-large extractor fine-tuned for paraphrase consistency, used on its own.** Each training question is
back-translated (EN→DE→EN, MarianMT) and kept when its sentence-embedding similarity to the original is ≥ 0.85
(68,439 of the 83,484 training questions that remain after the 5 % article hold-out). Question and paraphrase are trained on the same passage with
`loss = CE(q) + CE(q′) + λ · KL_sym(P_q, P_q′)` over the start/end distributions on passage tokens, λ = 1. At test time the
system reads only the question and returns the top-scoring span. On SQuAD v1.1 dev it reaches **90.0 EM / 95.4 F1**
with **92.7 %** agreement between a question and its paraphrase. Full analysis: `docs/RESULTS.md`. Every number from our
runs, there and here, is computed from the committed `runs/` artefacts and was independently re-checked against them.

### Design choices and the evidence behind them

- **Backbone: DeBERTa-v3-large.** The largest single lever. Plain cross-entropy fine-tuning gives 81.1 / 88.5 (BERT-base),
  88.3 / 93.9 (DeBERTa-v3-base) and 90.3 / 95.5 (DeBERTa-v3-large) EM / F1; no objective change moves
  accuracy by more than half a point. Training the large model took 43 min for plain CE on a 47 GB H100 MIG slice and
  108 min for the consistency run on a full H100 shared with another job (an upper bound); batch 8 × 2 accumulation
  steps, learning rate 1e-5, 2 epochs, bf16, peak GPU memory 16.4 GB (CE) and 25.3 GB (consistency).
- **Objective: consistency loss, not just augmentation.** Adding the KL term raises paraphrase agreement on every
  backbone (+4.6 BERT-base, +3.8 DeBERTa-v3-base, +3.3 DeBERTa-v3-large points) at an accuracy change within noise
  (+0.1 / -0.2 / -0.1 F1). Paraphrases as extra data alone (`aug`) give +3.2 of BERT-base's +4.6; the penalty adds the rest.
  On the large model it also improves robustness: AddSent F1 86.0 → 87.0, AddOneSent 89.3 → 90.2,
  adversarial paraphrases 85.6 → 90.3 (56 questions), with AddSent agreement 84.1 → 90.4 %. DeBERTa-v3-base did not
  share the distractor-F1 gain (`docs/RESULTS.md` §7), so this is a single-seed observation on the large model.
- **λ = 1, chosen on held-out training articles.** 5 % of the training articles (4,115 questions) are never trained on.
  The sweep λ ∈ {0.5, 1, 2} was run on BERT-base; the rule "largest λ whose held-out F1 is within 0.5 of the best" picks 1
  on both the selection sample and the full held-out set (the latter by a thin margin), and λ = 1 was then used unchanged
  for both DeBERTa backbones. Dev and the external sets were used only for the final numbers.
- **Extractor alone, no reranker in the proposed system.** The learned span reranker (an MLP over the extractor's own
  representations, trained on the held-out articles) was kept as the second system version and is reported throughout.
  On the proposed extractor the reranker of the main results trades -0.1 F1 for +2.4 agreement on dev and adds
  +0.6 / +0.8 F1 on AddSent / AddOneSent; a second training of the same reranker, used in the table below, gives
  +0.0 F1 for +1.4 agreement, which is the size of reranker-training noise. Its agreement gain comes from features that
  need a paraphrase at test time, and a learning-free average of the two views beats it on every consistency measure
  (table below). It adds a training stage and a test-time dependency for a benefit a simpler device exceeds, so the
  proposed system is the extractor by itself: the simplest model whose accuracy is within noise of the best and whose
  consistency is the highest among the trained extractors.

### Results against baselines

Official `evaluate-v2.0.py`, full SQuAD v1.1 dev set (10,570 questions); *agreement* = share of the 8,622 dev questions with an
accepted back-translated paraphrase that receive the same answer for question and paraphrase. Robustness columns: F1 on
Adversarial SQuAD AddSent / AddOneSent (Jia & Liang 2017) and on the human paraphrases of Gan & Ng (2019; 1,062 questions,
and the 56-question adversarial subset, so treat that column as indicative only).

| System (ours, single seed) | Dev EM | Dev F1 | Dev agreement | AddSent F1 | AddOneSent F1 | G&N para F1 | G&N adv para F1 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| BERT-base, CE (baseline) | 81.1 | 88.5 | 83.3 | 64.9 | 73.9 | 85.6 | 58.6 |
| BERT-base, CE + paraphrase augmentation | 81.0 | 88.2 | 86.5 | 65.9 | 73.8 | 85.7 | 54.1 |
| BERT-base, CE + KL consistency (λ = 1) | 81.4 | 88.6 | 87.9 | 66.1 | 74.5 | 86.1 | 63.6 |
| DeBERTa-v3-base, CE | 88.3 | 93.9 | 88.0 | 85.1 | 88.7 | 91.3 | 81.9 |
| DeBERTa-v3-base, CE + KL (λ = 1) | 88.3 | 93.7 | 91.8 | 84.2 | 87.9 | 91.0 | 82.9 |
| DeBERTa-v3-large, CE | 90.3 | 95.5 | 89.4 | 86.0 | 89.3 | 93.8 | 85.6 |
| **DeBERTa-v3-large, CE + KL (λ = 1) — proposed** | 90.0 | 95.4 | 92.7 | 87.0 | 90.2 | 93.8 | 90.3 |

Published SQuAD v1.1 dev results for reference (EM / F1): BERT-base 80.8 / 88.5 (Devlin et al. 2019, Table 2); DeBERTa-large
90.1 / 95.5 and DeBERTa-base 87.2 / 93.1 (He et al. 2021, DeBERTa v1, Tables 2–3; the DeBERTaV3 paper reports SQuAD v2.0 only).
Our plain-CE large run equals the published DeBERTa-large figure and the proposed consistency-trained run is within 0.1 of it,
from a single seed.

Consistency devices on top of the proposed extractor (dev; both from the same evaluation pass, `docs/RESULTS.md` §9b):

| Variant on the proposed extractor (dev) | EM | F1 | Agreement | Both views correct | Needs a paraphrase at test time |
| :-- | --: | --: | --: | --: | :-- |
| extractor alone (**proposed**) | 90.0 | 95.4 | 92.7 | 86.6 | no |
| + learned span reranker (pair features) | 90.2 | 95.4 | 94.1 | 87.4 | yes |
| + paraphrase ensemble (no learning) | 89.6 | 95.1 | 99.3 | 89.3 | yes |

*Both views correct* = share of paraphrase pairs where both answers are exact matches; it is the measure to read, because
99 % agreement is nearly automatic once both views choose from the same candidates. By that measure averaging the two
views' scores is the strongest consistency device (89.3 vs 87.4 %) and needs no training, at a -0.4 EM cost on the original
questions; the learned reranker sits in between. Neither is part of the proposed system, which answers from the question alone.

### Reproducing the proposed system

Everything below runs on **one machine with a single CUDA GPU**; no cluster account or Slurm is needed (the Slurm wrappers
in "SoC compute cluster" only batch the same scripts). Measured on one H100 in bf16: paraphrase caches 12 min (training
set) + 2 min (dev), extractor training 108 min with a 25.3 GB peak, optional reranker 2 min, all five evaluations 5 min.

```bash
# 0. environment and data: "Setup" below (pip install -r requirements.txt; provided/*.json from Canvas; download_eval_sets.py)

# 1. paraphrase caches (one-off, resumable): MarianMT EN->DE->EN + sentence-similarity filter >= 0.85.
#    ~1 h on Apple MPS, ~2.3 h on a laptop CPU, ~15 min on a CUDA GPU; evaluate.py builds the AddSent/AddOneSent caches itself.
python prepare_paraphrases.py --input provided/train-v1.1.json --output data/paraphrases/train.jsonl
python prepare_paraphrases.py --input provided/dev-v1.1.json   --output data/paraphrases/dev.jsonl

# 2. extractor: DeBERTa-v3-large, consistency loss, lambda = 1, effective batch 16 (8 x 2), lr 1e-5, 2 epochs
python train.py --run_name deb_large_kl --model microsoft/deberta-v3-large --loss_mode kl --lambda_kl 1.0 \
  --paraphrase_cache data/paraphrases/train.jsonl --batch_size 8 --grad_accum 2 --lr 1e-5 --precision bf16
#    needs ~26 GB of GPU memory as written. On a 24 GB card add --grad_checkpointing (~30 % slower, several-fold less
#    activation memory) or use --batch_size 4 --grad_accum 4 (same effective batch); neither variant was timed by us.
#    GPUs without bf16 (T4, V100) need --precision fp16, which we have not tested with this encoder.

# 3. official-scorer evaluation on dev, Gan & Ng (dev + adversarial) and Adversarial SQuAD (also runs on CPU/MPS, slower)
for S in dev gan_dev gan_adv addsent addonesent; do
  python evaluate.py --weights weights/deb_large_kl/best.pt --eval_set $S --precision bf16
done

# optional second version: learned span reranker (needs a paraphrase of each test question)
python train_reranker.py --weights weights/deb_large_kl/best.pt --paraphrase_cache data/paraphrases/train.jsonl --precision bf16
python evaluate.py --weights weights/deb_large_kl/best.pt --eval_set dev --reranker weights/deb_large_kl/reranker.pt --precision bf16

# steps 1-3 plus the reranker and figures as one resumable command on the same machine:
MODEL=microsoft/deberta-v3-large RUN_PREFIX=deb_large MODES=kl BATCH=8 GRAD_ACCUM=2 LR=1e-5 PRECISION=bf16 bash scripts/run_pipeline.sh
```

The trained checkpoint (`weights/deb_large_kl/best.pt`, about 1.7 GB) is not in git; the run's configuration, training history and
all evaluation outputs are, under `runs/deb_large_kl/`. To evaluate without retraining, obtain the checkpoint from the group,
place it at that path and run step 3.

## What the system does

1. **Extractor.** BERT-base (baseline), DeBERTa-v3-base or DeBERTa-v3-large (proposed) with a start/end span head over
   sliding windows. At inference it keeps each question's **top-20 distinct spans**.
2. **Paraphrase-consistency training.** Every training question is back-translated EN→DE→EN with open MarianMT
   models and kept when its sentence-embedding similarity to the original is ≥ 0.85. Both versions are trained on the
   same passage with
   `loss = CE(q) + CE(q′) + λ · KL_sym(P_q, P_q′)` over passage-token start/end distributions.
   1-D optimal-transport and Hungarian span-matching penalties are available as ablations of the KL term.
3. **Learned span reranker.** A small MLP scores the top-k candidates from the extractor's own hidden states and
   score features (including the paraphrase view's score for the same span) and picks one, so the answer is always
   an exact passage span. It is trained only on the held-out training articles (about 3,300 questions), never on
   the dev or test sets. **Every run is reported in two versions: extractor only (top-1 span, the proposed system) and
   extractor + reranker.** Every evaluation run produces both; reranker ablations are in `docs/RESULTS.md` §9.
4. **Protocol.** 5 % of the training *articles* are held out for model selection, λ tuning and reranker training;
   the dev set is used only for final numbers, always via the official `evaluate-v2.0.py`. We report EM, F1 and
   paraphrase agreement on SQuAD dev, on the paraphrased test sets of Gan & Ng (2019) and on Adversarial SQuAD
   (Jia & Liang, 2017).

Every reported number is generated from the committed JSON artefacts under `runs/`: `visualize.py` writes `figures/` and
`report/tables/`, `scripts/make_results_report.py` writes `docs/RESULTS.md`. Model checkpoints are not in git.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install -r requirements.txt
# course data (from Canvas): provided/train-v1.1.json, provided/dev-v1.1.json, provided/evaluate-v2.0.py
python download_eval_sets.py                             # Gan & Ng paraphrased sets + Adversarial SQuAD -> data/external/
python tests/test_core.py                                # under a minute; checks metrics against the official scorer etc.
bash scripts/smoke_test_cpu.sh                           # ~5 min on a laptop CPU: the whole pipeline with a tiny BERT
```

## Running the experiments

The one-shot version (sequential; GPU, Apple MPS or CPU):

```bash
MODEL=bert-base-uncased MODES="ce aug kl" bash scripts/run_pipeline.sh
MODEL=microsoft/deberta-v3-base RUN_PREFIX=deb MODES="kl" PRECISION=bf16 bash scripts/run_pipeline.sh
MODEL=microsoft/deberta-v3-large RUN_PREFIX=deb_large MODES="ce kl" BATCH=8 GRAD_ACCUM=2 LR=1e-5 PRECISION=bf16 \
  bash scripts/run_pipeline.sh                           # large encoder: GPU only (add EXTRA_TRAIN_ARGS=--grad_checkpointing if memory is short)
```

The same steps by hand:

```bash
# 1. paraphrase caches (resumable; MarianMT + similarity filter)
python prepare_paraphrases.py --input provided/train-v1.1.json --output data/paraphrases/train.jsonl
python prepare_paraphrases.py --input provided/dev-v1.1.json   --output data/paraphrases/dev.jsonl

# 2. extractors (best validation-F1 checkpoint -> weights/<run>/best.pt)
python train.py --run_name bert_ce  --loss_mode ce  --paraphrase_cache data/paraphrases/train.jsonl   # cache only used for validation agreement
python train.py --run_name bert_aug --loss_mode aug --paraphrase_cache data/paraphrases/train.jsonl
python train.py --run_name bert_kl  --loss_mode kl  --paraphrase_cache data/paraphrases/train.jsonl --lambda_kl 1.0
#    Validation reports F1 *and* paraphrase agreement on the held-out articles for every run (pass the cache to the
#    ce run too, so the baseline's agreement is measured). Tune lambda with e.g. --lambda_kl 0.5 / 1.0 / 2.0
#    (or KL_LAMBDAS="0.5 1.0 2.0" in scripts/run_pipeline.sh) and choose the largest lambda whose validation F1
#    stays within a small tolerance (e.g. 0.5) of the best: see runs/<run>/train_summary.json
#    (best_val_f1, best_val_agreement_em) or, after visualize.py, figures/table_lambda.md and lambda_sweep.pdf.

# 3. reranker (trained on the held-out training articles only)
python train_reranker.py --weights weights/bert_kl/best.pt --paraphrase_cache data/paraphrases/train.jsonl
#    Targets are graded by default (--target_mode graded): exact-match candidates get relevance 1, partial-F1
#    candidates get --partial_weight (0.3) x F1. Use --target_mode exact to train on EM candidates only, or
#    --target_mode f1 for the old best-F1 fallback. Training logs and runs/<run>/reranker_summary.json report the
#    reranker's ceilings: Recall@k EM and best-F1@k (evaluate.py also reports them under extractor.candidate_ceiling).

#    Reranker ablations on an existing extractor (no retraining): --no_pair_score drops the paraphrase-derived
#    features (pair_score, pair_gap, pair_missing); --tag keeps the variants' files apart. See
#    scripts/run_reranker_variants.sh (pair vs no-pair, pure vs residual fusion, optional gate) and
#    `evaluate.py --select_mode {reranker,residual,extractor} --alpha A --override_margin M --tag T`.

# 4. evaluation with the official scorer, with and without the reranker
#    (for dev/addsent/addonesent the consistency metric needs back-translated questions; evaluate.py generates and
#     caches them under data/paraphrases/ on first use, ~16 min on CPU for dev; --paraphrase_method none skips it)
for S in dev gan_dev gan_adv addsent addonesent; do
  python evaluate.py --weights weights/bert_kl/best.pt --eval_set $S --reranker weights/bert_kl/reranker.pt
done

# 5. figures / tables for the report, t-SNE of paraphrase pairs
python visualize.py --best bert_kl            # --best = run for the question-type table (bert_kl_lam1.0 for the committed runs);
#    reads every run under runs/ except smoke*, or a subset with --runs bert_ce bert_aug bert_kl
python tsne_pairs.py --weights weights/bert_ce/best.pt --paraphrase_cache data/paraphrases/dev.jsonl
python tsne_pairs.py --weights weights/bert_kl/best.pt --paraphrase_cache data/paraphrases/dev.jsonl
cp figures/tsne_bert_*.pdf report/figures/    # main.tex includes figures/tsne_bert_ce.pdf and tsne_bert_kl.pdf
```

Loss modes (`--loss_mode`):

| mode | objective | role |
| :-- | :-- | :-- |
| `ce` | CE(q) | plain extractor |
| `aug` | CE(q) + CE(q′) | paraphrase augmentation only (Gan & Ng style) |
| `kl` | CE(q) + CE(q′) + λ·KL_sym | **proposed** consistency training |
| `ot` / `hungarian` / `dual` | CE(q) + CE(q′) + 1-D Wasserstein and/or Hungarian span matching | ablation (proposal item 2a) |

Useful flags: `--precision fp16|bf16` (GPU), `--max_examples N` (subset), `--eval_every N`, `--device cpu|mps|cuda`,
`--grad_accum N` (effective batch = batch × N), `--grad_checkpointing` (activation checkpointing: ~30 % slower, far less GPU
memory; for `microsoft/deberta-v3-large`), `--val_max_examples N` (caps the validation run during training; the final
validation after training always scores all 4,115 held-out questions). `train_summary.json` records `peak_gpu_mem_gb` for GPU runs.

### SoC compute cluster (Slurm)

The scripts follow the course's SoC cluster quick-start guide (distributed on Canvas): Conda on a compute node, the
`job_template.sbatch` layout (`gpu-long` partition, `--gpus=h100-47:1`, logs in `logs/slurm/<job>-<id>.out`), and
`sbatch` from the repository root on `xlogin`. Use a plain terminal SSH client (no VS Code Remote-SSH).

```bash
# 1. copy the project (code + provided/ data) to the cluster, then log in through StuJump
rsync -avz --exclude={'.venv','.git','weights','runs','logs','data','__pycache__'} \
  -e "ssh -J <soc_username>@stujump.comp.nus.edu.sg" ./ <soc_username>@xlogin.comp.nus.edu.sg:~/cs4248-project/
ssh -J <soc_username>@stujump.comp.nus.edu.sg <soc_username>@xlogin.comp.nus.edu.sg

# 2. one-off environment setup ON A COMPUTE NODE (xlogin cannot run conda): Miniconda, env "cs4248",
#    requirements, external eval sets, model downloads, unit tests
salloc -t 120
srun --pty bash                               # if this fails with "Unable to satisfy cpu bind request": srun --cpu-bind=none --pty bash
cd ~/cs4248-project && bash scripts/setup_env_cluster.sh   # installs the CUDA-12 PyTorch build the cluster driver needs
exit; exit
#    (a GPU-less salloc shows "cuda available: False"; verify with  salloc -G h100-47 -t 10  if you want to see True)

# 3. submit jobs from the repository root on xlogin
bash scripts/submit_all_ablations.sh          # paraphrase-cache job, then one GPU job per run:
                                              #   bert ce / aug / kl (one job per lambda) / ot / dual, deberta ce / kl
bash scripts/submit_deberta_large.sh          # afterwards: deberta-v3-large ce / kl (one job each; batch 8x2, lr 1e-5, 10 h limit;
                                              #   GRAD_CKPT=1 after an out-of-memory failure, GPU=<type> for another GPU from gpufree)
#   or single jobs:
sbatch scripts/job.sbatch                     # bert-base: ce, aug, kl -> reranker -> all evaluations, in one job
sbatch --job-name=g17_deb --export=ALL,MODEL=microsoft/deberta-v3-base,RUN_PREFIX=deb,MODES="kl" scripts/job.sbatch
sbatch --gpus=a100-40:1 scripts/job.sbatch    # another GPU type (check availability with gpufree / squeue)

# 4. monitor (job names: g17_para, g17_bert_ce, g17_bert_kl_lam1.0, g17_deb_kl, ...; a bare job is g17_pipeline)
squeue --me -o "%.10i %.24j %.3t %.8M %.12R"
tail -f logs/slurm/<job-name>-<job_id>.out
wc -l data/paraphrases/train.jsonl            # liveness check during the paraphrase stage (grows to 87,599)
```

`scripts/job.sbatch` activates the conda env and runs `scripts/run_pipeline.sh`; `STAGES`, `MODEL`, `MODES`,
`RUN_PREFIX`, `KL_LAMBDAS`, `PRECISION` (bf16 on H100/A100, fp16 on T4/V100) are passed with `--export=ALL,...`.
Every stage except `figures` is skipped when its output already exists, so a job can simply be resubmitted after a time-out. Model
caches live in `~/.cache/huggingface`; if home space runs short, move `weights/` and `data/` to the 500 GB scratch
area (`/mnt/scratch/${USER:0:1}/$USER`) and symlink them.

## Outputs

```
runs/<run>/config.json, history.json, val_history.json, train_summary.json   training telemetry
runs/<run>/reranker_summary.json                                             reranker before/after on held-out articles
runs/<run>/reranker_summary_<tag>.json, rerank_variants/<variant>/eval_<set>/  reranker ablations (RESULTS.md §9, §9b)
runs/<run>/eval_<set>/results.json          EM/F1 (official scorer), consistency, per-question-type, config
runs/<run>/eval_<set>/official_eval*.json   raw output of evaluate-v2.0.py (top-1 / reranked / paired view)
runs/<run>/eval_<set>/predictions*.json     {id: answer} files that were scored
runs/<run>/eval_<set>/candidates.jsonl      top-k distinct spans with scores
runs/<run>/eval_<set>/disagreements.jsonl   question/paraphrase pairs with different answers (qualitative table)
weights/<run>/best.pt, last.pt, reranker.pt
figures/*.pdf|png|md, report/tables/*.tex, report/figures/*.pdf
```

## Repository layout

```
g17qa/              core package (data, paraphrase, model, losses, decode, reranker, metrics, utils)
prepare_paraphrases.py  train.py  train_reranker.py  evaluate.py  visualize.py  tsne_pairs.py  download_eval_sets.py
scripts/            smoke_test_cpu.sh, run_pipeline.sh, job.sbatch, submit_all_ablations.sh, submit_deberta_large.sh, setup_env_cluster.sh,
                    reranker ablations (run/submit_reranker_variants.sh, run/submit_consistency_baseline.sh, run/submit_consistency_reranker.sh,
                    summarize_reranker_variants.py), make_results_report.py (writes docs/RESULTS.md; --verified <date> stamps an independent check), benchmark_cpu.py
tests/              test_core.py (metrics vs official scorer, windows/labels, losses, decoding), test_reranker.py, test_consistency.py, test_consistency_reranker.py
runs/               committed JSON artefacts of the 11 runs and the reranker ablations (configs, training history, official-scorer results)
provided/           official scorer evaluate-v2.0.py (in git) + train/dev JSON from Canvas (not in git)
data/external/      Gan & Ng paraphrased sets, Adversarial SQuAD (download_eval_sets.py; not in git)
data/paraphrases/   paraphrase caches (prepare_paraphrases.py; not in git)
report/             ACL-format paper (main.tex), bib, style files; tables/ and figures/ are generated by visualize.py / tsne_pairs.py (committed copies match runs/)
docs/               RESULTS.md (full results analysis, generated from runs/) + results_digest.json (its data)
app.py              legacy marimo demo notebook from the first iteration (not used for any reported number)
```

## Contributions and external code

Written by the group: paraphrase generation and filtering pipeline, sliding-window features, the consistency
objective and its OT/Hungarian variants, top-k candidate decoding, the learned span reranker, evaluation, analysis
and plotting code. External: PyTorch, Hugging Face Transformers (BERT, DeBERTa-v3, MarianMT weights),
sentence-transformers (MiniLM), scikit-learn, SciPy, the official SQuAD scorer, and the released test sets of
Gan & Ng (2019) and Jia & Liang (2017). No closed-source models or APIs are used anywhere.

## References

- Rajpurkar et al. 2016. SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP.
- Devlin et al. 2019. BERT. NAACL. — He et al. 2021/2023. DeBERTa / DeBERTaV3. ICLR.
- Gan & Ng 2019. Improving the Robustness of Question Answering Systems to Question Paraphrasing. ACL.
- Jia & Liang 2017. Adversarial Examples for Evaluating Reading Comprehension Systems. EMNLP.
- Xie et al. 2020. Unsupervised Data Augmentation for Consistency Training. NeurIPS.
- Tiedemann & Thottingal 2020. OPUS-MT. EAMT. — Reimers & Gurevych 2019. Sentence-BERT. EMNLP.
