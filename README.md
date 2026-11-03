# Paraphrase-Consistent Extractive QA with a Learned Span Reranker

**CS4248 Introduction to Large Language Models (AY2026/27 S1), National University of Singapore — Group G17**
Chong Xin Wei, Ng Jian Yi, Soh Jun Ying Gavin, Wang Chengke, Yee Yan Hwee

Topic: Extractive question answering on SQuAD v1.1. Proposal: `docs/CS4248 Project.pdf`.

## What the system does

1. **Extractor.** BERT-base (baseline) or DeBERTa-v3-base with a start/end span head over sliding windows. At
   inference it keeps each question's **top-20 distinct spans**.
2. **Paraphrase-consistency training.** Every training question is back-translated EN→DE→EN with open MarianMT
   models and kept when its sentence-embedding similarity to the original is ≥ 0.85. Both versions are trained on the
   same passage with
   `loss = CE(q) + CE(q′) + λ · KL_sym(P_q, P_q′)` over passage-token start/end distributions.
   1-D optimal-transport and Hungarian span-matching penalties are available as ablations of the KL term.
3. **Learned span reranker.** A small MLP scores the top-k candidates from the extractor's own hidden states and
   score features (including the paraphrase view's score for the same span) and picks one, so the answer is always
   an exact passage span. It is trained only on the held-out training articles (about 3,300 questions), never on
   the dev or test sets. **The system is reported in two versions: extractor only (top-1 span) and extractor +
   reranker.** Every evaluation run produces both.
4. **Protocol.** 5 % of the training *articles* are held out for model selection, λ tuning and reranker training;
   the dev set is used only for final numbers, always via the official `evaluate-v2.0.py`. We report EM, F1 and
   paraphrase agreement on SQuAD dev, on the paraphrased test sets of Gan & Ng (2019) and on Adversarial SQuAD
   (Jia & Liang, 2017).

There are **no stored result numbers** in this repository: every table and figure is generated from `runs/` by
`visualize.py` after the experiments have been run.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install -r requirements.txt
# course data (from Canvas): provided/train-v1.1.json, provided/dev-v1.1.json, provided/evaluate-v2.0.py
python download_eval_sets.py                             # Gan & Ng paraphrased sets + Adversarial SQuAD -> data/external/
python tests/test_core.py                                # ~1 min, checks metrics against the official scorer etc.
bash scripts/smoke_test_cpu.sh                           # ~5 min on a laptop CPU: the whole pipeline with a tiny BERT
```

## Running the experiments

The one-shot version (sequential; GPU, Apple MPS or CPU):

```bash
MODEL=bert-base-uncased MODES="ce aug kl" bash scripts/run_pipeline.sh
MODEL=microsoft/deberta-v3-base RUN_PREFIX=deb MODES="kl" PRECISION=bf16 bash scripts/run_pipeline.sh
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

# 4. evaluation with the official scorer, with and without the reranker
#    (for dev/addsent/addonesent the consistency metric needs back-translated questions; evaluate.py generates and
#     caches them under data/paraphrases/ on first use, ~16 min on CPU for dev; --paraphrase_method none skips it)
for S in dev gan_dev gan_adv addsent addonesent; do
  python evaluate.py --weights weights/bert_kl/best.pt --eval_set $S --reranker weights/bert_kl/reranker.pt
done

# 5. figures / tables for the report, t-SNE of paraphrase pairs
python visualize.py --best bert_kl            # all runs under runs/ except smoke*; or --runs bert_ce bert_aug bert_kl
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

Useful flags: `--precision fp16|bf16` (GPU), `--max_examples N` (subset), `--eval_every N`, `--device cpu|mps|cuda`.

### SoC compute cluster (Slurm)

The scripts follow the course quick-start guide (`docs/SoC_Cluster_Guide.pdf`): Conda on a compute node, the
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
Every stage is skipped when its output already exists, so a job can simply be resubmitted after a time-out. Model
caches live in `~/.cache/huggingface`; if home space runs short, move `weights/` and `data/` to the 500 GB scratch
area (`/mnt/scratch/${USER:0:1}/$USER`) and symlink them.

## Can this run locally on a CPU?

Measured with `python scripts/benchmark_cpu.py` on an Apple M2 Pro (12 cores, 16 GB), `bert-base-uncased`,
L = 384, stride 128 (88,641 training windows, 10,833 dev windows):

| Task | CPU only | Apple-silicon GPU (`--device mps`) |
| :-- | :-- | :-- |
| Extractor training, `ce` (1 forward/step), one epoch | 8.0 s/step (batch 16) → ≈ 12 h | 0.44 s/step (batch 8) → ≈ 1.4 h |
| Extractor training, `kl` (2 forwards/step), one epoch | 26 s/step → ≈ 40 h | 1.07 s/step → ≈ 3.3 h |
| Dev-set inference (×2 when paraphrases are evaluated) | ≈ 6.5 min | ≈ 2 min |
| MarianMT back-translation of all 87.6k training questions (one-off, cached, resumable); ≈ 88 % of a 128-question dev sample passed the 0.85 filter | 11 q/s → ≈ 2.2 h | 23 q/s → ≈ 1.1 h |
| Similarity filter | 370 q/s | 660 q/s |
| Reranker (feature extraction on ≈ 4.1k held-out questions + MLP) | minutes | minutes |
| Tiny-model smoke test of the whole pipeline | ≈ 5 min | — |

What this means:

- **CPU only** is fine for the smoke test, building the paraphrase caches, evaluating trained checkpoints on every
  test set, training the reranker and producing figures. Full fine-tuning is not practical: one 2-epoch `ce` run
  needs about a day and one `kl` run several days.
- **Apple-silicon GPU** (`--device mps --batch_size 8 --grad_accum 2`): a 2-epoch BERT-base `ce` run is ≈ 3 h and a
  `kl` run ≈ 7 h, so single BERT-base runs are feasible overnight on a 16 GB Mac. Batch 16 runs out of GPU memory in
  paired modes, and DeBERTa-v3-base is roughly twice as slow and may not fit.
- The **full grid** (BERT `ce`/`aug`/`kl`, DeBERTa-v3 `ce`/`kl`, optional OT ablations, λ sweep) belongs on the SoC
  cluster; a T4 or A100 with fp16/bf16 is several times faster than MPS and runs the jobs in parallel.

## Outputs

```
runs/<run>/config.json, history.json, val_history.json, train_summary.json   training telemetry
runs/<run>/reranker_summary.json                                             reranker before/after on held-out articles
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
scripts/            smoke_test_cpu.sh, run_pipeline.sh, job.sbatch, submit_all_ablations.sh, setup_env_cluster.sh, benchmark_cpu.py
tests/test_core.py  unit tests (metrics vs official scorer, windows/labels, losses, decoding)
provided/           official scorer evaluate-v2.0.py (in git) + train/dev JSON from Canvas (not in git)
data/external/      Gan & Ng paraphrased sets, Adversarial SQuAD (download_eval_sets.py; not in git)
data/paraphrases/   paraphrase caches (prepare_paraphrases.py; not in git)
report/             ACL-format paper (main.tex), bib, style files; tables/ and figures/ are generated, not committed
docs/               proposal (CS4248 Project.pdf), course requirements, SoC cluster guide, CHANGES.md (what was
                    changed and why), status_review (snapshot of the repo before the rebuild), math.md (earlier notes)
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
