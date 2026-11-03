# G17 Results Report: Paraphrase-Consistent Extractive QA on SQuAD v1.1

*Generated 2026-11-04 by `scripts/make_results_report.py` from the run artefacts in `runs/`. Every number is read from `runs/<run>/config.json`, `train_summary.json`, `reranker_summary.json` or `eval_<set>/results.json` (which copy the official `evaluate-v2.0.py` output stored in `official_eval*.json`). Metrics are rounded to one decimal first and every difference quoted in the text is the difference of the rounded values. The exact values used are in `docs/results_digest.json`. An independent fact-check on 2026-11-04 recomputed every table cell and prose number from the raw files for the runs present on that date.*

## 1. Summary

- The BERT-base baseline reaches **81.1 EM / 88.5 F1** on the full 10,570-question dev set, matching the published BERT-base result (80.8 / 88.5, Devlin et al. 2019, Table 2), so the training recipe is validated.
- Consistency training (symmetric KL, λ = 1) raises **paraphrase agreement from 83.3 % to 87.9 %** (+4.6 points over the baseline, +1.4 over augmentation-only) while EM / F1 move by +0.3 / +0.1, i.e. within noise. DeBERTa-v3-base shows the same pattern (+3.8 agreement, -0.2 F1).
- Against distractor sentences the effect is small: AddSent F1 64.9 → 66.1 (+1.2, of which +1.0 comes from augmentation alone), AddOneSent 73.9 → 74.5 (+0.6); agreement on those sets rises by +6.2 and +4.1 points.
- The learned span reranker raises agreement further (87.9 → 92.9 % on dev) but costs -0.9 F1 / -1.4 EM on dev; its clearest benefit is on the 56-question adversarial-paraphrase set (63.6 → 75.6 F1), which is too small for firm conclusions.
- λ was chosen on held-out training articles with the rule "largest λ whose held-out F1 is within 0.5 points of the best": **λ = 1** (bert_kl_lam1.0). The rule gives λ = 1 as well when applied to all 4,115 held-out questions instead of the 2,000-question selection sample, where λ = 1 is nominally the best (by 0.08 points, well within noise).

## 2. What was run

All runs share the same recipe: 2 epochs, batch 16, learning rate 3e-05, sequence length 384, window stride 128, AdamW with 10 % warm-up, best checkpoint by validation F1. Training uses the 83,484 questions from 95 % of the SQuAD training articles (bert-base-uncased: 84,504–84,507 windows; deberta-v3-base: 84,286–84,289 windows; the count varies slightly with the tokeniser and with whether a paraphrase view is built). The 4,115 questions of the remaining 5 % of articles are held out: during training, validation is run on a seeded random sample of 2,000 of them (`--val_max_examples`), which is what checkpoint selection and the λ rule use; the selected checkpoint is then evaluated once on all 4,115 (`final_full_val`), and the reranker is trained on the same articles. The dev set and the external test sets are used only for the final numbers.

| Run | Model | Objective | λ | Optimiser steps | Training time (min) | Paraphrases attached to training questions |
| :-- | :-- | :-- | --: | --: | --: | --: |
| BERT ce (baseline) | bert-base-uncased | cross-entropy only (baseline) | – | 10,564 | 15 | not used |
| BERT aug | bert-base-uncased | paraphrases as extra training data | – | 10,564 | 19 | 68,439 / 83,484 |
| BERT kl λ=0.5 | bert-base-uncased | paraphrases + symmetric-KL consistency penalty | 0.5 | 10,564 | 22 | 68,439 / 83,484 |
| BERT kl λ=1 | bert-base-uncased | paraphrases + symmetric-KL consistency penalty | 1.0 | 10,564 | 19 | 68,439 / 83,484 |
| BERT kl λ=2 | bert-base-uncased | paraphrases + symmetric-KL consistency penalty | 2.0 | 10,564 | 22 | 68,439 / 83,484 |
| BERT ot | bert-base-uncased | paraphrases + 1-D Wasserstein penalty | – | 10,564 | 19 | 68,439 / 83,484 |
| BERT dual | bert-base-uncased | paraphrases + Wasserstein + Hungarian span matching | – | 10,564 | 22 | 68,439 / 83,484 |
| DeBERTa ce (baseline) | deberta-v3-base | cross-entropy only (baseline) | – | 10,536 | 17 | not used |
| DeBERTa kl λ=1 | deberta-v3-base | paraphrases + symmetric-KL consistency penalty | 1.0 | 10,538 | 33 | 68,439 / 83,484 |

Paraphrases come from MarianMT back-translation (EN→DE→EN) filtered at cosine similarity ≥ 0.85. On the dev set 8,622 of 10,570 questions (81.6 %) received an accepted paraphrase; the agreement metric is computed over those pairs only.

## 3. Main results on the dev set

Official scorer (`evaluate-v2.0.py`), all 10,570 questions. *Agreement* = percentage of (question, paraphrase) pairs that receive the same normalised answer. The `+reranker` columns are the second system version.

| Run | EM | F1 | Agreement | EM +reranker | F1 +reranker | Agreement +reranker |
| :-- | --: | --: | --: | --: | --: | --: |
| BERT ce (baseline) | 81.1 | 88.5 | 83.3 | 81.4 | 88.3 | 86.3 |
| BERT aug | 81.0 | 88.2 | 86.5 | 80.6 | 87.5 | 93.4 |
| BERT kl λ=0.5 | 81.1 | 88.5 | 87.6 | 81.3 | 88.3 | 89.1 |
| BERT kl λ=1 | 81.4 | 88.6 | 87.9 | 80.0 | 87.7 | 92.9 |
| BERT kl λ=2 | 81.3 | 88.4 | 88.8 | 80.4 | 87.8 | 93.7 |
| BERT ot | 80.9 | 88.2 | 86.3 | 78.4 | 86.8 | 91.3 |
| BERT dual | 81.1 | 88.3 | 86.7 | 81.1 | 88.1 | 90.0 |
| DeBERTa ce (baseline) | 88.3 | 93.9 | 88.0 | 87.4 | 93.3 | 89.7 |
| DeBERTa kl λ=1 | 88.3 | 93.7 | 91.8 | 88.0 | 93.7 | 92.6 |

**Reading.** Accuracy is flat across the 7 BERT objectives (EM 80.9–81.4, F1 88.2–88.6); the binomial standard error of one EM value over 10,570 questions is about 0.38 points and seed-to-seed variation of BERT fine-tuning is typically a few tenths of a point, so differences below roughly half a point should not be interpreted. Agreement is where the objectives separate: baseline 83.3 %, augmentation 86.5 %, consistency training 87.9 % (λ = 1) and 88.8 % (λ = 2). With 8,622 pairs the standard error of one agreement value is about 0.40 points and that of a difference between two runs about 0.57; the +4.6-point gain over the baseline is about 8 standard errors and clearly real, whereas the +1.4-point gain over augmentation-only is about 2.5 standard errors and, with one training seed per configuration, only suggestive.

## 4. Augmentation versus consistency loss

The `aug` run isolates the effect of the extra paraphrased examples from the effect of the penalty. Extra data alone moves agreement from 83.3 % to 86.5 % (+3.2); the KL penalty adds +1.4 points on top of that at λ = 1 and +2.3 at λ = 2 (both relative to `aug`). Accuracy does not pay for this: `aug` scores 81.0 / 88.2 and `kl` (λ = 1) 81.4 / 88.6 EM / F1. So roughly 70 % of the agreement gain over the baseline comes from the paraphrase data itself and the remainder from the penalty; the penalty's own share is the part that is only suggestive at one seed (Section 3).

## 5. Choosing λ on held-out data

The form of the rule (largest λ whose held-out F1 is within a tolerance of the best) and the example tolerance of 0.5 points were written into the README and the report draft before the runs finished; the value 0.5 was not formally pre-registered and is load-bearing (on the selection sample a tolerance below 0.44 would pick λ = 0.5 and one of 0.82 or more would pick λ = 2; on the full held-out set λ = 2 is 0.53 below the best). The rule is applied to unrounded F1 values. The table shows both the selection sample used during training and the full held-out set.

| Model | λ | Sample F1 (n=2,000) | Sample agreement | Full held-out F1 (n=4,115) | Full held-out agreement | Dev F1 | Dev agreement |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| bert-base-uncased | 0.5 | 82.5 | 89.5 | 81.9 | 89.8 | 88.5 | 87.6 |
| bert-base-uncased | 1 | 82.1 | 90.6 | 82.0 | 90.1 | 88.6 | 87.9 |
| bert-base-uncased | 2 | 81.7 | 91.0 | 81.5 | 90.7 | 88.4 | 88.8 |
| deberta-v3-base | 1 | 88.2 | 93.6 | 87.4 | 93.2 | 93.7 | 91.8 |

On the selection sample the best F1 is 82.5 (λ = 0.5) and λ = 0.5, 1 fall within 0.5 points, so the rule selects **λ = 1**. On all 4,115 held-out questions the F1 values are 81.9, 82.0, 81.5 for λ = 0.5, 1, 2: λ = 1 is nominally the best (by 0.08 points), λ = 0.5 within tolerance, and λ = 2 is 0.53 below the best, so the rule selects λ = 1 there too; agreement rises with λ on both the sample (89.5, 90.6, 91.0) and the full set (89.8, 90.1, 90.7). The sample differences in F1 (82.5, 82.1, 81.7) are within the sampling error of a 2,000-question sample (at most about 0.8 points for one value under a binomial bound, roughly 1.2 for a difference between two runs), so no accuracy trade-off is demonstrated between these λ values. On dev, λ = 1 also has the best F1 of the three (88.5, 88.6, 88.4).

## 6. Backbone: BERT-base versus DeBERTa-v3-base

DeBERTa-v3-base is far stronger as an extractor (88.3 / 93.9 EM / F1 versus 81.1 / 88.5 for BERT-base) and already more self-consistent (88.0 % agreement versus 83.3 %). Consistency training still adds +3.8 agreement points at -0.2 F1 on dev, so the effect is not specific to the weaker backbone. On the full held-out set the picture is the same: F1 87.1 → 87.4, agreement 90.8 → 93.2.

## 7. Robustness: paraphrased and adversarial test sets

F1 from the official scorer. For the Gan & Ng sets, *orig* is the model's F1 on the original questions and *para* on their human-written paraphrases; *agreement* is between the two answers. For AddSent / AddOneSent the agreement is with back-translated paraphrases of the adversarial questions.

| Run | G&N orig F1 | G&N para F1 | G&N agreement | G&N adv orig F1 | G&N adv para F1 | AddSent F1 | AddSent agreement | AddOneSent F1 | AddOneSent agreement |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| BERT ce (baseline) | 88.4 | 85.6 | 89.1 | 85.1 | 58.6 | 64.9 | 77.9 | 73.9 | 80.3 |
| BERT aug | 87.4 | 85.7 | 91.4 | 85.9 | 54.1 | 65.9 | 82.1 | 73.8 | 83.5 |
| BERT kl λ=0.5 | 88.5 | 86.0 | 90.8 | 81.8 | 61.7 | 65.8 | 83.7 | 74.1 | 85.4 |
| BERT kl λ=1 | 87.5 | 86.1 | 91.0 | 85.0 | 63.6 | 66.1 | 84.1 | 74.5 | 84.4 |
| BERT kl λ=2 | 88.5 | 86.9 | 91.9 | 84.4 | 63.5 | 65.7 | 84.1 | 73.8 | 86.0 |
| BERT ot | 87.2 | 85.6 | 90.3 | 85.0 | 57.4 | 65.8 | 81.8 | 73.9 | 83.4 |
| BERT dual | 87.5 | 85.7 | 90.0 | 85.8 | 55.6 | 66.0 | 81.9 | 74.2 | 82.8 |
| DeBERTa ce (baseline) | 93.5 | 91.3 | 91.8 | 93.9 | 81.9 | 85.1 | 85.5 | 88.7 | 86.2 |
| DeBERTa kl λ=1 | 92.4 | 91.0 | 93.7 | 95.4 | 82.9 | 84.2 | 90.2 | 87.9 | 89.9 |

**Distractor sentences (Jia & Liang).** Every model loses heavily when a distractor sentence is appended: the BERT baseline drops from 88.5 F1 on dev to 64.9 on AddSent (−23.6) and 73.9 on AddOneSent (−14.6). All 6 paraphrase-trained BERT runs land in a narrow band on AddSent (65.7–66.1 F1), so the +1.2 of the λ = 1 run over the baseline is mostly the paraphrase data (`aug` alone: +1.0); on AddOneSent `aug` is -0.1 and λ = 1 +0.6. What consistency training does change is stability: agreement on AddSent goes from 77.9 % to 84.1 % (+6.2), i.e. the model is fooled more consistently rather than less often. DeBERTa is much more robust to distractors (85.1 F1 on AddSent for the baseline) and consistency training slightly lowers its AddSent F1 (84.2) while raising agreement (85.5 → 90.2 %).

**Human paraphrases (Gan & Ng, 1,062 questions, 1,061 pairs).** Paraphrasing costs the baseline 2.8 F1 (88.4 on the original questions → 85.6 on the paraphrases) and the λ = 1 model 1.4 (87.5 → 86.1); note that the narrower gap comes from both directions, +0.5 on the paraphrases but -0.9 on the originals. Agreement between the two answers goes from 89.1 % to 91.0 % (+1.9). On the 56-question adversarial-paraphrase subset the baseline falls to 58.6 F1 and the λ = 1 model to 63.6; with 56 questions each one is worth 1.8 points, so these numbers are indicative only.

## 8. The learned span reranker

Held-out columns come from `reranker_summary.json`: the reranker is trained on 80 % of the held-out articles and early-stopped on the other 20 % (758 questions); the held-out *reranked* value is the best of 20 epochs on that same slice and is therefore optimistic. The remaining columns are the dev set. Δ = reranked − extractor.

| Run | Held-out top-1 EM / F1 | Held-out reranked EM / F1 (best epoch) | Last-epoch F1 | Oracle EM (correct span among top-20) | Dev Δ EM | Dev Δ F1 | Dev Δ agreement | AddSent Δ F1 | G&N adv Δ F1 |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| BERT ce (baseline) | 72.2 / 83.6 | 72.7 / 84.0 (ep. 1) | 78.9 | 96.0 | +0.3 | -0.2 | +3.0 | -0.1 | +4.8 |
| BERT aug | 70.7 / 82.7 | 71.1 / 82.6 (ep. 5) | 79.0 | 97.2 | -0.4 | -0.7 | +6.9 | -0.7 | +25.4 |
| BERT kl λ=0.5 | 70.6 / 82.8 | 70.4 / 82.4 (ep. 1) | 79.3 | 97.1 | +0.2 | -0.2 | +1.5 | +0.2 | +4.4 |
| BERT kl λ=1 | 70.6 / 82.4 | 71.9 / 83.0 (ep. 8) | 80.3 | 96.7 | -1.4 | -0.9 | +5.0 | -0.1 | +12.0 |
| BERT kl λ=2 | 69.9 / 81.8 | 71.1 / 82.5 (ep. 6) | 79.9 | 96.7 | -0.9 | -0.6 | +4.9 | +1.1 | +14.3 |
| BERT ot | 70.3 / 82.3 | 71.1 / 82.7 (ep. 11) | 81.9 | 96.7 | -2.5 | -1.4 | +5.0 | +0.0 | +23.2 |
| BERT dual | 70.7 / 82.4 | 71.4 / 82.7 (ep. 2) | 80.6 | 97.0 | +0.0 | -0.2 | +3.3 | +0.0 | +10.5 |
| DeBERTa ce (baseline) | 76.1 / 87.4 | 76.4 / 88.0 (ep. 1) | 86.1 | 98.2 | -0.9 | -0.6 | +1.7 | +0.2 | +3.9 |
| DeBERTa kl λ=1 | 76.5 / 87.6 | 76.3 / 87.5 (ep. 1) | 84.3 | 98.0 | -0.3 | +0.0 | +0.8 | -0.3 | +4.1 |

**Reading.** The oracle column shows that the correct span is among the top-20 candidates for 96–98 % of held-out questions, so there is room to gain; the reranker captures almost none of it. The best-epoch held-out EM changes of -0.2 to +1.3 points are measured on 758 questions (binomial standard error 1.5–1.7 points) and the last-epoch F1 is 0.8–5.1 points below the best epoch, i.e. the reranker overfits within a few epochs. The honest estimates are the dev deltas: F1 never rises (it falls in 8 of 9 runs and is unchanged in 1) while agreement rises in 9 of 9; for the selected run the trade is -0.9 F1 for +5.0 agreement. We hypothesise that the paraphrase-score feature drives this (the reranker prefers spans both phrasings score highly, which stabilises answers but sometimes overrides a correct top-1 span); no run without that feature was trained, so this is not established by ablation. Its largest effect is on the adversarial paraphrases of Gan & Ng, where the paired view is the easier original question (orig F1 85.0) and the reranker can lean on it (F1 63.6 → 75.6 for the selected run; small set). Conclusion: the reranker is a consistency device with an accuracy cost on in-distribution data; both system versions are reported.

## 9. Optimal-transport and span-matching ablation

Replacing the KL penalty by the 1-D Wasserstein penalty (`ot`) or by Wasserstein plus Hungarian span matching (`dual`), with the same paraphrase data, gives dev agreement of 86.3 % and 86.7 % against 87.9 % for KL at λ = 1, with EM / F1 of 80.9 / 88.2 and 81.1 / 88.3. Both are within 0.2 points of the augmentation-only run (-0.2 and +0.2), so at their default weights (λ_OT = 0.5, μ = 0.1, not tuned) these penalties add nothing beyond the paraphrase data, whereas KL adds +1.4. On AddSent they reach 65.8 and 66.0 F1 versus 66.1. KL remains the primary method; these runs are a negative ablation.

## 10. Question-type analysis (dev)

EM / F1 by question type for the baseline and the selected consistency run (extractor only), and F1 with the reranker for the selected run. Types are assigned from the question's wh-word; *how many/much* also covers *how long / old / far / often / tall / big / large*.

| Type | n | Baseline EM / F1 | kl λ=1 EM / F1 | Δ F1 | kl λ=1 F1 +reranker |
| :-- | --: | :-- | :-- | --: | --: |
| what | 5,812 | 79.5 / 87.5 | 79.8 / 87.6 | +0.1 | 87.1 |
| who | 1,204 | 88.9 / 92.2 | 88.6 / 91.8 | -0.4 | 90.6 |
| when | 924 | 93.4 / 95.5 | 93.0 / 95.3 | -0.2 | 93.9 |
| how many/much | 923 | 84.8 / 90.7 | 85.8 / 91.6 | +0.9 | 89.4 |
| which | 646 | 80.2 / 87.1 | 82.2 / 87.7 | +0.6 | 87.3 |
| where | 511 | 77.3 / 87.0 | 77.7 / 87.7 | +0.7 | 86.4 |
| how | 301 | 61.1 / 78.5 | 61.8 / 77.9 | -0.6 | 77.1 |
| why | 155 | 53.5 / 77.0 | 54.8 / 78.3 | +1.3 | 76.9 |
| other | 94 | 55.3 / 75.4 | 57.4 / 74.4 | -1.0 | 74.1 |

The hardest types are *other* (74.4 F1), *how* (77.9 F1), *why* (78.3 F1); the easiest are *when* (95.3), *who* (91.8). Consistency training changes per-type F1 by at most 1.3 points, i.e. it does not redistribute accuracy across question types; its effect is on stability, which this table does not measure. The reranker lowers F1 for 9 of the 9 types, consistent with Section 8.

## 11. Caveats

- Single training seed per configuration; no confidence intervals beyond the binomial standard errors quoted above. Dev EM/F1 differences below about half a point, the +1.4-point agreement gain of KL over augmentation alone, and all differences on the 56-question adversarial-paraphrase set should be treated as unconfirmed.
- Agreement is measured only on questions with an accepted back-translated paraphrase (8,622 of 10,570 on dev); the filter admits some meaning-changing paraphrases, which caps attainable agreement below 100 %.
- Checkpoint selection and the λ rule used a 2,000-question sample of the held-out articles; the full-set numbers are reported alongside and lead to the same λ.
- The OT / Hungarian weights were not tuned, so Section 9 compares them at default weights.
- Validation articles were used for checkpoint selection, λ selection and reranker training; dev and the external sets were never used for any decision.

## 12. Provenance

```
python visualize.py --best bert_kl_lam1.0        # regenerates figures/ and report/tables/ from runs/
python scripts/make_results_report.py              # regenerates this file and docs/results_digest.json
```
Per-run sources: `runs/<run>/config.json` (hyper-parameters, split sizes, paraphrase attachment), `runs/<run>/train_summary.json` (steps, time, selection-sample and full held-out F1/agreement), `runs/<run>/reranker_summary.json` (including the per-epoch history), `runs/<run>/eval_<set>/results.json` and `official_eval*.json` (official scorer output). Training, reranker fitting and the dev evaluations were produced by the training jobs on 3 Nov 2026; the four external sets were re-evaluated the same evening after a fix to the Gan & Ng original-question scoring (commit 3d62dc6). Hardware (NUS SoC cluster, H100 NVL MIG 3g.47gb slices) and Slurm job ids (training 1068703–1068711, re-evaluation 1069429–1069438) are taken from the cluster logs, which are not part of the repository.
