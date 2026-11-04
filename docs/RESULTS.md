# G17 Results Report: Paraphrase-Consistent Extractive QA on SQuAD v1.1

*Generated 2026-11-04 by `scripts/make_results_report.py` from `runs/`. Every number is read from `config.json`, `train_summary.json`, `reranker_summary.json` or `eval_<set>/results.json` (official `evaluate-v2.0.py` output). Metrics are rounded to one decimal first; every difference in the text is a difference of rounded values. Exact values: `docs/results_digest.json`. Independently fact-checked against the raw files on 2026-11-04.*

## 1. Summary

- **The baseline reproduces the literature.** BERT-base: 81.1 EM / 88.5 F1 on the 10,570-question dev set (published 80.8 / 88.5).
- **Consistency training buys stability, not accuracy.** Paraphrase agreement rises from 83.3 % to 87.9 % (+4.6 points, of which +1.4 beyond augmentation alone) at +0.3 EM / +0.1 F1. The same holds for DeBERTa-v3-base (+3.8) and DeBERTa-v3-large (+3.3).
- **The backbone is the accuracy lever.** DeBERTa-v3-base 88.3 / 93.9, DeBERTa-v3-large 90.3 / 95.5 EM / F1, against 81.1 / 88.5 for BERT-base; no objective change moves accuracy by more than half a point.
- **Robustness gains are small and come mostly from the paraphrase data.** AddSent F1 64.9 → 66.1 (+1.2, of which +1.0 from augmentation alone); AddSent agreement +6.2.
- **The reranker trades accuracy for agreement.** Selected run, dev: -0.9 F1 for +5.0 agreement; its largest gain is on the 56-question adversarial-paraphrase set (63.6 → 75.6 F1), which is too small for firm conclusions. On DeBERTa-v3-large the cost falls to -0.1 F1 and F1 rises on all four robustness sets, but with newer reranker targets (Section 8).
- **Reranker ablation.** The paraphrase features carry the agreement and adversarial-paraphrase gains; blending reranker and extractor scores instead of letting the reranker overrule removes the dev accuracy loss and adds a few tenths of EM on all three extractors tested, at the cost of most of the agreement. Neither is adopted (Section 9).
- **λ = 1**, chosen on held-out training articles as the largest λ with F1 within 0.5 points of the best (same choice on the full held-out set).

## 2. What was run

2 epochs, effective batch 16, learning rate 3e-05, sequence length 384, stride 128, AdamW with 10 % warm-up, best checkpoint by validation F1; the DeBERTa-v3-large runs use batch 8 × 2 accumulation steps and learning rate 1e-05. Training uses the 83,484 questions of 95 % of the training articles; the other 5 % (4,115 questions) are held out for checkpoint selection (on a 2,000-question sample), λ selection and reranker training. Dev and the external sets are used only for the final numbers. Paraphrases are MarianMT back-translations (EN→DE→EN) kept at cosine similarity ≥ 0.85; 8,622 of 10,570 dev questions (81.6 %) have one, and agreement is measured over those pairs.

| Run | Model | Objective | λ | Batch × accum. | LR | Optimiser steps | Training (min) | Paraphrases attached |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |
| BERT ce (baseline) | bert-base-uncased | cross-entropy only (baseline) | – | 16 × 1 | 3e-05 | 10,564 | 15 | not used |
| BERT aug | bert-base-uncased | paraphrases as extra training data | – | 16 × 1 | 3e-05 | 10,564 | 19 | 68,439 / 83,484 |
| BERT kl λ=0.5 | bert-base-uncased | paraphrases + symmetric-KL consistency penalty | 0.5 | 16 × 1 | 3e-05 | 10,564 | 22 | 68,439 / 83,484 |
| BERT kl λ=1 | bert-base-uncased | paraphrases + symmetric-KL consistency penalty | 1 | 16 × 1 | 3e-05 | 10,564 | 19 | 68,439 / 83,484 |
| BERT kl λ=2 | bert-base-uncased | paraphrases + symmetric-KL consistency penalty | 2 | 16 × 1 | 3e-05 | 10,564 | 22 | 68,439 / 83,484 |
| BERT ot | bert-base-uncased | paraphrases + 1-D Wasserstein penalty | – | 16 × 1 | 3e-05 | 10,564 | 19 | 68,439 / 83,484 |
| BERT dual | bert-base-uncased | paraphrases + Wasserstein + Hungarian span matching | – | 16 × 1 | 3e-05 | 10,564 | 22 | 68,439 / 83,484 |
| DeBERTa ce (baseline) | deberta-v3-base | cross-entropy only (baseline) | – | 16 × 1 | 3e-05 | 10,536 | 17 | not used |
| DeBERTa kl λ=1 | deberta-v3-base | paraphrases + symmetric-KL consistency penalty | 1 | 16 × 1 | 3e-05 | 10,538 | 33 | 68,439 / 83,484 |
| DeBERTa-large ce (baseline) | deberta-v3-large | cross-entropy only (baseline) | – | 8 × 2 | 1e-05 | 10,536 | 43 | not used |
| DeBERTa-large kl λ=1 | deberta-v3-large | paraphrases + symmetric-KL consistency penalty | 1 | 8 × 2 | 1e-05 | 10,538 | 108 | 68,439 / 83,484 |

## 3. Main results on the dev set

Official scorer, all 10,570 questions. *Agreement* = share of (question, paraphrase) pairs that get the same normalised answer. `+reranker` = the second system version.

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
| DeBERTa-large ce (baseline) | 90.3 | 95.5 | 89.4 | 90.2 | 95.4 | 93.4 |
| DeBERTa-large kl λ=1 | 90.0 | 95.4 | 92.7 | 90.0 | 95.3 | 95.1 |

**Reading.** Accuracy is flat across the 7 BERT objectives (EM 80.9–81.4, F1 88.2–88.6; one standard error of an EM value is about 0.4 points, so differences under half a point are noise). Agreement is where they separate: 83.3 % baseline, 86.5 % augmentation, 87.9 % consistency (λ = 1). The +4.6 gain over the baseline is about 8 standard errors of a difference; the +1.4 over augmentation-only is about 2.5 and, with one seed per run, only suggestive.

## 4. Augmentation versus consistency loss

The paraphrased examples alone (`aug`) move agreement 83.3 → 86.5 % (+3.2); the KL penalty adds +1.4 at λ = 1 and +2.3 at λ = 2, at no accuracy cost (EM / F1 81.0 / 88.2 for `aug`, 81.4 / 88.6 at λ = 1, 81.3 / 88.4 at λ = 2). Roughly 70 % of the gain over the baseline is therefore the data and the rest the penalty; the penalty's share is the part that is only suggestive at one seed.

## 5. Choosing λ on held-out data

Rule: the largest λ whose held-out F1 is within 0.5 points of the best, applied to unrounded values. The selection sample is what training used; the full held-out set is a check.

| Model | λ | Sample F1 (n=2,000) | Sample agreement | Full held-out F1 (n=4,115) | Full held-out agreement | Dev F1 | Dev agreement |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| bert-base-uncased | 0.5 | 82.5 | 89.5 | 81.9 | 89.8 | 88.5 | 87.6 |
| bert-base-uncased | 1 | 82.1 | 90.6 | 82.0 | 90.1 | 88.6 | 87.9 |
| bert-base-uncased | 2 | 81.7 | 91.0 | 81.5 | 90.7 | 88.4 | 88.8 |
| deberta-v3-base | 1 | 88.2 | 93.6 | 87.4 | 93.2 | 93.7 | 91.8 |
| deberta-v3-large | 1 | 88.8 | 95.4 | 88.2 | 94.6 | 95.4 | 92.7 |

Both the sample and the full set select **λ = 1** (sample F1 82.5, 82.1, 81.7 for λ = 0.5, 1, 2; all within the sampling error of a 2,000-question sample, about 1.2 points for a difference). Agreement rises monotonically with λ (89.5, 90.6, 91.0), so the choice rests on the tolerance: a tolerance under about 0.44 would pick λ = 0.5 and one of 0.82 or more λ = 2. On dev, λ = 1 also has the best F1 of the sweep (88.5, 88.6, 88.4; a 0.2-point spread, within noise).

## 6. Backbone: BERT-base, DeBERTa-v3-base and DeBERTa-v3-large

| Backbone | Objective | Dev EM | Dev F1 | Dev agreement | Full held-out F1 | Full held-out agreement | AddSent F1 | AddSent agreement | Training (min) |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| bert-base-uncased | ce | 81.1 | 88.5 | 83.3 | 81.6 | 87.3 | 64.9 | 77.9 | 15 |
| bert-base-uncased | kl λ=1 | 81.4 | 88.6 | 87.9 | 82.0 | 90.1 | 66.1 | 84.1 | 19 |
| deberta-v3-base | ce | 88.3 | 93.9 | 88.0 | 87.1 | 90.8 | 85.1 | 85.5 | 17 |
| deberta-v3-base | kl λ=1 | 88.3 | 93.7 | 91.8 | 87.4 | 93.2 | 84.2 | 90.2 | 33 |
| deberta-v3-large | ce | 90.3 | 95.5 | 89.4 | 88.2 | 90.9 | 86.0 | 84.1 | 43 |
| deberta-v3-large | kl λ=1 | 90.0 | 95.4 | 92.7 | 88.2 | 94.6 | 87.0 | 90.4 | 108 |

Each step up in backbone adds far more accuracy than any objective change (88.5 → 93.9 → 95.5 F1). Consistency training raises agreement on every backbone (+4.6, +3.8, +3.3) at +0.1, -0.2, -0.1 F1, so the effect is not specific to the weak encoder. Distractor robustness is the clearest place the backbones disagree: consistency training raises AddSent F1 for BERT-base (+1.2) and DeBERTa-v3-large (+1.0) but lowers it for DeBERTa-v3-base (-0.9); with one seed per run, F1 moves of about one point are not reliable, whereas the agreement gains are several standard errors (Section 7). The large runs peaked at 16.4 / 25.3 GB of GPU memory; the `kl` job ran on a full, shared H100 rather than a 47 GB MIG slice, so its 108 min is not comparable with the other times. Its reranker behaviour is discussed in Section 8.

## 7. Robustness: paraphrased and adversarial test sets

F1 from the official scorer. Gan & Ng: *orig* = original questions, *para* = their human paraphrases, *agreement* = same answer on both. AddSent / AddOneSent: agreement with back-translated paraphrases of the adversarial questions.

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
| DeBERTa-large ce (baseline) | 94.9 | 93.8 | 92.3 | 97.4 | 85.6 | 86.0 | 84.1 | 89.3 | 85.4 |
| DeBERTa-large kl λ=1 | 94.9 | 93.8 | 95.0 | 97.4 | 90.3 | 87.0 | 90.4 | 90.2 | 91.2 |

**Distractors (Jia & Liang).** The BERT baseline drops from 88.5 F1 on dev to 64.9 on AddSent and 73.9 on AddOneSent. All 6 paraphrase-trained BERT runs sit within 65.7–66.1 F1 on AddSent, so the +1.2 of λ = 1 is mostly the paraphrase data (`aug` alone +1.0); what changes is stability, AddSent agreement 77.9 → 84.1 % (+6.2), i.e. the model is fooled more consistently rather than less often. DeBERTa-v3-base is far more robust (85.1 F1) and consistency training moves it by -0.9; DeBERTa-v3-large gains +1.0 (86.0 → 87.0) with agreement 84.1 → 90.4 % (+6.3, about 7 standard errors over 3,007 pairs).

**Human paraphrases (Gan & Ng, 1,062 questions, 1,061 pairs).** Paraphrasing costs the baseline 2.8 F1 (88.4 → 85.6) and the λ = 1 model 1.4 (87.5 → 86.1); the narrower gap comes from both sides (+0.5 on the paraphrases, -0.9 on the originals). Agreement 89.1 → 91.0 % (+1.9). On the 56-question adversarial subset (1.8 points per question) the baseline scores 58.6 and λ = 1 63.6 F1; indicative only.

## 8. The learned span reranker

Held-out columns: the reranker is trained on 80 % of the held-out articles and early-stopped on the other 20 % (758 questions); the reranked value is the best of 20 epochs on that slice (by F1 for one-hot targets, by EM then F1 for graded) and therefore optimistic. Remaining columns: dev and external sets, Δ = reranked − extractor.

| Run | Targets | Held-out top-1 EM / F1 | Held-out reranked EM / F1 (best epoch) | Last-epoch F1 | Oracle EM (correct span in top-20) | Dev Δ EM | Dev Δ F1 | Dev Δ agreement | AddSent Δ F1 | G&N adv Δ F1 |
| :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| BERT ce (baseline) | one-hot | 72.2 / 83.6 | 72.7 / 84.0 (ep. 1) | 78.9 | 96.0 | +0.3 | -0.2 | +3.0 | -0.1 | +4.8 |
| BERT aug | one-hot | 70.7 / 82.7 | 71.1 / 82.6 (ep. 5) | 79.0 | 97.2 | -0.4 | -0.7 | +6.9 | -0.7 | +25.4 |
| BERT kl λ=0.5 | one-hot | 70.6 / 82.8 | 70.4 / 82.4 (ep. 1) | 79.3 | 97.1 | +0.2 | -0.2 | +1.5 | +0.2 | +4.4 |
| BERT kl λ=1 | one-hot | 70.6 / 82.4 | 71.9 / 83.0 (ep. 8) | 80.3 | 96.7 | -1.4 | -0.9 | +5.0 | -0.1 | +12.0 |
| BERT kl λ=2 | one-hot | 69.9 / 81.8 | 71.1 / 82.5 (ep. 6) | 79.9 | 96.7 | -0.9 | -0.6 | +4.9 | +1.1 | +14.3 |
| BERT ot | one-hot | 70.3 / 82.3 | 71.1 / 82.7 (ep. 11) | 81.9 | 96.7 | -2.5 | -1.4 | +5.0 | +0.0 | +23.2 |
| BERT dual | one-hot | 70.7 / 82.4 | 71.4 / 82.7 (ep. 2) | 80.6 | 97.0 | +0.0 | -0.2 | +3.3 | +0.0 | +10.5 |
| DeBERTa ce (baseline) | one-hot | 76.1 / 87.4 | 76.4 / 88.0 (ep. 1) | 86.1 | 98.2 | -0.9 | -0.6 | +1.7 | +0.2 | +3.9 |
| DeBERTa kl λ=1 | one-hot | 76.5 / 87.6 | 76.3 / 87.5 (ep. 1) | 84.3 | 98.0 | -0.3 | +0.0 | +0.8 | -0.3 | +4.1 |
| DeBERTa-large ce (baseline) | graded | 79.0 / 89.9 | 79.4 / 90.0 (ep. 2) | 89.0 | 98.4 | -0.1 | -0.1 | +4.0 | +0.2 | +7.4 |
| DeBERTa-large kl λ=1 | graded | 78.6 / 89.5 | 79.7 / 89.5 (ep. 5) | 88.1 | 98.9 | +0.0 | -0.1 | +2.4 | +0.6 | +4.5 |

**Reading.** The correct span is among the top-20 candidates for 96.0–98.9 % of held-out questions, but the reranker recovers almost none of that: on dev, F1 never rises (it falls in 10 of 11 runs, unchanged in 1) while agreement rises in 11 of 11; for the selected run the trade is -0.9 F1 for +5.0 agreement. The last-epoch F1 is 0.8–5.1 points below the best epoch, so it overfits within a few epochs. We hypothesise that the paraphrase-score feature drives the trade (it favours spans both phrasings score highly); Section 9 tests this by ablation. Its clearest benefit is on the adversarial paraphrases, where it can lean on the easier original question (63.6 → 75.6 F1 for the selected run; small set). Both system versions are reported.

**Graded targets.** `deb_large_ce` and `deb_large_kl` were trained after the target changed from one-hot (first exact-match candidate, else best F1) to graded relevance (exact match 1, partial overlap 0.3 × F1), so they are not strictly comparable with the other 9 rows. `deb_large_ce`: -0.1 F1 for +4.0 agreement on dev, F1 up on all four robustness sets. `deb_large_kl`: -0.1 F1 for +2.4 agreement on dev, F1 up on all four robustness sets. Among the base-model rerankers only `bert_kl_lam0.5` also gains on all four, mostly by smaller margins. Section 9 retrains the base rerankers with graded targets and separates the two effects.

## 9. Reranker ablation: features, decision rule, confidence gate

On `bert_kl_lam1.0`, `deb_kl`, `deb_large_kl` the reranker was retrained twice on the held-out articles (graded targets, same seed): with the three paraphrase-derived features (*pair*) and without them (*no pair*). Each was then used in two ways: the reranker's argmax replaces the extractor's answer (*argmax*, the current rule), or the two log-probabilities are added with weight α on the reranker (*blend*), optionally behind a *gate* that keeps the extractor's answer unless the reranker prefers another span by more than the margin in nats. Δ = variant − extractor, dev set; FIX / BREAK = answers turned right / wrong (exact match).

| Variant | BERT kl λ=1: Δ EM / Δ F1 / Δ agreement | DeBERTa kl λ=1: Δ EM / Δ F1 / Δ agreement | DeBERTa-large kl λ=1: Δ EM / Δ F1 / Δ agreement |
| :-- | :-- | :-- | :-- |
| A · pair · argmax | +0.1 / -0.2 / +4.9 | -1.4 / -0.8 / +1.6 | +0.2 / +0.0 / +1.5 |
| B · no pair · argmax | +0.3 / -0.2 / +0.9 | -0.4 / -0.4 / -0.1 | +0.3 / +0.1 / +0.5 |
| B · no pair · argmax · gate 0.1 | +0.6 / +0.0 / +0.5 | -0.1 / -0.3 / -0.3 | +0.2 / +0.0 / +0.4 |
| B · no pair · argmax · gate 0.25 | +0.6 / +0.1 / +0.4 | +0.0 / -0.2 / +0.0 | +0.2 / +0.0 / +0.4 |
| C · no pair · blend α=0.25 | +0.3 / +0.1 / +0.4 | +0.2 / +0.1 / +0.1 | +0.1 / +0.0 / +0.2 |
| D · no pair · blend α=0.5 | +0.4 / +0.1 / +0.6 | +0.3 / +0.1 / +0.3 | +0.1 / +0.0 / +0.3 |
| E · no pair · blend α=1 | +0.4 / +0.1 / +0.6 | +0.4 / +0.1 / +0.3 | +0.2 / +0.0 / +0.4 |
| E · no pair · blend α=1 · gate 0.1 | +0.5 / +0.1 / +0.5 | +0.5 / +0.1 / +0.3 | +0.1 / +0.0 / +0.4 |
| E · no pair · blend α=1 · gate 0.25 | +0.5 / +0.2 / +0.5 | +0.3 / +0.1 / +0.2 | +0.2 / +0.0 / +0.3 |

**Held-out selection.** Before looking at dev, the rule "best held-out EM, then F1" over all configurations of both rerankers (758 held-out questions, standard error 1.5–1.7 EM points) would choose the following; the chosen values are early-stopped on the same questions and therefore optimistic:

| Run | Held-out EM / F1, extractor | Chosen configuration | Held-out EM / F1, chosen | Dev Δ EM / Δ F1 / Δ agreement of the choice |
| :-- | :-- | :-- | :-- | :-- |
| BERT kl λ=1 | 70.6 / 82.4 | pair · argmax (= A) | 71.8 / 83.0 | +0.1 / -0.2 / +4.9 |
| DeBERTa kl λ=1 | 76.5 / 87.6 | no pair · argmax (= B) | 77.7 / 88.0 | -0.4 / -0.4 / -0.1 |
| DeBERTa-large kl λ=1 | 78.6 / 89.5 | no pair · argmax (= B) | 79.8 / 89.7 | +0.3 / +0.1 / +0.5 |

Robustness sets for the three main variants (Δ = variant − extractor):

| Run | Variant | AddSent Δ F1 | AddSent Δ agreement | AddOneSent Δ F1 | G&N para Δ F1 | G&N adv Δ F1 |
| :-- | :-- | --: | --: | --: | --: | --: |
| BERT kl λ=1 | A · pair · argmax | -0.1 | +5.5 | +0.0 | +1.1 | +7.9 |
| BERT kl λ=1 | B · no pair · argmax | -0.1 | -0.2 | +0.3 | +0.8 | +0.7 |
| BERT kl λ=1 | E · no pair · blend α=1 | +0.2 | +0.9 | +0.2 | +0.8 | -0.5 |
| DeBERTa kl λ=1 | A · pair · argmax | -0.4 | +2.7 | -0.3 | +0.0 | +7.7 |
| DeBERTa kl λ=1 | B · no pair · argmax | -0.4 | -0.2 | -0.2 | -0.2 | +2.4 |
| DeBERTa kl λ=1 | E · no pair · blend α=1 | -0.1 | -0.1 | -0.1 | +0.4 | +2.4 |
| DeBERTa-large kl λ=1 | A · pair · argmax | +0.6 | +2.1 | +0.6 | +0.4 | +0.9 |
| DeBERTa-large kl λ=1 | B · no pair · argmax | +0.5 | +0.6 | +0.7 | +0.1 | +0.0 |
| DeBERTa-large kl λ=1 | E · no pair · blend α=1 | +0.2 | +0.8 | +0.2 | +0.0 | +0.0 |

**Features.** The paraphrase features carry the agreement gain (+4.9, +1.6, +1.5 agreement points on dev with them, +0.9, -0.1, +0.5 without; order BERT kl λ=1, DeBERTa kl λ=1, DeBERTa-large kl λ=1) and the adversarial-paraphrase gain (+7.9, +7.7, +0.9 F1 with, +0.7, +2.4, +0.0 without). This confirms the hypothesis of Section 8.

**Decision rule.** Blending instead of taking the reranker's argmax (α = 1, no pair features) raises dev EM on all three extractors without lowering F1 (+0.4, +0.4, +0.2 EM, +0.1, +0.1, +0.0 F1; fixes : breaks 94:53, 82:41, 40:24), but by a few tenths of a point and keeping only +0.6, +0.3, +0.4 of the agreement. Gates reduce both fixes and breaks; relative to the ungated rule they shift dev EM, F1 or agreement by at most 0.5 points.

**Targets and noise.** Variant A is the committed reranker's configuration retrained with graded targets. Against the one-hot rerankers of Section 8 the target alone moves the dev trade: BERT kl λ=1 -0.9 F1 / +5.0 agreement → -0.2 / +4.9; DeBERTa kl λ=1 +0.0 F1 / +0.8 agreement → -0.8 / +1.6. For DeBERTa-large kl λ=1 both are graded, so the shift (-0.1 → +0.0 F1, +2.4 → +1.5 agreement) is the run-to-run noise of reranker training; differences of that size in this section mean nothing.

**Selection.** The held-out rule would pick A, B, B for the three runs (table above), 2 of which lose dev F1; 758 questions cannot separate configurations a few tenths apart. No variant is adopted: the pair-feature argmax reranker stays the agreement version and the extractor alone the accuracy version.

## 10. Optimal-transport and span-matching ablation

Replacing the KL penalty by the 1-D Wasserstein penalty (`ot`) or by Wasserstein plus Hungarian span matching (`dual`) gives dev agreement 86.3 % and 86.7 % against 87.9 % for KL, at EM / F1 80.9 / 88.2 and 81.1 / 88.3. Both are within 0.2 points of augmentation-only (-0.2, +0.2) whereas KL adds +1.4: at their default, untuned weights (λ_OT = 0.5, μ = 0.1) these penalties add nothing beyond the paraphrase data. KL remains the method; this is a negative ablation.

## 11. Question-type analysis (dev)

Types from the question's wh-word (*how many/much* also covers *how long / old / far / often / tall / big / large*). Baseline and selected run, extractor only; last column with the reranker.

| Type | n | Baseline EM / F1 | kl λ = 1 EM / F1 | Δ F1 | kl λ = 1 F1 +reranker |
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

Hardest: *other* (74.4), *how* (77.9), *why* (78.3); easiest: *when* (95.3), *who* (91.8). Consistency training changes per-type F1 by at most 1.3 points, so it does not redistribute accuracy across types; the reranker lowers F1 for 9 of 9 types, consistent with Section 8.

## 12. Caveats

- One training seed per configuration. Dev EM/F1 differences under about half a point, the +1.4-point agreement gain of KL over augmentation alone, and everything on the 56-question adversarial-paraphrase set are unconfirmed.
- Agreement is measured only on questions with an accepted paraphrase (8,622 of 10,570 on dev); the 0.85 filter admits some meaning-changing paraphrases, which caps attainable agreement.
- The λ rule's tolerance (0.5) was fixed before the runs but not formally pre-registered, and it is load-bearing (Section 5).
- OT / Hungarian weights were not tuned (Section 10).
- The reranker ablation selects on 758 held-out questions, too few to separate configurations a few tenths of a point apart; its dev differences below about half a point are unconfirmed (Section 9).
- The DeBERTa-v3-large runs use a smaller per-step batch and lower learning rate (Section 2) and graded reranker targets (Section 8), so large-versus-base is not a pure backbone comparison.
- Held-out training articles were used for every decision (checkpoint, λ, reranker); dev and the external sets never were.

## 13. Provenance

```
python visualize.py --best bert_kl_lam1.0        # regenerates figures/ and report/tables/ from runs/
python scripts/make_results_report.py              # regenerates this file and docs/results_digest.json
```
Sources per run: `config.json`, `train_summary.json`, `reranker_summary.json`, `eval_<set>/results.json` and `official_eval*.json`. NUS SoC cluster, H100 NVL; base-model training jobs 1068703–1068711 and re-evaluation jobs 1069429–1069438 (3 Nov 2026, after the Gan & Ng original-question scoring fix in commit 3d62dc6); DeBERTa-v3-large jobs 1072440 (`ce`) and 1072441 (`kl`) on 4 Nov 2026; reranker-ablation jobs 1073460–1073462 on 4 Nov 2026 (`scripts/submit_reranker_variants.sh`, outputs under `runs/<run>/rerank_variants/`). Cluster logs are not part of the repository.
