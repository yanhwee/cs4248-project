### deb_large_kl | addsent

Extractor only: EM 81.18 / F1 86.98 / agreement 90.32 (n=3560). Candidate ceilings: R@1 EM 81.18, R@3 EM 89.58, R@5 EM 91.49, R@10 EM 93.96, R@20 EM 95.96; best-F1@20 97.04

| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dEM 95% CI | dF1 | dAgreement | dAgreement 95% CI | Both views correct | FIX | BREAK |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | :-- | --: | --: | :-- | --: | --: | --: |
| A_pair_reranker | yes | reranker | - | - | 82.16 | 87.59 | 92.35 | +0.98 | n/a | +0.61 | +2.03 | n/a | n/a | 55 (1.54%) | 20 (0.56%) |
| B_nopair_reranker | no | reranker | - | - | 82.11 | 87.48 | 90.92 | +0.93 | n/a | +0.50 | +0.60 | n/a | n/a | 61 (1.71%) | 28 (0.79%) |
| B_nopair_reranker_gate0.1 | no | reranker | - | 0.1 | 82.05 | 87.41 | 90.72 | +0.87 | n/a | +0.43 | +0.40 | n/a | n/a | 47 (1.32%) | 16 (0.45%) |
| B_nopair_reranker_gate0.25 | no | reranker | - | 0.25 | 81.80 | 87.36 | 90.59 | +0.62 | n/a | +0.38 | +0.27 | n/a | n/a | 25 (0.70%) | 3 (0.08%) |
| CB_nopair_argmax | no | reranker | - | - | 82.13 | 87.48 | 90.92 | +0.96 | [+0.42, +1.49]* | +0.52 | +0.57 | [-0.20, +1.30] | 78.78 | 62 (1.74%) | 28 (0.79%) |
| CB_nopair_res1.0 | no | residual | 1.0 | - | 81.83 | 87.26 | 91.09 | +0.65 | [+0.31, +0.98]* | +0.29 | +0.73 | [+0.20, +1.33]* | 78.72 | 31 (0.87%) | 8 (0.22%) |
| CB_pair_argmax_ensemble | yes | reranker | - | - | 82.13 | 87.59 | 92.45 | +0.96 | [+0.45, +1.43]* | +0.62 | +2.10 | [+1.33, +2.89]* | 79.35 | 55 (1.54%) | 21 (0.59%) |
| CB_pair_argmax_ensemble / pair-ensemble | yes (extractor's) | ensemble | beta=1.0 | - | 81.18 | 86.98 | 99.33 | +0.00 | [-0.48, +0.48] | +0.01 | +8.98 | [+7.95, +10.08]* | 81.41 | 38 (1.07%) | 38 (1.07%) |
| CR_cw0.5_argmax | no | reranker | - | - | 81.80 | 87.26 | 90.95 | +0.62 | [+0.08, +1.15]* | +0.29 | +0.60 | [-0.20, +1.36] | 78.98 | 60 (1.69%) | 38 (1.07%) |
| CR_cw0.5_res1.0 | no | residual | 1.0 | - | 81.91 | 87.29 | 91.09 | +0.73 | [+0.42, +1.04]* | +0.32 | +0.73 | [+0.17, +1.30]* | 78.88 | 31 (0.87%) | 5 (0.14%) |
| CR_cw1.0_argmax | no | reranker | - | - | 81.91 | 87.39 | 91.02 | +0.73 | [+0.22, +1.26]* | +0.42 | +0.67 | [-0.10, +1.43] | 78.78 | 56 (1.57%) | 30 (0.84%) |
| CR_cw1.0_res1.0 | no | residual | 1.0 | - | 81.91 | 87.28 | 91.05 | +0.73 | [+0.42, +1.07]* | +0.31 | +0.70 | [+0.13, +1.30]* | 78.78 | 31 (0.87%) | 5 (0.14%) |
| CR_cw2.0_argmax | no | reranker | - | - | 80.42 | 85.47 | 90.19 | -0.76 | [-1.40, -0.11]* | -1.50 | -0.17 | [-1.03, +0.70] | 77.02 | 61 (1.71%) | 88 (2.47%) |
| CR_cw2.0_res1.0 | no | residual | 1.0 | - | 81.66 | 87.16 | 91.05 | +0.48 | [+0.17, +0.79]* | +0.19 | +0.70 | [+0.17, +1.23]* | 78.62 | 24 (0.67%) | 7 (0.20%) |
| C_nopair_res0.25 | no | residual | 0.25 | - | 81.49 | 87.17 | 90.92 | +0.31 | n/a | +0.19 | +0.60 | n/a | n/a | 12 (0.34%) | 1 (0.03%) |
| D_nopair_res0.5 | no | residual | 0.5 | - | 81.54 | 87.20 | 90.75 | +0.37 | n/a | +0.22 | +0.43 | n/a | n/a | 17 (0.48%) | 4 (0.11%) |
| E_nopair_res1.0 | no | residual | 1.0 | - | 81.83 | 87.24 | 91.12 | +0.65 | n/a | +0.26 | +0.80 | n/a | n/a | 31 (0.87%) | 8 (0.22%) |
| E_nopair_res1.0_gate0.1 | no | residual | 1.0 | 0.1 | 81.85 | 87.25 | 90.85 | +0.67 | n/a | +0.27 | +0.53 | n/a | n/a | 30 (0.84%) | 6 (0.17%) |
| E_nopair_res1.0_gate0.25 | no | residual | 1.0 | 0.25 | 81.71 | 87.30 | 90.59 | +0.53 | n/a | +0.32 | +0.27 | n/a | n/a | 19 (0.53%) | 0 (0.00%) |

95% paired-bootstrap CIs (resampling questions; agreement: pairs); `*` = interval excludes 0. Both views correct = share of pairs where the question's and the paraphrase's answers are both exact matches.
