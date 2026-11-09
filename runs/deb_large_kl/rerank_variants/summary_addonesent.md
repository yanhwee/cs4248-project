### deb_large_kl | addonesent

Extractor only: EM 84.56 / F1 90.22 / agreement 91.08 (n=1787). Candidate ceilings: R@1 EM 84.56, R@3 EM 92.28, R@5 EM 93.84, R@10 EM 95.69, R@20 EM 97.37; best-F1@20 98.19

| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dEM 95% CI | dF1 | dAgreement | dAgreement 95% CI | Both views correct | FIX | BREAK |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | :-- | --: | --: | :-- | --: | --: | --: |
| A_pair_reranker | yes | reranker | - | - | 85.62 | 90.84 | 93.29 | +1.06 | n/a | +0.62 | +2.21 | n/a | n/a | 23 (1.29%) | 4 (0.22%) |
| B_nopair_reranker | no | reranker | - | - | 85.62 | 90.87 | 91.35 | +1.06 | n/a | +0.65 | +0.27 | n/a | n/a | 27 (1.51%) | 8 (0.45%) |
| B_nopair_reranker_gate0.1 | no | reranker | - | 0.1 | 85.39 | 90.69 | 91.01 | +0.84 | n/a | +0.46 | -0.07 | n/a | n/a | 18 (1.01%) | 3 (0.17%) |
| B_nopair_reranker_gate0.25 | no | reranker | - | 0.25 | 84.95 | 90.50 | 91.42 | +0.39 | n/a | +0.28 | +0.34 | n/a | n/a | 7 (0.39%) | 0 (0.00%) |
| CB_nopair_argmax | no | reranker | - | - | 85.62 | 90.87 | 91.35 | +1.06 | [+0.45, +1.73]* | +0.67 | +0.13 | [-0.80, +1.07] | 82.29 | 27 (1.51%) | 8 (0.45%) |
| CB_nopair_res1.0 | no | residual | 1.0 | - | 85.06 | 90.41 | 91.68 | +0.50 | [+0.17, +0.84]* | +0.21 | +0.47 | [-0.20, +1.14] | 82.03 | 10 (0.56%) | 1 (0.06%) |
| CB_pair_argmax_ensemble | yes | reranker | - | - | 85.62 | 90.84 | 93.29 | +1.06 | [+0.50, +1.62]* | +0.64 | +2.08 | [+1.14, +3.09]* | 82.96 | 23 (1.29%) | 4 (0.22%) |
| CB_pair_argmax_ensemble / pair-ensemble | yes (extractor's) | ensemble | beta=1.0 | - | 84.56 | 90.32 | 99.13 | +0.00 | [-0.62, +0.62] | +0.12 | +7.91 | [+6.51, +9.32]* | 84.37 | 16 (0.90%) | 16 (0.90%) |
| CR_cw0.5_argmax | no | reranker | - | - | 85.34 | 90.52 | 91.28 | +0.78 | [+0.11, +1.51]* | +0.32 | +0.07 | [-1.01, +1.07] | 82.36 | 28 (1.57%) | 14 (0.78%) |
| CR_cw0.5_res1.0 | no | residual | 1.0 | - | 85.06 | 90.49 | 91.48 | +0.50 | [+0.17, +0.90]* | +0.29 | +0.27 | [-0.40, +0.87] | 81.96 | 10 (0.56%) | 1 (0.06%) |
| CR_cw1.0_argmax | no | reranker | - | - | 85.45 | 90.63 | 91.35 | +0.90 | [+0.28, +1.51]* | +0.43 | +0.13 | [-0.87, +1.07] | 82.43 | 25 (1.40%) | 9 (0.50%) |
| CR_cw1.0_res1.0 | no | residual | 1.0 | - | 85.11 | 90.46 | 91.55 | +0.56 | [+0.22, +0.95]* | +0.26 | +0.34 | [-0.34, +1.01] | 82.03 | 11 (0.62%) | 1 (0.06%) |
| CR_cw2.0_argmax | no | reranker | - | - | 84.50 | 89.47 | 91.01 | -0.06 | [-0.84, +0.78] | -0.73 | -0.20 | [-1.34, +0.87] | 81.49 | 28 (1.57%) | 29 (1.62%) |
| CR_cw2.0_res1.0 | no | residual | 1.0 | - | 85.00 | 90.44 | 91.62 | +0.45 | [+0.17, +0.78]* | +0.24 | +0.40 | [-0.20, +1.01] | 81.89 | 8 (0.45%) | 0 (0.00%) |
| C_nopair_res0.25 | no | residual | 0.25 | - | 84.78 | 90.36 | 91.55 | +0.22 | n/a | +0.14 | +0.47 | n/a | n/a | 4 (0.22%) | 0 (0.00%) |
| D_nopair_res0.5 | no | residual | 0.5 | - | 84.78 | 90.37 | 91.28 | +0.22 | n/a | +0.15 | +0.20 | n/a | n/a | 5 (0.28%) | 1 (0.06%) |
| E_nopair_res1.0 | no | residual | 1.0 | - | 85.06 | 90.41 | 91.55 | +0.50 | n/a | +0.19 | +0.47 | n/a | n/a | 10 (0.56%) | 1 (0.06%) |
| E_nopair_res1.0_gate0.1 | no | residual | 1.0 | 0.1 | 85.06 | 90.41 | 91.35 | +0.50 | n/a | +0.19 | +0.27 | n/a | n/a | 10 (0.56%) | 1 (0.06%) |
| E_nopair_res1.0_gate0.25 | no | residual | 1.0 | 0.25 | 84.83 | 90.40 | 91.48 | +0.28 | n/a | +0.18 | +0.40 | n/a | n/a | 5 (0.28%) | 0 (0.00%) |

95% paired-bootstrap CIs (resampling questions; agreement: pairs); `*` = interval excludes 0. Both views correct = share of pairs where the question's and the paraphrase's answers are both exact matches.
