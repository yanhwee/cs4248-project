### deb_large_kl | gan_dev

Extractor only: EM 88.32 / F1 93.81 / agreement 95.10 (n=1062). Candidate ceilings: R@1 EM 88.32, R@3 EM 96.23, R@5 EM 97.46, R@10 EM 98.59, R@20 EM 98.96; best-F1@20 99.38

| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dEM 95% CI | dF1 | dAgreement | dAgreement 95% CI | Both views correct | FIX | BREAK |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | :-- | --: | --: | :-- | --: | --: | --: |
| A_pair_reranker | yes | reranker | - | - | 89.08 | 94.20 | 96.98 | +0.75 | n/a | +0.40 | +1.89 | n/a | n/a | 16 (1.51%) | 8 (0.75%) |
| B_nopair_reranker | no | reranker | - | - | 88.79 | 93.91 | 95.66 | +0.47 | n/a | +0.11 | +0.57 | n/a | n/a | 13 (1.22%) | 8 (0.75%) |
| B_nopair_reranker_gate0.1 | no | reranker | - | 0.1 | 88.70 | 93.92 | 96.32 | +0.38 | n/a | +0.11 | +1.23 | n/a | n/a | 8 (0.75%) | 4 (0.38%) |
| B_nopair_reranker_gate0.25 | no | reranker | - | 0.25 | 88.61 | 93.89 | 95.76 | +0.28 | n/a | +0.08 | +0.66 | n/a | n/a | 4 (0.38%) | 1 (0.09%) |
| CB_nopair_argmax | no | reranker | - | - | 88.89 | 93.95 | 95.76 | +0.47 | [-0.38, +1.32] | +0.13 | +0.75 | [-0.29, +1.98] | 88.03 | 13 (1.22%) | 8 (0.75%) |
| CB_nopair_res1.0 | no | residual | 1.0 | - | 88.70 | 93.83 | 95.85 | +0.28 | [-0.28, +0.85] | +0.02 | +0.85 | [+0.09, +1.70]* | 88.12 | 6 (0.56%) | 3 (0.28%) |
| CB_pair_argmax_ensemble | yes | reranker | - | - | 89.08 | 94.20 | 96.98 | +0.66 | [-0.28, +1.60] | +0.39 | +1.98 | [+0.85, +3.11]* | 88.50 | 16 (1.51%) | 9 (0.85%) |
| CB_pair_argmax_ensemble / pair-ensemble | yes (extractor's) | ensemble | beta=1.0 | - | 89.08 | 94.38 | 99.43 | +0.66 | [-0.19, +1.41] | +0.56 | +4.43 | [+3.20, +5.75]* | 89.07 | 13 (1.22%) | 6 (0.56%) |
| CR_cw0.5_argmax | no | reranker | - | - | 89.08 | 94.10 | 95.19 | +0.66 | [-0.09, +1.41] | +0.29 | +0.19 | [-0.94, +1.41] | 88.22 | 13 (1.22%) | 6 (0.56%) |
| CR_cw0.5_res1.0 | no | residual | 1.0 | - | 88.51 | 93.64 | 95.66 | +0.09 | [-0.47, +0.66] | -0.18 | +0.66 | [-0.09, +1.41] | 87.94 | 5 (0.47%) | 4 (0.38%) |
| CR_cw1.0_argmax | no | reranker | - | - | 88.51 | 93.79 | 95.76 | +0.09 | [-0.66, +0.94] | -0.03 | +0.75 | [-0.38, +1.79] | 88.03 | 11 (1.04%) | 10 (0.94%) |
| CR_cw1.0_res1.0 | no | residual | 1.0 | - | 88.61 | 93.78 | 95.57 | +0.19 | [-0.28, +0.75] | -0.04 | +0.57 | [-0.19, +1.32] | 87.94 | 5 (0.47%) | 3 (0.28%) |
| CR_cw2.0_argmax | no | reranker | - | - | 87.66 | 93.06 | 95.85 | -0.75 | [-1.69, +0.19] | -0.75 | +0.85 | [-0.09, +1.79] | 87.28 | 8 (0.75%) | 16 (1.51%) |
| CR_cw2.0_res1.0 | no | residual | 1.0 | - | 88.32 | 93.66 | 95.66 | -0.09 | [-0.56, +0.38] | -0.16 | +0.66 | [+0.00, +1.32] | 87.75 | 3 (0.28%) | 4 (0.38%) |
| C_nopair_res0.25 | no | residual | 0.25 | - | 88.61 | 93.84 | 95.57 | +0.28 | n/a | +0.03 | +0.47 | n/a | n/a | 4 (0.38%) | 1 (0.09%) |
| D_nopair_res0.5 | no | residual | 0.5 | - | 88.61 | 93.84 | 95.66 | +0.28 | n/a | +0.03 | +0.57 | n/a | n/a | 4 (0.38%) | 1 (0.09%) |
| E_nopair_res1.0 | no | residual | 1.0 | - | 88.70 | 93.83 | 95.85 | +0.38 | n/a | +0.03 | +0.75 | n/a | n/a | 6 (0.56%) | 2 (0.19%) |
| E_nopair_res1.0_gate0.1 | no | residual | 1.0 | 0.1 | 88.61 | 93.83 | 95.85 | +0.28 | n/a | +0.02 | +0.75 | n/a | n/a | 5 (0.47%) | 2 (0.19%) |
| E_nopair_res1.0_gate0.25 | no | residual | 1.0 | 0.25 | 88.51 | 93.79 | 95.48 | +0.19 | n/a | -0.02 | +0.38 | n/a | n/a | 2 (0.19%) | 0 (0.00%) |

95% paired-bootstrap CIs (resampling questions; agreement: pairs); `*` = interval excludes 0. Both views correct = share of pairs where the question's and the paraphrase's answers are both exact matches.
