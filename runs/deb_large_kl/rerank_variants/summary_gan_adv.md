### deb_large_kl | gan_adv

Extractor only: EM 83.93 / F1 90.29 / agreement 91.07 (n=56). Candidate ceilings: R@1 EM 83.93, R@3 EM 91.07, R@5 EM 92.86, R@10 EM 94.64, R@20 EM 96.43; best-F1@20 97.62

| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dEM 95% CI | dF1 | dAgreement | dAgreement 95% CI | Both views correct | FIX | BREAK |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | :-- | --: | --: | :-- | --: | --: | --: |
| A_pair_reranker | yes | reranker | - | - | 85.71 | 91.18 | 92.86 | +1.79 | n/a | +0.89 | +1.79 | n/a | n/a | 1 (1.79%) | 0 (0.00%) |
| B_nopair_reranker | no | reranker | - | - | 83.93 | 90.29 | 91.07 | +0.00 | n/a | +0.00 | +0.00 | n/a | n/a | 0 (0.00%) | 0 (0.00%) |
| B_nopair_reranker_gate0.1 | no | reranker | - | 0.1 | 83.93 | 90.29 | 91.07 | +0.00 | n/a | +0.00 | +0.00 | n/a | n/a | 0 (0.00%) | 0 (0.00%) |
| B_nopair_reranker_gate0.25 | no | reranker | - | 0.25 | 83.93 | 90.29 | 91.07 | +0.00 | n/a | +0.00 | +0.00 | n/a | n/a | 0 (0.00%) | 0 (0.00%) |
| CB_nopair_argmax | no | reranker | - | - | 83.93 | 90.29 | 91.07 | +0.00 | [+0.00, +0.00] | +0.00 | +0.00 | [+0.00, +0.00] | 83.93 | 0 (0.00%) | 0 (0.00%) |
| CB_nopair_res1.0 | no | residual | 1.0 | - | 83.93 | 90.29 | 91.07 | +0.00 | [+0.00, +0.00] | +0.00 | +0.00 | [+0.00, +0.00] | 83.93 | 0 (0.00%) | 0 (0.00%) |
| CB_pair_argmax_ensemble | yes | reranker | - | - | 85.71 | 91.18 | 92.86 | +1.79 | [+0.00, +5.36] | +0.89 | +1.79 | [+0.00, +5.36] | 85.71 | 1 (1.79%) | 0 (0.00%) |
| CB_pair_argmax_ensemble / pair-ensemble | yes (extractor's) | ensemble | beta=1.0 | - | 89.29 | 94.20 | 96.43 | +5.36 | [+0.00, +12.50] | +3.91 | +5.36 | [+0.00, +12.50] | 89.29 | 3 (5.36%) | 0 (0.00%) |
| CR_cw0.5_argmax | no | reranker | - | - | 85.71 | 90.88 | 89.29 | +1.79 | [+0.00, +5.36] | +0.60 | -1.79 | [-5.36, +0.00] | 83.93 | 1 (1.79%) | 0 (0.00%) |
| CR_cw0.5_res1.0 | no | residual | 1.0 | - | 83.93 | 90.29 | 91.07 | +0.00 | [+0.00, +0.00] | +0.00 | +0.00 | [+0.00, +0.00] | 83.93 | 0 (0.00%) | 0 (0.00%) |
| CR_cw1.0_argmax | no | reranker | - | - | 83.93 | 90.29 | 91.07 | +0.00 | [+0.00, +0.00] | +0.00 | +0.00 | [+0.00, +0.00] | 83.93 | 0 (0.00%) | 0 (0.00%) |
| CR_cw1.0_res1.0 | no | residual | 1.0 | - | 83.93 | 90.29 | 91.07 | +0.00 | [+0.00, +0.00] | +0.00 | +0.00 | [+0.00, +0.00] | 83.93 | 0 (0.00%) | 0 (0.00%) |
| CR_cw2.0_argmax | no | reranker | - | - | 78.57 | 84.93 | 85.71 | -5.36 | [-12.50, +0.00] | -5.36 | -5.36 | [-12.50, +0.00] | 78.57 | 0 (0.00%) | 3 (5.36%) |
| CR_cw2.0_res1.0 | no | residual | 1.0 | - | 83.93 | 90.29 | 91.07 | +0.00 | [+0.00, +0.00] | +0.00 | +0.00 | [+0.00, +0.00] | 83.93 | 0 (0.00%) | 0 (0.00%) |
| C_nopair_res0.25 | no | residual | 0.25 | - | 83.93 | 90.29 | 91.07 | +0.00 | n/a | +0.00 | +0.00 | n/a | n/a | 0 (0.00%) | 0 (0.00%) |
| D_nopair_res0.5 | no | residual | 0.5 | - | 83.93 | 90.29 | 91.07 | +0.00 | n/a | +0.00 | +0.00 | n/a | n/a | 0 (0.00%) | 0 (0.00%) |
| E_nopair_res1.0 | no | residual | 1.0 | - | 83.93 | 90.29 | 91.07 | +0.00 | n/a | +0.00 | +0.00 | n/a | n/a | 0 (0.00%) | 0 (0.00%) |
| E_nopair_res1.0_gate0.1 | no | residual | 1.0 | 0.1 | 83.93 | 90.29 | 91.07 | +0.00 | n/a | +0.00 | +0.00 | n/a | n/a | 0 (0.00%) | 0 (0.00%) |
| E_nopair_res1.0_gate0.25 | no | residual | 1.0 | 0.25 | 83.93 | 90.29 | 91.07 | +0.00 | n/a | +0.00 | +0.00 | n/a | n/a | 0 (0.00%) | 0 (0.00%) |

95% paired-bootstrap CIs (resampling questions; agreement: pairs); `*` = interval excludes 0. Both views correct = share of pairs where the question's and the paraphrase's answers are both exact matches.
