### deb_large_kl | gan_adv

Extractor only: EM 83.93 / F1 90.29 / agreement 91.07 (n=56). Candidate ceilings: R@1 EM 83.93, R@3 EM 91.07, R@5 EM 92.86, R@10 EM 94.64, R@20 EM 96.43; best-F1@20 97.62

| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dF1 | dAgreement | FIX | BREAK | SAME_CORRECT | SAME_WRONG |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| A_pair_reranker | yes | reranker | - | - | 85.71 | 91.18 | 92.86 | +1.79 | +0.89 | +1.79 | 1 (1.79%) | 0 (0.00%) | 47 | 8 |
| B_nopair_reranker | no | reranker | - | - | 83.93 | 90.29 | 91.07 | +0.00 | +0.00 | +0.00 | 0 (0.00%) | 0 (0.00%) | 47 | 9 |
| B_nopair_reranker_gate0.1 | no | reranker | - | 0.1 | 83.93 | 90.29 | 91.07 | +0.00 | +0.00 | +0.00 | 0 (0.00%) | 0 (0.00%) | 47 | 9 |
| B_nopair_reranker_gate0.25 | no | reranker | - | 0.25 | 83.93 | 90.29 | 91.07 | +0.00 | +0.00 | +0.00 | 0 (0.00%) | 0 (0.00%) | 47 | 9 |
| C_nopair_res0.25 | no | residual | 0.25 | - | 83.93 | 90.29 | 91.07 | +0.00 | +0.00 | +0.00 | 0 (0.00%) | 0 (0.00%) | 47 | 9 |
| D_nopair_res0.5 | no | residual | 0.5 | - | 83.93 | 90.29 | 91.07 | +0.00 | +0.00 | +0.00 | 0 (0.00%) | 0 (0.00%) | 47 | 9 |
| E_nopair_res1.0 | no | residual | 1.0 | - | 83.93 | 90.29 | 91.07 | +0.00 | +0.00 | +0.00 | 0 (0.00%) | 0 (0.00%) | 47 | 9 |
| E_nopair_res1.0_gate0.1 | no | residual | 1.0 | 0.1 | 83.93 | 90.29 | 91.07 | +0.00 | +0.00 | +0.00 | 0 (0.00%) | 0 (0.00%) | 47 | 9 |
| E_nopair_res1.0_gate0.25 | no | residual | 1.0 | 0.25 | 83.93 | 90.29 | 91.07 | +0.00 | +0.00 | +0.00 | 0 (0.00%) | 0 (0.00%) | 47 | 9 |
