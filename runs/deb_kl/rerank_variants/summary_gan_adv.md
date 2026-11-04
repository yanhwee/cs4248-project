### deb_kl | gan_adv

Extractor only: EM 76.79 / F1 82.85 / agreement 82.14 (n=56). Candidate ceilings: R@1 EM 76.79, R@3 EM 91.07, R@5 EM 92.86, R@10 EM 98.21, R@20 EM 98.21; best-F1@20 99.40

| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dF1 | dAgreement | FIX | BREAK | SAME_CORRECT | SAME_WRONG |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| A_pair_reranker | yes | reranker | - | - | 85.71 | 90.62 | 96.43 | +8.93 | +7.77 | +14.29 | 5 (8.93%) | 0 (0.00%) | 43 | 8 |
| B_nopair_reranker | no | reranker | - | - | 80.36 | 85.26 | 89.29 | +3.57 | +2.41 | +7.14 | 2 (3.57%) | 0 (0.00%) | 43 | 11 |
| B_nopair_reranker_gate0.1 | no | reranker | - | 0.1 | 78.57 | 83.48 | 85.71 | +1.79 | +0.62 | +3.57 | 1 (1.79%) | 0 (0.00%) | 43 | 12 |
| B_nopair_reranker_gate0.25 | no | reranker | - | 0.25 | 78.57 | 83.48 | 85.71 | +1.79 | +0.62 | +3.57 | 1 (1.79%) | 0 (0.00%) | 43 | 12 |
| C_nopair_res0.25 | no | residual | 0.25 | - | 80.36 | 85.26 | 87.50 | +3.57 | +2.41 | +5.36 | 2 (3.57%) | 0 (0.00%) | 43 | 11 |
| D_nopair_res0.5 | no | residual | 0.5 | - | 80.36 | 85.26 | 87.50 | +3.57 | +2.41 | +5.36 | 2 (3.57%) | 0 (0.00%) | 43 | 11 |
| E_nopair_res1.0 | no | residual | 1.0 | - | 80.36 | 85.26 | 87.50 | +3.57 | +2.41 | +5.36 | 2 (3.57%) | 0 (0.00%) | 43 | 11 |
| E_nopair_res1.0_gate0.1 | no | residual | 1.0 | 0.1 | 80.36 | 85.26 | 87.50 | +3.57 | +2.41 | +5.36 | 2 (3.57%) | 0 (0.00%) | 43 | 11 |
| E_nopair_res1.0_gate0.25 | no | residual | 1.0 | 0.25 | 80.36 | 85.26 | 87.50 | +3.57 | +2.41 | +5.36 | 2 (3.57%) | 0 (0.00%) | 43 | 11 |
