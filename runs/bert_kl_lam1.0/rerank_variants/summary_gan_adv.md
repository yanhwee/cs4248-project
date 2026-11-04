### bert_kl_lam1.0 | gan_adv

Extractor only: EM 58.93 / F1 63.64 / agreement 62.50 (n=56). Candidate ceilings: R@1 EM 58.93, R@3 EM 78.57, R@5 EM 82.14, R@10 EM 91.07, R@20 EM 92.86; best-F1@20 93.76

| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dF1 | dAgreement | FIX | BREAK | SAME_CORRECT | SAME_WRONG |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| A_pair_reranker | yes | reranker | - | - | 64.29 | 71.51 | 78.57 | +5.36 | +7.87 | +16.07 | 3 (5.36%) | 0 (0.00%) | 33 | 20 |
| B_nopair_reranker | no | reranker | - | - | 58.93 | 64.32 | 67.86 | +0.00 | +0.68 | +5.36 | 0 (0.00%) | 0 (0.00%) | 33 | 23 |
| B_nopair_reranker_gate0.1 | no | reranker | - | 0.1 | 58.93 | 64.32 | 67.86 | +0.00 | +0.68 | +5.36 | 0 (0.00%) | 0 (0.00%) | 33 | 23 |
| B_nopair_reranker_gate0.25 | no | reranker | - | 0.25 | 58.93 | 64.32 | 67.86 | +0.00 | +0.68 | +5.36 | 0 (0.00%) | 0 (0.00%) | 33 | 23 |
| C_nopair_res0.25 | no | residual | 0.25 | - | 58.93 | 63.13 | 66.07 | +0.00 | -0.51 | +3.57 | 0 (0.00%) | 0 (0.00%) | 33 | 23 |
| D_nopair_res0.5 | no | residual | 0.5 | - | 58.93 | 63.13 | 67.86 | +0.00 | -0.51 | +5.36 | 0 (0.00%) | 0 (0.00%) | 33 | 23 |
| E_nopair_res1.0 | no | residual | 1.0 | - | 58.93 | 63.13 | 66.07 | +0.00 | -0.51 | +3.57 | 0 (0.00%) | 0 (0.00%) | 33 | 23 |
| E_nopair_res1.0_gate0.1 | no | residual | 1.0 | 0.1 | 58.93 | 63.13 | 66.07 | +0.00 | -0.51 | +3.57 | 0 (0.00%) | 0 (0.00%) | 33 | 23 |
| E_nopair_res1.0_gate0.25 | no | residual | 1.0 | 0.25 | 58.93 | 63.13 | 66.07 | +0.00 | -0.51 | +3.57 | 0 (0.00%) | 0 (0.00%) | 33 | 23 |
