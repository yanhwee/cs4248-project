### deb_large_kl | gan_dev

Extractor only: EM 88.32 / F1 93.81 / agreement 95.10 (n=1062). Candidate ceilings: R@1 EM 88.32, R@3 EM 96.23, R@5 EM 97.46, R@10 EM 98.59, R@20 EM 98.96; best-F1@20 99.38

| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dF1 | dAgreement | FIX | BREAK | SAME_CORRECT | SAME_WRONG |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| A_pair_reranker | yes | reranker | - | - | 89.08 | 94.20 | 96.98 | +0.75 | +0.40 | +1.89 | 16 (1.51%) | 8 (0.75%) | 930 | 108 |
| B_nopair_reranker | no | reranker | - | - | 88.79 | 93.91 | 95.66 | +0.47 | +0.11 | +0.57 | 13 (1.22%) | 8 (0.75%) | 930 | 111 |
| B_nopair_reranker_gate0.1 | no | reranker | - | 0.1 | 88.70 | 93.92 | 96.32 | +0.38 | +0.11 | +1.23 | 8 (0.75%) | 4 (0.38%) | 934 | 116 |
| B_nopair_reranker_gate0.25 | no | reranker | - | 0.25 | 88.61 | 93.89 | 95.76 | +0.28 | +0.08 | +0.66 | 4 (0.38%) | 1 (0.09%) | 937 | 120 |
| C_nopair_res0.25 | no | residual | 0.25 | - | 88.61 | 93.84 | 95.57 | +0.28 | +0.03 | +0.47 | 4 (0.38%) | 1 (0.09%) | 937 | 120 |
| D_nopair_res0.5 | no | residual | 0.5 | - | 88.61 | 93.84 | 95.66 | +0.28 | +0.03 | +0.57 | 4 (0.38%) | 1 (0.09%) | 937 | 120 |
| E_nopair_res1.0 | no | residual | 1.0 | - | 88.70 | 93.83 | 95.85 | +0.38 | +0.03 | +0.75 | 6 (0.56%) | 2 (0.19%) | 936 | 118 |
| E_nopair_res1.0_gate0.1 | no | residual | 1.0 | 0.1 | 88.61 | 93.83 | 95.85 | +0.28 | +0.02 | +0.75 | 5 (0.47%) | 2 (0.19%) | 936 | 119 |
| E_nopair_res1.0_gate0.25 | no | residual | 1.0 | 0.25 | 88.51 | 93.79 | 95.48 | +0.19 | -0.02 | +0.38 | 2 (0.19%) | 0 (0.00%) | 938 | 122 |
