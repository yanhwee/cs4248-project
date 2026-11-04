### deb_large_kl | addonesent

Extractor only: EM 84.56 / F1 90.22 / agreement 91.08 (n=1787). Candidate ceilings: R@1 EM 84.56, R@3 EM 92.28, R@5 EM 93.84, R@10 EM 95.69, R@20 EM 97.37; best-F1@20 98.19

| Variant | pair_score | mode | alpha | gate | EM | F1 | Agreement | dEM | dF1 | dAgreement | FIX | BREAK | SAME_CORRECT | SAME_WRONG |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| A_pair_reranker | yes | reranker | - | - | 85.62 | 90.84 | 93.29 | +1.06 | +0.62 | +2.21 | 23 (1.29%) | 4 (0.22%) | 1507 | 253 |
| B_nopair_reranker | no | reranker | - | - | 85.62 | 90.87 | 91.35 | +1.06 | +0.65 | +0.27 | 27 (1.51%) | 8 (0.45%) | 1503 | 249 |
| B_nopair_reranker_gate0.1 | no | reranker | - | 0.1 | 85.39 | 90.69 | 91.01 | +0.84 | +0.46 | -0.07 | 18 (1.01%) | 3 (0.17%) | 1508 | 258 |
| B_nopair_reranker_gate0.25 | no | reranker | - | 0.25 | 84.95 | 90.50 | 91.42 | +0.39 | +0.28 | +0.34 | 7 (0.39%) | 0 (0.00%) | 1511 | 269 |
| C_nopair_res0.25 | no | residual | 0.25 | - | 84.78 | 90.36 | 91.55 | +0.22 | +0.14 | +0.47 | 4 (0.22%) | 0 (0.00%) | 1511 | 272 |
| D_nopair_res0.5 | no | residual | 0.5 | - | 84.78 | 90.37 | 91.28 | +0.22 | +0.15 | +0.20 | 5 (0.28%) | 1 (0.06%) | 1510 | 271 |
| E_nopair_res1.0 | no | residual | 1.0 | - | 85.06 | 90.41 | 91.55 | +0.50 | +0.19 | +0.47 | 10 (0.56%) | 1 (0.06%) | 1510 | 266 |
| E_nopair_res1.0_gate0.1 | no | residual | 1.0 | 0.1 | 85.06 | 90.41 | 91.35 | +0.50 | +0.19 | +0.27 | 10 (0.56%) | 1 (0.06%) | 1510 | 266 |
| E_nopair_res1.0_gate0.25 | no | residual | 1.0 | 0.25 | 84.83 | 90.40 | 91.48 | +0.28 | +0.18 | +0.40 | 5 (0.28%) | 0 (0.00%) | 1511 | 271 |
