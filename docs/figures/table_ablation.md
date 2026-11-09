| Run | Model | Objective | Val F1 | Val Agreement | EM | F1 | Agreement | EM +reranker | F1 +reranker | Agreement +reranker |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| bert_aug | bert-base-uncased | CE + aug. | 81.5 | 88.7 | 81.0 | 88.2 | 86.5 | 80.6 | 87.5 | 93.4 |
| bert_ce | bert-base-uncased | CE | 82.0 | 86.9 | 81.1 | 88.5 | 83.3 | 81.4 | 88.3 | 86.3 |
| bert_dual | bert-base-uncased | CE + aug. + OT + Hung. | 81.8 | 89.4 | 81.1 | 88.3 | 86.7 | 81.1 | 88.1 | 90.0 |
| bert_kl_lam0.5 | bert-base-uncased | CE + aug. + KL | 82.5 | 89.5 | 81.1 | 88.5 | 87.6 | 81.3 | 88.3 | 89.1 |
| bert_kl_lam1.0 | bert-base-uncased | CE + aug. + KL | 82.1 | 90.6 | 81.4 | 88.6 | 87.9 | 80.0 | 87.7 | 92.9 |
| bert_kl_lam2.0 | bert-base-uncased | CE + aug. + KL | 81.7 | 91.0 | 81.3 | 88.4 | 88.8 | 80.4 | 87.8 | 93.7 |
| bert_ot | bert-base-uncased | CE + aug. + OT | 81.7 | 88.1 | 80.9 | 88.2 | 86.3 | 78.4 | 86.8 | 91.3 |
| deb_ce | deberta-v3-base | CE | 87.3 | 90.8 | 88.3 | 93.9 | 88.0 | 87.4 | 93.3 | 89.7 |
| deb_kl | deberta-v3-base | CE + aug. + KL | 88.2 | 93.6 | 88.3 | 93.7 | 91.8 | 88.0 | 93.7 | 92.6 |
| deb_large_ce | deberta-v3-large | CE | 88.6 | 90.7 | 90.3 | 95.5 | 89.4 | 90.2 | 95.4 | 93.4 |
| deb_large_kl | deberta-v3-large | CE + aug. + KL | 88.8 | 95.4 | 90.0 | 95.4 | 92.7 | 90.0 | 95.3 | 95.1 |
