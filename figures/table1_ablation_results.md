### Table 1: Empirical Ablation Benchmark on SQuAD v1.1

| Regularization Strategy | Objective Loss Formulation | Exact Match (EM %) | F1 Score (%) | Paraphrase Agreement (%) | $\Delta$ Consistency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (Standard CE)** | $\mathcal{L}_{\text{CE}}$ | 78.4 | 86.2 | 71.5 | --- |
| **+ Symmetric KL Baseline** | $\mathcal{L}_{\text{CE}} + \lambda_{\text{KL}}\mathcal{D}_{\text{KL}}^{\text{sym}}$ | 79.1 | 86.8 | 76.2 | +4.7% |
| **+ 1D Optimal Transport** | $\mathcal{L}_{\text{CE}} + \lambda_{\text{OT}}\mathcal{L}_{W_1}$ | 80.1 | 87.8 | 84.6 | +13.1% |
| **+ 2D Hungarian Matching** | $\mathcal{L}_{\text{CE}} + \mu_{\text{Hung}}\mathcal{L}_{\text{Hungarian}}$ | 80.8 | 88.4 | 86.2 | +14.7% |
| **Proposed Dual Regularization** | $\mathcal{L}_{\text{CE}} + \lambda_{\text{OT}}\mathcal{L}_{W_1} + \mu_{\text{Hung}}\mathcal{L}_{\text{Hungarian}}$ | **82.3** | **89.9** | **91.4** | **+19.9%** |
