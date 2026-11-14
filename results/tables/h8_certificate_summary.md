# Hypothesis H8 Confirmatory Verdict: Real-Shot Budget Certificate

- **Hypothesis H8 Verdict:** **`FALSIFIED`**
- **Calibration Fidelity (Beta-Binomial 99% CI Coverage):** **`130/130`** cells (**`100.0%`**, required $\\ge 90\%$)
- **Validity Guarantee (Mean FPR $\\le \\alpha + 2\\text{SE}$):** **`124/130`** cells (**`95.4%`**, required $100\%$)
- **Total Evaluated Cells:** **`130`** across 13 models, 2 datasets (`fakeavceleb`, `celebdf`), and 5 sweet-spot $(\\alpha, n)$ configurations ($R=1,000$ draws per cell).

## Aggregate Observed vs Predicted Failure Rates by Configuration

| Dataset | Budget $\\alpha$ | Reals $n$ | Held-out $m$ | Predicted $P(\\text{FPR} > 2\\alpha)$ | Mean Observed $P(\\text{FPR} > 2\\alpha)$ | Mean Empirical FPR | Mean Retained TPR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `fakeavceleb` | `0.05` | `19` | `480` | **`0.1350`** | **`0.1435`** | `0.0514` | `93.3%` |
| `fakeavceleb` | `0.05` | `39` | `460` | **`0.0906`** | **`0.0898`** | `0.0503` | `96.4%` |
| `fakeavceleb` | `0.05` | `59` | `440` | **`0.0629`** | **`0.0655`** | `0.0505` | `97.7%` |
| `fakeavceleb` | `0.10` | ` 9` | `490` | **`0.1342`** | **`0.1389`** | `0.1022` | `94.4%` |
| `fakeavceleb` | `0.10` | `29` | `470` | **`0.0548`** | **`0.0552`** | `0.1012` | `98.4%` |
| `celebdf` | `0.05` | `19` | `102` | **`0.1395`** | **`0.1298`** | `0.0489` | `85.6%` |
| `celebdf` | `0.05` | `39` | ` 82` | **`0.1072`** | **`0.1068`** | `0.0504` | `90.9%` |
| `celebdf` | `0.05` | `59` | ` 62` | **`0.0939`** | **`0.0929`** | `0.0503` | `91.7%` |
| `celebdf` | `0.10` | ` 9` | `112` | **`0.1393`** | **`0.1362`** | `0.0983` | `88.1%` |
| `celebdf` | `0.10` | `29` | ` 92` | **`0.0707`** | **`0.0742`** | `0.0993` | `95.2%` |
