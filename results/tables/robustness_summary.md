# Robustness Perturbation Benchmark: Celeb-DF-v2

*Checkpoint: `outputs/checkpoints_smsdt_full/seed1/best_ema.pt` (Clean Baseline AUC: 0.8884)*

| Perturbation | Severity / Quality | AUC | AUC Drop |
| :--- | :--- | :---: | :---: |
| Clean Baseline | None | 0.8884 | +0.0000 |
| JPEG Quality | 90 | 0.8972 | -0.0088 |
| JPEG Quality | 70 | 0.8829 | +0.0055 |
| JPEG Quality | 50 | 0.8482 | +0.0402 |
| JPEG Quality | 30 | 0.8048 | +0.0836 |
| Gaussian Blur (sigma) | 0.5 | 0.8920 | -0.0035 |
| Gaussian Blur (sigma) | 1.0 | 0.8732 | +0.0152 |
| Gaussian Blur (sigma) | 2.0 | 0.7987 | +0.0898 |
| Downscale Factor | 0.75 | 0.8907 | -0.0023 |
| Downscale Factor | 0.5 | 0.8734 | +0.0151 |
| Downscale Factor | 0.25 | 0.8217 | +0.0667 |
