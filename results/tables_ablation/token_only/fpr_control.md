# FPR control under dataset shift (H4, H5)

- source_quantile: threshold from source reals only (deployment default).
- target_gmm / target_gmm_anchored: label-free, fitted on unlabelled target logits.
- oracle: uses target labels; reference only.
- H4 failures (lower CI of FPR > 2 alpha under source_quantile): 0 of 6 rows
- H4a verdict (confirmatory seeds [3, 4, 5] only, conservative CI): undetermined (fewer than 3 confirmatory seeds)
- H5 verdicts: {'target_gmm@alpha=0.05': 'falsified (degenerate fits)', 'target_gmm@alpha=0.1': 'falsified', 'target_gmm_anchored@alpha=0.05': 'falsified (degenerate fits)', 'target_gmm_anchored@alpha=0.1': 'falsified'}

| model | seed | dataset | role | source | policy | alpha | fpr | fpr_ci_low | fpr_ci_high | fpr_over_alpha | tpr | degenerate | swapped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.1180 | 0.0699 | 0.1714 | 2.3596 | 0.6912 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0000 | 0.0000 | 0.0281 | 0.0000 | 0.1588 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0000 | 0.0000 | 0.0284 | 0.0000 | 0.1588 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0335 | 0.0500 | 0.8989 | 0.5206 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1292 | 0.0828 | 0.1833 | 1.2921 | 0.7471 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.0112 | 0.0000 | 0.0608 | 0.1124 | 0.3147 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.0112 | 0.0000 | 0.0608 | 0.1124 | 0.3147 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0839 | 0.1000 | 0.9551 | 0.6500 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.0899 | 0.0494 | 0.1397 | 1.7978 | 0.6588 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0000 | 0.0000 | 0.0108 | 0.0000 | 0.1206 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0000 | 0.0000 | 0.0108 | 0.0000 | 0.1206 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0337 | 0.0500 | 0.8989 | 0.5647 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1517 | 0.1000 | 0.2089 | 1.5169 | 0.7588 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.0056 | 0.0000 | 0.0398 | 0.0562 | 0.3735 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.0056 | 0.0000 | 0.0398 | 0.0562 | 0.3735 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0824 | 0.1000 | 0.9551 | 0.6676 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.1461 | 0.0944 | 0.2036 | 2.9213 | 0.6882 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0000 | 0.0000 | 0.2598 | 0.0000 | 0.1471 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0000 | 0.0000 | 0.5378 | 0.0000 | 0.1471 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0331 | 0.0500 | 0.8989 | 0.5324 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1685 | 0.1144 | 0.2281 | 1.6854 | 0.7441 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.0056 | 0.0000 | 0.3423 | 0.0562 | 0.3118 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.0056 | 0.0000 | 0.5671 | 0.0562 | 0.3118 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0833 | 0.1000 | 0.9551 | 0.6471 | False | False |
