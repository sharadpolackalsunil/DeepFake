# FPR control under dataset shift (H4, H5)

- source_quantile: threshold from source reals only (deployment default).
- target_gmm / target_gmm_anchored: label-free, fitted on unlabelled target logits.
- oracle: uses target labels; reference only.
- H4 failures (lower CI of FPR > 2 alpha under source_quantile): 0 of 6 rows
- H4a verdict (confirmatory seeds [3, 4, 5] only, conservative CI): undetermined (fewer than 3 confirmatory seeds)
- H5 verdicts: {'target_gmm@alpha=0.05': 'falsified (degenerate fits)', 'target_gmm@alpha=0.1': 'falsified', 'target_gmm_anchored@alpha=0.05': 'falsified (degenerate fits)', 'target_gmm_anchored@alpha=0.1': 'falsified'}

| model | seed | dataset | role | source | policy | alpha | fpr | fpr_ci_low | fpr_ci_high | fpr_over_alpha | tpr | degenerate | swapped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.1236 | 0.0745 | 0.1784 | 2.4719 | 0.7059 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0056 | 0.0000 | 0.0183 | 0.1124 | 0.2088 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0056 | 0.0000 | 0.0183 | 0.1124 | 0.2088 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0335 | 0.0500 | 0.8989 | 0.5324 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1292 | 0.0828 | 0.1833 | 1.2921 | 0.7471 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.0112 | 0.0000 | 0.0362 | 0.1124 | 0.3206 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.0112 | 0.0000 | 0.0362 | 0.1124 | 0.3206 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0843 | 0.1000 | 0.9551 | 0.6559 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.0899 | 0.0494 | 0.1397 | 1.7978 | 0.6471 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0000 | 0.0000 | 0.0117 | 0.0000 | 0.2294 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0000 | 0.0000 | 0.0117 | 0.0000 | 0.2294 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0337 | 0.0500 | 0.8989 | 0.5559 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1461 | 0.0952 | 0.2012 | 1.4607 | 0.7618 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.0112 | 0.0000 | 0.0428 | 0.1124 | 0.3735 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.0112 | 0.0000 | 0.0428 | 0.1124 | 0.3735 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0824 | 0.1000 | 0.9551 | 0.6706 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.1461 | 0.0944 | 0.2036 | 2.9213 | 0.6882 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0000 | 0.0000 | 0.0159 | 0.0000 | 0.2000 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0000 | 0.0000 | 0.0160 | 0.0000 | 0.2000 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0331 | 0.0500 | 0.8989 | 0.5206 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1685 | 0.1138 | 0.2278 | 1.6854 | 0.7441 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.0056 | 0.0000 | 0.0556 | 0.0562 | 0.3235 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.0056 | 0.0000 | 0.0562 | 0.0562 | 0.3235 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0828 | 0.1000 | 0.9551 | 0.6441 | False | False |
