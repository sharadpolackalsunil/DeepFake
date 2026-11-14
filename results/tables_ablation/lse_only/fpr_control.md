# FPR control under dataset shift (H4, H5)

- source_quantile: threshold from source reals only (deployment default).
- target_gmm / target_gmm_anchored: label-free, fitted on unlabelled target logits.
- oracle: uses target labels; reference only.
- H4 failures (lower CI of FPR > 2 alpha under source_quantile): 0 of 6 rows
- H4a verdict (confirmatory seeds [3, 4, 5] only, conservative CI): undetermined (fewer than 3 confirmatory seeds)
- H5 verdicts: {'target_gmm@alpha=0.05': 'falsified', 'target_gmm@alpha=0.1': 'falsified', 'target_gmm_anchored@alpha=0.05': 'falsified', 'target_gmm_anchored@alpha=0.1': 'falsified'}

| model | seed | dataset | role | source | policy | alpha | fpr | fpr_ci_low | fpr_ci_high | fpr_over_alpha | tpr | degenerate | swapped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.0955 | 0.0555 | 0.1429 | 1.9101 | 0.6500 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0225 | 0.0000 | 0.1467 | 0.4494 | 0.3824 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0225 | 0.0000 | 0.1467 | 0.4494 | 0.3824 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0330 | 0.0500 | 0.8989 | 0.5265 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1348 | 0.0857 | 0.1943 | 1.3483 | 0.7324 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.1348 | 0.0773 | 0.2337 | 1.3483 | 0.7059 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.1348 | 0.0773 | 0.2337 | 1.3483 | 0.7059 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0828 | 0.1000 | 0.9551 | 0.6529 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.1236 | 0.0778 | 0.1761 | 2.4719 | 0.7088 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0506 | 0.0000 | 0.1701 | 1.0112 | 0.5353 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0506 | 0.0000 | 0.1701 | 1.0112 | 0.5353 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0326 | 0.0500 | 0.8989 | 0.5147 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1517 | 0.0978 | 0.2096 | 1.5169 | 0.7647 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.1517 | 0.0782 | 0.2619 | 1.5169 | 0.7647 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.1517 | 0.0782 | 0.2619 | 1.5169 | 0.7647 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0833 | 0.1000 | 0.9551 | 0.6324 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.1461 | 0.0966 | 0.2044 | 2.9213 | 0.6735 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0169 | 0.0000 | 0.1338 | 0.3371 | 0.3676 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0169 | 0.0000 | 0.1338 | 0.3371 | 0.3676 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0337 | 0.0500 | 0.8989 | 0.5559 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1798 | 0.1206 | 0.2426 | 1.7978 | 0.7294 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.1517 | 0.0674 | 0.2573 | 1.5169 | 0.6794 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.1517 | 0.0674 | 0.2573 | 1.5169 | 0.6794 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0829 | 0.1000 | 0.9551 | 0.6412 | False | False |
