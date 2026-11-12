# FPR control under dataset shift (H4, H5)

- source_quantile: threshold from source reals only (deployment default).
- target_gmm / target_gmm_anchored: label-free, fitted on unlabelled target logits.
- oracle: uses target labels; reference only.
- H4 failures (lower CI of FPR > 2 alpha under source_quantile): 0 of 6 rows
- H4a verdict (confirmatory seeds [3, 4, 5] only, conservative CI): undetermined (fewer than 3 confirmatory seeds)
- H5 verdicts: {'target_gmm@alpha=0.05': 'falsified', 'target_gmm@alpha=0.1': 'falsified', 'target_gmm_anchored@alpha=0.05': 'falsified', 'target_gmm_anchored@alpha=0.1': 'falsified'}

| model | seed | dataset | role | source | policy | alpha | fpr | fpr_ci_low | fpr_ci_high | fpr_over_alpha | tpr | degenerate | swapped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.0674 | 0.0339 | 0.1074 | 1.3483 | 0.4324 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.2753 | 0.1173 | 0.3690 | 5.5056 | 0.6882 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.2753 | 0.1153 | 0.3690 | 5.5056 | 0.6882 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0341 | 0.0500 | 0.8989 | 0.3471 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1573 | 0.1017 | 0.2169 | 1.5730 | 0.5588 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.2978 | 0.1601 | 0.3977 | 2.9775 | 0.7559 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.2978 | 0.1604 | 0.3977 | 2.9775 | 0.7559 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0778 | 0.1000 | 0.9551 | 0.4676 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.0281 | 0.0056 | 0.0542 | 0.5618 | 0.2176 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.1236 | 0.0423 | 0.3530 | 2.4719 | 0.3824 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.1236 | 0.0414 | 0.3429 | 2.4719 | 0.3824 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0337 | 0.0500 | 0.8989 | 0.2559 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.0899 | 0.0497 | 0.1391 | 0.8989 | 0.3118 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.1629 | 0.0804 | 0.4026 | 1.6292 | 0.4471 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.1629 | 0.0778 | 0.3955 | 1.6292 | 0.4471 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0833 | 0.1000 | 0.9551 | 0.3176 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.0506 | 0.0225 | 0.0857 | 1.0112 | 0.2853 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.1011 | 0.0321 | 0.2295 | 2.0225 | 0.3706 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.1011 | 0.0319 | 0.2287 | 2.0225 | 0.3706 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0328 | 0.0500 | 0.8989 | 0.2441 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.0843 | 0.0476 | 0.1302 | 0.8427 | 0.3529 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.1404 | 0.0660 | 0.2892 | 1.4045 | 0.4441 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.1404 | 0.0656 | 0.2887 | 1.4045 | 0.4441 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0829 | 0.1000 | 0.9551 | 0.3529 | False | False |
