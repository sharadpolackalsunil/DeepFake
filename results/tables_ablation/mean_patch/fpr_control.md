# FPR control under dataset shift (H4, H5)

- source_quantile: threshold from source reals only (deployment default).
- target_gmm / target_gmm_anchored: label-free, fitted on unlabelled target logits.
- oracle: uses target labels; reference only.
- H4 failures (lower CI of FPR > 2 alpha under source_quantile): 0 of 6 rows
- H4a verdict (confirmatory seeds [3, 4, 5] only, conservative CI): undetermined (fewer than 3 confirmatory seeds)
- H5 verdicts: {'target_gmm@alpha=0.05': 'falsified (degenerate fits)', 'target_gmm@alpha=0.1': 'falsified', 'target_gmm_anchored@alpha=0.05': 'falsified (degenerate fits)', 'target_gmm_anchored@alpha=0.1': 'falsified'}

| model | seed | dataset | role | source | policy | alpha | fpr | fpr_ci_low | fpr_ci_high | fpr_over_alpha | tpr | degenerate | swapped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.1180 | 0.0699 | 0.1714 | 2.3596 | 0.7000 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0000 | 0.0000 | 0.0323 | 0.0000 | 0.1735 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0000 | 0.0000 | 0.0330 | 0.0000 | 0.1735 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0335 | 0.0500 | 0.8989 | 0.5353 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1348 | 0.0860 | 0.1875 | 1.3483 | 0.7471 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.0112 | 0.0000 | 0.0678 | 0.1124 | 0.3206 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.0112 | 0.0000 | 0.0663 | 0.1124 | 0.3206 | False | False |
| smsdt_full | seed1 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0843 | 0.1000 | 0.9551 | 0.6500 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.0955 | 0.0543 | 0.1444 | 1.9101 | 0.6647 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0000 | 0.0000 | 0.0114 | 0.0000 | 0.1118 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0000 | 0.0000 | 0.0114 | 0.0000 | 0.1118 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0337 | 0.0500 | 0.8989 | 0.5618 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1461 | 0.0964 | 0.2023 | 1.4607 | 0.7588 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.0056 | 0.0000 | 0.0455 | 0.0562 | 0.3765 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.0056 | 0.0000 | 0.0455 | 0.0562 | 0.3765 | False | False |
| smsdt_full | seed2 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0833 | 0.1000 | 0.9551 | 0.6706 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | source_quantile | 0.0500 | 0.1461 | 0.0944 | 0.2036 | 2.9213 | 0.7000 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm | 0.0500 | 0.0000 | 0.0000 | 0.5029 | 0.0000 | 0.1382 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.0500 | 0.0000 | 0.0000 | 0.5378 | 0.0000 | 0.1382 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | oracle | 0.0500 | 0.0449 | 0.0331 | 0.0498 | 0.8989 | 0.5265 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | source_quantile | 0.1000 | 0.1854 | 0.1278 | 0.2455 | 1.8539 | 0.7471 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm | 0.1000 | 0.0056 | 0.0000 | 0.5540 | 0.0562 | 0.3118 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | target_gmm_anchored | 0.1000 | 0.0056 | 0.0000 | 0.5672 | 0.0562 | 0.3118 | False | False |
| smsdt_full | seed3 | celebdf | confirmatory | ffppval | oracle | 0.1000 | 0.0955 | 0.0829 | 0.1000 | 0.9551 | 0.6471 | False | False |
