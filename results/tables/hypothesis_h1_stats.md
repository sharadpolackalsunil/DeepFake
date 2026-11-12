# Hypothesis H1: Paired Cluster Bootstrap Delta-AUC

- Verdict (pre-registered rule): **supported**
- delta = 0.0100 (pre-registered default); n_boot = 10000; Bonferroni m = 2, per-comparison level = 0.9750
- Negatives: pure-source; seeds: [0, 1, 2, 3]
- Only rows with Kind = confirmatory decide H1. 'N multiplier to resolve' is the rough factor by which the evaluation set must grow before an equivalence verdict is possible.

| Dataset | Role | Kind | Model A | Model B | Comparison | AUC (A) | AUC (B) | dAUC (A-B) | CI level | CI Lower | CI Upper | Classification | N multiplier to resolve | Seeds | Clusters | Samples | Dropped (unaligned) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CELEBDF | exploratory | exploratory | smsdt_st | smsdt_s | H1: ST vs S | 0.8704 | 0.8826 | -0.0122 | 0.9500 | -0.0290 | 0.0045 | inconclusive | inf | 3 | 124 | 518 | 0 |
| DFD | exploratory | exploratory | smsdt_st | smsdt_s | H1: ST vs S | 0.8622 | 0.8773 | -0.0151 | 0.9500 | -0.0254 | -0.0047 | inconclusive | inf | 3 | 273 | 2847 | 0 |
| FAKEAVCELEB | confirmatory | confirmatory | smsdt_st | smsdt_s | H1: ST vs S | 0.9392 | 0.9437 | -0.0045 | 0.9750 | -0.0096 | 0.0006 | equivalent | 0.8780 | 3 | 991 | 2998 | 0 |
| FFPP | in_domain | exploratory | smsdt_st | smsdt_s | H1: ST vs S | 0.9629 | 0.9627 | 0.0002 | 0.9500 | -0.0066 | 0.0072 | equivalent | 0.4957 | 3 | 140 | 840 | 0 |
| FFPPVAL | exploratory | exploratory | smsdt_st | smsdt_s | H1: ST vs S | 0.9482 | 0.9377 | 0.0105 | 0.9500 | 0.0035 | 0.0175 | inconclusive | inf | 3 | 178 | 1304 | 0 |
| CELEBDF | exploratory | exploratory | smsdt_full | smsdt_s | H1: Full vs S | 0.8882 | 0.8826 | 0.0056 | 0.9500 | -0.0118 | 0.0235 | inconclusive | 16.0172 | 3 | 124 | 518 | 0 |
| DFD | exploratory | exploratory | smsdt_full | smsdt_s | H1: Full vs S | 0.8447 | 0.8773 | -0.0325 | 0.9500 | -0.0450 | -0.0207 | worse | inf | 3 | 273 | 2847 | 0 |
| FAKEAVCELEB | confirmatory | confirmatory | smsdt_full | smsdt_s | H1: Full vs S | 0.9449 | 0.9437 | 0.0012 | 0.9750 | -0.0050 | 0.0073 | equivalent | 0.4915 | 3 | 991 | 2998 | 0 |
| FFPP | in_domain | exploratory | smsdt_full | smsdt_s | H1: Full vs S | 0.9625 | 0.9627 | -0.0001 | 0.9500 | -0.0078 | 0.0073 | equivalent | 0.5892 | 3 | 140 | 840 | 0 |
| FFPPVAL | exploratory | exploratory | smsdt_full | smsdt_s | H1: Full vs S | 0.9410 | 0.9377 | 0.0034 | 0.9500 | -0.0041 | 0.0107 | inconclusive | 1.2449 | 3 | 178 | 1304 | 0 |
| CELEBDF | exploratory | exploratory | smsdt_full | smsdt_st | Exploratory: Full vs ST (forensic stream + LSE readout) | 0.8882 | 0.8704 | 0.0178 | 0.9500 | 0.0050 | 0.0318 | inconclusive | inf | 3 | 124 | 518 | 0 |
| DFD | exploratory | exploratory | smsdt_full | smsdt_st | Exploratory: Full vs ST (forensic stream + LSE readout) | 0.8447 | 0.8622 | -0.0175 | 0.9500 | -0.0284 | -0.0068 | inconclusive | inf | 3 | 273 | 2847 | 0 |
| FAKEAVCELEB | confirmatory | exploratory | smsdt_full | smsdt_st | Exploratory: Full vs ST (forensic stream + LSE readout) | 0.9449 | 0.9392 | 0.0057 | 0.9500 | 0.0015 | 0.0101 | inconclusive | 1.0169 | 3 | 991 | 2998 | 0 |
| FFPP | in_domain | exploratory | smsdt_full | smsdt_st | Exploratory: Full vs ST (forensic stream + LSE readout) | 0.9625 | 0.9629 | -0.0004 | 0.9500 | -0.0059 | 0.0054 | equivalent | 0.3455 | 3 | 140 | 840 | 0 |
| FFPPVAL | exploratory | exploratory | smsdt_full | smsdt_st | Exploratory: Full vs ST (forensic stream + LSE readout) | 0.9410 | 0.9482 | -0.0071 | 0.9500 | -0.0140 | -0.0002 | inconclusive | 5.6907 | 3 | 178 | 1304 | 0 |
| CELEBDF | exploratory | exploratory | smsdt_s | gend_ours | Exploratory: S vs gend_ours (training recipe, same pipeline) | 0.8826 | 0.8294 | 0.0532 | 0.9500 | 0.0210 | 0.0879 | better | inf | 3 | 124 | 518 | 0 |
| DFD | exploratory | exploratory | smsdt_s | gend_ours | Exploratory: S vs gend_ours (training recipe, same pipeline) | 0.8773 | 0.8656 | 0.0117 | 0.9500 | -0.0122 | 0.0372 | inconclusive | inf | 3 | 273 | 2847 | 0 |
| FAKEAVCELEB | confirmatory | exploratory | smsdt_s | gend_ours | Exploratory: S vs gend_ours (training recipe, same pipeline) | 0.9437 | 0.8907 | 0.0530 | 0.9500 | 0.0398 | 0.0668 | better | inf | 3 | 991 | 2998 | 0 |
| FFPP | in_domain | exploratory | smsdt_s | gend_ours | Exploratory: S vs gend_ours (training recipe, same pipeline) | 0.9627 | 0.9044 | 0.0583 | 0.9500 | 0.0453 | 0.0714 | better | inf | 3 | 140 | 840 | 0 |
| FFPPVAL | exploratory | exploratory | smsdt_s | gend_ours | Exploratory: S vs gend_ours (training recipe, same pipeline) | 0.9377 | 0.9007 | 0.0370 | 0.9500 | 0.0198 | 0.0536 | better | inf | 3 | 178 | 1304 | 0 |
| CELEBDF | exploratory | exploratory | smsdt_full | gend_ours | Exploratory: Full vs gend_ours | 0.8882 | 0.8294 | 0.0588 | 0.9500 | 0.0239 | 0.0957 | better | inf | 3 | 124 | 518 | 0 |
| DFD | exploratory | exploratory | smsdt_full | gend_ours | Exploratory: Full vs gend_ours | 0.8447 | 0.8656 | -0.0208 | 0.9500 | -0.0498 | 0.0092 | inconclusive | inf | 3 | 273 | 2847 | 0 |
| FAKEAVCELEB | confirmatory | exploratory | smsdt_full | gend_ours | Exploratory: Full vs gend_ours | 0.9449 | 0.8907 | 0.0542 | 0.9500 | 0.0403 | 0.0688 | better | inf | 3 | 991 | 2998 | 0 |
| FFPP | in_domain | exploratory | smsdt_full | gend_ours | Exploratory: Full vs gend_ours | 0.9625 | 0.9044 | 0.0582 | 0.9500 | 0.0431 | 0.0733 | better | inf | 3 | 140 | 840 | 0 |
| FFPPVAL | exploratory | exploratory | smsdt_full | gend_ours | Exploratory: Full vs gend_ours | 0.9410 | 0.9007 | 0.0404 | 0.9500 | 0.0220 | 0.0585 | better | inf | 3 | 178 | 1304 | 0 |
| CELEBDF | exploratory | exploratory | smsdt_s | gend_official | G1: S vs official GenD weights (same pipeline) | 0.8826 | 0.9327 | -0.0501 | 0.9500 | -0.0852 | -0.0210 | worse | inf | 3 | 124 | 518 | 0 |
| DFD | exploratory | exploratory | smsdt_s | gend_official | G1: S vs official GenD weights (same pipeline) | 0.8773 | 0.8840 | -0.0067 | 0.9500 | -0.0266 | 0.0142 | inconclusive | 38.5854 | 3 | 273 | 2847 | 0 |
| FAKEAVCELEB | confirmatory | exploratory | smsdt_s | gend_official | G1: S vs official GenD weights (same pipeline) | 0.9437 | 0.9745 | -0.0308 | 0.9500 | -0.0391 | -0.0227 | worse | inf | 3 | 991 | 2998 | 0 |
| FFPP | in_domain | exploratory | smsdt_s | gend_official | G1: S vs official GenD weights (same pipeline) | 0.9627 | 0.9452 | 0.0175 | 0.9500 | 0.0060 | 0.0293 | inconclusive | inf | 3 | 140 | 840 | 0 |
| FFPPVAL | exploratory | exploratory | smsdt_s | gend_official | G1: S vs official GenD weights (same pipeline) | 0.9545 | 0.9371 | 0.0174 | 0.9500 | 0.0073 | 0.0276 | inconclusive | inf | 3 | 140 | 840 | 464 |
| CELEBDF | exploratory | exploratory | gend_ours | gend_official | G1: gend_ours vs official GenD weights (same pipeline) | 0.8294 | 0.9327 | -0.1033 | 0.9500 | -0.1393 | -0.0723 | worse | inf | 3 | 124 | 518 | 0 |
| DFD | exploratory | exploratory | gend_ours | gend_official | G1: gend_ours vs official GenD weights (same pipeline) | 0.8656 | 0.8840 | -0.0184 | 0.9500 | -0.0324 | -0.0060 | inconclusive | inf | 3 | 273 | 2847 | 0 |
| FAKEAVCELEB | confirmatory | exploratory | gend_ours | gend_official | G1: gend_ours vs official GenD weights (same pipeline) | 0.8907 | 0.9745 | -0.0837 | 0.9500 | -0.0985 | -0.0704 | worse | inf | 3 | 991 | 2998 | 0 |
| FFPP | in_domain | exploratory | gend_ours | gend_official | G1: gend_ours vs official GenD weights (same pipeline) | 0.9044 | 0.9452 | -0.0408 | 0.9500 | -0.0568 | -0.0247 | worse | inf | 3 | 140 | 840 | 0 |
| FFPPVAL | exploratory | exploratory | gend_ours | gend_official | G1: gend_ours vs official GenD weights (same pipeline) | 0.8988 | 0.9371 | -0.0383 | 0.9500 | -0.0527 | -0.0232 | worse | inf | 3 | 140 | 840 | 464 |
