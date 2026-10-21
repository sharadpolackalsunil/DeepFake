# S-MSDT → SOTA: Diagnosis, Answers and Phased Implementation Roadmap

**Responds to:** `architecture (1).md` (S-MSDT architecture specification and SOTA roadmap request)
**Repository checked:** `sharadpolackalsunil/deepfake`, `main` @ `341e0ee` (read on 2026-10-19)
**Scope:** generalization and detection accuracy only. Export, ONNX/TensorRT and latency are excluded, as the spec requests.

**Roles used throughout**

| Tag | Who | Meaning |
|---|---|---|
| **[AG]** | Antigravity (coding agent) | Code change: exact file, equations, shapes, snippet |
| **[USER]** | Human on the RTX 5090 | Commands to run, downloads, disk, judgement calls |
| **[VERIFY]** | Antigravity | Check that must pass before the next phase starts |

> **About the code in this document.** Every snippet passes a Python syntax check. None of it could be executed here: there is no GPU, dataset or PyTorch in my environment. Treat each snippet as a reviewed draft. The `[VERIFY]` step after each phase is what proves it works on your machine.

---

## 0. Executive summary

1. **Test AUC 0.71 on FF++ is an in-domain number, and it is about 25 points below a plain baseline.** In DeepfakeBench, a vanilla EfficientNet-B4 trained and tested on FF++ c23 reaches **0.957 frame-level AUC**, and Xception reaches **0.964**. S-MSDT gets 0.71 on the same kind of test set (the same dataset, but unseen identities). A gap this large on an *in-domain* test usually comes from a protocol or pipeline defect plus overfitting, not from the choice of backbone. **Gate 0 (Section 3) must answer "is it the data or the model?" before any redesign.**
2. **The test set is not the FF++ protocol any paper reports.** DeepFakeDetection (DFD) contributes **67,175 of 124,363 test fake chunks (54.0%)**, yet DFD is absent from validation. The reported counts (val 307 videos, test 1,027) do not match an official 140/140 split (840 FF++ videos each). Validation and test therefore measure different distributions, and that alone explains part of the EER-threshold drift (0.74 → 0.93).
3. **The split code can still leak identities.** The default split groups fakes by the *first* id of `target_source` (for example `042` in `042_056`). A swap fake carries the *source* face (`056`), which can sit in a different split. DFD/actors videos fall through to `"train"` whenever they are missing from a split file.
4. **Training saw roughly 2 passes of the data, not 23 epochs.** From your own test counts (~88 chunks per FF++ video), the FF++ training set is about **380k chunks**. A 32k-chunk "epoch" covers about **8%** of it, not the 25–35% the spec estimates. 23 such epochs come to about 1.9 passes. OneCycleLR was still at **~60% of peak LR** at step 23k, so the best checkpoint was selected mid-schedule.
5. **The model's design discards the evidence the failing methods leave.** NeuralTextures and Face2Face change small, local regions (mouth, expression) and keep identity. The 7×7 average pooling and the global mean over 392 tokens dilute exactly those signals, and JPEG augmentation down to Q=15 erases them. This matches the identity-leakage mechanism documented by Dong et al. (CVPR 2023).
6. **The SBI function exists but is never called, and it keeps the label "real".** If it were switched on as written it would *teach* the model that blending boundaries are real. It must emit label 1.
7. **Recommended path to a publishable result.**
   - Phase 0: a leak-free protocol and a map-style video dataset with exact epochs and paired real/fake sampling.
   - Phase 1: proper, temporally consistent SBI; dense tokens; attention pooling; simpler losses.
   - Phase 2: a CLIP ViT-L/14 frame encoder with LayerNorm-only tuning, feeding the S-MSDT temporal head.
   - Phase 3: cross-dataset evaluation and the paper's ablations.

   The current published bar for FF++ → Celeb-DF v2 video AUC is about **0.93** (SBI, CADDM: frame models with blending), **0.895** (AltFreezing: video model) and **0.965** (LNCLIP-DF: CLIP with LayerNorm tuning).

---

## 1. What the code actually does vs. what the spec says

Checked on `main` @ `341e0ee`. Rows marked "mismatch" change how results should be read.

| # | Spec claim | Code reality | Consequence |
|---|---|---|---|
| 1 | Identity-disjoint 720/140/140 split | `build_split_manifests` uses the split file if given. Otherwise it groups by `raw_id.split("_")[0]`, the **target** id only. Unmatched ids default to `"train"`. | **Mismatch.** Swap fakes can carry a source identity from another split. Reported video counts (307 / 1,027) don't match 840 / 840. |
| 2 | FF++ protocol | `FFPP_DATASETS` includes `original_sequences/actors` and `DeepFakeDetection` | **Mismatch.** DFD is 54% of test fakes and absent from val. DFD is normally a *cross-dataset* test set (DeepfakeBench, SBI). |
| 3 | Δ₀ = 0 at t=0 | `prev = zeros` at t=0, so **Δ₀ = x_norm₀**, not 0 | Frame 0 gets a large, unique difference vector. It acts as an accidental "first frame" marker. |
| 4 | No temporal position encoding | Confirmed: `pos_embed` is `(1,1,49,D)` | Correct concern (Section 4, doubt 3) |
| 5 | Worker duplication? | webdataset 1.0.2 defaults: `workersplitter=split_by_worker`. `ResampledShards` seeds each worker with `pid`, `time_ns` and `urandom`. | No duplication of exact sequences, but no epoch guarantee. Moot after Phase 0. |
| 6 | SBI augmentation | `simulate_sbi_blend` is defined but **never called**. It returns blended frames with the **original label**. | Must not be enabled as written: it would label blends as real. Replace it (Phase 1). |
| 7 | Confidence-gated temporal residual | `x = x + gated * conf` with SCRFD `det_score` | **Shortcut risk.** If fakes get systematically different detector scores, `conf` leaks the label. Test in Gate 0 (D2). |
| 8 | Readout | Global mean over N=49 and T=8 confirmed | Confirms the dilution concern |
| 9 | Alignment | Causal EMA (α=0.6) on 5 points. Frames where alignment fails fall back to an **unaligned bbox crop** in the same chunk. | EMA lags about α/(1−α) = 1.5 frames on head motion. Mixed aligned/unaligned crops inside one chunk create geometry jumps. |
| 10 | Augmentation | JPEG Q∈[15,80] at p=0.4, then blur, resize, crop, flip; independent per sample | Q=15 removes most of NeuralTextures' signal. Aggressive for in-domain c23. |
| 11 | Schedule | OneCycleLR, `total_steps=50000`, run stopped at 23k | LR at 23k ≈ 0.60 × peak (backbone ≈ 1.8e-5, head ≈ 1.8e-4) |

---

## 2. Reading the results correctly

### 2.1 Where S-MSDT sits against published numbers

| Setting | Reference | AUC | S-MSDT now |
|---|---|---|---|
| FF++ c23 in-domain, frame-level | EfficientNet-B4 (DeepfakeBench) | 0.957 | 0.71 (chunk) |
| FF++ c23 in-domain, frame-level | Xception (DeepfakeBench) | 0.964 | — |
| FF++ → Celeb-DF v2, frame-level | EfficientNet-B4 (DeepfakeBench) | 0.749 | not run |
| FF++ → Celeb-DF v2, video-level | SBI, EfficientNet-B4 (CVPR 2022) | 0.932 | not run |
| FF++ → Celeb-DF v2 (level as reported) | CADDM, EfficientNet-B4 (CVPR 2023) | 0.939 | not run |
| FF++ → Celeb-DF v2, video-level | AltFreezing, 3D ResNet-50 (CVPR 2023) | 0.895 | not run |
| FF++ → Celeb-DF v2, video-level | Forensics Adapter, CLIP ViT-L/14 (CVPR 2025) | 0.957 | not run |
| FF++ → Celeb-DF v2, video-level | LNCLIP-DF, CLIP ViT-L/14 LN-tuning (2025) | 0.965 | not run |

The table mixes frame-level and video-level metrics, as the original papers report them. Compare like with like.

### 2.2 What the failure pattern tells you

| Observation | Most likely cause | How to confirm (Section 3) |
|---|---|---|
| Train 0.9998, test 0.71 *in-domain* | Memorization plus a protocol defect | D1: a plain EfficientNet-B4 on the same shards |
| NT 0.59, F2F 0.66 vs FSh 0.80, DF 0.79 | Model relies on identity/blend cues. Local mouth/expression evidence is pooled away and destroyed by Q=15 JPEG. | D3 identity probe; per-method AUC after Phase 1 pooling changes |
| FPR@0.5 = 47.5% on test reals | Unseen-domain reals (DFD actor videos) look "fake"; distribution shift | D4: FPR separately for `youtube` vs `actors` reals |
| EER threshold 0.74 (val) → 0.93 (test) | Val and test have different compositions (DFD only in test) | D4: re-evaluate on matched compositions |
| Accuracy 73% at 0.5 | Threshold not chosen on val | Report AUC first; choose operating points on val only |

---

## 3. Gate 0: diagnostics to run before changing the architecture (≤ 1 day)

Each diagnostic has a decision rule. Run them on the existing shards and `checkpoints_v1_epoch23/best.pt`.

| ID | Question | Experiment | Decision |
|---|---|---|---|
| D1 | Data or model? | Train a frame-level EfficientNet-B4 (timm, BCE, 224, 8 frames per video, 5 epochs) on the **same** train shards; test on the same test shards | AUC ≥ 0.93 → the data is fine and S-MSDT is the problem. AUC ≈ 0.7 → fix data/protocol first (Phase 0). |
| D2 | Is `conf` a shortcut? | (a) AUC of mean `conf` alone as the score. (b) Re-evaluate S-MSDT with `conf=1`. | (a) > 0.60 or (b) changes AUC by > 0.02 → remove `conf` from the model |
| D3 | Does the model encode identity? | Linear probe on frozen pooled features of **real** videos: predict identity (720 classes, or target id) | Top-1 far above chance (1/720) → identity is encoded. Track this number after each phase; it should fall. |
| D4 | Is the test set coherent? | Per-source FPR (`youtube` vs `actors` reals). Per-split counts of videos, identities and methods. | Big FPR gap → domain shift. Counts ≠ 140/140/720 per method → protocol bug. |
| D5 | Does the split leak? | `tools/check_split.py` below | Any identity overlap → regenerate shards (Phase 0) |

### 3.1 [AG] `tools/check_split.py`: identity-overlap check

```python
"""Fail loudly if any face identity appears in more than one split.

Usage: python tools/check_split.py data/splits/index.csv
index.csv columns: video_id, split, label, method, target_id, source_id
(source_id is empty for real videos and for reenactment methods).
"""
import sys
import csv
from collections import defaultdict


def identities(row):
    ids = {row["target_id"]}
    if row.get("source_id"):
        ids.add(row["source_id"])
    return ids


def main(path):
    owner = defaultdict(set)          # identity -> set of splits it appears in
    counts = defaultdict(lambda: defaultdict(int))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            counts[row["split"]][row["method"]] += 1
            for ident in identities(row):
                owner[ident].add(row["split"])
    leaks = {i: s for i, s in owner.items() if len(s) > 1}
    for split, per_method in sorted(counts.items()):
        print(split, dict(per_method))
    if leaks:
        print(f"LEAK: {len(leaks)} identities in >1 split, e.g. {list(leaks.items())[:5]}")
        sys.exit(1)
    print("OK: splits are identity-disjoint")


if __name__ == "__main__":
    main(sys.argv[1])
```

### 3.2 [AG] `tools/diag_conf_shortcut.py`: is detector confidence predictive of the label?

```python
"""AUC of detector confidence alone as a fake score (D2a).

Reads conf.npy and label.cls from shards; no model needed.
"""
import glob
import sys
import numpy as np
import webdataset as wds
from sklearn.metrics import roc_auc_score


def main(pattern):
    scores, labels = [], []
    ds = wds.WebDataset(sorted(glob.glob(pattern)), shardshuffle=False).decode()
    for s in ds:
        conf = np.asarray(s["conf.npy"], dtype=np.float32)
        lab = s["label.cls"]
        lab = float(lab.decode() if isinstance(lab, bytes) else lab)
        scores.append(-conf.mean())   # lower confidence -> "more fake"
        labels.append(lab)
    auc = roc_auc_score(labels, scores)
    print(f"conf-only AUC = {auc:.4f}  (|AUC-0.5| > 0.10 means conf leaks the label)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/cache/ffpp/test-*.tar")
```

### 3.3 [USER] Gate 0 commands

```powershell
python tools/diag_conf_shortcut.py "data/cache/ffpp/test-*.tar"
python tools/diag_conf_shortcut.py "data/cache/ffpp/train-*.tar"
# D1 baseline: see Phase 1 "frame baseline" config, or DeepfakeBench's EfficientNet-B4 recipe
```

**[VERIFY] Gate 0 is done when** D1–D5 each have a recorded number in `outputs/diagnostics.md`.

---

## 4. Answers to the six technical doubts (spec §9)

### Doubt 1: Is the residual temporal-difference update harmful?

**Verdict: the idea is sound. The implementation has three problems.**

- **Δ₀ is not zero.** Zero-padding `prev` makes Δ₀ = x̂₀ (norm 1) while Δ_t for t>0 is small. Fix it with replicate padding and a learned first-frame embedding if you want frame 0 marked.
- **Unnormalized, un-gated injection.** `gated` comes out of a fresh `Conv1d` with no normalization and is added at full scale from step 0. Use zero-initialized output weights (the block starts as identity) and a LayerNorm on the input.
- **`conf` multiplies the update.** That makes the residual magnitude a function of detector score, which is a shortcut channel (D2). Drop it, or use `conf` only as a key-padding mask for frames with no detected face.

Cross-attention to difference tokens is not proven better. It is a reasonable ablation, not a default. Revised block:

```python
# smsdt/models/nmsss_vit.py  (replace the temporal-difference part of NMSSSBlock.forward)
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDiff(nn.Module):
    """x: (B, T, N, D) -> x + W([x_hat, delta]),  delta_0 = 0, zero-init so it starts as identity."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x_hat = F.normalize(self.norm(x), dim=-1)                     # (B, T, N, D)
        prev = torch.cat([x_hat[:, :1], x_hat[:, :-1]], dim=1)        # replicate pad -> delta_0 = 0
        delta = x_hat - prev                                          # (B, T, N, D)
        return x + self.proj(torch.cat([x_hat, delta], dim=-1))       # (B, T, N, D)
```

`nn.Linear` on the last dimension is equivalent to the current `Conv1d(2D, D, 1)` and needs no rearranging.

### Doubt 2: Does `abs()` on Haar subbands erase phase?

**Verdict: partly a misframing, but drop the `abs()` anyway.**

- A single-level real Haar DWT has no Fourier "phase". What `abs()` removes is the **sign (edge polarity)** of each coefficient.
- `abs()` was introduced to stop signed coefficients cancelling under *average pooling*. The branch now uses a *learned* strided conv (k=16, s=16), which can learn signed filters, so the original reason no longer applies.
- **Action:** feed signed coefficients, or signed and `abs` stacked as 18 channels. Make it an ablation row.
- If phase is the hypothesis, test it properly with a DCT or FFT phase input, or a dual-tree complex wavelet. Note too that a 16×16 stride-16 patch stem cannot localize 1–3 pixel seams better than its patch size. Use k=8, s=8 → 14×14 to match the denser token grid (Phase 1).

In Phase 2 a CLIP encoder replaces most of the frequency branch's job. Keep the branch only if the ablation shows a gain on cross-dataset AUC.

### Doubt 3: No temporal position encoding

**Verdict: confirmed. Add it.** The Δ term injects some order information (Δ_t depends on t−1), and the bug in doubt 1 marks frame 0, so attention is not fully order-blind. It is still a bag of frames for every other ordering. Add a learnable `(1, T, 1, D)` embedding, initialized with `trunc_normal_(std=0.02)`, before the first block. For T changes between training and test, interpolate it linearly.

```python
# smsdt/models/smsdt.py  (__init__)
self.time_embed = nn.Parameter(torch.zeros(1, max_frames, 1, embed_dim))   # max_frames = 16
nn.init.trunc_normal_(self.time_embed, std=0.02)
# forward, after z = rearrange(...):  z = z + self.pos_embed + self.time_embed[:, :T]
```

### Doubt 4: WebDataset workers on Windows

**Verdict: no duplication; the real problem is the missing epoch guarantee.** I checked webdataset 1.0.2's source. `WebDataset` defaults to `workersplitter=split_by_worker`, so with `resampled=False` each worker reads a disjoint subset of shards. With `resampled=True`, `ResampledShards` seeds each worker from `pid`, `time_ns` and `os.urandom`, so workers do not replay identical streams, but sampling is with replacement and there is no epoch.

`resampled=False` also has a catch: `empty_check=True` raises an error when a worker receives zero shards, so the number of shards must be at least `num_workers`. Phase 0 replaces the iterable pipeline with a map-style dataset, which removes the issue entirely.

### Doubt 5: Can affine warping create artifacts the model learns?

**Verdict: not a label shortcut by itself, but two real risks.**

- Real and fake frames go through the same warp, so interpolation artifacts alone don't separate the classes.
- **Risk 1:** fakes can have noisier landmarks (the detector is running on a manipulated face). That means more frame-to-frame warp jitter on fakes. The temporal module can learn the jitter instead of manipulation. Causal EMA reduces jitter but lags on motion.
- **Risk 2:** chunks mix aligned crops with unaligned bbox-fallback crops, which causes big geometry jumps.
- **Fix (Phase 0):** a loose 1.3× crop using a similarity transform (scale, rotation, translation) from **non-causally smoothed** landmarks (centered Savitzky–Golay or bidirectional EMA), applied identically to a real video and its fakes. Drop frames where detection fails instead of falling back. For paired training, compute the crop from the **real** video's landmarks and reuse it for the fake frames. FF++ fakes are frame-aligned to their target video, so geometry then becomes identical across a pair and cannot leak.

### Doubt 6: Does the contrastive loss dominate?

**Verdict: measure it; probably drop it in Phase 1.**

- At B=8 with 7 classes there are very few positives, and τ=0.07 gives high-variance gradients. The logged magnitudes (0.1 × 0.3–0.5 = 0.03–0.05 vs focal 0.05–0.10) mean SupCon contributes a comparable share of the gradient late in training.
- It probably does not cluster by identity: its positives are same-method, *different-identity* pairs.
- **Action:** log per-term gradient norms on shared parameters (snippet below). Set `lambda_contrastive=0` in Phase 1, and reintroduce a real-vs-fake SupCon only with paired batches of ≥ 32 clips.
- Remove the decorrelation loss. Forcing orthogonality between RGB and wavelet features penalizes both branches for agreeing on a true seam, and nothing in the results shows it helps.

```python
# smsdt/train.py: per-term gradient norm on shared parameters (run every 200 steps)
def term_grad_norms(model, terms):
    """terms: dict name -> scalar loss (graph retained). Returns L2 grad norm per term on the fusion layer."""
    shared = [p for p in model.fusion.parameters() if p.requires_grad]
    out = {}
    for name, loss in terms.items():
        grads = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        out[name] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None)))
    return out
```

---

## 5. Decisions on the ten requested areas (spec §10)

| Area | Decision | Why |
|---|---|---|
| 1. Crop | **Loose 1.3× crop**, similarity transform from smoothed landmarks, 256 stored → 224 random crop for training | Swap seams sit at the jaw, hairline and cheeks. DeepfakeBench uses a 1.3 margin; SBI uses a random 4–20% margin for training and 12.5% at test. Tight ArcFace crops cut the seam off. |
| 1. Mesh tracking | Store **106 landmarks** (InsightFace `buffalo_l` includes a 2d106 model) | Needed for SBI convex-hull masks; 5 points are not enough |
| 2. Backbone | Phase 1: keep EfficientNet-B4 (it reaches 0.93 CDF with SBI). Phase 2: **CLIP ViT-L/14, LayerNorm-only tuning**. | LNCLIP-DF: LN tuning beat full fine-tuning and LoRA (which overfit quickly) and reached 0.965 CDF. DINOv2 is a reasonable ablation, not the default. |
| 3. Temporal model | Keep factorized spatial→temporal attention (it is your contribution), with the fixes in Section 4. No Mamba. | At T=8–16 attention is cheap. Mamba adds risk without a known gain on these benchmarks. AltFreezing's insight (alternate spatial/temporal training) is worth borrowing. |
| 4. Frequency branch | Signed coefficients, k=8 stem to a 14×14 grid; keep only if ablation shows cross-dataset gain | See doubt 2 |
| 5. Tokens and readout | 14×14 tokens via learned strided conv (no average pooling); **gated attention pooling + top-k patch logits**; patch-level mask supervision | Preserves local evidence; directly targets the NT/F2F failure |
| 6. Hardware | B = 8 pairs (16 clips); T = 8 (try 16 with frame stride 2); 224 input; dense 196–256 tokens | Section 9 has the arithmetic |
| 7. Imbalance | **Paired real/fake sampling** (each batch is 50/50 by construction); plain BCE; drop focal α | Removes the need for re-weighting and removes identity/background as a cue. LNCLIP-DF found paired training overfits less. |
| 7. External reals | Optional, Phase 3 only, **always with SBI applied to them** so each source contains both labels | Prevents "source = label" shortcuts |
| 8. Loss | BCE (label smoothing 0.05) + patch-mask BCE (λ=0.5) + optional uniformity/alignment on L2-normalized features. Drop focal, SupCon and decorrelation. | Simpler, with supervision aimed at localized artifacts |
| 8. Schedule | AdamW, linear warmup 3% of steps, cosine to 1% of peak, total steps computed from the dataset | Section 6.4 |
| 8. LR | EffNet: backbone 5e-5, head 5e-4. CLIP LN params: sweep {5e-5, 1e-4, 3e-4}, head 5e-4. Weight decay 0.05 (0 on norms and biases). Clip grad norm 1.0. | Standard fine-tuning ranges; choose by val AUC |
| 9. NT/F2F | Mouth-region evidence: dense tokens, top-k pooling, mask supervision, JPEG Q∈[50,100], optional mouth crop stream (Phase 3) | Local, low-amplitude artifacts |
| 10. Epochs | **Map-style dataset: one epoch = every real video once, each paired with one random fake** | Exact coverage; schedule computed from `len(loader)` |

---

## 6. Phase 0: Protocol and data pipeline (Days 1–3)

**Goal:** a leak-free, standard protocol that any reviewer can reproduce, and exact, balanced epochs.

### 6.1 [USER] Downloads and disk

1. Official FF++ split files: `train.json`, `val.json`, `test.json` from the FaceForensics GitHub repo, `dataset/splits/`. Put them in `data/splits/official/`.
2. Optional, for mask supervision: FF++ manipulation masks with `download1.py ... -t masks` for each method.
3. Disk: storing 128 aligned frames per video at 256×256×3 uint8 needs about **109 GB** for the 4,320 FF++ train videos (720 real + 5 methods), plus about 21 GB each for val and test. The current overlapping-chunk shards would need about 458 GB for the same data, because every frame is stored twice under stride 4.

### 6.2 [AG] Protocol rules (write these into `configs/data.yaml` comments and the paper)

- Train/val/test: **FF++ c23, five methods** (Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter) and the youtube reals, official pair split (360/70/70 pairs → 720/140/140 real videos).
- `original_sequences/actors` and `DeepFakeDetection` are **excluded from FF++ splits** and used only as the cross-dataset test "DFD".
- Validation selects the checkpoint and any thresholds. Test is touched once per final run.

### 6.3 [AG] `tools/make_ffpp_index.py`: official split → index CSV

```python
"""Build data/splits/index.csv from the official FF++ pair splits.

Each official file is a list of [a, b] pairs. Real videos are 'a' and 'b'.
Fake videos are named 'a_b' (target a, source b) and 'b_a'. Both members of a
pair, and every video made from them, go to the same split.
"""
import csv
import json
import os

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
SWAPS = {"Deepfakes", "FaceSwap", "FaceShifter"}       # source identity appears in the fake


def main(split_dir="data/splits/official", raw_root="data/raw/ffpp_c23", comp="c23",
         out="data/splits/index.csv"):
    rows = []
    for split in ("train", "val", "test"):
        with open(os.path.join(split_dir, f"{split}.json")) as f:
            pairs = json.load(f)
        for a, b in pairs:
            for vid in (a, b):
                rows.append(dict(video_id=f"real_{vid}", split=split, label=0, method="real",
                                 target_id=vid, source_id="",
                                 path=f"{raw_root}/original_sequences/youtube/{comp}/videos/{vid}.mp4"))
            for m in METHODS:
                for t, s in ((a, b), (b, a)):
                    rows.append(dict(video_id=f"{m}_{t}_{s}", split=split, label=1, method=m,
                                     target_id=t, source_id=s if m in SWAPS else "",
                                     path=f"{raw_root}/manipulated_sequences/{m}/{comp}/videos/{t}_{s}.mp4"))
    missing = [r["path"] for r in rows if not os.path.exists(r["path"])]
    print(f"{len(rows)} videos indexed, {len(missing)} missing on disk")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
```

Expected output: 6,000 videos indexed (1,000 real + 5 × 1,000 fake), split 4,320 / 840 / 840.

### 6.4 [AG] Preprocessing: per-video frame store (replaces chunk shards)

File: `smsdt/preprocess/build_store.py` (new). For each video in `index.csv`:

1. Read **all** frames; run SCRFD on every frame (or every 2nd, interpolating boxes in between). Get 5-point and 106-point landmarks (`buffalo_l` → `face.kps`, `face.landmark_2d_106`).
2. Track one identity: pick the largest face in frame 0, then the face with highest IoU to the previous box.
3. Smooth landmarks **non-causally**: `scipy.signal.savgol_filter(kps, window_length=7, polyorder=2, axis=0)`.
4. Crop: a similarity transform mapping the smoothed 5 points to the ArcFace template **scaled to leave a 1.3× margin**, output 256×256. For fake videos, **reuse the crop transform of the matching real (target) video frame**, so pairs share geometry.
5. Keep up to 128 frames as 4 segments of 32 contiguous frames, spread evenly across the video.
6. Save to `data/store/<video_id>/`:
   - `frames.npy`: `(F, 256, 256, 3)` uint8 RGB
   - `lmk106.npy`: `(F, 106, 2)` float32, in crop coordinates
   - `fidx.npy`: `(F,)` int32, original frame numbers
   - `mask.npy` (optional): `(F, 64, 64)` uint8 FF++ manipulation mask, warped with the same transform

Core of step 4:

```python
import cv2
import numpy as np

ARC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def loose_template(out=256, margin=1.3):
    """ArcFace 5-point template (112 px) rescaled so the face occupies 1/margin of the crop."""
    t = (ARC - 56.0) / margin + 56.0          # shrink about the template centre
    return t * (out / 112.0)


def crop_frame(frame_bgr, kps5, out=256, margin=1.3):
    M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), loose_template(out, margin),
                                       method=cv2.LMEDS)
    if M is None:
        return None, None
    crop = cv2.warpAffine(frame_bgr, M, (out, out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), M


def warp_points(pts, M):
    """Apply a 2x3 affine M to (K, 2) points."""
    return pts @ M[:, :2].T + M[:, 2]
```

### 6.5 [AG] `smsdt/data/video_dataset.py`: paired, map-style dataset

```python
"""Map-style video datasets with exact epochs.

Train: one item = (real clip, matching fake clip) with identical frame indices
and identical augmentation. One epoch = every real training video exactly once.
Eval:  one item = K evenly spaced clips from one video.
"""
import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

METHOD_ID = {"real": 0, "Deepfakes": 1, "Face2Face": 2, "FaceSwap": 3,
             "NeuralTextures": 4, "FaceShifter": 5, "SBI": 6}


def load_index(path, split):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


class VideoStore:
    """Lazy, worker-safe access to one video's stored arrays."""

    def __init__(self, root, video_id):
        d = os.path.join(root, video_id)
        self.dir = d
        self.fidx = np.load(os.path.join(d, "fidx.npy"))
        self._frames = None

    @property
    def frames(self):
        if self._frames is None:                                  # opened inside the worker
            self._frames = np.load(os.path.join(self.dir, "frames.npy"), mmap_mode="r")
        return self._frames

    def lmk(self):
        return np.load(os.path.join(self.dir, "lmk106.npy"))


def contiguous_starts(fidx, T, stride):
    """Positions s where fidx[s : s+T*stride : stride] is a run of consecutive frame numbers."""
    span = (T - 1) * stride
    ok = [s for s in range(len(fidx) - span) if fidx[s + span] - fidx[s] == span]
    return ok


class Augment:
    """Sample parameters once, apply identically to every clip in a pair."""

    def __init__(self, size=224, jpeg=(50, 100), p_jpeg=0.3, p_blur=0.1, p_flip=0.5):
        self.size, self.jpeg, self.p_jpeg, self.p_blur, self.p_flip = size, jpeg, p_jpeg, p_blur, p_flip

    def sample(self, in_size):
        scale = random.uniform(0.9, 1.1)
        crop = int(round(self.size * scale))
        crop = min(crop, in_size)
        return dict(
            y=random.randint(0, in_size - crop), x=random.randint(0, in_size - crop), crop=crop,
            flip=random.random() < self.p_flip,
            q=random.randint(*self.jpeg) if random.random() < self.p_jpeg else None,
            blur=random.choice([3, 5]) if random.random() < self.p_blur else None,
        )

    def apply(self, clip_u8, p):
        """clip_u8: (T, H, W, 3) uint8 RGB -> (T, 3, size, size) float32 in [0, 1]."""
        import cv2
        out = []
        for fr in clip_u8:
            fr = np.ascontiguousarray(fr[p["y"]:p["y"] + p["crop"], p["x"]:p["x"] + p["crop"]])
            if p["q"] is not None:
                ok, enc = cv2.imencode(".jpg", fr[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), p["q"]])
                fr = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
            if p["blur"] is not None:
                fr = cv2.GaussianBlur(fr, (p["blur"], p["blur"]), 0)
            if p["flip"]:
                fr = fr[:, ::-1]
            out.append(np.ascontiguousarray(fr))
        x = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
        return F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)


class PairedClipDataset(Dataset):
    def __init__(self, index_csv, store_root, T=8, stride=1, p_sbi=0.5, sbi_fn=None,
                 methods=("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"),
                 augment=None, normalize=None, mask_grid=14):
        rows = load_index(index_csv, "train")
        self.root, self.T, self.stride, self.mask_grid = store_root, T, stride, mask_grid
        self.reals = [r for r in rows if r["method"] == "real"]
        self.fakes = defaultdict(list)                     # target_id -> fake rows
        for r in rows:
            if r["method"] in methods:
                self.fakes[r["target_id"]].append(r)
        self.p_sbi, self.sbi_fn = p_sbi, sbi_fn
        self.aug = augment or Augment()
        self.normalize = normalize                          # callable on (T, 3, H, W) or None

    def __len__(self):
        return len(self.reals)                              # exact epoch over real identities

    def __getitem__(self, i):
        real = self.reals[i]
        rs = VideoStore(self.root, real["video_id"])
        use_sbi = self.sbi_fn is not None and (random.random() < self.p_sbi
                                               or not self.fakes[real["target_id"]])
        if use_sbi:
            starts = contiguous_starts(rs.fidx, self.T, self.stride)
            s = random.choice(starts)
            sel = np.arange(s, s + self.T * self.stride, self.stride)
            real_u8 = np.asarray(rs.frames[sel])
            fake_u8, fake_mask = self.sbi_fn(real_u8, rs.lmk()[sel])   # (T,H,W,3), (T,H,W) float
            method = "SBI"
        else:
            fake = random.choice(self.fakes[real["target_id"]])
            fs = VideoStore(self.root, fake["video_id"])
            common, ri, fi = np.intersect1d(rs.fidx, fs.fidx, return_indices=True)
            starts = contiguous_starts(common, self.T, self.stride)
            if not starts:                                   # rare: no common contiguous run; use the next real video
                return self.__getitem__((i + 1) % len(self))
            s = random.choice(starts)
            sel = np.arange(s, s + self.T * self.stride, self.stride)
            real_u8 = np.asarray(rs.frames[ri[sel]])
            fake_u8 = np.asarray(fs.frames[fi[sel]])
            fake_mask = None
            method = fake["method"]

        p = self.aug.sample(real_u8.shape[1])
        clips = torch.stack([self.aug.apply(real_u8, p), self.aug.apply(fake_u8, p)])  # (2,T,3,S,S)
        if self.normalize is not None:
            clips = torch.stack([self.normalize(c) for c in clips])
        g = self.mask_grid
        if fake_mask is not None:
            m = torch.from_numpy(np.ascontiguousarray(fake_mask)).float()[:, None]      # (T,1,H,W)
            m = m[:, :, p["y"]:p["y"] + p["crop"], p["x"]:p["x"] + p["crop"]]
            if p["flip"]:
                m = m.flip(-1)
            m = F.interpolate(m, size=(g, g), mode="area")[:, 0]                         # (T,g,g)
            has_fake_mask = 1.0
        else:
            m = torch.zeros(self.T, g, g)                    # unknown for FF++ fakes without masks
            has_fake_mask = 0.0
        return {
            "frames": clips,
            "label": torch.tensor([0.0, 1.0]),
            "method": torch.tensor([0, METHOD_ID[method]]),
            "mask": torch.stack([torch.zeros_like(m), m]),   # (2,T,g,g); real mask is known to be empty
            "has_mask": torch.tensor([1.0, has_fake_mask]),  # (2,) which items the mask loss may use
        }


def paired_collate(batch):
    """Flatten pairs: frames (2B,T,3,S,S), label (2B,), method (2B,), mask (2B,T,g,g), has_mask (2B,)."""
    return {k: torch.cat([b[k] for b in batch])
            for k in ("frames", "label", "method", "mask", "has_mask")}
```

`mask_grid` is 14 for the EfficientNet model (Phase 1) and 16 for CLIP ViT-L/14 (Phase 2). If you download the FF++ manipulation masks, load them in the non-SBI branch the same way and set `has_fake_mask = 1.0`.

**Evaluation dataset:** `EvalClipDataset(split)` returns, for each video, `K=8` evenly spaced clips of T frames (stride 1), shape `(K, T, 3, 224, 224)`, using a deterministic center crop. Video score = **mean** of clip probabilities (primary, fixed before seeing test results); also report top-50% mean.

### 6.6 [AG] Exact-epoch scheduler

```python
import math
import torch


def cosine_with_warmup(optimizer, total_steps, warmup_frac=0.03, min_ratio=0.01):
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# in train.py, after building the DataLoader:
# steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
# total_steps     = epochs * steps_per_epoch
# sched           = cosine_with_warmup(opt, total_steps)
```

Parameter groups for weight decay:

```python
def param_groups(model, backbone_prefix, lr_backbone, lr_head, wd=0.05):
    groups = {(b, d): [] for b in (True, False) for d in (True, False)}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = n.startswith(backbone_prefix)
        no_decay = p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "embed" in n
        groups[(is_bb, not no_decay)].append(p)
    out = []
    for (is_bb, decay), ps in groups.items():
        if ps:
            out.append({"params": ps, "lr": lr_backbone if is_bb else lr_head,
                        "weight_decay": wd if decay else 0.0})
    return out
```

Use `torch.optim.AdamW(param_groups(model, "rgb_branch.backbone", 5e-5, 5e-4), betas=(0.9, 0.999))`.

### 6.7 [USER] Phase 0 commands

```powershell
python tools/make_ffpp_index.py
python tools/check_split.py data/splits/index.csv
python -m smsdt.preprocess.build_store --index data/splits/index.csv --out data/store --size 256 --margin 1.3 --workers 8
```

### 6.8 [VERIFY] Phase 0

| Check | Pass condition |
|---|---|
| `check_split.py` | Prints `OK`; counts 720/140/140 real and 720/140/140 per method |
| `len(PairedClipDataset)` | 720 |
| One batch from `DataLoader(..., batch_size=8, collate_fn=paired_collate, num_workers=8)` | `frames.shape == (16, 8, 3, 224, 224)`, `label.mean() == 0.5` |
| Pair geometry | Mean absolute pixel difference between real and fake clips of a **NeuralTextures** pair is concentrated in the mouth region (save a heat map) |
| Epoch length | Iterating the loader once yields exactly `ceil(720 / 8) = 90` batches |
| Re-run D1 on the new store | EfficientNet-B4 frame baseline in-domain AUC ≥ 0.93. If not, stop and debug before Phase 1. |

---

## 7. Phase 1: Anti-overfitting and model fixes on the current backbone (Days 3–7)

**Goal:** in-domain FF++ AUC ≥ 0.95, with NT/F2F ≥ 0.90, and a first Celeb-DF v2 number. The backbone stays EfficientNet-B4, so the changes are isolated and each can be ablated.

### 7.1 [AG] Temporally consistent SBI (`smsdt/data/sbi_video.py`)

SBI (Shiohara & Yamasaki, CVPR 2022) creates a fake from a real frame: it blends a slightly transformed copy of the face back onto itself, which produces boundary and statistical inconsistencies with no identity change. For a **video** model, sample the transform parameters **once per clip**. Otherwise the per-frame randomness creates flicker that the temporal module learns as a trivial cue. The mask follows the landmarks frame by frame.

```python
"""Clip-level self-blended images. Same blend parameters for all T frames."""
import random

import cv2
import numpy as np


def _hull_mask(lmk, h, w):
    m = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(lmk.astype(np.int32)), 1.0)
    return m


def _params(h):
    return dict(
        ksize=random.choice([15, 25, 35]) | 1,          # mask blur (odd)
        dilate=random.randint(-int(0.04 * h), int(0.04 * h)),
        ratio=random.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0]),
        hsv=(random.uniform(-0.03, 0.03), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)),
        bc=(random.uniform(0.9, 1.1), random.uniform(-15, 15)),
        scale=random.uniform(0.95, 1.05),
        shift=(random.uniform(-0.02, 0.02) * h, random.uniform(-0.02, 0.02) * h),
        sharpen=random.random() < 0.5,
    )


def _source(frame, p):
    """Slightly altered copy of the face: colour, contrast, sharpness, small scale/shift."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + p["hsv"][0] * 180) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + p["hsv"][1]), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + p["hsv"][2]), 0, 255)
    src = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    src = np.clip(src * p["bc"][0] + p["bc"][1], 0, 255)
    if p["sharpen"]:
        blur = cv2.GaussianBlur(src, (0, 0), 1.0)
        src = np.clip(src * 1.5 - blur * 0.5, 0, 255)
    M = np.float32([[p["scale"], 0, p["shift"][0] + (1 - p["scale"]) * w / 2],
                    [0, p["scale"], p["shift"][1] + (1 - p["scale"]) * h / 2]])
    return cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_REFLECT101)


def sbi_clip(clip_u8, lmk106):
    """clip_u8: (T, H, W, 3) uint8 RGB; lmk106: (T, 106, 2) crop coords.
    Returns fake clip (T, H, W, 3) uint8 and blend masks (T, H, W) float32 in [0, 1]."""
    T, h, w, _ = clip_u8.shape
    p = _params(h)
    kernel = np.ones((3, 3), np.uint8)
    fakes, masks = [], []
    for t in range(T):
        tgt = clip_u8[t].astype(np.float32)
        m = _hull_mask(lmk106[t], h, w)
        if p["dilate"] > 0:
            m = cv2.dilate(m, kernel, iterations=p["dilate"])
        elif p["dilate"] < 0:
            m = cv2.erode(m, kernel, iterations=-p["dilate"])
        m = cv2.GaussianBlur(m, (p["ksize"], p["ksize"]), 0) * p["ratio"]
        src = _source(clip_u8[t], p)
        out = src * m[..., None] + tgt * (1 - m[..., None])
        fakes.append(np.clip(out, 0, 255).astype(np.uint8))
        masks.append(m)
    return np.stack(fakes), np.stack(masks)
```

This is a condensed version of SBI. The original also uses elastic mask deformation and several mask shapes; add them once the basic version trains. Pass `sbi_fn=sbi_clip` to `PairedClipDataset`. **Delete `simulate_sbi_blend` from `dataset.py`.**

### 7.2 [AG] Dense tokenizer (replaces the 7×7 average pooling in PSIT)

Shapes for a 224 input with EfficientNet-B4 `out_indices=(2,3,4)`: P2 `(BT,256,28,28)`, P3 `(BT,256,14,14)`, P4 `(BT,256,7,7)`.

```latex
z = \mathrm{Conv}_{1\times1}\Big(\mathrm{Conv}^{s=2}_{3\times3}(P_2) + P_3 + \mathrm{Up}_{2\times}(P_4)\Big) \in \mathbb{R}^{BT\times 256\times 14\times 14} \;\rightarrow\; (BT, 196, 256)
```

```python
import torch.nn as nn
import torch.nn.functional as F


class DensePSIT(nn.Module):
    """Multi-scale tokens on a 14x14 grid with learned (not average) downsampling."""

    def __init__(self, dim=256):
        super().__init__()
        self.down2 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)    # 28 -> 14
        self.merge = nn.Conv2d(dim, dim, 1)

    def forward(self, fpn):                                          # [P2, P3, P4]
        p2, p3, p4 = fpn
        up4 = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        z = self.merge(self.down2(p2) + p3 + up4)                    # (BT, D, 14, 14)
        return z.flatten(2).transpose(1, 2)                          # (BT, 196, D)
```

Wavelet stem to match: `nn.Conv2d(9, D, kernel_size=8, stride=8)` on the 112×112 subbands → `(BT, D, 14, 14)`, with signed coefficients (doubt 2). Spatial position embedding becomes `(1, 1, 196, D)`.

Cost: spatial attention over 196 tokens vs 49 is 16× more attention-score compute per frame. At D=256 and 4 blocks this is still small next to the backbone.

### 7.3 [AG] Readout: gated attention pooling with top-k patch evidence

```python
import torch
import torch.nn as nn


class EvidenceReadout(nn.Module):
    """z: (B, T, N, D) -> clip logit (B,), patch logits (B, T, N).

    logit = w * head(attention-pooled z) + (1 - w) * mean(top-k patch logits)
    """

    def __init__(self, dim=256, topk_frac=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.att_v = nn.Linear(dim, 128)
        self.att_u = nn.Linear(dim, 128)
        self.att_w = nn.Linear(128, 1)
        self.patch_head = nn.Linear(dim, 1)
        self.head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(dim // 2, 1))
        self.mix = nn.Parameter(torch.zeros(()))         # sigmoid(0) = 0.5
        self.topk_frac = topk_frac

    def forward(self, z):
        B, T, N, D = z.shape
        h = self.norm(z).reshape(B, T * N, D)
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h)))   # (B, TN, 1)
        a = torch.softmax(a.float(), dim=1).to(h.dtype)
        pooled = (a * h).sum(1)                                                    # (B, D)
        patch = self.patch_head(h).squeeze(-1)                                     # (B, TN)
        k = max(1, int(self.topk_frac * T * N))
        top = patch.topk(k, dim=1).values.mean(1)                                  # (B,)
        w = torch.sigmoid(self.mix)
        logit = w * self.head(pooled).squeeze(-1) + (1 - w) * top
        return logit, patch.view(B, T, N)
```

The gated attention follows Ilse et al. (ICML 2018). Top-k over 5% of the T·N patches keeps a small local anomaly from being averaged away, while the softmax pooling path keeps gradients spread over many tokens, which avoids max-pooling saturation.

### 7.4 [AG] Model changes in `smsdt/models/smsdt.py`

1. `self.psit = DensePSIT(embed_dim)`; wavelet stem k=8, s=8; `pos_embed (1,1,196,D)`.
2. Add `time_embed (1,16,1,D)` (doubt 3).
3. Replace the temporal-difference code with `TemporalDiff` (doubt 1). **Stop multiplying by `conf`**; pass `conf` only to build a key-padding mask if frames can be missing (they cannot in the new store).
4. Replace the mean-pool head with `EvidenceReadout`. `forward` returns `(logit, patch_logits)`.

### 7.5 [AG] Loss (`smsdt/losses.py`)

```latex
\mathcal{L} = \mathrm{BCE}_{\varepsilon=0.05}(\hat y, y) \;+\; \lambda_m \cdot \frac{1}{|\mathcal{P}|}\sum_{(b,t,n)\in\mathcal{P}} \mathrm{BCE}\big(s_{b,t,n},\, \mathbb{1}[m_{b,t,n} > 0.1]\big), \qquad \lambda_m = 0.5
```

$s_{b,t,n}$ is the patch logit, $m$ the 14×14 blend mask (zeros for reals), and $\mathcal{P}$ the set of patches of items with `has_mask = 1`.

```python
import torch
import torch.nn.functional as F


def detection_loss(logit, label, patch_logits=None, mask=None, has_mask=None,
                   smoothing=0.05, lam_mask=0.5):
    """logit (B,), label (B,), patch_logits (B,T,N), mask (B,T,g,g), has_mask (B,) in {0,1}."""
    target = label * (1 - smoothing) + 0.5 * smoothing
    loss = F.binary_cross_entropy_with_logits(logit.float(), target)
    logs = {"cls": float(loss)}
    if patch_logits is not None and mask is not None:
        sel = has_mask.bool() if has_mask is not None else torch.ones_like(label, dtype=torch.bool)
        if sel.any():
            m = (mask[sel].flatten(2) > 0.1).float()                # (B', T, g*g)
            lm = F.binary_cross_entropy_with_logits(patch_logits[sel].float(), m)
            loss = loss + lam_mask * lm
            logs["mask"] = float(lm)
    return loss, logs
```

Remove focal, SupCon and decorrelation from the default config. Keep the code for ablation rows.

### 7.6 [AG] Configuration for Phase 1

```yaml
# configs/train.yaml (Phase 1)
epochs: 60                 # 60 x 90 batches (720 pairs / 8) = 5,400 optimizer steps at grad_accum 1
batch_pairs: 8             # 16 clips per step
grad_accum: 1
lr: 5.0e-4                 # head
backbone_lr: 5.0e-5
weight_decay: 0.05
warmup_frac: 0.03
min_lr_ratio: 0.01
grad_clip: 1.0
label_smoothing: 0.05
lambda_mask: 0.5
p_sbi: 0.5
jpeg_quality: [50, 100]
p_jpeg: 0.3
ema_decay: 0.999           # keep an EMA copy of weights; evaluate the EMA model
```

5,400 steps is small. If val AUC is still rising at epoch 60, raise epochs; the schedule recomputes itself. SBI trained for 100 epochs.

Two further cheap wins:

- **Model EMA:** `torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))`, updated after every optimizer step; select checkpoints on the EMA model's val AUC.
- **Frame baseline config:** the same pipeline with T=1, no temporal blocks. This is ablation row "frame-only" and the D1 sanity check.

### 7.7 [USER] Phase 1 runs

```powershell
python -m smsdt.train --config-dir configs --override train.epochs=60 data.store=data/store
python -m smsdt.eval --ckpt outputs/checkpoints/best_ema.pt --split test
```

Run the ablations in Section 11 on 2 seeds each, using the same command with overrides.

### 7.8 [VERIFY] Phase 1

```python
# tests/test_phase1.py
import torch
from smsdt.models.smsdt import SMSDT
from smsdt.losses import detection_loss


def test_shapes_and_gradients():
    model = SMSDT(embed_dim=256, depth=4, heads=8)
    x = torch.randn(4, 8, 3, 224, 224)
    conf = torch.ones(4, 8)
    logit, patch = model(x, conf)
    assert logit.shape == (4,)
    assert patch.shape == (4, 8, 196)
    label = torch.tensor([0.0, 1.0, 0.0, 1.0])
    mask = torch.zeros(4, 8, 14, 14)
    mask[1, :, 5:9, 5:9] = 1.0
    has_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    loss, logs = detection_loss(logit, label, patch, mask, has_mask)
    assert "mask" in logs
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"no gradient reaches: {dead[:5]}"
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients in: {bad[:5]}"


def test_temporal_order_matters():
    model = SMSDT(embed_dim=256, depth=4, heads=8).eval()
    x = torch.randn(1, 8, 3, 224, 224)
    conf = torch.ones(1, 8)
    with torch.no_grad():
        a = model(x, conf)[0]
        b = model(x.flip(1), conf)[0]
    assert not torch.allclose(a, b), "model is invariant to frame order"


def test_temporal_diff_starts_as_identity():
    from smsdt.models.nmsss_vit import TemporalDiff
    td = TemporalDiff(256)
    z = torch.randn(2, 8, 196, 256)
    assert torch.allclose(td(z), z)
```

Also check: an SBI item has `label == 1` and a nonzero mask (save 8 sample images to `outputs/sbi_samples/`), and D3's identity probe accuracy drops compared with the v1 checkpoint.

**Exit criteria:** in-domain FF++ c23 test video AUC ≥ 0.95; NT and F2F ≥ 0.90; Celeb-DF v2 video AUC reported (Phase 3 explains how to get the data).

---

## 8. Phase 2: Foundation backbone (Weeks 2–3)

**Goal:** Celeb-DF v2 video AUC ≥ 0.93 (the SBI/CADDM level), with a path to ≥ 0.95.

### 8.1 Why CLIP ViT-L/14 with LayerNorm-only tuning

LNCLIP-DF (2025) trains only CLIP's LayerNorm affine parameters plus a linear head (about 0.03% of parameters) on FF++ c23. It reports video-level AUC of 0.965 on Celeb-DF v2, 0.973 on DFD, 0.870 on DFDC and 0.935 on FFIW. Its ablation: linear probe 76.1 → LN tuning 90.8 mean AUC across test sets, the largest single gain. It also found full fine-tuning and rank-1 LoRA overfit quickly. Effort (ICML 2025) and Forensics Adapter (CVPR 2025) reach similar numbers with other parameter-efficient schemes.

The design consequence for S-MSDT: **keep the large pretrained encoder nearly frozen, and make your contribution the temporal and local-evidence head on top of its patch tokens.**

### 8.2 [USER] Setup

```powershell
pip install "transformers>=4.40" safetensors
python -c "from transformers import CLIPVisionModel; CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')"
```

The download is about 1.7 GB, into the Hugging Face cache (set `HF_HOME` to a drive with space).

### 8.3 [AG] `smsdt/models/clip_encoder.py`

CLIP uses its own normalization: mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`. **Do not use ImageNet statistics with CLIP.**

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPFrameEncoder(nn.Module):
    """(BT, 3, 224, 224) in CLIP normalization -> cls (BT, 1024), patches (BT, 256, 1024)."""

    def __init__(self, name="openai/clip-vit-large-patch14", tune="layernorm", grad_ckpt=True):
        super().__init__()
        self.vit = CLIPVisionModel.from_pretrained(name)
        for n, p in self.vit.named_parameters():
            p.requires_grad = (tune == "layernorm" and "norm" in n) or tune == "full"
        if grad_ckpt:
            # non-reentrant checkpointing: gradients still reach the LayerNorms even though
            # the input image and most weights do not require grad
            self.vit.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, x):
        out = self.vit(pixel_values=x)
        cls = out.pooler_output                        # post-LayerNorm CLS, (BT, 1024)
        patches = out.last_hidden_state[:, 1:]         # (BT, 256, 1024) for 224 / 14 = 16x16
        return cls, patches
```

`"norm" in n` matches `pre_layrnorm`, `layer_norm1`, `layer_norm2` and `post_layernorm` in the Hugging Face CLIP module names. Print the trainable parameter count once; it should be in the tens of thousands, not millions.

### 8.4 [AG] `SMSDTv2`: CLIP tokens into the S-MSDT temporal head

```latex
\begin{aligned}
c_t, P_t &= \mathrm{CLIP}(x_t), \quad P_t \in \mathbb{R}^{256\times1024} \\
z_t &= \mathrm{LN}(P_t)\,W_p + e^{\mathrm{space}} + e^{\mathrm{time}}_t, \quad W_p \in \mathbb{R}^{1024\times D},\ D=384 \\
z &\leftarrow \mathrm{NMSSS}^{\times 2}(z) \quad \text{(spatial attn} \rightarrow \text{TemporalDiff} \rightarrow \text{temporal attn)} \\
\hat y &= \sigma(\beta)\,\mathrm{Readout}(z) + (1-\sigma(\beta))\, w^\top \tfrac{1}{T}\textstyle\sum_t \hat c_t, \quad \hat c_t = c_t/\lVert c_t\rVert
\end{aligned}
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SMSDTv2(nn.Module):
    def __init__(self, encoder, dim=384, depth=2, heads=6, max_frames=16, grid=16,
                 block_cls=None, readout_cls=None):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, dim))
        self.space = nn.Parameter(torch.zeros(1, 1, grid * grid, dim))
        self.time = nn.Parameter(torch.zeros(1, max_frames, 1, dim))
        nn.init.trunc_normal_(self.space, std=0.02)
        nn.init.trunc_normal_(self.time, std=0.02)
        self.blocks = nn.ModuleList([block_cls(dim, heads) for _ in range(depth)])
        self.readout = readout_cls(dim)
        self.cls_head = nn.Linear(1024, 1)
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, frames):                                     # (B, T, 3, 224, 224)
        B, T = frames.shape[:2]
        cls, patches = self.encoder(rearrange(frames, "b t c h w -> (b t) c h w"))
        z = rearrange(self.proj(patches), "(b t) n d -> b t n d", b=B, t=T)
        z = z + self.space + self.time[:, :T]
        for blk in self.blocks:
            z = blk(z)                                              # (B, T, N, D)
        logit_tok, patch_logits = self.readout(z)
        cls = F.normalize(cls, dim=-1).view(B, T, -1).mean(1)
        logit_cls = self.cls_head(cls).squeeze(-1)
        w = torch.sigmoid(self.beta)
        return w * logit_tok + (1 - w) * logit_cls, patch_logits
```

`block_cls` is the revised `NMSSSBlock` (spatial attention → `TemporalDiff` → temporal attention, no `conf` argument). `readout_cls` is `EvidenceReadout`. The patch grid is 16×16, so resize masks to 16×16 instead of 14×14 in the dataset for this model.

### 8.5 Training schedule for Phase 2 (two stages)

| Stage | What trains | Input | Batch | Epochs | LR |
|---|---|---|---|---|---|
| A: frame | CLIP LayerNorms + `cls_head` (set `beta` so w≈0, or skip the temporal head) | T=1, 8 random frames per pair per item | 32 pairs (64 frames) | 20–40 | LN: sweep {5e-5, 1e-4, 3e-4}; head 5e-4 |
| B: video | Temporal head + readout; LayerNorms at 0.1× their Stage A LR, or frozen | T=8 (then 16, stride 2) | 4–8 pairs | 20–40 | head 3e-4 |

Stage A should reproduce the LNCLIP-DF level on its own. If it doesn't (Celeb-DF v2 video AUC well below 0.94), stop and debug data and normalization before Stage B. Stage B then has to show a measurable gain over Stage A; that gain is the paper's claim.

### 8.6 [VERIFY] Phase 2

```python
# tests/test_phase2.py  (needs the CLIP weights downloaded)
import torch
from smsdt.models.clip_encoder import CLIPFrameEncoder


def test_only_layernorms_train():
    enc = CLIPFrameEncoder(grad_ckpt=False)
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert trainable and all("norm" in n for n in trainable)
    n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in enc.parameters())
    print(f"trainable {n_train:,} / {n_all:,} ({100 * n_train / n_all:.3f}%)")
    assert n_train / n_all < 0.001


def test_patch_grid():
    enc = CLIPFrameEncoder(grad_ckpt=False).eval()
    with torch.no_grad():
        cls, patches = enc(torch.randn(2, 3, 224, 224))
    assert cls.shape == (2, 1024) and patches.shape == (2, 256, 1024)
```

Memory probe before long runs:

```python
import torch


def peak_memory_gb(model, batch_clips, T, steps=3):
    model.cuda().train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        x = torch.randn(batch_clips, T, 3, 224, 224, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, _ = model(x)
        logit.float().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return torch.cuda.max_memory_allocated() / 1e9
```

Pick the largest `batch_clips` that stays under about 28 GB on the 5090. Even with only LayerNorms trainable, gradients must flow back through every layer, so activation memory is that of full training; gradient checkpointing is what makes T=8 batches fit.

**Exit criteria:** Stage A Celeb-DF v2 video AUC ≥ 0.93; Stage B ≥ Stage A + 0.5 points on the average across test sets, over 3 seeds.

---

## 9. Hardware budget (RTX 5090, 32 GB; Core Ultra 9 285K; 64 GB RAM)

All compute figures are my estimates (multiply-accumulate counts), not measurements.

| Item | Estimate | Implication |
|---|---|---|
| CLIP ViT-L/14 @224, per image | ≈ 81 GMAC | 64 images per step ≈ 5.2 TMAC forward |
| EfficientNet-B4 @224 + FPN, per image | ≈ 2.1 GMAC | Phase 1 is cheap; batch size is limited by data loading |
| Spatial attention, 196 vs 49 tokens | 16× the score compute | Negligible next to the backbone at D=256 |
| T=16 vs T=8 | 2× backbone cost per clip | Use T=8 with frame stride 2 to span 15 frames (~0.5 s at 30 fps) for the same cost |
| 336 px CLIP (`clip-vit-large-patch14-336`) | 576 tokens, ≈ 2.4× the cost of 224 | Ablation only after the 224 result is solid |
| Store, 128 frames/video, 256² | ≈ 109 GB train | NVMe; memory-mapped reads; OS page cache uses the 64 GB RAM |

Recommendations:

- **Batch:** Phase 1, 8 pairs (16 clips) without accumulation. Larger batches matter mainly for contrastive losses, which this plan drops.
- **T:** 8 by default; T=16 or stride 2 as an ablation targeted at blink and lip-sync cues.
- **Resolution:** 224 by default; the loose 1.3× crop matters more than more pixels.
- **DataLoader:** `num_workers=8`, `persistent_workers=True`, `pin_memory=True`, `prefetch_factor=4`. On Windows wrap the entry point in `if __name__ == "__main__":` (already done) and keep dataset objects picklable (the classes above are).
- **GPU augmentation:** only if profiling shows the loader as the bottleneck. With memory-mapped uint8 frames and simple ops, 8 workers will likely keep up for Phase 1. Kornia on GPU is the fallback.

---

## 10. Phase 3: Cross-dataset robustness and the paper (Weeks 3–6)

### 10.1 [USER] Datasets to obtain (all require request forms or terms acceptance)

| Dataset | Role | Notes |
|---|---|---|
| Celeb-DF v2 | Primary cross-dataset test | Use its official test list (518 videos) |
| DFD (from FF++ `actors` + `DeepFakeDetection`) | Cross-dataset test | Already downloaded; remove it from FF++ splits |
| DFDC (test set) and DFDC-Preview | Cross-dataset test | Large; use the official test labels |
| FFIW-10K | Multi-face, in-the-wild test | Score the most-fake face per frame, as SBI does |
| WildDeepfake | In-the-wild test | Frame sequences, no video files |
| DeeperForensics-1.0 | Robustness under perturbations | Use its perturbation levels |

Process every test set with **the same** `build_store` pipeline (detector, tracking, 1.3× crop, frame selection).

### 10.2 [AG] Cross-dataset evaluation script

`smsdt/eval_cross.py`: for each dataset, load its index, build `EvalClipDataset`, and write one CSV row with frame AUC, video AUC (mean aggregation), video AUC (top-50%), EER and the number of videos. Fix the aggregation rule **before** looking at test results and report it in the paper.

### 10.3 External real videos (optional)

If you add CelebV-HQ or VoxCeleb2 reals to widen the real-identity distribution:

1. Re-encode with FF++-like settings: `ffmpeg -i in.mp4 -c:v libx264 -crf 23 -preset medium -an out.mp4`, at a resolution similar to FF++ sources.
2. Run the identical `build_store` pipeline.
3. **Apply SBI to the external reals as well**, so the external source supplies both labels and cannot become a "real" cue.
4. Run a source-discrimination test: train a small classifier to tell FF++ reals from external reals. AUC far above 0.5 means a domain signal remains that SBI pairing must neutralize.

### 10.4 [AG] Robustness curves

Evaluate the final model on FF++ test and Celeb-DF v2 under JPEG quality {90, 70, 50, 30}, Gaussian blur σ {0.5, 1, 2}, downscale {0.75, 0.5, 0.25} (then upscale back), and H.264 CRF {23, 30, 35, 40}. Plot AUC against severity for S-MSDTv2 and the strongest baseline.

---

## 11. Experiment plan for the paper

### 11.1 Main table (all trained on FF++ c23, video-level AUC, mean ± std over 3 seeds)

| Method | FF++ (in-domain) | CDF-v2 | DFD | DFDC | DFDC-P | FFIW | WDF | Avg (cross) |
|---|---|---|---|---|---|---|---|---|
| EfficientNet-B4 (your rerun) | | | | | | | | |
| SBI (official weights or rerun) | | | | | | | | |
| AltFreezing (published) | | | | | | | | |
| LNCLIP-DF / Effort / Forensics Adapter (rerun if weights available) | | | | | | | | |
| S-MSDT v1 (current) | | | | | | | | |
| **S-MSDTv2 (ours)** | | | | | | | | |

Mark which numbers are reruns and which are copied from papers, and state the frame/video aggregation for each.

### 11.2 Ablations (FF++ → CDF-v2 and the cross-dataset average; 2 seeds on a fixed budget, 3 for the final rows)

| Group | Variants |
|---|---|
| Data | unpaired vs **paired**; SBI off / SBI only / **FF++ fakes + SBI**; per-frame vs **clip-consistent** SBI; tight vs **1.3× crop** |
| Tokens | 7×7 average pool vs **14×14 learned** (Phase 1) |
| Readout | mean vs attention vs top-k vs **attention + top-k** |
| Supervision | no mask vs **mask loss** |
| Temporal | frame-only (T=1) vs factorized attention vs **+ TemporalDiff** vs + temporal PE; T=8 vs 16 |
| Frequency | none vs Haar abs vs **Haar signed** (keep only if it helps) |
| Backbone | EfficientNet-B4 vs CLIP linear probe vs **CLIP LN-tune** vs CLIP LoRA vs DINOv2 LN-tune |
| Losses | + focal; + SupCon; + decorrelation (show each removed term's effect) |
| `conf` | gated by `conf` vs **no `conf`** (with D2's shortcut evidence) |

### 11.3 Analysis figures

- Per-method AUC on FF++ test (bar chart): NT and F2F before and after Phase 1.
- Identity linear-probe accuracy (D3) for v1, Phase 1 and Phase 2 models. A falling curve supports the "identity-agnostic" claim.
- Patch-logit heat maps on NT and F2F examples: the evidence should sit on the mouth.
- Robustness curves (Section 10.4).
- Temporal sensitivity: AUC when frames are shuffled within a clip. A drop shows the model uses temporal order.

### 11.4 Paper framing

A defensible contribution, given what already exists, is:

> A temporally consistent self-blending scheme for video detectors, combined with an identity-agnostic, local-evidence temporal head over frozen foundation-model patch tokens. It closes the reenactment gap (NeuralTextures, Face2Face) that identity-driven detectors leave, and it improves video-level cross-dataset AUC over the frame-level foundation baseline.

To claim SOTA on Celeb-DF v2, the final number needs to reach the current published range (about 0.96 video-level). If Stage B ends below that, frame the paper as an analysis of when temporal modelling adds value on top of foundation features. That is still publishable if the ablations are clean.

---

## 12. Order of work and exit gates

| Step | Days | Output | Gate to continue |
|---|---|---|---|
| Gate 0 diagnostics | 1 | D1–D5 numbers | Root cause identified |
| Phase 0 protocol and store | 2 | `index.csv`, `data/store`, paired loader | `check_split` OK; EfficientNet-B4 baseline ≥ 0.93 in-domain |
| Phase 1 model fixes and SBI | 4 | S-MSDT v1.5 | In-domain ≥ 0.95; NT, F2F ≥ 0.90 |
| Phase 2 Stage A (CLIP frame) | 4–5 | Frame encoder | CDF-v2 video ≥ 0.93 |
| Phase 2 Stage B (temporal) | 5–7 | S-MSDTv2 | Beats Stage A on the cross-dataset average, 3 seeds |
| Phase 3 evaluation and ablations | 10–14 | Tables 11.1–11.3 | All rows filled |

## 13. Open questions for you

1. Is the test in the spec's Section 8 from the official split file or the default grouping? The video counts suggest the latter. If so, every v1 number must be re-measured after Phase 0.
2. Do you want DFD kept as a cross-dataset test (recommended), or merged into training for a deployment model that is not used for the paper?
3. Do you have, or can you request, Celeb-DF v2, DFDC and FFIW now? Requests can take days, so start them during Phase 0.

---

## Sources

- DeepfakeBench (Yan et al., NeurIPS 2023 Datasets & Benchmarks): in-domain and cross-dataset frame AUCs; 32 frames per video; 1.3 crop margin. [arXiv 2307.01426](https://arxiv.org/pdf/2307.01426)
- Detecting Deepfakes with Self-Blended Images (Shiohara & Yamasaki, CVPR 2022): CDF 93.18, DFD 97.56, DFDC 72.42, DFDCP 86.15, FFIW 84.83 video AUC; training details. [arXiv 2204.08376](https://arxiv.org/pdf/2204.08376) · [CVF page](https://openaccess.thecvf.com/content/CVPR2022/html/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.html)
- Implicit Identity Leakage (Dong et al., CVPR 2023): evidence that binary classifiers encode identity; CADDM CDF 93.88. [arXiv 2210.14457](https://arxiv.org/pdf/2210.14457) · [CVF page](https://openaccess.thecvf.com/content/CVPR2023/html/Dong_Implicit_Identity_Leakage_The_Stumbling_Block_to_Improving_Deepfake_Detection_CVPR_2023_paper.html)
- AltFreezing for More General Video Face Forgery Detection (Wang et al., CVPR 2023): CDF 89.5; FTCN 86.9; LipForensics 82.4. [arXiv 2307.08317](https://arxiv.org/pdf/2307.08317v1)
- Effort: Orthogonal Subspace Decomposition (Yan et al., ICML 2025): CDF-v2 0.956, DFDC 0.843, 0.19M trainable parameters. [arXiv 2411.15633](https://arxiv.org/html/2411.15633v4)
- Forensics Adapter (Cui et al., CVPR 2025): frozen CLIP ViT-L/14 with a 5.7M-parameter adapter and blending-boundary supervision. [arXiv 2411.19715](https://arxiv.org/html/2411.19715v3) · [CVF page](https://openaccess.thecvf.com/content/CVPR2025/html/Cui_Forensics_Adapter_Adapting_CLIP_for_Generalizable_Face_Forgery_Detection_CVPR_2025_paper.html)
- Deepfake Detection that Generalizes Across Benchmarks (LNCLIP-DF, Yermakov et al., 2025): LayerNorm-only CLIP tuning; CDF-v2 96.5, DFD 97.3, DFDC 87.0, FFIW 93.5; paired-data and PEFT findings; Forensics Adapter CDF 95.7 as reported there. [arXiv 2508.06248](https://arxiv.org/html/2508.06248v1)
- Gated attention pooling: Ilse, Tomczak & Welling, "Attention-based Deep Multiple Instance Learning", ICML 2018 (cited from memory, not re-fetched).
- webdataset 1.0.2 source (installed and inspected locally): `WebDataset` defaults and `ResampledShards` seeding.
