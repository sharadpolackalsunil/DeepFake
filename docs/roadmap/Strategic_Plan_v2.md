# S-MSDTv2 — Strategic Research & Engineering Plan (for Antigravity)

Repo: `sharadpolackalsunil/DeepFake` @ `5c3d9f4` · Hardware: RTX 5090 32 GB · Workspace `E:\DEEPFAKE`
Targets: IEEE TIFS / CVPR / ICCV / ECCV

**Status legend** used throughout: **[VERIFIED]** = read in the code or recomputed from your numbers · **[LIKELY]** = strong inference, needs the 3-line check given · **[HYPOTHESIS]** = expected effect, must be measured. I could not run PyTorch in my sandbox, so all code below is syntax-checked only, not executed on GPU.

---

## 0. Executive summary (read this first)

1. **Three code-level defects probably explain more of the NT/F2F gap than the architecture does.** Fix them *before* adding modules, otherwise you cannot tell what helped.
   - **D1 [VERIFIED] One-sided patch-mask loss.** With `p_sbi=0`, FF++ fakes get `has_mask=0`, so `detection_loss` selects only *real* items, whose masks are all zero. The patch head only ever sees "everything is 0". It never learns where a manipulation is. The `λ=0.5` term is a negatives-only regularizer.
   - **D2 [LIKELY] `set_epoch` is a no-op with `persistent_workers=True`.** `train_loader.dataset.set_epoch()` changes the main-process copy; persistent workers hold their own copy made at first iteration. If so, `fakes_list[self.epoch % 5]` is frozen at the value from epoch 0, and every pair uses the *same* fake method for all 40 epochs. That is consistent with FS 0.994 / DF 0.989 (seen) and NT 0.626 (never seen). Check in §6 step 1 (30 seconds).
   - **D3 [VERIFIED] EMA 0.999 vs. step count.** Stage A has ~330 steps: `0.999^330 = 0.72`, so 72 % of the "best_ema" weights are still the *initial* weights. Stage B is 90 steps/epoch × 40 = 3600 steps (fine), but it is initialized from that Stage A EMA.
2. **Your headline numbers are less secure than they look.**
   - **[VERIFIED]** The Master AUC is the fake-count-weighted mean of the per-method AUCs (I reproduced 0.8620 and 0.8669 exactly), and ~79 % of the fakes are DFD. So "0.862 → 0.867" is mostly DFD (0.865 → 0.865) and is within noise (SE ≈ 0.007). Do not claim it as an improvement.
   - **[VERIFIED]** `eval_cross.py` computes the EER threshold *on the evaluated set itself* (`compute_eer(labels, scores_mean)`), so "EER thr 0.027 → balanced acc 76.4 %" on Celeb-DF is an **oracle** number. A reviewer who sees this will call it test-set tuning.
   - **[VERIFIED]** `results.md` uses τ = 0.70 for Phase 1 and τ = 0.029 for Stage B. The optimal threshold moved ×24 between models, which is exactly why τ must be fitted per model on validation and frozen.
3. **Why AUC 0.87/0.86 coexists with Acc@0.5 60 % / 52 %** [VERIFIED mechanism, §4]: the CLIP model emits extremely sharp, conservative scores (reals sit at 1e-4–1.5e-2; many fakes sit at 0.03–0.4). Rank order is good (AUC), but 0.5 is far above where fakes live (NT Acc@0.5 = 2.9 %). On top of that, accuracy depends on class prior (89.6 % fake on Master, 65.6 % on Celeb-DF).
4. **Do these in order:** (a) fix D1–D3 + evaluation protocol (≈ 1 day, no new modules); (b) re-run Stage B; (c) only then add the residual stream / mouth prior / new readout (Stage C). Expected effect sizes are hypotheses, not promises (§5.4).

---

## 1. Deep diagnosis

### 1.1 Why NeuralTextures collapsed in Stage B (0.822 → 0.626)

Phase 1 (EffNet + wavelet) had NT 0.822; Stage A CLIP T=1 had 0.541; Stage B CLIP T=8 recovered only to 0.626. So the loss happened at the *CLIP swap*, and temporal modeling did not rescue it. Causes, ranked:

| # | Cause | Evidence | Status |
|---|---|---|---|
| 1 | **NT is a low-amplitude, mouth-only, texture-level edit.** NeuralTextures re-renders the mouth/lower-face with a neural texture; identity, pose and outline are unchanged. CLIP ViT-L/14 features are semantic and are pretrained to be invariant to exactly this kind of high-frequency change. | Stage A (frozen CLIP + LN-only) NT = 0.54, near chance; the same model scores DF 0.99 / FS 0.97, which add identity/boundary cues CLIP *does* see. | Strong |
| 2 | **Almost no trainable capacity at low level.** LN-only PEFT = ~96k params; it can re-scale features but cannot create a new texture-sensitive feature. Phase 1's wavelet branch (which gave NT 0.822) was dropped. | Param count; Phase 1 vs. Stage A. | Strong |
| 3 | **D2: the model may never see NT during training.** | `video_dataset.py:161` + `persistent_workers: true`. | [LIKELY] — check §6.1 |
| 4 | **D1: no positive localization supervision**, so the patch head has no reason to focus on the mouth. | `losses.py` + `fake_mask=None` for FF++ fakes. | [VERIFIED] |
| 5 | **TemporalDiff is computed on semantic CLIP tokens after a global projection.** NT flicker lives in a ~10 % area of the face and is small vs. head-motion-induced token change; the difference signal is dominated by motion, not by artifacts. Stage B gained +0.085 on NT only. | Code read; NT 0.54 → 0.63. | Plausible |
| 6 | **Saturated training set.** 720 reals/epoch, train AUC 1.0000, final loss 0.1203 vs. label-smoothing BCE floor 0.1169 (ε = 0.05). The model memorizes train identities and uses shortcuts (blend seams, identity), which transfer to DF/FS but not to NT. | `results.md` | [VERIFIED] |

### 1.2 Why Face2Face (0.874) and FaceShifter (0.887) lag FaceSwap (0.994) / Deepfakes (0.989)

- **FS and DF are identity-swap methods with a pasted face region** → a visible blending boundary and identity inconsistency. CLIP's 16×16 patch tokens resolve that seam (your own `results.md` explanation, and I agree).
- **F2F is expression reenactment** (source identity unchanged, mouth/expression geometry changed): the cue is *mouth/jaw region texture and temporal consistency*, not a boundary. Same family as NT; lifting NT lifts F2F.
- **FaceShifter** is a high-quality swap with learned blending (AEI-Net + HEAR-Net): the boundary is soft and the identity is blended; low-level residual statistics matter more.
- If D2 is real, DF/FS "dominance" may simply be "the only method trained on" and F2F/FSh numbers are partly zero-shot transfer.

### 1.3 Component flaws

**`EvidenceReadout`** (gated attention pooling + `patch_head` + top-k mean with learnable scalar `mix`):
- Top-k mean with `topk_frac=0.05` of 256 patches ⇒ k = 12 patches. For NT the manipulated area at 16×16 resolution is ≈ 20–40 patches, but with weak per-patch signal; for DF/FS it is the whole face with strong signal. A hard top-k over-weights single-patch outliers (hurts reals: false positives) and discards the diffuse evidence NT produces.
- Without positive mask supervision (D1) the patch logits are unconstrained: nothing makes them spatially meaningful.
- `mix`/`beta` are 0-dim parameters; `param_groups` uses `p.ndim == 1` for no-decay, so these scalars get weight decay 0.05 and are pulled to 0 ⇒ the token/cls mix and top-k mix are biased toward 0.5. [VERIFIED code read; effect small but free to fix.]

**`TemporalDiff`**:
- Operates on tokens already projected from CLIP; no alignment/motion compensation; differences of *semantic* features are not artifact-specific.
- Only adjacent-frame differences at stride 1 — NT/F2F per-frame jitter is high-frequency; stride 1 and 2 diffs would give both.

**Multi-task loss**:
- Label-smoothed BCE (ε = 0.05) sets a *floor* of 0.1169 and a soft target 0.975/0.025 — harmless for AUC but it is why train loss "converges" while train AUC is already 1.0: no gradient signal is left to fix hard cases.
- No class/method balancing, no hard-example emphasis; patch BCE negatives-only (D1).
- Model selection by *pooled* val AUC (DFD-dominated) ⇒ you pick checkpoints that are best at DFD, not at NT/F2F.
- `init_ckpt` loaded with `strict=False` — a silent key mismatch would pass unnoticed. Print `missing`/`unexpected` and assert.

---

## 2. SMSDTv2 → **SMSDTv3 (Stage C)** modifications

Design rule: **every new branch is gated and zero-initialized**, so at step 0 the network computes exactly what `best_ema.pt` computes. FS 0.994 / DF 0.989 cannot regress at initialization, and any later regression is attributable to training, which you guard with per-method floors (§3.5).

### 2.1 Dual-stream: high-frequency residual injection into CLIP tokens

**Idea.** Keep frozen-CLIP semantics (stream S). Add a light, trainable *residual stream* R that sees only high-frequency content, computed on the 224 input, tokenized onto the same 16×16 grid, and *injected* into the projected CLIP tokens by a zero-initialized gate.

**Math.** Let `x ∈ R^{3×224×224}` (denormalized to [0,1]).

1. Fixed high-pass bank (3 SRM kernels `K_SRM ∈ R^{3×5×5}`, applied per channel with truncation to [-2, 2]) and level-1/level-2 Haar subbands:
   `r = [ trunc(K_SRM * x) , LH¹(x), HL¹(x), HH¹(x) ↑2 , HH²(x) ↑4 ]` with `↑` nearest-neighbour upsampling to 112×112 (Haar halves resolution). Channels: 9 (SRM) + 9 (L1 subbands) + 3 (HH²) = 21.
2. Small CNN `φ`: stride-2 conv ×2 → 28×28, then adaptive average pool to 16×16 → `R ∈ R^{C_r×16×16}`, `C_r = 128`, then `1×1` conv to `d = 384`.
3. Token-wise gated injection into projected CLIP tokens `s_i ∈ R^{384}`:
   `g_i = σ(W_g [s_i ; R_i] + b_g)`, `W_g` zero-init, `b_g = -4` (⇒ g ≈ 0.018 at start)
   `ẑ_i = s_i + g_i ⊙ LN(R_i)`
4. Deep supervision on R alone (frequency supervision): `ℓ_freq = BCE( w_r · mean_i LN(R_i) , y )`, which forces R to be discriminative by itself (prevents it from being ignored behind the strong CLIP path).

Notes: no LN-only restriction applies to R (it is new and trained at the head LR). Cost: a ~0.4 M-param CNN; negligible vs. ViT-L.

### 2.2 Landmark-guided perioral attention prior

Landmarks are already stored per frame as `lmk106.npy` in **crop coordinates of the 256×256 store** (`build_store.py:221`). Training uses random scale 0.9–1.1 crops and 224 center crops at eval, so the prior must be warped with the *same* crop/flip: eval offset is `(256-224)/2 = 16 px` and train applies scale + flip — apply the identical affine to the landmarks inside `Augment` and return `lmk224`.

Mouth block of InsightFace `2d106det` is commonly indices **52–71** (outer + inner lips) — **verify by plotting once** (`MOUTH_IDX` is a config, don't trust my memory). Build a soft prior on the 16×16 grid:

`m_i = exp(-d(c_i, Hull_mouth)² / 2σ²)`, `σ = 1.2 patches`, `c_i` = patch centers, d = 0 inside the hull. Dilate by 1 patch so jaw/chin transition is included (F2F/NT also edit these).

Use it as an *additive attention bias* in the readout (§2.4) and as an *extra supervised mask target* for NT/F2F pairs (§2.3). Also a **mouth-ROI token**: native-resolution (store 256) 96×96 mouth crop → the residual CNN → one pooled vector added as an extra token to the temporal transformer. This gives native-resolution, no CLIP resampling blur on the region NT edits.

### 2.3 Difference-derived manipulation masks (fixes D1)

Real and fake store crops are **pixel-aligned** (fake reuses the real's `transforms.npy`), so the manipulated region is recoverable from `|fake − real|`:

```
d = max over C of |fake_u8 − real_u8|  →  gaussian blur (σ=3)  →  per-clip normalize by 95th pct
mask = d > τ_m   (τ_m ≈ 0.15, tune on 20 clips visually)
pool to 16×16 (mean) → soft target in [0,1]
```

Then the mask loss uses **both** classes: positives with their diff-mask, negatives with zeros. Sanity-check by dumping 20 overlays before training.

### 2.4 Reformulated `EvidenceReadout` (v3)

Changes: (i) K = 4 learned queries cross-attend over all T×256 tokens (diffuse + local evidence), (ii) additive log-prior from §2.2, (iii) **log-sum-exp (LSE) pooling** replaces hard top-k, (iv) temporal pooling by LSE over frames, (v) scalars excluded from weight decay.

`s = (1/r) · log( (1/N) Σ_i exp(r · ℓ_i) )`, with `r` learned, `r₀ = 3` (r→0 gives the mean, r→∞ gives max; this interpolates between "diffuse" NT and "local" DF evidence).

### 2.5 PyTorch modules

```python
# smsdt/models/v3_modules.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_bank(device, dtype):
    ll = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device) / 2
    lh = torch.tensor([[1, 1], [-1, -1]], dtype=dtype, device=device) / 2
    hl = torch.tensor([[1, -1], [1, -1]], dtype=dtype, device=device) / 2
    hh = torch.tensor([[1, -1], [-1, 1]], dtype=dtype, device=device) / 2
    return ll, lh, hl, hh


def haar_dwt(x):
    """x: (N,C,H,W) -> LL, LH, HL, HH each (N,C,H/2,W/2)."""
    N, C, H, W = x.shape
    ll, lh, hl, hh = _haar_bank(x.device, x.dtype)
    k = torch.stack([ll, lh, hl, hh]).unsqueeze(1)          # (4,1,2,2)
    k = k.repeat(C, 1, 1, 1)                                  # (4C,1,2,2)
    y = F.conv2d(x, k, stride=2, groups=C)                    # (N,4C,H/2,W/2)
    y = y.view(N, C, 4, H // 2, W // 2)
    return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


def _srm_kernels():
    k1 = torch.tensor([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
                       [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=torch.float32) / 4
    k2 = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12
    k3 = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0],
                       [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32) / 2
    return torch.stack([k1, k2, k3])                           # (3,5,5)


class ResidualStream(nn.Module):
    """High-frequency (SRM + Haar) residual tokens on the 16x16 CLIP grid,
    injected into projected CLIP tokens through a zero-init gate."""

    def __init__(self, dim=384, c_r=128, grid=16):
        super().__init__()
        self.grid = grid
        self.register_buffer("srm", _srm_kernels().unsqueeze(1).repeat(3, 1, 1, 1))  # (9,1,5,5)
        self.cnn = nn.Sequential(
            nn.Conv2d(21, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, c_r, 3, 2, 1), nn.GroupNorm(8, c_r), nn.GELU(),
        )
        self.to_dim = nn.Conv2d(c_r, dim, 1)
        self.ln_r = nn.LayerNorm(dim)
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        self.aux = nn.Linear(dim, 1)  # frequency-only deep-supervision head

    def residual_input(self, x01):
        N, C, H, W = x01.shape
        srm = F.conv2d(x01, self.srm, padding=2, groups=3).clamp(-2, 2)           # (N,9,H,W)
        _, lh1, hl1, hh1 = haar_dwt(x01)
        l1 = torch.cat([lh1, hl1, hh1], 1)                                         # (N,9,H/2,W/2)
        _, _, _, hh2 = haar_dwt(haar_dwt(x01)[0])
        hh2 = F.interpolate(hh2, size=(H // 2, W // 2), mode="nearest")           # (N,3,..)
        srm = F.avg_pool2d(srm, 2)                                                 # to H/2
        return torch.cat([srm, l1, hh2], 1)                                        # (N,21,H/2,W/2)

    def forward(self, x01, s):
        """x01: (B*T,3,224,224) in [0,1]; s: (B*T,N,dim) projected CLIP tokens."""
        r = self.cnn(self.residual_input(x01))
        r = F.adaptive_avg_pool2d(r, self.grid)
        r = self.to_dim(r).flatten(2).transpose(1, 2)                              # (N,256,dim)
        r = self.ln_r(r)
        g = torch.sigmoid(self.gate(torch.cat([s, r], -1)))
        aux_logit = self.aux(r.mean(1)).squeeze(-1)                                # (B*T,)
        return s + g * r, aux_logit


class EvidenceReadoutV3(nn.Module):
    """K learned queries + log-prior (mouth) + LSE pooling over patches and time."""

    def __init__(self, dim=384, heads=6, n_queries=4, r0=3.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.kv = nn.Linear(dim, 2 * dim)
        self.heads = heads
        self.patch_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2),
                                        nn.GELU(), nn.Linear(dim // 2, 1))
        self.prior_gain = nn.Parameter(torch.zeros(()))          # starts neutral
        self.log_r = nn.Parameter(torch.tensor(math.log(r0)))
        self.out = nn.Sequential(nn.LayerNorm(dim * n_queries), nn.Linear(dim * n_queries, 1))
        self.mix = nn.Parameter(torch.zeros(()))

    @staticmethod
    def lse(x, r, dim):
        n = x.shape[dim]
        return (torch.logsumexp(r * x, dim=dim) - math.log(n)) / r

    def forward(self, z, prior=None):
        """z: (B,T,N,D); prior: (B,T,N) soft mouth prior in [0,1] or None.
        Returns clip logit (B,), patch_logits (B,T,N)."""
        B, T, N, D = z.shape
        patch_logits = self.patch_head(z).squeeze(-1)                              # (B,T,N)
        k, v = self.kv(z.reshape(B, T * N, D)).chunk(2, -1)
        H = self.heads
        dh = D // H
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        qh = q.view(B, -1, H, dh).transpose(1, 2)
        kh = k.view(B, T * N, H, dh).transpose(1, 2)
        vh = v.view(B, T * N, H, dh).transpose(1, 2)
        att = (qh @ kh.transpose(-1, -2)) / math.sqrt(dh)                          # (B,H,K,TN)
        if prior is not None:
            att = att + self.prior_gain * torch.log(prior.reshape(B, 1, 1, T * N) + 1e-3)
        att = att.softmax(-1)
        pooled = (att @ vh).transpose(1, 2).reshape(B, -1)                          # (B,K*D)
        logit_att = self.out(pooled).squeeze(-1)
        r = self.log_r.exp()
        logit_lse = self.lse(self.lse(patch_logits, r, dim=2), r, dim=1)           # patches, then time
        w = torch.sigmoid(self.mix)
        return w * logit_lse + (1 - w) * logit_att, patch_logits
```

### 2.6 Wiring into `SMSDTv2.forward`

```python
# in __init__:   self.res = ResidualStream(dim); self.readout = EvidenceReadoutV3(dim, heads)
# in forward(frames, conf=None, prior=None):
#   frames are CLIP-normalized -> undo normalization to get [0,1] for the residual stream
x = rearrange(frames, "b t c h w -> (b t) c h w")
cls, patches = self.encoder(x)
s = self.proj(patches)                                            # (BT,N,384)
x01 = (x * self.clip_std + self.clip_mean).clamp(0, 1)            # buffers (1,3,1,1)
s, aux_logit = self.res(x01, s)
z = rearrange(s, "(b t) n d -> b t n d", b=B, t=T)
# ... existing space/time embeddings, NMSSS blocks ...
logit, patch_logits = self.readout(z, prior)
return logit, patch_logits, rearrange(aux_logit, "(b t) -> b t", b=B, t=T).mean(1)
```

Keep the old `cls_head` path in place for warm-start; gate it with `mix` as today. Load `best_ema.pt` with `strict=False`, **print and assert** that only keys starting with `res.`, `readout.` (and any new buffers) are missing.

---

## 3. Stage C: loss and training strategy (fine-tune `outputs/checkpoints_v2_stage_b/best_ema.pt`)

### 3.1 Data fixes (do these first; they alone may move NT/F2F)

```python
# smsdt/data/video_dataset.py  — replace round-robin by per-item balanced sampling (no epoch state)
METHOD_W = {"Deepfakes": 0.15, "FaceSwap": 0.15, "FaceShifter": 0.20,
            "Face2Face": 0.25, "NeuralTextures": 0.25}   # tilt toward weak methods; re-tune per val AUC

def pick_fake(fakes_list, rng):
    ws = [METHOD_W.get(f["method"], 0.1) for f in fakes_list]
    return rng.choices(fakes_list, weights=ws, k=1)[0]
# inside __getitem__: rng = random.Random((self.seed, idx, self.epoch_counter.value))  # or just random.Random()
# fake = pick_fake(fakes_list, rng)
```

Stateless sampling needs no `set_epoch`, so persistent workers are fine. Also set `persistent_workers: false` for any config that still relies on `set_epoch`.

Other data changes: (a) sample **2 different temporal offsets per real/fake pair per epoch** (more data per identity, same pairing); (b) add **mouth-SBI** pseudo-fakes at `p=0.2`: take the real clip, apply a random low-pass / downsample-×4–8-upsample / JPEG(30–70) re-synthesis inside a mouth-hull mask, blend, label = fake, mask = the hull (exact). The repo already has `sbi_clip(clip_u8, lmk106)`; copy its structure but swap the face hull for the mouth hull `MOUTH_IDX`. This creates NT-like labeled positives with perfect masks.

### 3.2 Losses

Per clip `i`, label `y_i`, method `m_i`, logit `ℓ_i`:

```
L = L_cls + λ_rank · L_rank + λ_mask · L_mask + λ_freq · L_freq
```

| Term | Definition | Weight |
|---|---|---|
| `L_cls` | **Focal** BCE on fakes, plain BCE on reals, **no label smoothing**: `-α(1-p)^γ log p` for y=1 (γ=2, α=0.5), `-(1-α) p^γ log(1-p)` for y=0 (γ=1) multiplied by method weight `w_m` (below) | 1.0 |
| `L_rank` | Pairwise logistic over the *paired* real/fake of the same identity: `softplus(-(ℓ_fake − ℓ_real − margin))`, margin 1.0. Direct AUC surrogate; uses the pairing that already exists in the batch | 0.5 |
| `L_mask` | BCE(patch_logits, soft mask) on **all** items; positives from §2.3 (or mouth-SBI hull), negatives zeros; `pos_weight = 3` because fake area is the minority | 0.5 → 1.0 (ramp over first 3 epochs) |
| `L_freq` | BCE(aux_logit from `ResidualStream`, y) | 0.3 |

**Method-balanced weights** `w_m`: recomputed each epoch from val per-method AUC: `w_m ∝ (1 − AUC_m + 0.1)`, normalized to mean 1, clipped to [0.5, 2.0], with `FS, DF` floored at 0.7 so they are not abandoned.

**Hard-negative mining.** Within each batch of 8 pairs, keep all reals and the **top 50 % highest-loss fakes** (OHEM) after a 2-epoch burn-in. Reals that score high are hard negatives; with 99 % specificity there are few, so also keep a small **memory bank** of the top-64 highest-scoring reals from the previous epoch and mix 2 of them per batch.

```python
# smsdt/losses_v3.py
import torch
import torch.nn.functional as F


def focal_bce(logit, y, alpha=0.5, g_pos=2.0, g_neg=1.0):
    logit = logit.float()
    p = torch.sigmoid(logit)
    lp = F.logsigmoid(logit)
    ln = F.logsigmoid(-logit)
    pos = -alpha * (1 - p).pow(g_pos) * lp
    neg = -(1 - alpha) * p.pow(g_neg) * ln
    return y * pos + (1 - y) * neg


def detection_loss_v3(logit, label, method_w, patch_logits, soft_mask, aux_logit, pair_idx,
                      lam_rank=0.5, lam_mask=0.5, lam_freq=0.3, margin=1.0,
                      ohem_frac=0.5, pos_weight=3.0):
    """label: (B,) float; method_w: (B,) float; patch_logits/soft_mask: (B,T,N) or (B,N);
    pair_idx: (B,) index of the paired partner (real<->fake) inside the batch."""
    b = label.shape[0]
    cls = focal_bce(logit, label) * method_w
    if ohem_frac < 1.0:
        fake = label > 0.5
        n_keep = max(1, int(ohem_frac * int(fake.sum())))
        fake_loss = cls.masked_fill(~fake, float("-inf"))
        thr = torch.topk(fake_loss, n_keep).values.min()
        keep = (~fake) | (cls >= thr)
        cls = (cls * keep).sum() / keep.sum().clamp(min=1)
    else:
        cls = cls.mean()

    is_fake = label > 0.5
    fk = torch.nonzero(is_fake).squeeze(1)
    if fk.numel() > 0:
        d = logit[fk].float() - logit[pair_idx[fk]].float()
        rank = F.softplus(-(d - margin)).mean()
    else:
        rank = logit.new_zeros(())

    pl = patch_logits.reshape(b, -1).float()
    sm = soft_mask.reshape(b, -1).float().clamp(0, 1)
    w = 1.0 + (pos_weight - 1.0) * sm
    mask = (F.binary_cross_entropy_with_logits(pl, sm, reduction="none") * w).mean()

    freq = F.binary_cross_entropy_with_logits(aux_logit.float(), label.float())
    total = cls + lam_rank * rank + lam_mask * mask + lam_freq * freq
    return total, {"cls": cls.detach(), "rank": rank.detach(),
                   "mask": mask.detach(), "freq": freq.detach()}
```

### 3.3 Optimizer / schedule (exact)

| Group | Params | LR | WD |
|---|---|---|---|
| CLIP LayerNorms (existing) | `encoder.*ln*` | **5e-6** | 0 |
| Existing head (`proj`, NMSSS blocks, old readout) | | **1e-4** | 0.05 (not on `ndim<2` or scalars) |
| New modules (`res.*`, `readout.*`) | | **3e-4** | 0.05 (not on `ndim<2` or scalars) |

- Optimizer **AdamW**, betas (0.9, 0.999), eps 1e-8; bf16 autocast; grad-clip 1.0.
- **Phase C1 (epochs 1–3):** freeze CLIP LN and existing head; train only new modules. Reason: new modules are zero-gated, this lets them learn without disturbing FS/DF.
- **Phase C2 (epochs 4–28):** unfreeze all, linear warmup 5 % then cosine to `min_lr_ratio = 0.01`.
- **Total 28 epochs × 90 steps ≈ 2,520 steps** (batch_pairs 8, T=8). Wall-clock should be roughly Stage B × 0.7 plus ~15 % for the residual stream. I have no timing of your Stage B run, so budget from your own log.
- **EMA:** decay warm-up `d_t = min(0.998, (1+t)/(10+t))` (never lets initial weights dominate); evaluate the EMA only.
- Fix no-decay rule: `no_decay = p.ndim < 2 or name.endswith(("beta", "mix", "log_r", "prior_gain"))`.
- Seeds: 3 seeds for the final model (mean ± std in the paper).

### 3.4 What to log every epoch
Per-method val AUC (6 sets), pooled and **macro** AUC, mean logit of reals/fakes per method, `g` gate mean (should rise from 0.018 slowly), `prior_gain`, `r`, fraction of each method actually sampled in the epoch (this would have exposed D2).

### 3.5 Model selection and regression guards
Select the checkpoint by **macro-mean of per-method AUC over {DF, F2F, FS, FSh, NT}** on FF++ val, *subject to* `AUC_FS ≥ 0.985` and `AUC_DF ≥ 0.98`; DFD stays a separate cross-domain monitor and is excluded from selection. This prevents DFD from steering the selection.

---

## 4. Scientific protocol: accuracy, calibration, thresholds

### 4.1 Why high AUC and low Acc@0.5 coexist

AUC measures only *ranking*: `P(score_fake > score_real)`. Accuracy at 0.5 additionally needs (i) the score distributions to straddle 0.5 and (ii) is prior-dependent. Here: (i) reals cluster at 1e-4–1.5e-2 and a large part of the fake mass is at 0.03–0.4, so 0.5 is above most fakes (NT Acc@0.5 = 2.9 %, FF++ 60.2 %); (ii) a threshold chosen on a 90 %-fake set is not optimal on a 66 %-fake set (Celeb-DF). Temperature scaling and Platt scaling are monotone maps ⇒ **AUC is unchanged**; they only move where 0.5 falls.

### 4.2 What top venues do (and what to do)

- DeepfakeBench-style protocol (which most CVPR/ICCV/TIFS comparison tables follow): **frame-level and/or video-level ACC (fixed 0.5), AUC, AP, EER**; cross-dataset tables are essentially **AUC-only**. Nobody serious tunes the threshold on the test set; doing so invites a data-snooping rejection.
- Therefore: **AUC is your headline metric**; Acc is secondary, and any accuracy shown must use a threshold fixed *before* looking at the test set.

### 4.3 The protocol to adopt (freeze it in the paper)

1. **Calibration set = FF++ validation only** (140 reals + 700 fakes), not including DFD actors (a different domain; it would contaminate τ). Per-method balanced.
2. Fit **Platt scaling with class-balanced sample weights** on the video-level mean logit `u`: `p = σ(a·u + b)`. Balanced weights ⇒ the calibrated 0.5 equals the balanced-error optimum under a 50/50 prior. (Temperature-only `a·u` is a special case, `b=0`, and cannot shift the operating point; use Platt.)
3. Choose **τ on validation** by one rule, stated in advance: **maximize Youden J = TPR − FPR** (equivalent to maximizing balanced accuracy). Report EER-τ as an alternative, not as the primary.
4. **Freeze** `(a, b, τ)` and apply them unchanged to FF++ test, DFD, and Celeb-DF v2. Cross-dataset accuracy under frozen τ will be lower (covariate shift) — report it honestly; do **not** re-tune on Celeb-DF.
5. If you also want to show the upper bound, add an **"oracle τ"** column explicitly labeled *not a valid operating point*.

### 4.4 Main-table metrics

| Role | Metric |
|---|---|
| **Primary** | Video-level AUC (with 95 % CI) |
| Secondary | AP, EER, **TPR @ FPR = 1 % and 5 %**, balanced accuracy @ frozen τ |
| Calibration | ECE (15 bins) before/after Platt, reliability diagram in the appendix |
| Appendix only | Acc@0.5 (uncalibrated, for comparability with DeepfakeBench-style tables), threshold sweep, frame-level numbers |

Report **per-method** AUC with CIs, and never the pooled "Master" alone — pooling is dominated by the method with the most videos (here DFD ≈ 79 % of fakes).

### 4.5 Presenting ROC and threshold sweeps without cherry-picking

- State the threshold rule before results; show the **full ROC** (log-x axis to expose the low-FPR region) with the frozen operating point marked.
- Show the sweep as a **figure from validation**, with the test point at the validation-chosen τ — the test curve is displayed but never used to choose.
- 95 % CIs: **DeLong** for AUC (or video-level bootstrap with 2,000 resamples), bootstrap for TPR@FPR and balanced accuracy. With 518 Celeb-DF videos the AUC SE is ≈ 0.015–0.02, so differences < 0.02 between your variants are not significant — use paired bootstrap on the same videos.
- Release the per-video score CSVs and the exact `(a, b, τ)` with the code.

```python
# tools/calibrate.py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def fit_platt(u, y, iters=200):
    """u: (n,) mean logits, y: (n,) in {0,1}. Class-balanced weighted logistic regression."""
    u = torch.tensor(u, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    w = torch.where(y > 0.5, 0.5 / y.sum(), 0.5 / (1 - y).sum())
    a = torch.ones((), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = a * u + b
        loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(a), float(b)


def youden_tau(p, y):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def tpr_at_fpr(p, y, target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target, fpr, tpr))


def bal_acc(p, y, tau):
    pred = p >= tau
    return 0.5 * ((pred[y == 1]).mean() + (~pred[y == 0]).mean())


def boot_ci(fn, p, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx_r, idx_f = np.where(y == 0)[0], np.where(y == 1)[0]
    for _ in range(n):   # stratified, video-level resampling
        s = np.concatenate([rng.choice(idx_r, len(idx_r)), rng.choice(idx_f, len(idx_f))])
        vals.append(fn(p[s], y[s]))
    return np.percentile(vals, [2.5, 97.5])


# usage:
# a, b = fit_platt(u_val, y_val)
# p_val = 1/(1+np.exp(-(a*u_val+b)));  tau = youden_tau(p_val, y_val)      # FROZEN
# p_ce  = 1/(1+np.exp(-(a*u_cdf+b)));  bal_acc(p_ce, y_cdf, tau); roc_auc_score(y_cdf, p_ce)
```

Also patch `eval_cross.py`: add `--calib path.json` (with `a, b, tau`) so that all printed accuracy uses the frozen values, and rename the current `eer_thresh`-based rows as `oracle_*`.

---

## 5. Celeb-DF v2 cross-dataset protocol and SOTA targets

### 5.1 Protocol

- **Train:** FF++ c23 (the five methods you use + their reals), **no DFD, no Celeb-DF, no external data**. State this explicitly: several competitors train on FF++ c23 only, which is the apples-to-apples setting.
- **Test:** official Celeb-DF v2 test list (518 videos = 178 real + 340 fake, prevalence 65.6 % — matches your N = 518).
- Same preprocessing as FF++ (1.3 margin, SCRFD + 106-pt landmarks, 224 crop), **K = 8 clips × T = 8**, same aggregation as FF++ eval. Aggregation chosen *before* seeing Celeb-DF (mean over clips; also report top-50 % as ablation on FF++ val only).
- Report **video-level AUC (primary)**, AP, EER, TPR@FPR 1 %/5 %, balanced acc @ frozen τ. Also report **frame-level AUC** separately: DeepfakeBench-style baselines are reported at frame level, and mixing the two in one column is the most common reviewer objection.

### 5.2 Baselines and how 0.856 compares

Figures gathered from the papers/benchmarks I could open this session (re-verify each against the original table before camera-ready; setups differ in backbone, training set and level):

| Method | Celeb-DF v2 AUC | Level / note |
|---|---|---|
| RECCE, SPSL, F3-Net, UCF (DeepfakeBench) | 0.73 – 0.77 | frame-level, FF++ c23 trained |
| P&P | 94.7 | video-level |
| DFD-FCG | 95.0 | video-level |
| LAA-Net | 95.4 | video-level |
| RAE | 95.5 | video-level |
| Effort | 95.6 | video-level, CLIP + SVD adapters |
| ForAda (Forensics Adapter) | 95.7 | video-level, CLIP |
| SBI | 93.2 | video-level, self-blended pseudo-fakes |
| LNCLIP-DF | 96.5 | video-level, CLIP LN-tuning |

**Reading:** your **0.856 is above the DeepfakeBench frame-level numbers of RECCE/SPSL/F3-Net (0.73–0.77), but those are a different aggregation level, so it is not a like-for-like win**; against the video-level group it is **~10 pts below ForAda (95.7) and ~10.9 below LNCLIP-DF (96.5)**. Because LNCLIP-DF uses the same LN-only idea on CLIP, your direct competitor is LNCLIP-DF/Effort/ForAda; a TIFS/CVPR claim needs ≥ 0.95, and a "SOTA" claim needs > 96.5 (or a clear robustness/efficiency story if you stay at ~0.95).

### 5.3 AUC targets

| Claim | Celeb-DF v2 video AUC |
|---|---|
| Competitive (workshop / lower-tier) | ≥ 0.90 |
| Parity with ForAda/Effort (credible main-track) | ≥ 0.955 |
| SOTA claim | > 0.965 with CI excluding LNCLIP-DF's 0.965 |

### 5.4 What can raise 0.856 → 0.95+ (ordered by my expected leverage — all [HYPOTHESIS], measure each as an ablation)

1. **SBI / pseudo-fake training** (`p_sbi ≥ 0.3`, currently 0.0). SBI alone reaches 93.2 on Celeb-DF; it is the single strongest published lever for cross-dataset because it removes method-specific shortcuts. Your repo already has `sbi_clip`; it is switched off. Turn it on *with proper masks* (D1 fix).
2. **Fix D1–D3** (positive mask supervision, real method balance, honest EMA).
3. **Residual stream + mouth prior** (§2) for F2F/NT-type cues.
4. **More adaptation capacity on CLIP**: LN-only (~96k) is the weakest point vs. Effort/ForAda; add rank-8 LoRA on the last 8 attention blocks (~0.6 M) at LR 1e-4 — ablate LN-only vs. LN+LoRA.
5. **Less memorization**: train AUC 1.0 / loss at the floor means the data is exhausted; use 2 clips/pair/epoch (§3.1), stronger augmentation (compression, color, down-up resampling), and early stop by macro val AUC.
6. **Test-time**: K = 16 clips, mean of logits (not probabilities), hflip TTA.

Honest expectation: I would not promise 0.95+. A realistic planning range after 1–4 is 0.90–0.94; reaching ≥ 0.955 probably needs SBI plus the LoRA ablation to work. Decide the paper's story by the measured result (§6 gates).

---

## 6. Step-by-step roadmap for Antigravity

Rules: work in `E:\DEEPFAKE`; one git branch per step; never touch the test/Celeb-DF lists for any decision; log every run in `outputs/runs/<name>/` with config + seed + git hash. Each step has a **gate** — do not proceed until it passes.

**Step 1 — Verify D2 (≈ 5 min).** Add to the training loop, first 3 epochs, and also after each `set_epoch`:
```python
import collections
cnt = collections.Counter()
for i, batch in enumerate(train_loader):
    m = batch["method"][1::2]          # fakes sit at odd positions per the paired layout; adapt if not
    cnt.update(m.tolist() if hasattr(m, "tolist") else list(m))
    if i == 20: break
print(epoch, cnt)
```
**Gate:** if one method dominates across epochs ⇒ D2 confirmed. Either way, apply the stateless sampler of §3.1 and set the 5 method fractions logged per epoch.

**Step 2 — Hygiene fixes (no new modules).**
- Stateless balanced fake sampler (§3.1); `persistent_workers: false` where `set_epoch` remains.
- Difference masks (§2.3); dump 20 overlays to `outputs/debug/masks/`; eyeball. **Gate:** masks cover mouth for NT/F2F and face for DF/FS.
- Mask loss on all items (replace the `sel = has_mask` branch).
- `strict=False` → print/assert missing/unexpected keys. EMA warm-up (§3.3). No-decay rule for scalars.
- Selection by macro per-method AUC with FS/DF floors (§3.5).

**Step 3 — Evaluation protocol patch (can run in parallel).** Add `tools/calibrate.py`; `eval_cross.py --calib`; rename EER-on-test outputs to `oracle_*`; add bootstrap CIs, TPR@FPR, AP, ECE; output per-video score CSV. **Gate:** re-evaluate the *existing* Stage B and Celeb-DF scores with the frozen val-fitted `(a, b, τ)`; record the new honest accuracy numbers. This is your baseline for the paper.

**Step 4 — Re-run Stage B with Step-2 fixes only (the "v2-fixed" ablation row).**
`python -m smsdt.train_v2 --config configs/train_v2_stage_b_fixed.yaml` (same hyperparameters as `train_v2_stage_b.yaml`, init from Stage A `best_ema.pt`; if Stage A is also affected by EMA, re-run Stage A with `ema_decay: 0.99`, ~330 steps).
**Gate:** NT and F2F val AUC. If NT ≥ 0.75 here, D2 was the main cause and the architecture work is a bonus, not a rescue. Record Celeb-DF.

**Step 5 — Implement v3 modules** (`v3_modules.py`, `losses_v3.py`; wire as §2.6). Unit tests before training: (a) with gates at init, `model_v3(x)` equals `model_v2(x)` within 1e-4 on 8 random clips; (b) shapes for T = 1 and T = 8; (c) mouth-hull overlay on 20 crops confirms `MOUTH_IDX`; (d) landmark transform matches augmentation (flip + scale).

**Step 6 — Stage C training** (`configs/train_v3_stage_c.yaml`): init `outputs/checkpoints_v2_stage_b/best_ema.pt` (or the Step-4 checkpoint); §3.3 schedule; §3.2 losses; §3.4 logging. Run 3 seeds only after seed 0 passes the gate. **Gate:** macro val AUC > Step-4 by ≥ 0.02 and FS ≥ 0.985, DF ≥ 0.98.

**Step 7 — Ablations (each = one config flag, same seed; paired bootstrap on val + Celeb-DF):**
(1) Step-4 baseline → (2) +residual stream → (3) +mouth prior/ROI token → (4) +readout v3 → (5) +rank/focal/OHEM → (6) +mouth-SBI (`p=0.2`) → (7) +global SBI (`p=0.3`) → (8) +LoRA last-8. Report each row's per-method AUC and Celeb-DF AUC.

**Step 8 — Final protocol run.** Fit `(a, b, τ)` on FF++ val; freeze; evaluate FF++ test (per method), DFD, Celeb-DF v2 (video + frame), robustness (JPEG 30–90, blur, resize) with CIs. Save scores CSV, ROC plot (log-x), threshold-sweep from val, reliability diagram.

**Step 9 — Paper decision gate.**
- Celeb-DF ≥ 0.955 → main-track submission, "SOTA/parity with LN-tuned CLIP baselines + temporal/forensic residual stream".
- 0.90–0.95 → TIFS/second-tier framing: *efficient, calibrated, evidence-localizing detector* with the honest protocol as a contribution.
- < 0.90 → investigate Steps 4/7 ablations; do not submit on the current story.

**Standing checks for Antigravity:** never select checkpoints or thresholds with test/Celeb-DF; report seeds; keep `results.md` numbers tagged with the commit hash; flag any evaluation that mixes frame- and video-level numbers.

---

## Sources

- [DeepfakeBench (SCLBD)](https://github.com/SCLBD/DeepfakeBench) — evaluation protocol and frame-level baselines (RECCE, SPSL, F3-Net, UCF) [as summarized from my earlier session reads; direct re-fetch was refused]
- ForAda (Forensics Adapter, arXiv 2411.19715), Effort, LNCLIP-DF, RAE, LAA-Net, DFD-FCG, P&P: figures taken from the papers/benchmark tables I opened earlier; I have not attached links because I could not re-open each one in this session. [SBI, Shiohara & Yamasaki](https://arxiv.org/abs/2204.08376). Verify every number and citation against the original paper before camera-ready.
- Repo files read: `smsdt/models/smsdt_v2.py`, `smsdt/train_v2.py`, `smsdt/data/video_dataset.py`, `smsdt/losses.py`, `smsdt/eval_cross.py`, `smsdt/preprocess/build_store.py`, `results.md`, configs `train_v2_stage_{a,b}.yaml`
