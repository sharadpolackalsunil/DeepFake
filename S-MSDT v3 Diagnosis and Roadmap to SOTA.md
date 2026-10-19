# S-MSDT v3: Diagnosis and Roadmap to SOTA

Review of the S-MSDT deepfake detection architecture (FF++ c23, v3 at 0.72 AUC after 5 epochs)

## Summary

**0.72 AUC on FF++ c23 is far too low for an in-distribution result.** A plain EfficientNet-B4 with frame-averaged logits usually reaches 0.90+ AUC within a few epochs. Your three versions also landed in a narrow band (v1 0.76, v2 0.69, v3 0.72) even though they differ a lot architecturally. That pattern usually means the bottleneck is something they all share (data pipeline, loss setup, or the shared front end), not the fusion or attention design. Fix that before pursuing SOTA ideas.

## 1. Diagnostics to run first

1. **Compare train AUC with val AUC.** If both are around 0.75, you have an optimization or pipeline bug. If train is high and val is 0.72, you have identity memorization or a split problem.
2. **Run a stripped baseline:** B4 only, plain BCE with `pos_weight`, mean over frames, no wavelet branch, no SupCon, no decorrelation. If it can't reach about 0.90, the problem is in the data or training setup.
3. **Overfit 64 clips.** It should hit about 100% training accuracy in minutes.
4. **Check the pipeline:**
   - Splits follow the official FF++ identity lists.
   - The WebDataset shuffle buffer is large enough, since shards written per video produce badly correlated batches.
   - Normalization is identical at train and eval.
   - BatchNorm is in eval mode at validation.
   - The `OneCycleLR` epoch-5 point (probably still warmup with `total_steps=50000`).

## 2. Likely culprits in the current design

- **`AdaptiveAvgPool2d` to 7×7 is a low-pass filter.** On P2 (56×56) it is an 8×8 box blur, which erases the local blending artifacts you want. You also drop EfficientNet's stage 5 (448 ch, native 7×7), which is your most semantic feature.
- **SupCon runs on a micro-batch of 8.** Gradient accumulation doesn't enlarge the contrastive batch. At 1:4 real:fake you often have 1–2 reals per batch, so the loss is noisy or degenerate. Use a memory queue, a real-balanced sampler, or drop it at first.
- **ρ is near-constant when you train only on c23**, so the gate has nothing to condition on. It only makes sense if you train on mixed compression (raw/c23/c40 or random re-encoding).
- **The decorrelation loss fights the fusion.** Forcing the pooled RGB and DWT embeddings to be orthogonal doesn't make the wavelet stream useful. Set λ\_align=0 until the baseline works.
- **The stride-4 depthwise conv in the wavelet stem aliases the high frequencies**, and resizing the crops to 224 already alters the compression fingerprint.
- **Verify the focal α convention.** α=0.2 should apply to the fake (majority) class. Plain BCE with `pos_weight` is more stable for debugging.

## 3. Answers to the six research questions

### Spatial backbone

A foundation backbone is the highest-leverage change. CLIP ViT-B/16 or DINOv2, fine-tuned with LoRA or adapters (not fully fine-tuned) at a low LR, has consistently generalized better than ImageNet CNNs in recent work (UnivFD, Forensics Adapter, Effort-style SVD adapters). Use the last-layer patch tokens (14×14 for ViT-B/16 at 224) plus a few intermediate layers. This replaces the FPN and pooling.

### Frequency domain

Treat it as an ablation, not a core component. Frequency fingerprints tend to be generator- and compression-specific, so they help in-dataset but often hurt cross-dataset. There is no strong evidence that DTCWT beats Haar for c40, and heavy compression removes the high-frequency cues regardless. If you keep it, use a late, light fusion instead of a heavy stem.

### Temporal modeling

At T=8 consecutive frames (about 0.27 s), attention is cheap and Mamba buys little. Sampling with a stride of 2–4 frames and applying temporal modules to the foundation features usually helps more. Temporal modeling gives modest gains on FF++ compared with the data and backbone choices.

### Generalization

The biggest levers are **Self-Blended Images (SBI)** and video-level blending, with strong augmentation (JPEG/H.264 re-encoding, blur, resize). Domain-adversarial disentanglement (UCF-style) is second-order.

### Losses

BCE plus a well-batched SupCon is enough. An ArcFace-style compact-real margin is a reasonable later experiment, but not a priority.

## 4. Roadmap

| Phase | Goal | Actions |
| --- | --- | --- |
| 0 | Working baseline | Fix the pipeline until the stripped baseline exceeds 0.90 val AUC |
| 1 | Stronger features | Swap in a LoRA-tuned CLIP/DINOv2, remove the pooling bottleneck, add temporal attention on top |
| 2 | Generalization | Add SBI plus compression augmentation; evaluate cross-dataset on Celeb-DF v2 and DFDC |
| 3 | Refinement | Re-add the frequency branch and the contrastive/margin losses one at a time, keeping each only if it helps |

## Notes

Verify the exact numbers in the papers mentioned (SBI, UnivFD, Forensics Adapter, Effort, UCF), since they are cited from memory. To narrow down the most likely suspect, check the train AUC, the epoch size in steps, and how the shards are shuffled.
