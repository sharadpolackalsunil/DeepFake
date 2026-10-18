# S-MSDT: Spatio-Temporal Scale-Invariant Multi-Scale Deepfake Transformer with Asymmetric Frequency-Guided Differencing

**Author:** S-MSDT Research & Engineering Team  
**Affiliation:** Deepfake Detection & Media Forensics Lab  
**Target Architecture:** NVIDIA GeForce RTX 5090 (32 GB GDDR7, Blackwell `sm_120`, Native `bfloat16`)  
**Benchmark Target:** FaceForensics++ (`c23` Light Compression)  
**Document Version:** 2.0 (Design, Formulation, and Training Specification)

---

## Abstract

Deepfake video creation tools have evolved from crude whole-face swaps to photorealistic facial reenactment and neural texture synthesis. Detecting subtle, localized manipulations—such as micro-expression rendering in *NeuralTextures* and boundary blend artifacts in *Face2Face*—remains a persistent challenge for standard spatio-temporal Vision Transformers (ViTs). Conventional architectures suffer from three critical bottlenecks: (i) spatial self-attention blurs localized inter-frame inconsistencies prior to temporal subtraction; (ii) coarse token grids ($7 \times 7 = 49$ tokens) severely under-sample localized manipulation boundaries (e.g., mouth regions covering only $2\text{--}3$ tokens); and (iii) uniform spatial-temporal mean pooling dilutes concentrated forgery traces across large areas of authentic facial skin.

In this work, we present **S-MSDT (Scale-Invariant Multi-Scale Deepfake Transformer)**, an end-to-end spatio-temporal dual-stream architecture engineered for real-time deepfake detection ($<8\text{ ms}$ per 8-frame chunk). S-MSDT incorporates four core contributions:
1. **High-Resolution Pyramidal Scale-Invariant Tokenizer (PSIT):** Quadruples spatial token resolution ($14 \times 14 = 196$ tokens) using multi-scale FPN features ($P_2, P_3, P_4$), expanding mouth-region coverage from $2$ to $16$ tokens.
2. **Spatio-Temporal Wavelet Frequency Guard:** Employs 2D Haar Discrete Wavelet Transform (DWT) combined with 1D inter-frame temporal high-pass filtering and a 3-stage Depthwise-Separable Squeeze-and-Excitation (DW-SE) stem, dynamically weighted by a parameter-free spectral energy ratio ($\rho_t$).
3. **Temporal-First Decomposed ViT with Asymmetric $Q/K$ Self-Subtraction:** Inverts the execution order by applying Temporal Multi-Head Self-Attention (MHSA) *prior* to Spatial MHSA. Queries ($Q_{\text{temp}}$) and Keys ($K_{\text{temp}}$) are projected from uncertainty-gated temporal residuals $\tilde{\Delta}_t$, while Values ($V_{\text{temp}}$) are projected from un-subtracted spatial tokens, preserving static texture fidelity.
4. **Sharp Attentional Readout:** Replaces uniform mean pooling with a temperature-scaled softmax spatial-temporal attention mechanism ($\tau = 0.2$), initialized to uniform identity via zero-initialization to ensure smooth convergence.

Trained with a composite objective combining class-balanced Focal Binary Cross-Entropy ($\alpha = 0.2, \gamma = 2.0$), Supervised Contrastive Loss ($\tau = 0.07$), and Cross-Domain Subspace Orthogonality Regularization, S-MSDT achieves stable optimization across disparate manipulation categories on an NVIDIA GeForce RTX 5090 Blackwell platform.

---

## 1. Introduction & Motivation

Digital facial manipulation algorithms fall broadly into two operational paradigms:
* **Identity Replacement (Face Swapping):** Examples include *Deepfakes* (autoencoder-based swapping) and *FaceShifter* (non-linear GAN-based identity swapping). These approaches modify the entire central facial region and leave pronounced frequency discrepancies and boundary blending artifacts.
* **Expression & Motion Re-enactment:** Examples include *Face2Face* (dense deformation tracking) and *NeuralTextures* (neural rendering of mouth and lip movements). These methods keep the authentic identity, modifying only a localized fraction ($10\%\text{--}15\%$) of the face.

```
                      ┌──────────────────────────────────────────────┐
                      │   Digital Facial Manipulation Taxonomy       │
                      └──────────────────────┬───────────────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
  [Identity Replacement]                                          [Expression Re-enactment]
  • Deepfakes (Autoencoder)                                       • Face2Face (Deformation Tracking)
  • FaceSwap (Graphics/Blending)                                  • NeuralTextures (Neural Rendering)
  • FaceShifter (GAN-based Swap)                                  
  ─────────────────────────────────                               ───────────────────────────────────
  Footprint: Macro-scale swap; strong                             Footprint: Micro-scale localized
  spectral & blending boundary artifacts.                         jitter; 85% of face remains authentic.
```

### The Spatial Dilution Paradox
In baseline Vision Transformer architectures, feature maps are pooled into a coarse token grid (typically $7 \times 7 = 49$ tokens, where each token corresponds to a $32 \times 32$ pixel block on a $224 \times 224$ crop). Under this resolution:
* A localized mouth reenactment covers only **$2$ to $3$ tokens**.
* Uniform global average pooling ($z_{\text{video}} = \frac{1}{T \cdot N} \sum_{t, p} z_{t, p}$) averages these $2$ anomalous tokens with $47$ authentic skin, cheek, and forehead tokens.
* The forgery signal is attenuated by a factor of $\frac{2}{49} \approx 4\%$, blinding the linear classification head.

### The Temporal Smoothing Problem
Standard decomposed video transformers execute intra-frame spatial self-attention before temporal cross-attention. When spatial attention runs first, self-attention coefficients cause authentic cheek and background tokens to diffuse into the manipulated lip tokens. By the time temporal differencing is computed ($\Delta_t = X_t - X_{t-1}$), the boundary jitter has already been spatially low-pass filtered.

### Contributions
To overcome these limitations, we design and implement S-MSDT v2:
1. **Asymmetric Temporal-First Processing:** Inter-frame temporal self-attention operates directly on unsmoothed features, using directional differences exclusively to guide query-key correlation.
2. **High-Density $14 \times 14$ Tokenization:** Pyramidal multi-scale feature pooling captures micro-expressions with $16$ dedicated tokens.
3. **Learned Attention Spotlight:** Sharp Attentional Readout automatically weights the most suspicious facial sub-regions without hand-crafted bounding boxes.
4. **Hardware-Co-Designed Pipeline:** Zero-copy WebDataset streaming, GPU-resident detection, and native `bfloat16` tensor arithmetic optimized for Blackwell architectures.

---

## 2. Problem Formulation & Preliminary Concepts

Let a video chunk be defined as $\mathcal{V} = \{I_t\}_{t=0}^{T-1}$, consisting of $T = 8$ contiguous RGB face crops $I_t \in \mathbb{R}^{3 \times H \times W}$ aligned via ArcFace 5-point landmark homography to spatial resolution $H = W = 224$.

Each chunk is associated with:
* A ground-truth binary label $y \in \{0, 1\}$, where $y = 0$ denotes pristine authentic video and $y = 1$ denotes manipulated media.
* A categorical manipulation identifier $m \in \{0, 1, 2, 3, 4, 5\}$ corresponding to `{Real, Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter}`.
* A landmark tracking confidence vector $c = [c_0, c_1, \dots, c_{T-1}] \in [0, 1]^T$, extracted per frame by the SCRFD detector.

### 2D Discrete Haar Wavelet Transform (DWT)
To extract frequency-domain boundary cues without spatial phase loss, each frame $I_t$ is decomposed using orthogonal 2D Haar filters into four subbands:
$$LL_t, LH_t, HL_t, HH_t = \text{DWT}_{\text{Haar}}(I_t) \in \mathbb{R}^{3 \times \frac{H}{2} \times \frac{W}{2}}$$

The decomposition filters are defined as:
$$h_{LL} = \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}, \quad h_{LH} = \frac{1}{2}\begin{bmatrix} 1 & 1 \\ -1 & -1 \end{bmatrix}, \quad h_{HL} = \frac{1}{2}\begin{bmatrix} 1 & -1 \\ 1 & -1 \end{bmatrix}, \quad h_{HH} = \frac{1}{2}\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

The low-frequency approximation $LL_t$ captures illumination and low-frequency structure, while $\{LH_t, HL_t, HH_t\}$ capture horizontal, vertical, and diagonal high-frequency spectral components where synthetic upsampling and blending checkerboards reside.

---

## 3. S-MSDT Architecture

```
Input Chunk: (B, T=8, 3, 224, 224) Aligned Crops + (B, T) SCRFD Confidences
  │
  ├──► [Branch A: RGB Multi-Scale Stream]
  │      EfficientNet-B4 Backbone (Pretrained ImageNet)
  │      └─► Top-Down Feature Pyramid Network (FPN)
  │            P2 (56x56), P3 (28x28), P4 (14x14)
  │            └─► High-Res PSIT (Adaptive Pool 14x14)
  │                  └─► z_rgb ∈ ℝ^(B·T × 196 × 256)
  │
  ├──► [Branch B: Spatio-Temporal Wavelet Guard]
  │      2D Haar DWT (LH, HL, HH subbands — 9 channels)
  │      + 1D Temporal High-Pass Filter on LL across T (3 channels)
  │      └─► Combined 12-Channel High-Frequency Tensor (112x112)
  │            └─► 3-Stage Depthwise-Separable + SE Stem
  │                  (112x112) → (56x56) → (28x28) → (14x14)
  │                  └─► z_dwt ∈ ℝ^(B·T × 196 × 256)
  │                  └─► Spectral Energy Ratio ρ_t ∈ ℝ^(B·T × 1)
  │
  ├──► [Compression-Aware Gated Fusion (CASG)]
  │      Wavelet-to-RGB Local Cross-Attention
  │      Gate G = σ(MLP([z_rgb, z_dwt_refined, ρ_t]))
  │      z_fused = G ⊙ z_rgb + (1 - G) ⊙ z_dwt_refined
  │      Add Learned Spatial Positional Embedding E_pos ∈ ℝ^(1 × 1 × 196 × 256)
  │
  ├──► [N-MSSS ViT Backbone (x4 Decomposed Stages)]
  │      For each stage blk = 1 ... 4:
  │      ├─ 1. Hyperspherical Differencing (Δ_0 = X_1 - X_0; Δ_t = X_t - X_{t-1})
  │      │     Detector Uncertainty Gating: Δ_gated = Δ ⊙ σ(MLP([c_t, c_{t-1}]))
  │      ├─ 2. Asymmetric Temporal Attention (Inter-frame across T=8):
  │      │     Q = W_Q(Δ_gated), K = W_K(Δ_gated), V = W_V(X_unsubtracted)
  │      │     X ← X + Attn(Q, K, V)
  │      ├─ 3. Spatial Self-Attention (Intra-frame across N=196 patches):
  │      │     X ← X + MHSA_spatial(LN(X))
  │      └─ 4. Feed-Forward Network:
  │             X ← X + MLP(LN(X))
  │
  └──► [Sharp Attentional Readout & Head]
         Spatial Attention Pooling: α_{t, p} = Softmax(w_s^⊤ tanh(W_s z_{t, p}) / τ)
         Temporal Attention Pooling: β_t = Softmax(w_t^⊤ tanh(W_t z_{frame, t}) / τ)
         z_video = ∑_t β_t (∑_p α_{t, p} z_{t, p}) ∈ ℝ^(B × 256)
         └─► LayerNorm → Linear(256→128) → GELU → Dropout(0.1) → Linear(128→1) → Logit
```

### 3.1 RGB Feature Pyramid Network (FPN) Stream
Branch A utilizes a full-capacity **EfficientNet-B4** backbone ($\sim 19.3\text{M}$ parameters) initialized with ImageNet weights. Three feature maps are tapped across increasing receptive fields:
* Level 2 ($C_2$): Stride 4, $56 \times 56$ resolution, capturing fine edge and blending boundaries.
* Level 3 ($C_3$): Stride 8, $28 \times 28$ resolution, capturing localized part geometries (eyes, nostrils, lips).
* Level 4 ($C_4$): Stride 16, $14 \times 14$ resolution, capturing facial global layout.

Lateral $1 \times 1$ convolutions project channels to a uniform embedding dimension $D = 256$. A top-down pathway merges features using nearest-neighbor $2\times$ upsampling, followed by $3 \times 3$ smoothing convolutions to eliminate aliasing:
$$P_4 = \text{Smooth}_4(\text{Lateral}_4(C_4))$$
$$P_3 = \text{Smooth}_3(\text{Lateral}_3(C_3) + \text{Upsample}(P_4))$$
$$P_2 = \text{Smooth}_2(\text{Lateral}_2(C_2) + \text{Upsample}(P_3))$$

### 3.2 High-Resolution Pyramidal Scale-Invariant Tokenizer (PSIT)
To standardize multi-scale features into token sequences for the transformer, the Pyramidal Scale-Invariant Tokenizer (PSIT) performs adaptive spatial pooling to a uniform grid of size $G \times G = 14 \times 14$ ($N = 196$ tokens):
$$z_{\text{rgb}} = \text{Conv}_{1 \times 1}\left(\sum_{k \in \{2, 3, 4\}} \text{AdaptiveAvgPool}_{14 \times 14}(P_k)\right) \in \mathbb{R}^{(B \cdot T) \times 196 \times D}$$

Each token corresponds to a localized $16 \times 16$ pixel region of the original $224 \times 224$ face, providing the spatial granularity needed to capture mouth boundary artifacts.

### 3.3 Spatio-Temporal Wavelet Guard & Compression Sensor
Branch B isolates frequency-domain artifacts without relying on spatial convolutions alone. 

#### 1. 1D Temporal High-Pass Filtering
Deepfake synthesis exhibits frame-to-frame intensity and phase instability that manifests along the temporal dimension. For each video chunk, low-frequency $LL$ frames are analyzed across time $T$:
$$F_{\text{temp}, t} = \begin{cases} |LL_1 - LL_0| & \text{if } t = 0 \\ |LL_t - LL_{t-1}| & \text{if } t \ge 1 \end{cases} \in \mathbb{R}^{3 \times 112 \times 112}$$

Concatenating spatial subbands and the temporal filter response produces a 12-channel high-frequency representation:
$$W_{\text{cat}, t} = \text{Concat}\big(|LH_t|, |HL_t|, |HH_t|, F_{\text{temp}, t}\big) \in \mathbb{R}^{12 \times 112 \times 112}$$

#### 2. Parameter-Free Spectral Energy Ratio ($\rho_t$)
Video compression attenuates high-frequency subbands while leaving the low-frequency approximation intact. We compute the instantaneous high-to-low spectral ratio:
$$\rho_t = \frac{\frac{1}{3 \cdot 112^2} \sum_{c, h, w} (|LH_t| + |HL_t| + |HH_t|)}{\frac{1}{3 \cdot 112^2} \sum_{c, h, w} |LL_t| + \epsilon} \in \mathbb{R}^{(B \cdot T) \times 1}$$

$\rho_t$ provides a scalar measure of image sharpness versus quantization blur.

#### 3. 3-Stage Depthwise-Separable Frequency Stem
Rather than using a crude $16 \times 16$ strided convolution, $W_{\text{cat}}$ passes through three successive stride-2 depthwise-separable stages with Squeeze-and-Excitation (SE) channel recalibration:
$$\text{Stage 1: } (12 \times 112 \times 112) \xrightarrow{\text{Conv } 3\times 3, s=2} (64 \times 56 \times 56) \rightarrow \text{BN} \rightarrow \text{GELU} \rightarrow \text{SE}$$
$$\text{Stage 2: } (64 \times 56 \times 56) \xrightarrow{\text{DWConv } 3\times 3, s=2} \xrightarrow{\text{PWConv } 1\times 1} (128 \times 28 \times 28) \rightarrow \text{BN} \rightarrow \text{GELU} \rightarrow \text{SE}$$
$$\text{Stage 3: } (128 \times 28 \times 28) \xrightarrow{\text{DWConv } 3\times 3, s=2} \xrightarrow{\text{PWConv } 1\times 1} (256 \times 14 \times 14) \rightarrow \text{BN} \rightarrow \text{GELU} \rightarrow \text{SE}$$

The resulting tensor matches the spatial grid $14 \times 14$ and is flattened into $z_{\text{dwt}} \in \mathbb{R}^{(B \cdot T) \times 196 \times D}$.

### 3.4 Compression-Aware Spectral Gating (CASG)
Spatial and frequency tokens are fused via a two-stage mechanism:
1. **Local Cross-Attention:** Frequency tokens query RGB tokens to align textural edges with spectral anomalies:
   $$\tilde{z}_{\text{dwt}} = z_{\text{dwt}} + \text{MultiHeadAttention}(Q=\text{LN}(z_{\text{dwt}}), K=\text{LN}(z_{\text{rgb}}), V=\text{LN}(z_{\text{rgb}}))$$
2. **Spectral-Ratio-Conditioned Gate:**
   $$G = \sigma\Big(\text{MLP}\big([z_{\text{rgb}}, \tilde{z}_{\text{dwt}}, \rho_t]\big)\Big) \in (0, 1)^{(B \cdot T) \times 196 \times D}$$
   $$z_{\text{fused}} = G \odot z_{\text{rgb}} + (1 - G) \odot \tilde{z}_{\text{dwt}}$$

$z_{\text{fused}}$ is augmented with a learned 2D spatial positional embedding $E_{\text{pos}} \in \mathbb{R}^{1 \times 1 \times 196 \times D}$.

### 3.5 Temporal-First Decomposed ViT with Asymmetric Attention
The core backbone consists of $M = 4$ identical decomposed stages.

#### 1. Hyperspherical Normalization & Boundary-Consistent Differencing
Tokens are normalized onto the unit hypersphere along the channel dimension $D$:
$$\hat{X}_t = \frac{X_t}{\|X_t\|_2 + \epsilon}$$

To prevent an artificial norm spike at $t=0$ (where a naive difference $\hat{X}_0 - 0$ yields a vector of unit length), we implement forward differencing at $t=0$ and backward differencing for $t \ge 1$:
$$\Delta_t = \begin{cases} \hat{X}_1 - \hat{X}_0 & \text{if } t = 0 \\ \hat{X}_t - \hat{X}_{t-1} & \text{if } t \in \{1, \dots, T-1\} \end{cases}$$

#### 2. Detector Uncertainty Gating
SCRFD face detector confidences modulate the difference stream. If tracking is unstable, feature differentials reflect bounding box jitter rather than facial manipulation. A lightweight gating network modulates the residual:
$$g_{\text{rel}, t} = \sigma\Big(\text{MLP}_{\text{gate}}\big([c_t, c_{t-1}]\big)\Big) \in (0, 1)$$
$$\tilde{\Delta}_t = g_{\text{rel}, t} \cdot \Delta_t$$

#### 3. Asymmetric Temporal Self-Attention
For each spatial patch index $p \in \{1, \dots, 196\}$, temporal self-attention across the $T = 8$ frames is defined as:
$$Q_{\text{temp}} = \text{LayerNorm}(\tilde{\Delta}) W_Q^{(t)}, \quad K_{\text{temp}} = \text{LayerNorm}(\tilde{\Delta}) W_K^{(t)}$$
$$V_{\text{temp}} = \text{LayerNorm}(X) W_V^{(t)}$$
$$X' = X + \text{Dropout}\left(\text{Softmax}\left(\frac{Q_{\text{temp}} K_{\text{temp}}^\top}{\sqrt{D / H_{\text{heads}}}}\right)\right) V_{\text{temp}} W_O^{(t)}$$

Because $Q$ and $K$ are driven by $\tilde{\Delta}$, the attention matrix measures *temporal anomalies*. Because $V$ is driven by the original tokens $X$, the features transported remain the *un-subtracted spatial textures*.

#### 4. Spatial Attention & FFN
Following temporal attention, standard intra-frame spatial self-attention and a 2-layer MLP (expansion ratio $4.0$) are applied:
$$Y = X' + \text{MHSA}_{\text{spatial}}(\text{LayerNorm}(X'))$$
$$Z = Y + \text{FFN}(\text{LayerNorm}(Y))$$

### 3.6 Sharp Attentional Readout Head
To map the final tensor $Z \in \mathbb{R}^{B \times T \times 196 \times D}$ to a chunk logit $\hat{y} \in \mathbb{R}$, we use a two-tier temperature-scaled attention pooling mechanism.

#### Spatial Pooling
$$a_{t, p} = w_s^\top \tanh(W_s Z_{t, p}), \quad \alpha_{t, p} = \frac{\exp(a_{t, p} / \tau)}{\sum_{j=1}^{196} \exp(a_{t, j} / \tau)}, \quad z_{\text{frame}, t} = \sum_{p=1}^{196} \alpha_{t, p} Z_{t, p}$$

#### Temporal Pooling
$$b_t = w_t^\top \tanh(W_t z_{\text{frame}, t}), \quad \beta_t = \frac{\exp(b_t / \tau)}{\sum_{k=0}^{T-1} \exp(b_k / \tau)}, \quad z_{\text{video}} = \sum_{t=0}^{T-1} \beta_t z_{\text{frame}, t} \in \mathbb{R}^{B \times D}$$

Here, $\tau = 0.2$ sharpens the distribution, enabling the model to focus on the small subset of patches containing manipulations.

#### Zero-Initialization Guarantee
The final linear projection layers ($w_s, w_t$) are initialized to zero ($W = \mathbf{0}, b = 0$). As a result, at step 0:
$$a_{t, p} = 0 \implies \alpha_{t, p} = \frac{1}{196}, \quad b_t = 0 \implies \beta_t = \frac{1}{8}$$
The readout initializes as exact uniform average pooling, preventing random patch blindness and ensuring stable gradient propagation during warmup.

#### Classification MLP
$$\hat{y} = W_2 \cdot \text{Dropout}_{0.1}\big(\text{GELU}(W_1 \cdot \text{LayerNorm}(z_{\text{video}}))\big) \in \mathbb{R}$$

---

## 4. Multi-Objective Loss Formulation

The optimization objective balances binary classification, latent cluster formation, and cross-modal feature specialization:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \lambda_{\text{con}} \mathcal{L}_{\text{contrastive}} + \lambda_{\text{align}} \mathcal{L}_{\text{align}}$$
where $\lambda_{\text{con}} = 0.1$ and $\lambda_{\text{align}} = 0.05$.

```
                       Total Composite Objective L_total
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
  [L_focal (1.0)]               [L_con (0.1)]                 [L_align (0.05)]
  Class-Balanced Focal Loss     Supervised Contrastive        Subspace Orthogonality
  α=0.2, γ=2.0                  Temperature τ=0.07            Decorrelates RGB vs Wavelet
  Balances 1:4 Real:Fake        Clusters same-forgery fakes   Forces spectral specialization
  ratio; downweights easy       and pushes Real/Fake          prior to cross-attention
  samples.                      apart in ViT space.           fusion.
```

### 4.1 Class-Balanced Focal Loss ($\mathcal{L}_{\text{focal}}$)
FaceForensics++ has an inherent $1:4$ real-to-fake class imbalance (1,000 real videos vs. 5,000 manipulated videos across 5 methods). To prevent the classifier from collapsing toward the majority fake class, we employ class-balanced Focal Loss with label smoothing ($\epsilon_{\text{smooth}} = 0.05$):
$$\tilde{y} = y(1 - \epsilon_{\text{smooth}}) + 0.5 \epsilon_{\text{smooth}}$$
$$p = \sigma(\hat{y})$$
$$p_t = p \tilde{y} + (1 - p)(1 - \tilde{y})$$
$$\alpha_t = \alpha \tilde{y} + (1 - \alpha)(1 - \tilde{y})$$
$$\mathcal{L}_{\text{focal}} = -\frac{1}{B} \sum_{i=1}^B \alpha_t (1 - p_t)^\gamma \log(p_t + \epsilon)$$

Setting $\alpha = 0.2$ and focusing parameter $\gamma = 2.0$ balances effective gradient contributions:
$$\text{Effective Weight: } 4 \text{ Fakes} \times 0.2 = \mathbf{0.8} \quad \text{vs.} \quad 1 \text{ Real} \times (1 - 0.2) = \mathbf{0.8}$$

### 4.2 Supervised Contrastive Loss ($\mathcal{L}_{\text{contrastive}}$)
Applied to the post-ViT learned representations $z_{\text{video}}$, normalized to unit length $\hat{z} = z / \|z\|_2$. For batch size $B$, positive pairs are formed between samples sharing the same manipulation type ($m_i = m_j$):
$$\mathcal{L}_{\text{contrastive}} = \sum_{i=1}^B \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\hat{z}_i^\top \hat{z}_p / \tau_{\text{con}})}{\sum_{a \in A(i)} \exp(\hat{z}_i^\top \hat{z}_a / \tau_{\text{con}})}$$
where $A(i) \equiv \{1, \dots, B\} \setminus \{i\}$, $P(i) \equiv \{p \in A(i) : m_p = m_i\}$, and temperature $\tau_{\text{con}} = 0.07$.

This objective clusters representations of identical generative techniques while separating real videos from fake variants.

### 4.3 Subspace Orthogonality Alignment Loss ($\mathcal{L}_{\text{align}}$)
To ensure the frequency branch learns features complementary to the RGB stream rather than duplicating textural patterns, an orthogonality penalty is applied to pre-fusion pooled tokens:
$$\mathcal{L}_{\text{align}} = \frac{1}{B} \sum_{i=1}^B \left| \frac{z_{\text{rgb}, i}^\top z_{\text{dwt}, i}}{\|z_{\text{rgb}, i}\|_2 \|z_{\text{dwt}, i}\|_2 + \epsilon} \right|$$
Minimizing $\mathcal{L}_{\text{align}}$ drives cosine correlation toward zero, enforcing subspace decorrelation.

---

## 5. Preprocessing & Data Pipeline

```
Raw MP4 Video ──► SCRFD Face Detector (Buffalo_L ONNX) ──► ArcFace 5-Point Alignment
                       │                                         │
                       ▼                                         ▼
            Detection Confidences (c_t)                Aligned 224x224 RGB Crops
                       │                                         │
                       └───────────────────┬─────────────────────┘
                                           │
                                           ▼
                             Chunk Partitioning (T=8, Stride=4)
                                           │
                                           ▼
                             WebDataset Shard Writer (.tar)
                                 500 chunks per shard
```

### 5.1 Landmark Tracking & Alignment
1. **SCRFD Detection:** Frame faces are localized using `det_10g.onnx` (`buffalo_l`) via `CUDAExecutionProvider`. Model sessions are cached in worker memory to eliminate process startup overhead.
2. **ArcFace Homography:** Each detected face is normalized using 5 facial landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner) mapped to canonical ArcFace coordinates:
   $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} s \cos \theta & -s \sin \theta & t_x \\ s \sin \theta & s \cos \theta & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
   Faces are cropped and scaled to $224 \times 224$ pixels.

### 5.2 Streaming WebDataset Shards
To avoid random read latency across millions of individual frame files, aligned chunks ($T=8$, stride $=4$) are packaged into sequential uncompressed `.tar` archives containing 500 chunks each ($\sim 300\text{ MB}$ per shard):
* `frames.npy`: $(8, 3, 224, 224)$ `uint8` array.
* `conf.npy`: $(8,)$ float array of landmark tracking confidences.
* `label.cls`: Integer label ($0$ for real, $1$ for fake).
* `manip_type.cls`: Integer manipulation ID ($0\text{--}5$).

### 5.3 Training Data Pipeline
```
WebDataset Shards (.tar) ──► Shard Resampling ──► Shuffle Buffer (1,000 chunks)
                                                        │
                                                        ▼
                                             Sample Decoding & Albumentations
                                             (HorizontalFlip, Compression, Blur)
                                                        │
                                                        ▼
                                             Micro-batch Assembly (B=8)
```

During training, shards are streamed with continuous resampling. An in-memory shuffle buffer ($B_{\text{buf}} = 1,000$ chunks) ensures decorrelated batch assembly across video sources.

---

## 6. Training Configuration & Optimization Setup

### 6.1 Hardware Specifications
* **GPU:** NVIDIA GeForce RTX 5090 (Blackwell architecture, `sm_120`)
* **VRAM:** 32 GB GDDR7 (Memory bandwidth $\sim 1,792\text{ GB/s}$)
* **Arithmetic Precision:** Native hardware `torch.bfloat16` via automatic mixed precision (`autocast`)
* **Host System:** Windows 10/11, 32 GB RAM, 4 worker subprocesses

### 6.2 Optimization Profile
* **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 0.05$)
* **Decoupled Learning Rates:**
  * EfficientNet-B4 Backbone: $\eta_{\text{backbone}} = 3.0 \times 10^{-5}$
  * ViT Backbone & Readout Head: $\eta_{\text{head}} = 3.0 \times 10^{-4}$
* **Learning Rate Schedule:** Cosine decay with a 2,000-step linear warmup ($2$ epochs):
  $$\eta(s) = \begin{cases} \eta_{\text{max}} \cdot \frac{s}{s_{\text{warmup}}} & \text{if } s < s_{\text{warmup}} \\ \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{s - s_{\text{warmup}}}{s_{\text{total}} - s_{\text{warmup}}} \pi\right)\right) & \text{if } s \ge s_{\text{warmup}} \end{cases}$$
* **Gradient Accumulation:** Micro-batch size $8 \times 4$ accumulation steps = **Effective Batch Size $32$**.
* **Gradient Clipping:** Maximum $L_2$ norm $\le 1.0$.
* **Crash-Resilient Checkpointing:** State snapshots written to `outputs/checkpoints/latest.pt` every 500 optimization steps.

---

## 7. Empirical Progression: Baseline (v1) to Upgraded (v2)

### 7.1 Quantitative Baseline Results (S-MSDT v1 at Epoch 23)
Evaluated on the official FaceForensics++ `c23` held-out validation set ($N = 307$ videos, $>30,000$ chunks):

| Metric | Overall Chunk-Level | Overall Video-Level (Top-K, $K=3$) |
|:---|:---:|:---:|
| **AUC** | **0.7649** | **0.7818** |
| **Equal Error Rate (EER)** | **0.2997** | **0.2692** |
| **Accuracy @ 0.5** | **0.7258** | **0.7883** |
| **F1 Score @ 0.5** | **0.8149** | **0.8708** |

### Per-Manipulation Category Breakdown
Scoring each manipulation method against all real videos:

| Manipulation Category | Manipulation Type | AUC | EER | F1 @ 0.5 | Forgery Footprint |
|:---|:---|:---:|:---:|:---:|:---|
| **Deepfakes** | Autoencoder Face Swap | **0.8854** | 0.2012 | 0.8111 | Macro-scale swap; high spectral disparity |
| **FaceShifter** | Advanced GAN Face Swap | **0.8748** | 0.2167 | 0.7966 | Macro-scale swap; subtle boundary blending |
| **FaceSwap** | Graphics 3D Mesh Swap | **0.7495** | 0.3183 | 0.6543 | Color calibration & seam anomalies |
| **Face2Face** | Expression Deformation | **0.6671** | 0.3732 | 0.6037 | Localized mouth/brow re-enactment |
| **NeuralTextures** | Neural Texture Rendering | **0.5886** | 0.4355 | 0.4926 | Micro-scale mouth expression synthesis |

```
                       S-MSDT v1 Per-Category AUC Performance
  1.00 ┌─────────────────────────────────────────────────────────────────┐
       │                                                                 │
  0.88 │   █████████              █████████                              │
       │   ██ 0.885█              ██ 0.875█                              │
  0.75 │   ██      █              ██      █     █████████                │
       │   ██      █              ██      █     ██ 0.750█                │
  0.66 │   ██      █              ██      █     ██      █    █████████   │
       │   ██      █              ██      █     ██      █    ██ 0.667█   │
  0.58 │   ██      █              ██      █     ██      █    ██      █   │  █████████
       │   ██      █              ██      █     ██      █    ██      █   │  ██ 0.589█
  0.50 └───┴───────┴──────────────┴───────┴─────┴───────┴────┴───────┴───┴──┴───────┴──
          Deepfakes              FaceShifter    FaceSwap     Face2Face    NeuralTextures
```

### 7.2 Analysis of Empirical Findings
1. **Macro vs. Micro Forgeries:** S-MSDT v1 reliably detects whole-face replacement methods (*Deepfakes* AUC $0.885$, *FaceShifter* AUC $0.875$) through its dual-stream FPN and DWT decomposition.
2. **The Re-enactment Gap:** Performance degrades substantially on *Face2Face* ($0.667$) and *NeuralTextures* ($0.589$). Because identity is preserved in these methods, global identity features cannot serve as a reliable discriminator.
3. **Architectural Justification for v2:** The empirical results from v1 directly motivated the v2 architectural overhaul:
   * **$14 \times 14$ grid** provides 4× more spatial samples over the mouth area.
   * **Temporal-First ordering** preserves frame-to-frame mouth movement anomalies before spatial self-attention can blur them.
   * **Sharp Attentional Readout** prevents the 85% authentic face area from diluting localized mouth manipulation signals.

---

## 8. Summary & Current Status

All components of the S-MSDT v2 architecture have been implemented, verified, and committed to the `v2` development branch:
* **Token Grid:** Upgraded to $14 \times 14 = 196$ tokens via `PSIT` and `FrequencyStem`.
* **Wavelet Stream:** 3-stage DW-SE stem with 1D temporal filtering.
* **Transformer Backbone:** Temporal-first execution with asymmetric $Q/K$ differencing.
* **Readout Head:** Temperature-scaled attention pooling with zero-initialization.
* **Loss Objectives:** Multi-task formulation incorporating balanced Focal Loss, Supervised Contrastive Loss ($\tau = 0.07$), and Subspace Orthogonality Regularization.

---

## References

1. **Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M.** (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images*. IEEE/CVF International Conference on Computer Vision (ICCV).
2. **Tan, M., & Le, Q. V.** (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. International Conference on Machine Learning (ICML).
3. **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). *Focal Loss for Dense Object Detection*. IEEE International Conference on Computer Vision (ICCV).
4. **Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D.** (2020). *Supervised Contrastive Learning*. Advances in Neural Information Processing Systems (NeurIPS).
5. **Zhao, H., Zhou, W., Chen, D., Wei, T., Zheng, L., & Yu, N.** (2021). *Multi-Attentional Deepfake Detection*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
6. **Guo, J., Deng, J., Lattas, A., & Zafeiriou, S.** (2021). *Sample and Computation Redistribution for Efficient Face Detection (SCRFD)*. arXiv:2105.04714.
7. **Ilse, M., Tomczak, J., & Welling, M.** (2018). *Attention-based Deep Multiple Instance Learning*. International Conference on Machine Learning (ICML).
