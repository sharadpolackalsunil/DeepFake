"""Test-Time Adaptation (TTA) Engine for S-MSDT TMLR Study.

Implements:
1. LayerNorm-only Entropy-Minimization TTA (TENT-style) for CLIP vision models
2. Two operational modes: Online (sequential streaming) and Batch-Episodic (reset after each batch)
3. Deterministic state reset guaranteeing bit-identical parameters between datasets
4. Explicit collapse detection (AUC drop > 0.03 or degenerate majority prediction > 95%)
5. Saerens-style Expectation-Maximization prior shift correction
6. Clip-consistency objective (hypothesis H6, docs/decisions.md): the K clips of one
   video share one label, so their logits should agree. This is a label-free signal that
   single-image TTA does not have. Loss = Var_k(clip logits) + lambda_e * H(video prob)
   - lambda_d * H(batch-mean prob). Entropy-only (pre-registered H3b) stays the default.
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Generator

from tools.stats import fast_auc


def get_layernorm_params(model: nn.Module) -> List[nn.Parameter]:
    """Collect only LayerNorm affine parameters (gamma, beta) from the vision tower and heads.
    
    Freezes all non-LayerNorm parameters and ensures LayerNorm parameters have requires_grad=True.
    """
    ln_params = []
    for name, param in model.named_parameters():
        if "norm" in name.lower() or "ln" in name.lower():
            param.requires_grad = True
            ln_params.append(param)
        else:
            param.requires_grad = False
    return ln_params


def binary_entropy_loss(logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Shannon entropy of Bernoulli distribution parameterized by sigmoid(logits).
    
    H(p) = - p * log(p) - (1 - p) * log(1 - p)
    Minimizing entropy encourages high-confidence predictions on unlabeled test samples.
    """
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    entropy = - (p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return entropy.mean()


def clip_consistency_loss(
    clip_logits: torch.Tensor,
    lambda_entropy: float = 1.0,
    lambda_div: float = 0.1,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Label-free loss for a batch of videos, each scored as K clips.

    Args:
        clip_logits: (B, K) clip logits for B videos
        lambda_entropy: weight of the video-level prediction entropy
        lambda_div: weight of the anti-collapse term (maximise the entropy of the
            batch-mean prediction). Note it pulls the batch towards 50% fake; keep it small
            and report the lambda_div = 0 ablation.

    Returns:
        (loss, parts) where parts are the float values of each term.
    """
    if clip_logits.ndim != 2:
        raise ValueError(f"clip_logits must be (B, K), got {tuple(clip_logits.shape)}")
    consist = clip_logits.var(dim=1, unbiased=False).mean() if clip_logits.shape[1] > 1 else clip_logits.new_zeros(())
    video_logit = clip_logits.mean(dim=1)
    p = torch.sigmoid(video_logit).clamp(eps, 1 - eps)
    ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
    pm = p.mean().clamp(eps, 1 - eps)
    marg_ent = -(pm * torch.log(pm) + (1 - pm) * torch.log(1 - pm))
    loss = consist + lambda_entropy * ent - lambda_div * marg_ent
    return loss, {"consistency": float(consist.detach()), "entropy": float(ent.detach()),
                  "marginal_entropy": float(marg_ent.detach())}


class TTAAdapter:
    """Manages test-time adaptation across evaluation datasets with strict reset guarantees."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        mode: str = "online",
        batch_size: int = 32,
        seed: int = 42,
        objective: str = "entropy",
        lambda_entropy: float = 1.0,
        lambda_div: float = 0.1,
    ):
        """
        Args:
            model: Target neural network model
            lr: TTA learning rate (pre-registered: 1e-4)
            mode: 'online' (adapts continuously across stream) or 'batch_episodic' (resets after each batch)
            batch_size: Test stream batch size (default 32 videos)
            seed: Deterministic RNG seed for stream ordering
        """
        assert mode in ["online", "batch_episodic"], f"Unknown mode: {mode}"
        assert objective in ["entropy", "consistency"], f"Unknown objective: {objective}"
        self.objective = objective
        self.lambda_entropy = lambda_entropy
        self.lambda_div = lambda_div
        self.model = model
        self.lr = lr
        self.mode = mode
        self.batch_size = batch_size
        self.seed = seed
        
        # Save clean snapshot of initial model state dict for deterministic reset
        self.source_state_dict = copy.deepcopy(model.state_dict())
        self.ln_params = get_layernorm_params(self.model)
        self.optimizer = torch.optim.Adam(self.ln_params, lr=self.lr)

    def reset(self):
        """Reset model parameters bit-identically to source checkpoint."""
        self.model.load_state_dict(copy.deepcopy(self.source_state_dict))
        self.ln_params = get_layernorm_params(self.model)
        self.optimizer = torch.optim.Adam(self.ln_params, lr=self.lr)

    def adapt_batch(self, x: torch.Tensor, prior: Optional[torch.Tensor] = None, max_chunk: int = 4) -> Tuple[torch.Tensor, float]:
        """Perform one adaptation step on an unlabeled test batch.

        entropy:     x is whatever the model takes; loss = mean binary entropy.
        consistency: x is (B, K, ...) = K clips per video; the model is run on the
                     (B*K, ...) clips; returns video logits (B,) = mean over clips.
        """
        self.model.eval()
        self.optimizer.zero_grad()

        if self.objective == "consistency":
            if x.ndim < 3:
                raise ValueError("consistency objective needs x of shape (B, K, ...)")
            b, k = x.shape[:2]
            x_flat = x.reshape(b * k, *x.shape[2:])
            if prior is not None:
                p_flat = prior.reshape(b * k, *prior.shape[2:])
                out = self.model(x_flat, prior=p_flat)
            else:
                out = self.model(x_flat)
            clip_logits = (out[0] if isinstance(out, (tuple, list)) else out).reshape(b, k)
            loss, _ = clip_consistency_loss(clip_logits, self.lambda_entropy, self.lambda_div)
            loss.backward()
            self.optimizer.step()
            return clip_logits.detach().mean(dim=1), float(loss.item())
        
        N = x.size(0)
        if N <= max_chunk:
            if prior is not None:
                out = self.model(x, prior=prior)
            else:
                out = self.model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = binary_entropy_loss(logits)
            loss.backward()
            self.optimizer.step()
            return logits.detach(), float(loss.item())

        # Sub-batch gradient accumulation across chunks of max_chunk clips
        logits_list = []
        total_loss = 0.0
        for i in range(0, N, max_chunk):
            x_chunk = x[i:i + max_chunk]
            p_chunk = prior[i:i + max_chunk] if prior is not None else None
            
            if p_chunk is not None:
                out = self.model(x_chunk, prior=p_chunk)
            else:
                out = self.model(x_chunk)
                
            chunk_logits = out[0] if isinstance(out, (tuple, list)) else out
            chunk_weight = x_chunk.size(0) / float(N)
            chunk_loss = binary_entropy_loss(chunk_logits) * chunk_weight
            chunk_loss.backward()
            
            logits_list.append(chunk_logits.detach())
            total_loss += float(chunk_loss.item())

        self.optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        all_logits = torch.cat(logits_list, dim=0)
        return all_logits, total_loss

    def adapt_stream(
        self,
        test_loader: Generator,
        labels_all: np.ndarray,
        pre_auc: float
    ) -> Dict:
        """Run full test-time adaptation on test dataset.
        
        Args:
            test_loader: Iterable yielding test batches
            labels_all: Ground truth labels (used strictly for post-hoc evaluation)
            pre_auc: Unadapted source model AUC on this dataset
            
        Returns:
            dict: Adaptation metrics, post_auc, collapse flags
        """
        self.reset()
        adapted_logits = []
        losses = []
        
        for batch in test_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits, loss_val = self.adapt_batch(x)
            adapted_logits.append(logits.cpu().numpy())
            losses.append(loss_val)
            
            if self.mode == "batch_episodic":
                self.reset()
                
        all_logits = np.concatenate(adapted_logits, axis=0)
        all_scores = 1.0 / (1.0 + np.exp(-all_logits))
        
        post_auc = fast_auc(labels_all, all_scores)
        
        # Collapse check criteria:
        # 1. AUC drops by more than 0.03
        auc_drop = (pre_auc - post_auc) > 0.03
        
        # 2. At least 95% of test videos predicted as one class while true share is <= 85%
        pred_pos_fraction = float(np.mean(all_scores >= 0.50))
        true_pos_fraction = float(np.mean(labels_all == 1))
        degenerate = (pred_pos_fraction >= 0.95 or pred_pos_fraction <= 0.05) and (0.15 <= true_pos_fraction <= 0.85)
        
        collapsed = bool(auc_drop or degenerate)
        
        return {
            "pre_auc": float(pre_auc),
            "post_auc": float(post_auc),
            "delta_auc": float(post_auc - pre_auc),
            "mean_entropy": float(np.mean(losses)),
            "pred_pos_fraction": pred_pos_fraction,
            "true_pos_fraction": true_pos_fraction,
            "auc_drop_collapse": bool(auc_drop),
            "degenerate_collapse": bool(degenerate),
            "collapsed": collapsed,
            "mode": self.mode,
            "objective": self.objective,
        }


TestTimeAdapter = TTAAdapter
