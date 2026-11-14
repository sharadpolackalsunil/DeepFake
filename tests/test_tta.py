"""Unit tests for smsdt/tta.py: state reset, parameter selection, collapse detection."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import torch
import torch.nn as nn

from smsdt.tta import TTAAdapter, get_layernorm_params, binary_entropy_loss


class ToyModel(nn.Module):
    """Simple toy model with Linear and LayerNorm layers."""
    def __init__(self, dim=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, 1)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.ln(h)
        return self.fc2(h).squeeze(-1)

import torch.nn.functional as F


def test_layernorm_param_selection():
    model = ToyModel(dim=16)
    # Freeze linear layers
    for p in model.fc1.parameters():
        p.requires_grad = False
    for p in model.fc2.parameters():
        p.requires_grad = False
    for p in model.ln.parameters():
        p.requires_grad = True
        
    ln_params = get_layernorm_params(model)
    assert len(ln_params) == 2  # ln.weight and ln.bias
    assert all(p.requires_grad for p in ln_params)


def test_tta_bit_identical_reset():
    torch.manual_seed(42)
    model = ToyModel(dim=16)
    
    # Save original parameter tensors
    orig_params = {k: v.clone() for k, v in model.state_dict().items()}
    
    adapter = TTAAdapter(model, lr=1e-2, mode="online")
    
    # Run adaptation step with dummy data
    x_dummy = torch.randn(10, 16)
    adapter.adapt_batch(x_dummy)
    
    # Confirm parameters changed
    changed = False
    for k, v in model.state_dict().items():
        if not torch.equal(v, orig_params[k]):
            changed = True
            break
    assert changed, "Parameters should have updated during adapt_batch"
    
    # Reset and verify bit-identical equality
    adapter.reset()
    for k, v in model.state_dict().items():
        assert torch.equal(v, orig_params[k]), f"Param {k} does not match original after reset!"


def test_collapse_detection_auc_drop():
    torch.manual_seed(42)
    model = ToyModel(dim=16)
    adapter = TTAAdapter(model, mode="online")
    
    # Synthetic test batches
    batches = [torch.randn(20, 16) for _ in range(3)]
    labels_all = np.array([0] * 30 + [1] * 30)
    
    # Pre-AUC high (0.95), post-AUC low (0.50) -> should trigger collapse
    res = adapter.adapt_stream(batches, labels_all, pre_auc=0.95)
    
    assert res["auc_drop_collapse"] is True
    assert res["collapsed"] is True


def test_collapse_detection_degenerate_predictions():
    torch.manual_seed(42)
    # Model that predicts extreme positive logits
    class DegenerateModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(16)
            self.bias = nn.Parameter(torch.tensor([10.0]))
        def forward(self, x):
            return self.ln(x).mean(dim=-1) + self.bias
            
    deg_model = DegenerateModel()
    adapter = TTAAdapter(deg_model, mode="online")
    
    batches = [torch.randn(20, 16) for _ in range(3)]
    labels_all = np.array([0] * 30 + [1] * 30)  # 50/50 balanced ground truth
    
    res = adapter.adapt_stream(batches, labels_all, pre_auc=0.50)
    
    assert res["pred_pos_fraction"] > 0.95
    assert res["degenerate_collapse"] is True
    assert res["collapsed"] is True
