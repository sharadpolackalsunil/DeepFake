"""Measure trainable and total parameter counts per module for all models.

Provides empirical parameter figures to replace estimates in manuscript tables.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from typing import Dict, Tuple

from smsdt.models.smsdt_v3 import SMSDTv3
from smsdt.models.smsdt_v2 import SMSDTv2
from smsdt.models.smsdt import SMSDT as SMSDTv1
from smsdt.models.clip_encoder import CLIPFrameEncoder


def count_parameters(model: nn.Module) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
    """Count total and trainable parameters overall and by top-level submodule."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    breakdown = {}
    for name, child in model.named_children():
        c_tot = sum(p.numel() for p in child.parameters())
        c_trn = sum(p.numel() for p in child.parameters() if p.requires_grad)
        breakdown[name] = (c_tot, c_trn)
        
    return total, trainable, breakdown


def build_s_model(encoder: nn.Module) -> nn.Module:
    """Build single-frame S model (CLIP ViT-L/14 with LN-only tuning)."""
    class SMSDTS(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.encoder = enc
            self.head = nn.Linear(1024, 1)
        def forward(self, x):
            cls, _ = self.encoder(x)
            return self.head(cls).squeeze(-1)
    return SMSDTS(encoder)


def main():
    print("=" * 80, flush=True)
    print("S-MSDT PARAMETER AUDIT REPORT (Measured Values)", flush=True)
    print("=" * 80, flush=True)
    
    models = {}
    
    try:
        print("\n[1/4] Auditing S-MSDT v1 (Phase 1 Baseline)...", flush=True)
        m_v1 = SMSDTv1()
        models["SMSDT v1 (EffNet-B4 + Wavelet)"] = m_v1
        print("  ✓ S-MSDT v1 initialized.", flush=True)
    except Exception as e:
        print(f"  Note: S-MSDT v1 initialization: {e}", flush=True)
        
    print("\nLoading shared CLIP ViT-L/14 backbone (LN-only)...", flush=True)
    shared_encoder = CLIPFrameEncoder(name="openai/clip-vit-large-patch14", tune="layernorm")
    print("  ✓ Backbone loaded.", flush=True)

    try:
        print("\n[2/4] Auditing S-MSDT-S (Static LN-CLIP)...", flush=True)
        m_s = build_s_model(shared_encoder)
        models["S-MSDT-S (Static LN-CLIP)"] = m_s
        print("  ✓ S-MSDT-S built.", flush=True)
    except Exception as e:
        print(f"  Note: S-MSDT-S initialization: {e}", flush=True)
        
    try:
        print("\n[3/4] Auditing S-MSDT-ST (v2 Video ViT)...", flush=True)
        m_st = SMSDTv2(encoder=shared_encoder, dim=384, depth=2, heads=6)
        models["S-MSDT-ST (v2 Video ViT)"] = m_st
        print("  ✓ S-MSDT-ST built.", flush=True)
    except Exception as e:
        print(f"  Note: S-MSDT-ST initialization: {e}", flush=True)
        
    try:
        print("\n[4/4] Auditing S-MSDT Full / v3 (Stage C)...", flush=True)
        m_v3 = SMSDTv3(encoder=shared_encoder, dim=384, depth=2, heads=6)
        models["S-MSDT Full / v3 (Stage C)"] = m_v3
        print("  ✓ S-MSDT Full / v3 built.", flush=True)
    except Exception as e:
        print(f"  Note: S-MSDT-v3 initialization: {e}", flush=True)
        
    print("\n" + "=" * 80, flush=True)
    print(f"{'Model Name':<35} | {'Total Params':<14} | {'Trainable':<12} | {'Trainable %':<10}", flush=True)
    print("-" * 80, flush=True)
    
    for name, model in models.items():
        total, trainable, breakdown = count_parameters(model)
        pct = (trainable / total) * 100.0 if total > 0 else 0.0
        print(f"{name:<35} | {total:>14,d} | {trainable:>12,d} | {pct:>9.2f}%", flush=True)
        for sub_name, (sub_tot, sub_trn) in breakdown.items():
            sub_pct = (sub_trn / sub_tot) * 100.0 if sub_tot > 0 else 0.0
            print(f"   ├─ {sub_name:<30} : {sub_tot:>12,d} | {sub_trn:>10,d} ({sub_pct:5.2f}%)", flush=True)
            
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
