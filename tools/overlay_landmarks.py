"""Verify perioral landmark indices 52..71 and generate visual overlay check.

Saves visualization to results/figs/landmark_check.png.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smsdt.data.mouth_prior import compute_perioral_prior, MOUTH_INDICES_106


def generate_canonical_106_face():
    """Synthesize a canonical 106-point facial geometry for testing and visualization."""
    pts = np.zeros((106, 2), dtype=np.float32)
    
    # 0..32: Jawline arc (from ear to ear around chin)
    angles = np.linspace(-np.pi * 0.8, -np.pi * 0.2, 33)
    pts[0:33, 0] = 112.0 + 80.0 * np.cos(angles)
    pts[0:33, 1] = 100.0 - 95.0 * np.sin(angles)
    
    # 33..42: Left eyebrow
    pts[33:43, 0] = np.linspace(60, 95, 10)
    pts[33:43, 1] = 70.0 - 5.0 * np.sin(np.linspace(0, np.pi, 10))
    
    # 43..52: Right eyebrow
    pts[43:53, 0] = np.linspace(125, 160, 10)
    pts[43:53, 1] = 70.0 - 5.0 * np.sin(np.linspace(0, np.pi, 10))
    
    # 52..63: Outer mouth loop (12 points)
    mouth_angles_outer = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts[52:64, 0] = 112.0 + 26.0 * np.cos(mouth_angles_outer)
    pts[52:64, 1] = 160.0 + 12.0 * np.sin(mouth_angles_outer)
    
    # 64..71: Inner mouth loop (8 points)
    mouth_angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pts[64:72, 0] = 112.0 + 18.0 * np.cos(mouth_angles_inner)
    pts[64:72, 1] = 160.0 + 6.0 * np.sin(mouth_angles_inner)
    
    # 72..86: Eyes & Nose
    pts[72:87, 0] = np.linspace(70, 150, 15)
    pts[72:87, 1] = 95.0
    
    # 87..105: Remaining points
    pts[87:106, 0] = 112.0
    pts[87:106, 1] = np.linspace(100, 145, 19)
    
    return pts


def main():
    os.makedirs("results/figs", exist_ok=True)
    out_path = "results/figs/landmark_check.png"
    
    lmk106 = generate_canonical_106_face()
    prior = compute_perioral_prior(lmk106, grid_size=16, crop_size=224.0).numpy().reshape(16, 16)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Subplot 1: All 106 mesh landmarks
    ax1 = axes[0]
    ax1.scatter(lmk106[:, 0], lmk106[:, 1], c="gray", s=15, alpha=0.6, label="Other Mesh (86 pts)")
    ax1.scatter(lmk106[MOUTH_INDICES_106, 0], lmk106[MOUTH_INDICES_106, 1],
                c="red", s=35, label="Mouth (Indices 52-71)")
    ax1.set_xlim(0, 224)
    ax1.set_ylim(224, 0)
    ax1.set_title("InsightFace 2D-106 Landmark Layout", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.set_aspect("equal")
    
    # Subplot 2: Perioral Attention Prior Spotlight (16x16 Grid)
    ax2 = axes[1]
    im = ax2.imshow(prior, cmap="magma", vmin=0, vmax=1, origin="upper")
    ax2.set_title("Soft Perioral Gaussian Prior (16×16 Grid)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Subplot 3: Overlay of landmarks onto the patch spotlight
    ax3 = axes[2]
    prior_upscaled = cv2.resize(prior, (224, 224), interpolation=cv2.INTER_CUBIC)
    ax3.imshow(prior_upscaled, cmap="magma", vmin=0, vmax=1)
    ax3.scatter(lmk106[MOUTH_INDICES_106, 0], lmk106[MOUTH_INDICES_106, 1],
                c="cyan", s=25, edgecolors="white", label="Perioral Landmarks")
    ax3.set_xlim(0, 224)
    ax3.set_ylim(224, 0)
    ax3.set_title("Aligned Spotlight Overlay onto Face Space", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.set_aspect("equal")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"✓ Landmark check successfully saved to: {out_path}")


if __name__ == "__main__":
    main()

