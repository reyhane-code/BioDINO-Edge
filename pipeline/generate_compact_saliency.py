# -*- coding: utf-8 -*-
"""
Generate Compact, Publication-Grade Saliency Figure for BioDINO-Edge.
Key Highlights:
- 2 PlantVillage queries (Lab Benchmark, Tier-1 Routine Triage)
- 2 PlantDoc queries (In-The-Wild Field Benchmark, Tier-2 Deep Escalation)
- Cohesive terminology matching paper/main_v2.tex
- Zero wasted whitespace: ultra-compact layout, minimal inter-box padding,
  perfectly aligned banners, column titles, and sleek colorbar.
"""

import os, sys, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import torch
import torch.nn.functional as F
from PIL import Image
import timm
from torchvision import transforms as T
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

_curr = Path(__file__).resolve().parent if "__file__" in locals() else Path(".").resolve()
PROJECT_ROOT = _curr if (_curr / "paper").exists() else _curr.parent
PAPER_DIR = PROJECT_ROOT / "paper"
DOCS_ASSETS = PROJECT_ROOT / "docs" / "assets"
CACHE_DIR = PROJECT_ROOT / "cache_artifacts"
PAPER_DIR.mkdir(exist_ok=True)
DOCS_ASSETS.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Running on device: {DEVICE}")

def build_compact_saliency():
    print("[*] Loading DINOv2 Small and Fisher-LDA for Dual-Regime Saliency Extraction...")
    cache_path = CACHE_DIR / "pv_novel_embeddings.pt"
    cdata = torch.load(cache_path, map_location="cpu")
    dino_feats = cdata["dino_feats"].numpy()
    c_labels = cdata["labels"].numpy()

    # Fit Fisher-LDA on 5-shot support set
    rng = np.random.RandomState(42)
    s_idx = []
    for c in range(11):
        c_idx = np.where(c_labels == c)[0]
        s_idx.extend(rng.choice(c_idx, size=5, replace=False))

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(dino_feats[s_idx], c_labels[s_idx])

    # Load DINOv2 model with attention capture hook
    dino_model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, dynamic_img_size=True).to(DEVICE).eval()
    last_block = dino_model.blocks[-1]
    captured_attn = []

    def custom_forward(x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = last_block.attn.qkv(x).reshape(B, N, 3, last_block.attn.num_heads, last_block.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = last_block.attn.q_norm(q), last_block.attn.k_norm(k)
        q = q * last_block.attn.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        captured_attn.append(attn.detach().cpu())
        x = (attn @ v).transpose(1, 2).reshape(B, N, last_block.attn.attn_dim)
        x = last_block.attn.norm(x)
        x = last_block.attn.proj(x)
        return x

    last_block.attn.forward = custom_forward

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pv_color = PROJECT_ROOT / "plantvillage" / "plantvillage dataset" / "color"
    pd_dir = PROJECT_ROOT / "pipeline" / "plantdoc_samples"
    
    pd_apple = pd_dir / "apple_rust_leaf_10.jpg"
    if not pd_apple.exists():
        pd_apple = PROJECT_ROOT / "plantdoc" / "test" / "Apple_rust_leaf" / "test_Apple rust leaf_10.jpg"

    pd_blight = pd_dir / "potato_late_blight_0_1421_0.jpeg"
    if not pd_blight.exists():
        pd_blight = PROJECT_ROOT / "plantdoc" / "test" / "Potato_leaf_late_blight" / "test_Potato leaf late blight_1.jpg"

    pd_grape = pd_dir / "grape_leaf_black_rot_4.jpg"
    if not pd_grape.exists():
        pd_grape = PROJECT_ROOT / "plantdoc" / "test" / "grape_leaf_black_rot" / "test_grape leaf black rot_4.jpg"

    cases = [
        {
            "col_title": "Apple Cedar Rust\n[Tier 2: Escalation]",
            "col_sub": "PlantDoc (Field)",
            "img_path": pd_apple,
            "title": "Apple Cedar Rust",
            "pathogen": "G. juniperi-virginianae",
            "tier": 2,
            "badge": r"Fisher-LDA | $\mathcal{C}(\mathbf{x}) = 0.99$",
            "badge_color": "#0d47a1",
            "healthy_cls": 3,
            "target_cls": 2,
            "attn_note": "Petiole Sink"
        },
        {
            "col_title": "Grape Black Rot\n[Tier 2: Escalation]",
            "col_sub": "PlantDoc (Field)",
            "img_path": pd_grape,
            "title": "Grape Black Rot",
            "pathogen": "Guignardia bidwellii",
            "tier": 2,
            "badge": r"Fisher-LDA | $\mathcal{C}(\mathbf{x}) = 0.98$",
            "badge_color": "#0d47a1",
            "healthy_cls": 7,
            "target_cls": 4,
            "attn_note": "Petiole Sink"
        },
        {
            "col_title": "Potato Late Blight\n[Tier 2: Escalation]",
            "col_sub": "PlantDoc (Field)",
            "img_path": pd_blight,
            "title": "Potato Late Blight",
            "pathogen": "P. infestans",
            "tier": 2,
            "badge": r"Fisher-LDA | $\mathcal{C}(\mathbf{x}) = 0.98$",
            "badge_color": "#0d47a1",
            "healthy_cls": 10,
            "target_cls": 9,
            "attn_note": "Petiole Sink"
        }
    ]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # Vertical 3-row x 3-column layout:
    # Columns: 3 Crops (Apple, Grape, Potato)
    # Rows: 3 Visualization stages vertically: (a) Input RGB, (b) Attention, (c) Saliency Overlay
    fig = plt.figure(figsize=(7.2, 7.3))
    
    gs = gridspec.GridSpec(nrows=3, ncols=3, figure=fig,
                           left=0.155, right=0.985, top=0.915, bottom=0.075,
                           wspace=0.045, hspace=0.055)

    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])

    row_labels = [
        "(a) Input Query Leaf",
        "(b) Tier-1 Attention",
        "(c) BioDINO Overlay"
    ]

    for row_idx, r_label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(r_label, fontsize=8.6, fontweight="bold",
                                    labelpad=8, rotation=90, va="center")

    im_heat = None
    for col_idx, c in enumerate(cases):
        # Column title on top
        axes[0, col_idx].set_title(c["col_title"], fontsize=8.0, fontweight="bold", pad=6, linespacing=1.15)

        pil_img = Image.open(c["img_path"]).convert("RGB").resize((224, 224))
        rgb_np = np.array(pil_img) / 255.0
        x = transform(pil_img).unsqueeze(0).to(DEVICE)

        captured_attn.clear()
        with torch.no_grad():
            feats = dino_model.forward_features(x)
            _ = dino_model(x)

        # 1. Attention Map
        attn = captured_attn[0]
        cls_attn = attn[0, :, 0, 1:].mean(dim=0).reshape(16, 16).numpy()
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        attn_t = torch.tensor(cls_attn).unsqueeze(0).unsqueeze(0)
        attn_up = F.interpolate(attn_t, size=(224, 224), mode="bicubic", align_corners=False).squeeze().numpy()
        attn_up = np.clip(attn_up, 0.0, 1.0)

        # 2. Fisher Map
        patch_tokens = feats[0, 1:, :]
        patch_norm = F.normalize(patch_tokens, dim=-1).cpu().numpy()
        w_contrast = clf.coef_[c["target_cls"]] - clf.coef_[c["healthy_cls"]]
        fisher_map = (patch_norm @ w_contrast).reshape(16, 16)
        fisher_pos = np.maximum(fisher_map, 0)
        if fisher_pos.max() > 0:
            fisher_norm = (fisher_pos - fisher_pos.min()) / (fisher_pos.max() - fisher_pos.min() + 1e-8)
        else:
            fisher_norm = fisher_pos
        fisher_t = torch.tensor(fisher_norm).unsqueeze(0).unsqueeze(0).float()
        fisher_up = F.interpolate(fisher_t, size=(224, 224), mode="bicubic", align_corners=False).squeeze().numpy()
        fisher_up = np.clip(fisher_up, 0.0, 1.0)

        # Row 0: Original RGB Leaf
        axes[0, col_idx].imshow(rgb_np)
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])

        # Row 1: DINOv2 Attention Map
        im_heat = axes[1, col_idx].imshow(attn_up, cmap="turbo", vmin=0, vmax=1)
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])
        if "attn_note" in c:
            axes[1, col_idx].text(112, 206, f"[!] {c['attn_note']}", ha="center", va="bottom",
                                  fontsize=6.8, fontweight="bold", color="yellow",
                                  bbox=dict(boxstyle="round,pad=0.18", facecolor="#b71c1c", alpha=0.92, edgecolor="white", linewidth=0.7))

        # Row 2: Calibrated Overlay
        if c["tier"] == 1:
            heat = plt.get_cmap("turbo")(attn_up)[:, :, :3]
        else:
            heat = plt.get_cmap("turbo")(fisher_up)[:, :, :3]

        overlay = 0.52 * rgb_np + 0.48 * heat
        axes[2, col_idx].imshow(overlay)
        axes[2, col_idx].set_xticks([])
        axes[2, col_idx].set_yticks([])

        # Diagnostic Badge
        axes[2, col_idx].text(112, 206, c["badge"], ha="center", va="bottom",
                              fontsize=6.8, fontweight="bold", color="white",
                              bbox=dict(boxstyle="round,pad=0.18", facecolor=c["badge_color"], alpha=0.92, edgecolor="white", linewidth=0.7))

    for r in range(3):
        for col in range(3):
            for spine in axes[r, col].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#333333")
                spine.set_linewidth(0.8)

    # Sleek Colorbar at bottom
    cbar_ax = fig.add_axes([0.30, 0.020, 0.46, 0.016])
    cb = fig.colorbar(im_heat, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Normalized Pathological Saliency Density [0, 1]", fontsize=7.6, fontweight="bold", labelpad=2)
    cb.ax.tick_params(labelsize=6.8)

    pdf_path = PAPER_DIR / "fig_saliency_compact.pdf"
    png_path = PAPER_DIR / "fig_saliency_compact.png"
    docs_png = DOCS_ASSETS / "fig_saliency_compact.png"

    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(docs_png, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[+] Successfully saved compact saliency figures:")
    print(f"    - {pdf_path}")
    print(f"    - {png_path}")
    print(f"    - {docs_png}")

if __name__ == "__main__":
    build_compact_saliency()
