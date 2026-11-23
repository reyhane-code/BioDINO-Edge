# -*- coding: utf-8 -*-
"""
BioDINO-Edge Algorithmic Flowchart Generator - Collision-Free & Publication-Grade
Flowchart Standard:
- Oval / Pill: Start, Terminals, and Final Outputs
- Rectangle: Deterministic Computational Steps (Process)
- Diamond: Conditional Branching & Gating Decisions
- Strictly Orthogonal (Straight Horizontal / Vertical) Connectors: Zero diagonal slashes, zero text intersections
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import shutil

OUT_PNG = Path("D:/Research/uncertainty-gated-paper/docs/assets/fig_algorithmic_flowchart.png")
ARTIFACT_PNG = Path(r"C:\Users\reyhan\.gemini\antigravity\brain\65b28a87-71ad-4fe1-a04e-17777a37c218\fig_algorithmic_flowchart.png")

def draw_oval(ax, x, y, w, h, text, color="#2563EB", text_color="#FFFFFF", border_color="#0F172A"):
    oval = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={h/2}",
        facecolor=color, edgecolor=border_color, linewidth=1.5, zorder=4
    )
    ax.add_patch(oval)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.2, fontweight="bold", color=text_color, zorder=5)

def draw_rect(ax, x, y, w, h, title, details="", color="#FFFFFF", border_color="#334155", title_color="#0F172A"):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="square,pad=0",
        facecolor=color, edgecolor=border_color, linewidth=1.4, zorder=4
    )
    ax.add_patch(rect)
    if details:
        ax.text(x, y + 0.32, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color=title_color, zorder=5)
        ax.text(x, y - 0.32, details, ha="center", va="center", fontsize=7.2, color="#475569", zorder=5)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color=title_color, zorder=5)

def draw_diamond(ax, x, y, w, h, text, color="#FEF3C7", border_color="#D97706", text_color="#92400E"):
    pts = [[x, y + h/2], [x + w/2, y], [x, y - h/2], [x - w/2, y]]
    poly = patches.Polygon(pts, closed=True, facecolor=color, edgecolor=border_color, linewidth=1.5, zorder=4)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", fontsize=7.6, fontweight="bold", color=text_color, zorder=5)

def draw_arrow(ax, x1, y1, x2, y2, label="", color="#334155", label_offset_x=0.0, label_offset_y=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=10, shrinkA=0, shrinkB=0),
        zorder=3
    )
    if label:
        mx = (x1 + x2) / 2 + label_offset_x
        my = (y1 + y2) / 2 + label_offset_y
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.2, fontweight="bold",
                color=color, bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="none", alpha=0.92), zorder=6)

def build_flowchart():
    fig = plt.figure(figsize=(19, 23), dpi=300)
    ax = fig.add_axes((0.015, 0.015, 0.97, 0.97))
    ax.axis("off")
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 46)

    # Main Header
    ax.text(24.0, 45.2, "BioDINO-Edge: Complete Algorithmic & Decision Flowchart",
            ha="center", va="center", fontsize=15.5, fontweight="bold", color="#0F172A")
    ax.text(24.0, 44.2, "End-to-End Pipeline: Support Manifold Calibration -> Dual-Path Cascaded Routing -> On-Device Transductive Recalibration",
            ha="center", va="center", fontsize=9.2, color="#475569")

    # =========================================================================
    # STAGE BOUNDING CONTAINERS (Background Subgraphs)
    # =========================================================================
    # Stage 1 Container (Left)
    c1 = patches.FancyBboxPatch((1.5, 14.2), 13.5, 29.0, boxstyle="round,pad=0.3,rounding_size=0.6",
                                fc="#F8FAFC", ec="#94A3B8", lw=1.3, ls="--", zorder=1)
    ax.add_patch(c1)
    ax.text(8.25, 42.4, "STAGE 1: OFFLINE SUPPORT CALIBRATION",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0369A1")
    ax.text(8.25, 41.7, "(One-time setup on-device / ~1.2s total)",
            ha="center", va="center", fontsize=7.0, color="#64748B")

    # Stage 2 Container (Right / Center + Exit Column)
    c2 = patches.FancyBboxPatch((16.5, 14.2), 30.0, 29.0, boxstyle="round,pad=0.3,rounding_size=0.6",
                                fc="#F0FDF4", ec="#86EFAC", lw=1.3, ls="--", zorder=1)
    ax.add_patch(c2)
    ax.text(31.5, 42.4, "STAGE 2: ONLINE QUERY ROUTING & CASCADED INFERENCE",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#15803D")
    ax.text(31.5, 41.7, "(Real-time edge execution / 21 ms Fast-Path vs 83 ms Deep Companion)",
            ha="center", va="center", fontsize=7.0, color="#64748B")

    # Stage 3 Container (Bottom)
    c3 = patches.FancyBboxPatch((1.5, 1.2), 45.0, 12.0, boxstyle="round,pad=0.3,rounding_size=0.6",
                                fc="#FFF1F2", ec="#FECDD3", lw=1.3, ls="--", zorder=1)
    ax.add_patch(c3)
    # Stage 3 Header placed on top-left so center channel (X=23.5) is 100% free of text collision
    ax.text(3.0, 12.3, "STAGE 3: TRANSDUCTIVE RECALIBRATION (ON-DEVICE SECONDARY LEARNING)",
            ha="left", va="center", fontsize=8.5, fontweight="bold", color="#BE123C")
    ax.text(3.0, 11.5, "Field-session manifold adaptation using unlabeled queries | Closed-form <25 ms | Zero fine-tuning loss",
            ha="left", va="center", fontsize=7.2, color="#881337")

    # =========================================================================
    # STAGE 1: SUPPORT FLOW (Column 1: Center X = 8.25, Width = 11.0)
    # =========================================================================
    cx1 = 8.25
    w1 = 11.0
    h1 = 1.5

    draw_oval(ax, cx1, 40.2, w1, 1.4, "START: 5-Shot Support Set (S)", color="#0284C7")
    draw_rect(ax, cx1, 37.0, w1, h1, "Multi-View Augmentation", "Generate 8 views per leaf (N_s = 440 total)", color="#F0F9FF", border_color="#0284C7")
    draw_rect(ax, cx1, 33.6, w1, h1, "Dual ViT Extraction (Frozen)", "DINOv2 (384-d) + BioCLIP (512-d)", color="#F0F9FF", border_color="#0284C7")
    draw_rect(ax, cx1, 30.2, w1, h1, "Joint Manifold Concatenation", "z_i = [f_bio, f_dino] in R^896 (L2-norm)", color="#F0F9FF", border_color="#0284C7")
    draw_rect(ax, cx1, 26.8, w1, h1, "Compute Class Centroids & Scatter", "Centroids mu_c and Scatter Matrix S_W", color="#F0F9FF", border_color="#0284C7")
    draw_rect(ax, cx1, 23.4, w1, h1, "Ledoit-Wolf Covariance Shrinkage", "Sigma_hat = (1 - rho*)S_W + rho* nu I (<2 ms)", color="#E0F2FE", border_color="#0284C7")
    draw_rect(ax, cx1, 20.0, w1, h1, "Solve Closed-Form Fisher LDA", "W = Sigma_hat^-1 * mu  |  b = -0.5 diag(...)", color="#E0F2FE", border_color="#0284C7")
    draw_rect(ax, cx1, 16.6, w1, h1, "Save W, b to Edge Model Buffer", "Baked into ONNX runtime weights", color="#E2E8F0", border_color="#64748B")

    # Connect Stage 1 Nodes (Strictly Vertical)
    draw_arrow(ax, cx1, 39.5, cx1, 37.75)
    draw_arrow(ax, cx1, 36.25, cx1, 34.35)
    draw_arrow(ax, cx1, 32.85, cx1, 30.95)
    draw_arrow(ax, cx1, 29.45, cx1, 27.55)
    draw_arrow(ax, cx1, 26.05, cx1, 24.15)
    draw_arrow(ax, cx1, 22.65, cx1, 20.75)
    draw_arrow(ax, cx1, 19.25, cx1, 17.35)

    # =========================================================================
    # STAGE 2: ONLINE QUERY INFERENCE & CASCADED ROUTING
    # Column 2 Center X = 23.5 (Width = 11.0)
    # Column 3 Center X = 39.0 (Width = 8.6)
    # Gutter between Col 2 & Col 3: 34.7 - 29.0 = 5.7 units (Completely collision-free)
    # =========================================================================
    cx2 = 23.5
    cx3 = 39.0
    w2 = 11.0
    w3 = 8.6

    # Step 0: Query Input
    draw_oval(ax, cx2, 40.2, w2, 1.4, "START: Unknown Query Leaf (x_q)", color="#16A34A")

    # Step 1: DINOv2 Fast Feature Extraction
    draw_rect(ax, cx2, 37.0, w2, h1, "Extract DINOv2 Features (Fast-Path)", "f_dino in R^384 (Edge ONNX / 12 ms)", color="#DCFCE7", border_color="#16A34A")
    draw_arrow(ax, cx2, 39.5, cx2, 37.75)

    # Decision 1: Foliage Quality Gate
    draw_diamond(ax, cx2, 33.6, 8.5, 2.2, "Valid Foliage?\n(S_fol >= 0.35)", color="#FEF3C7", border_color="#D97706")
    draw_arrow(ax, cx2, 36.25, cx2, 34.7)

    # Decision 1 -> FAIL: Reject OOD (Directly Right to Col 3)
    draw_oval(ax, cx3, 33.6, w3, 1.5, "REJECT QUERY\n(Out-of-Distribution / Non-Leaf)", color="#DC2626")
    draw_arrow(ax, cx2 + 4.25, 33.6, cx3 - w3/2, 33.6, label="No (OOD)", label_offset_y=0.28, color="#DC2626")

    # Decision 1 -> PASS: Compute Fast Margin & Shannon Confidence
    draw_rect(ax, cx2, 30.2, w2, h1, "Compute Shannon Confidence C(x)", "Margin = P_(1) - P_(2)  |  Entropy = -sum p ln(p)", color="#DCFCE7", border_color="#16A34A")
    draw_arrow(ax, cx2, 32.5, cx2, 30.95, label="Yes", label_offset_x=0.45, color="#15803D")

    # Decision 2: Confidence Threshold
    draw_diamond(ax, cx2, 26.8, 8.5, 2.2, "Confident?\n(C(x) >= 0.35)", color="#FEF3C7", border_color="#D97706")
    draw_arrow(ax, cx2, 29.45, cx2, 27.9)

    # Decision 2 -> YES: Fast-Path Exit (Directly Right to Col 3)
    draw_oval(ax, cx3, 26.8, w3, 1.5, "FAST-PATH EXIT (21 ms)\nAccept DINO Prediction ŷ_fast", color="#15803D")
    draw_arrow(ax, cx2 + 4.25, 26.8, cx3 - w3/2, 26.8, label="Yes (High Conf)", label_offset_y=0.28, color="#15803D")

    # Decision 2 -> NO: Ambiguous, escalate to Deep Companion
    draw_rect(ax, cx2, 23.4, w2, h1, "Activate BioCLIP ViT-B (System 2)", "f_bio in R^512 (Plant semantic prior / 62 ms)", color="#F3E8FF", border_color="#9333EA")
    draw_arrow(ax, cx2, 25.7, cx2, 24.15, label="No (Ambiguous)", label_offset_x=1.1, color="#9333EA")

    # Step 3: Concatenation
    draw_rect(ax, cx2, 20.0, w2, h1, "Concatenate: z_q = [f_bio, f_dino]", "896-d Multimodal Joint Manifold vector", color="#F3E8FF", border_color="#9333EA")
    draw_arrow(ax, cx2, 22.65, cx2, 20.75)

    # Step 4: Full Fisher LDA + Zero-Shot Text Prior
    draw_rect(ax, cx2, 16.6, w2, h1, "Evaluate 896-d Fisher LDA + Text Prior", "P_ens = 0.75 Softmax(z W^T + b) + 0.25 P_text", color="#F3E8FF", border_color="#9333EA")
    draw_arrow(ax, cx2, 19.25, cx2, 17.35)

    # Cross-Stage Connector: Feed W, b from Stage 1 into Stage 2 (Strictly Horizontal)
    draw_arrow(ax, cx1 + w1/2, 16.6, cx2 - w2/2, 16.6, label="Feed W, b", label_offset_y=0.28, color="#2563EB")

    # =========================================================================
    # STAGE 3: TRANSDUCTIVE RECALIBRATION FLOW (Bottom Container)
    # Clean 2-Row Orthogonal Topology
    # Upper Row Y = 8.8, Lower Row Y = 4.0
    # =========================================================================
    # Stage 2 to Stage 3 Connector: Clean vertical drop through free center space
    draw_arrow(ax, cx2, 15.85, cx2, 9.9, label="P_ens Logits", label_offset_x=0.9, label_offset_y=1.8, color="#334155")

    # Decision 3: Batch or Single Shot?
    draw_diamond(ax, cx2, 8.8, 8.5, 2.2, "Unlabeled Batch\nAvailable?", color="#FEF3C7", border_color="#D97706")

    # Branch NO: Single query image only -> Inductive Final Output (Strictly Horizontal to Col 3)
    draw_oval(ax, cx3, 8.8, w3, 1.5, "INDUCTIVE OUTPUT\nŷ = argmax P_ens (Acc: 96.7%)", color="#7C3AED")
    draw_arrow(ax, cx2 + 4.25, 8.8, cx3 - w3/2, 8.8, label="No (Single)", label_offset_y=0.28, color="#7C3AED")

    # Branch YES: Batch / Stream Available -> Anchor Filtering (Strictly Horizontal to Left)
    w3_1 = 9.5
    draw_rect(ax, 9.0, 8.8, w3_1, h1, "Step 3.1: Select Confident Anchors", "Q_anc = {x in Q | C(x) >= 0.70}", color="#FFE4E6", border_color="#E11D48")
    draw_arrow(ax, cx2 - 4.25, 8.8, 9.0 + w3_1/2, 8.8, label="Yes (Batch)", label_offset_y=0.28, color="#BE123C")

    # Connector from Step 3.1 to Lower Row (Strictly Vertical Drop)
    # Step 3.2 is centered at X=7.5 with width 11.0 (spans 2.0 to 13.0).
    # Arrow drops vertically at X=9.0 from Y=8.05 to Y=4.9 cleanly into Step 3.2!
    draw_arrow(ax, 9.0, 8.05, 9.0, 4.9, label="Augmented S_trans", label_offset_x=1.3, color="#BE123C")

    # --- Lower Row (Y = 4.0): Clean Left-to-Right Orthogonal Sequence ---
    # Step 3.2: Soft-Anchored Re-fitting
    draw_rect(ax, 7.5, 4.0, 11.0, 1.8, "Step 3.2: Soft-Anchored Re-fitting", "S_trans = {3x S} U Q_anc -> Solve W_trans, b_trans (<25 ms)", color="#FFE4E6", border_color="#E11D48")

    # Step 3.3: Re-evaluate Full Query Batch
    draw_rect(ax, 19.5, 4.0, 10.5, 1.8, "Step 3.3: Re-evaluate Query Batch", "P_trans = 0.75 Softmax(z W^T + b) + 0.25 P_text", color="#FFE4E6", border_color="#E11D48")
    draw_arrow(ax, 7.5 + 5.5, 4.0, 19.5 - 5.25, 4.0, color="#BE123C")

    # Step 3.4: Temperature Scaling
    draw_rect(ax, 30.5, 4.0, 8.5, 1.8, "Step 3.4: Temperature Scaling", "P_cal = Softmax(z_logits / 0.50)", color="#FFE4E6", border_color="#E11D48")
    draw_arrow(ax, 19.5 + 5.25, 4.0, 30.5 - 4.25, 4.0, color="#BE123C")

    # Final Decision Terminal (Transductive)
    draw_oval(ax, 40.5, 4.0, 9.0, 1.8, "FINAL DECISION (TRANSDUCTIVE)\nŷ_trans (Acc: 98.7% - Zero Loss)", color="#BE123C")
    draw_arrow(ax, 30.5 + 4.25, 4.0, 40.5 - 4.5, 4.0, color="#BE123C")

    # Save outputs
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PNG), format="png", dpi=300, facecolor="#FFFFFF", edgecolor="none")
    fig.savefig(str(ARTIFACT_PNG), format="png", dpi=300, facecolor="#FFFFFF", edgecolor="none")
    plt.close(fig)
    print(f"[SUCCESS] Flowchart generated at: {OUT_PNG}")
    print(f"[SUCCESS] Artifact updated at: {ARTIFACT_PNG}")

if __name__ == "__main__":
    build_flowchart()
