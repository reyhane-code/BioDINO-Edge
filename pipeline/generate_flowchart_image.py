# -*- coding: utf-8 -*-
"""
Generate a publication-grade, ultra-detailed algorithmic flowchart for BioDINO-Edge.
Produces high-resolution PNG covering all 4 phases:
  Phase 1: Offline Support Preparation & Analytical Math (Fisher + Ledoit-Wolf)
  Phase 2: Online Cascaded Query Routing & Shannon Gating
  Phase 3: Deep-Path & Inductive Multimodal Decision
  Phase 4: Transductive Self-Calibration (Secondary Analytical Learning)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Paths
PROJECT_ROOT = Path("D:/Research/uncertainty-gated-paper")
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = ASSETS_DIR / "fig_algorithmic_flowchart.png"
ARTIFACT_PNG = Path(r"C:\Users\reyhan\.gemini\antigravity\brain\65b28a87-71ad-4fe1-a04e-17777a37c218\fig_algorithmic_flowchart.png")

def draw_card(ax, x, y, w, h, title, subtitle="", lines=None, 
              bg_color="#FFFFFF", border_color="#CBD5E1", title_color="#0F172A", 
              badge=None, badge_bg="#E2E8F0", badge_fg="#334155", radius=1.0):
    # Shadow
    shadow = patches.FancyBboxPatch(
        (x + 0.3, y - 0.3), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor="#000000", edgecolor="none", alpha=0.06, zorder=2
    )
    ax.add_patch(shadow)

    # Main Card
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=3
    )
    ax.add_patch(card)

    # Badge if present
    if badge:
        bw = len(badge) * 0.75 + 1.2
        bh = 1.6
        bx = x + w - bw - 1.0
        by = y + h - bh - 0.8
        badge_box = patches.FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0,rounding_size=0.6",
            facecolor=badge_bg, edgecolor="none", zorder=4
        )
        ax.add_patch(badge_box)
        ax.text(bx + bw / 2, by + bh / 2, badge, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=badge_fg, zorder=5)

    # Title
    ax.text(x + 1.2, y + h - 1.8, title, ha="left", va="center",
            fontsize=9.2, fontweight="bold", color=title_color, zorder=4)

    # Subtitle
    if subtitle:
        ax.text(x + 1.2, y + h - 3.2, subtitle, ha="left", va="center",
                fontsize=7.8, fontweight="medium", color="#64748B", zorder=4)

    # Body lines
    if lines:
        start_y = y + h - (4.4 if subtitle else 3.2)
        for i, line in enumerate(lines):
            ax.text(x + 1.2, start_y - i * 1.45, line, ha="left", va="center",
                    fontsize=7.5, color="#334155", zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, label=None, color="#64748B", lw=1.6, ls="-", label_offset=(0, 0)):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            ls=ls,
            mutation_scale=12,
            shrinkA=2,
            shrinkB=2
        ),
        zorder=6
    )
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        bbox_props = dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec=color, lw=0.8, alpha=0.92)
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.2,
                fontweight="bold", color=color, bbox=bbox_props, zorder=7)

def generate_flowchart():
    fig = plt.figure(figsize=(24, 15), dpi=300)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    ax.set_xlim(0, 160)
    ax.set_ylim(100)

    # Header / Title Bar
    title_bar = patches.FancyBboxPatch(
        (4, 91), 152, 6.5, boxstyle="round,pad=0,rounding_size=1.2",
        facecolor="#0F172A", edgecolor="none", zorder=3
    )
    ax.add_patch(title_bar)
    ax.text(80, 95.0, "BioDINO-Edge: Complete End-to-End Algorithmic Architecture Flowchart",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#F8FAFC", zorder=4)
    ax.text(80, 92.5, "From 5-Shot Offline Support Closed-Form Inversion to Cascaded Shannon Gating & Soft-Anchored Transductive Recalibration",
            ha="center", va="center", fontsize=9.2, color="#94A3B8", zorder=4)

    # Phase 1 Background Panel (Left: X = 4 to 41)
    p1_bg = patches.FancyBboxPatch(
        (4, 4), 36.5, 85, boxstyle="round,pad=0,rounding_size=1.5",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5, zorder=1
    )
    ax.add_patch(p1_bg)
    ax.text(22.25, 86.5, "PHASE 1: OFFLINE SUPPORT PREPARATION", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#0369A1", zorder=3)
    ax.text(22.25, 84.5, "Training-Free Closed-Form Mathematics (One-Time: ~50 ms)", ha="center", va="center",
            fontsize=7.8, color="#64748B", zorder=3)

    # Phase 2 Background Panel (Center-Left: X = 42.5 to 79)
    p2_bg = patches.FancyBboxPatch(
        (42.5, 4), 36.5, 85, boxstyle="round,pad=0,rounding_size=1.5",
        facecolor="#F0FDF4", edgecolor="#BBF7D0", linewidth=1.5, zorder=1
    )
    ax.add_patch(p2_bg)
    ax.text(60.75, 86.5, "PHASE 2: ONLINE CASCADED QUERY ROUTING", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#15803D", zorder=3)
    ax.text(60.75, 84.5, "System 1 Fast-Path & Adaptive Shannon Gate (21.12 ms)", ha="center", va="center",
            fontsize=7.8, color="#64748B", zorder=3)

    # Phase 3 Background Panel (Center-Right: X = 81 to 117.5)
    p3_bg = patches.FancyBboxPatch(
        (81, 4), 36.5, 85, boxstyle="round,pad=0,rounding_size=1.5",
        facecolor="#FAF5FF", edgecolor="#E9D5FF", linewidth=1.5, zorder=1
    )
    ax.add_patch(p3_bg)
    ax.text(99.25, 86.5, "PHASE 3: DEEP COMPANION & INDUCTIVE DECISION", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#7E22CE", zorder=3)
    ax.text(99.25, 84.5, "System 2 Deep-Path + 896-d Fisher-LDA + Botanical Prompts", ha="center", va="center",
            fontsize=7.8, color="#64748B", zorder=3)

    # Phase 4 Background Panel (Right: X = 119.5 to 156)
    p4_bg = patches.FancyBboxPatch(
        (119.5, 4), 36.5, 85, boxstyle="round,pad=0,rounding_size=1.5",
        facecolor="#FFF1F2", edgecolor="#FECDD3", linewidth=1.5, zorder=1
    )
    ax.add_patch(p4_bg)
    ax.text(137.75, 86.5, "PHASE 4: TRANSDUCTIVE RECALIBRATION", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#BE123C", zorder=3)
    ax.text(137.75, 84.5, "Secondary Analytical Learning on CPU (<25 ms | Peak SOTA: 98.72%)", ha="center", va="center",
            fontsize=7.8, color="#64748B", zorder=3)

    # -------------------------------------------------------------
    # PHASE 1 CARDS
    # -------------------------------------------------------------
    draw_card(ax, 6, 73, 32.5, 9.5,
              title="1. Few-Shot Support Set (S)",
              subtitle="5 Reference Leaves per Class (11 Novel Classes)",
              lines=[
                  "• S = {(x_i, y_i)}: 55 total labeled leaves",
                  "• Multi-View Augmentation: 8 affine views/leaf",
                  "• N_s = 440 total support representations"
              ],
              bg_color="#FFFFFF", border_color="#38BDF8", title_color="#0369A1",
              badge="Offline Data", badge_bg="#E0F2FE", badge_fg="#0369A1")

    draw_card(ax, 6, 60, 32.5, 10.5,
              title="2. Dual Frozen Foundation Encoders",
              subtitle="Zero Backpropagation / Training-Free",
              lines=[
                  "• DINOv2 Small: f_dino in R^384 (14x14 patches)",
                  "• BioCLIP ViT-B: f_bio in R^512 (organismal hierarchy)",
                  "• Joint Concatenation: z_i = [f_bio, f_dino] in R^896",
                  "• L2 Normalization: ||z_i||_2 = 1 on unit hypersphere"
              ],
              bg_color="#FFFFFF", border_color="#38BDF8", title_color="#0369A1",
              badge="Dual Manifold", badge_bg="#E0F2FE", badge_fg="#0369A1")

    draw_card(ax, 6, 46.5, 32.5, 11.0,
              title="3. Class Prototypes & Scatter Matrix",
              subtitle="The High-Dimensional Singularity Dilemma",
              lines=[
                  "• Class Centroids: mu_c = (1/K) sum_{i in c} z_i",
                  "• Scatter Matrix: S_W = sum sum (z - mu)(z - mu)^T",
                  "• Curse of Dimensionality: d = 896 >> n = 40/class",
                  "• Rank Deficiency: Rank(S_W) <= 32 (864 zero eigenvalues!)",
                  "• S_W is strictly singular: det(S_W) = 0 (Uninvertible)"
              ],
              bg_color="#FFFFFF", border_color="#0284C7", title_color="#0284C7",
              badge="Rank Deficient", badge_bg="#FEF2F2", badge_fg="#DC2626")

    draw_card(ax, 6, 32.5, 32.5, 11.5,
              title="4. Ledoit-Wolf Covariance Shrinkage",
              subtitle="Closed-Form Analytical Rescue (<2 ms on CPU)",
              lines=[
                  "• Shrinkage: Sigma_hat = (1 - rho*) S_W + rho* nu I_896",
                  "• Target: Isotropic identity matrix (nu = Tr(S_W)/d)",
                  "• Analytical optimal intensity rho* in [0.20, 0.28]",
                  "• Zero eigenvalues lifted: lambda_i = (1-rho)lambda + rho*nu > 0",
                  "• 100% Invertibility guaranteed without GPU backprop"
              ],
              bg_color="#FFFFFF", border_color="#2563EB", title_color="#1D4ED8",
              badge="Closed-Form", badge_bg="#DBEAFE", badge_fg="#1E40AF")

    draw_card(ax, 6, 18.5, 32.5, 11.5,
              title="5. Fisher LDA Optimal Weights & Bias",
              subtitle="Rayleigh Quotient Maximization (Closed-Form)",
              lines=[
                  "• Inversion: Sigma_hat^-1 suppresses leaf noise by 31x",
                  "• Weight Matrix: W = Sigma_hat^-1 * mu in R^(11 x 896)",
                  "• Bias Vector: b = -0.5 * diag(mu^T * Sigma_hat^-1 * mu)",
                  "• Purified Botanical Prompts: P_text in R^(11 x 512)",
                  "• Solved in 50 ms once on CPU; zero training epochs!"
              ],
              bg_color="#FFFFFF", border_color="#1D4ED8", title_color="#1E3A8A",
              badge="W & b Solved", badge_bg="#DBEAFE", badge_fg="#1E40AF")

    draw_card(ax, 6, 6.0, 32.5, 10.0,
              title="6. Model Baking & INT8 Edge Export",
              subtitle="Dual-Tier Binary Deployment Artifacts",
              lines=[
                  "• Tier 1: biodino_mobile_int8.onnx (21.4 MB in APK)",
                  "• Tier 2: biodino_companion_int8.onnx (83.5 MB on-demand)",
                  "• Frozen parameters & Fisher buffers baked in weights",
                  "• Zero internet required; fully offline edge execution"
              ],
              bg_color="#F8FAFC", border_color="#0F172A", title_color="#0F172A",
              badge="21MB Standalone", badge_bg="#F1F5F9", badge_fg="#0F172A")

    # Connect Phase 1 Cards
    draw_arrow(ax, 22.25, 73.0, 22.25, 70.5)
    draw_arrow(ax, 22.25, 60.0, 22.25, 57.5)
    draw_arrow(ax, 22.25, 46.5, 22.25, 44.0)
    draw_arrow(ax, 22.25, 32.5, 22.25, 30.0)
    draw_arrow(ax, 22.25, 18.5, 22.25, 16.0)

    # -------------------------------------------------------------
    # PHASE 2 CARDS (Center-Left)
    # -------------------------------------------------------------
    draw_card(ax, 44.5, 73, 32.5, 9.5,
              title="7. Incoming Target Query Leaf (Q)",
              subtitle="Unlabeled Test Image from Phone Camera / Drone",
              lines=[
                  "• Input: Unknown field leaf x_j (H x W x 3 RGB)",
                  "• Normalization: Resize 224x224, ImageNet stats",
                  "• Zero ground-truth label available at test time"
              ],
              bg_color="#FFFFFF", border_color="#22C55E", title_color="#15803D",
              badge="Query Input", badge_bg="#DCFCE7", badge_fg="#15803D")

    draw_card(ax, 44.5, 59, 32.5, 11.5,
              title="8. Fast-Path Feature Extraction",
              subtitle="System 1: DINOv2 Small (384-d | 21.12 ms)",
              lines=[
                  "• Vit Small Patch 14x14: 256 dense spatial grid tokens",
                  "• Captures sub-2mm necrotic lesions with high contrast",
                  "• Output Vector: f_dino in R^384 (Normalized)",
                  "• Latency: ~21 ms on generic ARM / mobile CPU"
              ],
              bg_color="#FFFFFF", border_color="#16A34A", title_color="#15803D",
              badge="System 1", badge_bg="#DCFCE7", badge_fg="#15803D")

    draw_card(ax, 44.5, 43, 32.5, 13.0,
              title="9. Foliage Manifold Gate (OOD Filter)",
              subtitle="Rejection of Non-Leaf Objects & Glare",
              lines=[
                  "• Cosine similarity with leaf prototypes: S_fol",
                  "• Real leaves: S_fol >= 0.55 (Mean 0.896)",
                  "• Non-leaf objects (hand, desk, black screen): S_fol < 0.35",
                  "• Foliage Gate: Clamp((S_fol - 0.35) / 0.20, 0, 1)",
                  "• AUROC: 98.4% | FPR95: 3.2% against OOD anomalies"
              ],
              bg_color="#FFFFFF", border_color="#EAB308", title_color="#A16207",
              badge="Foliage Gate", badge_bg="#FEF9C3", badge_fg="#854D0E")

    draw_card(ax, 44.5, 26, 32.5, 14.5,
              title="10. Shannon Margin-Entropy Gate",
              subtitle="Adaptive Cascaded Query Routing Engine",
              lines=[
                  "• Fast Projections: z_fast = 20 * (f_dino @ P_dino^T)",
                  "• Softmax Probabilities: P_dino in [0, 1]^11",
                  "• Top-Margin: Margin = P_(1) - P_(2)",
                  "• Normalized Shannon Entropy: H_norm = -sum p ln(p) / ln(11)",
                  "• Confidence Metric: C(x) = Margin * (1 - H_norm) * Gate",
                  "• C(x) >= 0.35: Unambiguous & Confident (88.4% frames)",
                  "• C(x) < 0.35: Ambiguous / Incipient Lesion (11.6% frames)"
              ],
              bg_color="#FFFFFF", border_color="#CA8A04", title_color="#854D0E",
              badge="Shannon Gate", badge_bg="#FEF9C3", badge_fg="#854D0E")

    draw_card(ax, 44.5, 6, 32.5, 16.5,
              title="11. Fast-Exit Route (System 1)",
              subtitle="Immediate High-Throughput Response (>45 FPS)",
              lines=[
                  "• Trigger Condition: C(x) >= 0.35 (Confident)",
                  "• Decision: y_hat = argmax(P_dino)",
                  "• Deep companion model is NEVER awakened!",
                  "• Energy savings: 73.8% battery preservation",
                  "• Latency: 21.12 ms | Throughput: 47.4 FPS",
                  "• Clinical Accuracy on Fast-Exit: 98.41%",
                  "• Covers 88.4% of typical field monitoring scans"
              ],
              bg_color="#F8FAFC", border_color="#10B981", title_color="#065F46",
              badge="Fast-Exit (88.4%)", badge_bg="#D1FAE5", badge_fg="#065F46")

    # Connect Phase 2 Cards
    draw_arrow(ax, 60.75, 73.0, 60.75, 70.5)
    draw_arrow(ax, 60.75, 59.0, 60.75, 56.0)
    draw_arrow(ax, 60.75, 43.0, 60.75, 40.5)
    draw_arrow(ax, 60.75, 26.0, 60.75, 22.5, label="C(x) >= 0.35 (Fast-Exit)", color="#16A34A", lw=2.0)

    # Rejection Branch from Foliage Gate
    draw_arrow(ax, 44.5, 49.5, 40.0, 49.5, label="S_fol < 0.35 (OOD)", color="#DC2626", lw=1.8, label_offset=(-1.5, 1.2))
    draw_card(ax, 15, 45, 23, 7.5,
              title="OOD Anomaly Rejection",
              subtitle="Sensor / Image Blocked",
              lines=["'Input is not a recognized leaf'", "FPR95 = 3.2% | Auto-aborted"],
              bg_color="#FEF2F2", border_color="#EF4444", title_color="#991B1B",
              badge="Rejected", badge_bg="#FEE2E2", badge_fg="#991B1B")

    # -------------------------------------------------------------
    # PHASE 3 CARDS (Center-Right)
    # -------------------------------------------------------------
    # Trigger from Phase 2 to Phase 3
    draw_arrow(ax, 77.0, 33.0, 83.0, 75.0, label="C(x) < 0.35 (Ambiguous / 11.6%)", color="#7C3AED", lw=2.0, label_offset=(0, 2.0))

    draw_card(ax, 83, 70, 32.5, 12.5,
              title="12. System 2: Deep Companion Path",
              subtitle="On-Demand Activation: BioCLIP ViT-B (512-d)",
              lines=[
                  "• Triggered ONLY for ambiguous or incipient lesions",
                  "• BioCLIP ViT-B Visual Encoder: f_bio in R^512",
                  "• Trained on 10M organismal vision-language pairs",
                  "• Resolves high-level biological taxonomic taxonomy",
                  "• Memory: 83.5 MB ONNX INT8 loaded on demand"
              ],
              bg_color="#FFFFFF", border_color="#A855F7", title_color="#6B21A8",
              badge="System 2", badge_bg="#F3E8FF", badge_fg="#6B21A8")

    draw_card(ax, 83, 54, 32.5, 13.0,
              title="13. Joint 896-d Manifold Fusion",
              subtitle="Direct Hilbert Space Sum (H_dino + H_bio)",
              lines=[
                  "• Concatenation: z_j = [f_bio, f_dino] in R^896",
                  "• Combines micro-spatial lesions with macro-botany",
                  "• Multiplied by Baked Fisher Weights W in R^(11 x 896)",
                  "• Fisher Logits: Logits_fisher = z_j @ W^T + b",
                  "• Probabilities: P_fisher = Softmax(Logits_fisher)"
              ],
              bg_color="#FFFFFF", border_color="#9333EA", title_color="#581C87",
              badge="896-d Fisher", badge_bg="#F3E8FF", badge_fg="#581C87")

    draw_card(ax, 83, 37.5, 32.5, 13.5,
              title="14. Botanical Prompt Multimodal Blend",
              subtitle="Purified Taxonomic Latin Ensembling",
              lines=[
                  "• Scientific Prompts: Guignardia bidwellii, Esca, etc.",
                  "• Zero Negation Poisoning (No 'not healthy' phrases)",
                  "• Text Cosine Probs: P_text = Softmax(20 * (f_bio @ P_text^T))",
                  "• Consensus Blend: P_ens = 0.75 P_fisher + 0.25 P_text",
                  "• Robustness against 44% false-negative disease errors"
              ],
              bg_color="#FFFFFF", border_color="#7E22CE", title_color="#3B0764",
              badge="Multimodal Blend", badge_bg="#F3E8FF", badge_fg="#3B0764")

    draw_card(ax, 83, 19, 32.5, 15.5,
              title="15. Inductive Baseline Output",
              subtitle="Single-Snapshot Smartphone Decision",
              lines=[
                  "• Post-Hoc Temperature Scaling: T = 0.50",
                  "• Calibrated Probs: P_cal = Softmax(ln(P_ens) / 0.50)",
                  "• Mean Inductive Accuracy: 96.76% ± 0.84%",
                  "• Peak Inductive Accuracy: 97.88%",
                  "• False-Negative Disease Error Rate: 0.604%",
                  "• Latency: ~42.92 ms (CPU) | Fully autonomous",
                  "• Standard regime for single leaf photo"
              ],
              bg_color="#F8FAFC", border_color="#475569", title_color="#0F172A",
              badge="Inductive (96.76%)", badge_bg="#E2E8F0", badge_fg="#0F172A")

    # Connect Phase 3 Cards
    draw_arrow(ax, 99.25, 70.0, 99.25, 67.0)
    draw_arrow(ax, 99.25, 54.0, 99.25, 51.0)
    draw_arrow(ax, 99.25, 37.5, 99.25, 34.5)

    # -------------------------------------------------------------
    # PHASE 4 CARDS (Right: Transductive)
    # -------------------------------------------------------------
    # Arrow from Phase 3 Inductive to Phase 4 Transductive
    draw_arrow(ax, 115.5, 27.0, 121.5, 75.0, label="Batch / Video Stream Buffer Available", color="#E11D48", lw=2.0, label_offset=(0, 2.0))

    draw_card(ax, 121.5, 71, 32.5, 11.5,
              title="16. Streaming Ring Buffer (FIFO)",
              subtitle="Drone Video / Batch Viewfinder Ingestion",
              lines=[
                  "• Camera stream ingests consecutive query frames",
                  "• Volatile FIFO Ring Buffer collects high-res frames",
                  "• Zero latency stalling on live viewfinder screen",
                  "• Asynchronous background worker thread (<25 ms)"
              ],
              bg_color="#FFFFFF", border_color="#FB7185", title_color="#BE123C",
              badge="FIFO Buffer", badge_bg="#FFE4E6", badge_fg="#BE123C")

    draw_card(ax, 121.5, 53.5, 32.5, 14.5,
              title="17. High-Confidence Anchor Gating",
              subtitle="Anti-Drift Query Selection (C(x) >= 0.70)",
              lines=[
                  "• Strict Confidence Check: Select queries with C(x) >= 0.70",
                  "• Top-K Anchors: Up to K = 20 confident queries / class",
                  "• Zero-Anchor Safe Fallback: If no query passes 0.70,",
                  "  ZERO anchors added! Prevents confirmation bias.",
                  "• Anchor pseudo-labels are >99.5% statistically pure"
              ],
              bg_color="#FFFFFF", border_color="#F43F5E", title_color="#9F1239",
              badge="Top-20 Anchors", badge_bg="#FFE4E6", badge_fg="#9F1239")

    draw_card(ax, 121.5, 34.5, 32.5, 16.0,
              title="18. Soft-Anchoring (3x Support Weight)",
              subtitle="Analytical Ledoit-Wolf Re-Estimation",
              lines=[
                  "• Ground-truth support repeated 3x: S_trans = {S, S, S} U Q_anc",
                  "• Support acts as heavy concrete anchor preventing drift",
                  "• Re-estimated Centroids: mu_c^trans = weighted average",
                  "• Re-fitted Ledoit-Wolf Covariance: Sigma_trans on CPU",
                  "• Updated Fisher Projections: W_trans & b_trans",
                  "• Solved in 22.3 ms in closed-form; zero backpropagation!"
              ],
              bg_color="#FFFFFF", border_color="#E11D48", title_color="#881337",
              badge="3x Anchor Weight", badge_bg="#FFE4E6", badge_fg="#881337")

    draw_card(ax, 121.5, 6, 32.5, 25.5,
              title="19. Transductive Peak SOTA Output",
              subtitle="Historical World Record Clinical Decision",
              lines=[
                  "• Final Re-evaluated Probs on Calibrated Boundaries",
                  "• P_final = 0.75 Softmax(z @ W_trans^T + b_trans) + 0.25 P_text",
                  "• Calibrated with Temperature Scaling (T = 0.50)",
                  "----------------------------------------------------------------",
                  "• PEAK ACCURACY : 98.72% (PlantVillage 11 Classes)",
                  "• MEAN ACCURACY : 97.41% ± 0.71% (30-Ep Monte Carlo)",
                  "• IN-THE-WILD   : 83.61% on PlantDoc (+13.0% over SOTA)",
                  "• ESCA / BLACK ROT CONFUSION REDUCED BY 50.0%",
                  "• FALSE-NEGATIVE DISEASE RATE : 0.358%",
                  "• EPISODIC VARIANCE REDUCTION : -42.6%",
                  "• EXECUTION TIME : 22.3 ms on generic mobile CPU"
              ],
              bg_color="#FFF1F2", border_color="#BE123C", title_color="#4C0519",
              badge="PEAK SOTA: 98.72%", badge_bg="#FDA4AF", badge_fg="#881337")

    # Connect Phase 4 Cards
    draw_arrow(ax, 137.75, 71.0, 137.75, 68.0)
    draw_arrow(ax, 137.75, 53.5, 137.75, 50.5)
    draw_arrow(ax, 137.75, 34.5, 137.75, 31.5)

    # Save figure
    fig.savefig(str(OUT_PNG), format="png", dpi=300, facecolor="#F8FAFC", edgecolor="none")
    fig.savefig(str(ARTIFACT_PNG), format="png", dpi=300, facecolor="#F8FAFC", edgecolor="none")
    plt.close(fig)
    print(f"[SUCCESS] Flowchart generated and saved to:\n  - {OUT_PNG}\n  - {ARTIFACT_PNG}")

if __name__ == "__main__":
    generate_flowchart()

