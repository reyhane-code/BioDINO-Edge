# -*- coding: utf-8 -*-
"""
Generate an ultra-clean, publication-grade architecture diagram for BioDINO-Edge.
Completely eliminates all hidden, overlapping, or clipped text:
- Compact height (46 units) for optimal paper page layout.
- Strictly aligned, uniform card dimensions across all tiers and columns.
- All arrows connect directly edge-to-edge from the box borders.
- Labels are styled as clean, centered badges positioned directly over the connector lines.
- Refined, cohesive publication color palette (Deep Slate, Terracotta, Emerald, Royal Violet, Ocean Blue).
- Clean scientific notation and formulas strictly aligned with the paper text.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import shutil

_curr = Path(__file__).resolve().parent
PROJECT_ROOT = _curr if (_curr / "paper").exists() else _curr.parent
PAPER_DIR = PROJECT_ROOT / "paper"
PAPER_DIR.mkdir(exist_ok=True)
WEB_FIG_DIR = PROJECT_ROOT / "web" / "public" / "figures"


def create_architecture_diagram():
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'font.size': 8.5,
        'axes.labelsize': 9,
        'figure.titlesize': 11
    })

    # Aspect ratio: 16.0 x 7.2 (Canvas: 100 x 46)
    fig = plt.figure(figsize=(16.0, 7.2), dpi=300)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)

    # -------------------------------------------------------------------------
    # 0. Outer Canvas Framing
    # -------------------------------------------------------------------------
    outer_box = patches.Rectangle((0.6, 0.6), 98.8, 44.8, fill=False,
                                  edgecolor="#94A3B8", linewidth=1.1, zorder=1)
    ax.add_patch(outer_box)

    # Helper: Draw container panels with solid top header banners
    def draw_container(x, y, w, h, bg, border, title,
                       banner_bg, banner_txt_color="#FFFFFF", banner_h=2.3, banner_w=None, lw=1.3):
        # Outer panel
        panel = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.7",
                                       facecolor=bg, edgecolor=border, linewidth=lw, zorder=2)
        ax.add_patch(panel)

        # Header banner pill: if banner_w is specified, center it over container
        if banner_w is None:
            bw = w - 1.2
            bx = x + 0.6
        else:
            bw = banner_w
            bx = x + (w - bw) / 2.0

        banner = patches.FancyBboxPatch((bx, y + h - banner_h - 0.4), bw, banner_h,
                                        boxstyle="round,pad=0.10,rounding_size=0.4",
                                        facecolor=banner_bg, edgecolor=border, linewidth=0.85, zorder=3)
        ax.add_patch(banner)
        ax.text(bx + bw / 2, y + h - banner_h / 2 - 0.4, title, ha='center', va='center',
                fontsize=7.8, fontweight='bold', color=banner_txt_color, zorder=4)

    # Helper: Draw inner component cards with evenly distributed content
    def draw_card(x, y, w, h, bg, border, title="", items=None, radius=0.5, lw=1.0,
                  title_color="#0F172A", title_fs=7.4, zorder=4):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.14,rounding_size={radius}",
                                     facecolor=bg, edgecolor=border, linewidth=lw, zorder=zorder)
        ax.add_patch(box)

        if items is None:
            items = []

        total_elements = 1 + len(items) if title else len(items)
        if total_elements == 0:
            return

        step = h / (total_elements + 1)
        curr_y = y + h - step

        if title:
            ax.text(x + w / 2, curr_y, title, ha='center', va='center',
                    fontsize=title_fs, fontweight='bold', color=title_color, zorder=zorder + 1)
            curr_y -= step

        for text_str, color, fs, is_bold in items:
            ax.text(x + w / 2, curr_y, text_str, ha='center', va='center',
                    fontsize=fs, color=color, fontweight='bold' if is_bold else 'normal', zorder=zorder + 1)
            curr_y -= step

    # Helper: Draw straight arrow with centered badge on the line
    def draw_arrow(x1, y1, x2, y2, color="#2563EB", lw=1.4, ls='-',
                   label="", label_pos=0.5,
                   label_color="#1D4ED8", label_bg="white", label_fs=6.2):
        if ls == '-':
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.36",
                                        color=color, lw=lw, linestyle='-', shrinkA=0, shrinkB=0), zorder=5)
        else:
            # 1. Draw dashed/dotted body line
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=ls, zorder=5)
            # 2. Draw crisp solid arrowhead so dash/dot pattern never cuts off arrow wings
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx**2 + dy**2)**0.5
            ux = dx / max(1e-6, dist)
            uy = dy / max(1e-6, dist)
            ax.annotate('', xy=(x2, y2), xytext=(x2 - ux * 0.05, y2 - uy * 0.05),
                        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.36",
                                        color=color, lw=lw, linestyle='-', shrinkA=0, shrinkB=0), zorder=6)

        # Text badge sitting centered on the line
        if label:
            lx = x1 + (x2 - x1) * label_pos
            ly = y1 + (y2 - y1) * label_pos
            ax.text(lx, ly, label, ha='center', va='center', fontsize=label_fs,
                    color=label_color, fontweight='bold', zorder=7, multialignment='center',
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=label_bg, edgecolor=color, alpha=0.98, lw=0.75))

    # Helper: Draw multi-segment 90-degree orthogonal path with arrow on final segment
    def draw_orthogonal_arrow(waypoints, color="#2563EB", lw=1.4, ls='-',
                              label="", label_seg=1, label_pos=0.5,
                              label_color="#1D4ED8", label_bg="white", label_fs=6.1):
        if ls == '-':
            for i in range(len(waypoints) - 2):
                ax.plot([waypoints[i][0], waypoints[i+1][0]],
                        [waypoints[i][1], waypoints[i+1][1]],
                        color=color, lw=lw, linestyle='-', zorder=5)
            ax.annotate('', xy=waypoints[-1], xytext=waypoints[-2],
                        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.36",
                                        color=color, lw=lw, linestyle='-', shrinkA=0, shrinkB=0), zorder=5)
        else:
            # 1. Draw all segments with dashed/dotted line style
            for i in range(len(waypoints) - 1):
                ax.plot([waypoints[i][0], waypoints[i+1][0]],
                        [waypoints[i][1], waypoints[i+1][1]],
                        color=color, lw=lw, linestyle=ls, zorder=5)
            # 2. Draw crisp solid arrowhead so dash/dot pattern never cuts off arrow wings
            p_prev = waypoints[-2]
            p_end = waypoints[-1]
            dx = p_end[0] - p_prev[0]
            dy = p_end[1] - p_prev[1]
            dist = (dx**2 + dy**2)**0.5
            ux = dx / max(1e-6, dist)
            uy = dy / max(1e-6, dist)
            ax.annotate('', xy=p_end, xytext=(p_end[0] - ux * 0.05, p_end[1] - uy * 0.05),
                        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.36",
                                        color=color, lw=lw, linestyle='-', shrinkA=0, shrinkB=0), zorder=6)

        if label:
            p_prev = waypoints[label_seg]
            p_curr = waypoints[label_seg + 1]
            lx = p_prev[0] + (p_curr[0] - p_prev[0]) * label_pos
            ly = p_prev[1] + (p_curr[1] - p_prev[1]) * label_pos
            ax.text(lx, ly, label, ha='center', va='center', fontsize=label_fs,
                    color=label_color, fontweight='bold', zorder=7, multialignment='center',
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=label_bg, edgecolor=color, alpha=0.98, lw=0.75))

    # =========================================================================
    # 1. COLUMN 1 (LEFT): Live Inputs (Top) & Calibration Engine (Bottom)
    #    x = 1.6 to 21.6 (Width = 20.0)
    # =========================================================================

    # --- Box 1: Live Foliar Query Stream (Top-Left) ---
    # Runtime stream evaluated directly across the cascaded edge tiers.
    # Container y: 32.5 to 44.4 (h = 11.9). Banner: Deep Slate Navy
    draw_container(x=1.6, y=32.5, w=20.0, h=11.9,
                   bg="#F8FAFC", border="#334155",
                   title="LIVE QUERY STREAM (ONLINE)",
                   banner_bg="#1E293B", banner_txt_color="#FFFFFF")

    # Card 1.1: Live Field Query Stream (Aligned with Tier-1 center y=36.7)
    draw_card(2.4, 33.6, 18.4, 6.6, "#FFFFFF", "#3B82F6",
              title="Live Foliar Query Stream",
              items=[
                  (r"Leaf Photos ($\mathbf{x}_q$) | Video Frames", "#0F172A", 6.9, True),
                  ("Uncontrolled Field Acquisition", "#475569", 6.5, False),
                  (r"Direct Input $\to$ Tier-1 DINOv2 Fast Path", "#1D4ED8", 6.6, True)
              ], title_color="#1D4ED8")

    # --- Box 2: Support Calibration & Domain Knowledge (Bottom-Left) ---
    # Pre-computed offline: Support views & curated taxonomic prompt templates.
    # Container y: 1.6 to 31.2 (h = 29.6). Banner: Rich Terracotta
    draw_container(x=1.6, y=1.6, w=20.0, h=29.6,
                   bg="#FFF7ED", border="#EA580C",
                   title="PHASE 0: OFFLINE CALIBRATION & SETUP",
                   banner_bg="#C2410C", banner_txt_color="#FFFFFF")

    # Card 2.1: Predefined Botanical Prompts (Offline)
    draw_card(2.4, 22.4, 18.4, 5.5, "#F0F9FF", "#0284C7",
              title=r"Predefined Class Prompts (Offline)",
              items=[
                  (r"Fixed Ensemble: $M=3$ Templates / Class", "#0369A1", 6.8, True),
                  (r"Taxonomic Botany: Host + Pathogen Etiology", "#475569", 6.5, False),
                  (r"BioCLIP Text Enc. $\phi_{\mathrm{bio}} \to \mathbf{P}_{\mathrm{text}}^{(c)}$ (Cached)", "#0F172A", 6.7, True)
              ], title_color="#0369A1", title_fs=7.1)

    # Card 2.2: Few-Shot Support Foliage
    draw_card(2.4, 15.6, 18.4, 5.4, "#FFFFFF", "#F97316",
              title="Few-Shot Support Foliage",
              items=[
                  (r"$\mathcal{S}_c = \{(\mathbf{x}_i, c)\}_{i=1}^5$ ($K=5$ Shots/Class, 11 Classes)", "#0F172A", 6.8, True),
                  (r"8$\times$ Multi-View Foliar Augmentation", "#C2410C", 6.7, True),
                  ("Generates $N=440$ Synthetic Field Views", "#475569", 6.4, False)
              ], title_color="#C2410C")

    # Arrow 1: Support Foliage -> Extraction (edge-to-edge)
    draw_arrow(11.6, 15.6, 11.6, 14.2, color="#EA580C", lw=1.3)

    # Card 2.3: Dual Foundation Extraction
    draw_card(2.4, 9.0, 18.4, 5.2, "#FFFFFF", "#EA580C",
              title="Dual Foundation Extraction",
              items=[
                  (r"DINOv2 (384-D) + BioCLIP ViT-B (512-D)", "#0F172A", 6.7, False),
                  (r"Joint Foliar Space: $\mathbf{Z} \in \mathbb{R}^{440 \times 896}$", "#9A3412", 6.9, True),
                  (r"Visual Prototypes: $\mathbf{P}_{\mathrm{bio, vis}}^{(c)} \in \mathbb{R}^{512}$", "#475569", 6.4, False)
              ], title_color="#C2410C")

    # Arrow 2: Extraction -> Fisher-LDA
    draw_arrow(11.6, 9.0, 11.6, 7.6, color="#D97706", lw=1.3)

    # Card 2.4: Ledoit-Wolf & Fisher-LDA
    draw_card(2.4, 2.4, 18.4, 5.2, "#FEF3C7", "#D97706",
              title="Ledoit--Wolf & Fisher-LDA",
              items=[
                  (r"Analytical Shrinkage: $\hat{\mathbf{\Sigma}}^{-1}$ (Resolves $N \ll d$)", "#0F172A", 6.7, False),
                  (r"Exact Projection: $(\mathbf{W}, \mathbf{b})$ in $<50$ ms (Zero Grads)", "#B45309", 6.9, True),
                  (r"Suppresses Background Variance ($>31\times$)", "#475569", 6.4, False)
              ], title_color="#92400E")

    # =========================================================================
    # 2. COLUMN 2 (CENTER): Cascaded Inference & Recalibration
    #    x = 26.6 to 72.6 (Width = 46.0)
    #    Gutter 1: x = 21.6 to 26.6 (Width = 5.0)
    # =========================================================================

    # --- Box 3: Tier-1 Edge Fast Path (Top-Center) ---
    # Container y: 31.0 to 44.4 (h = 13.4). Banner: Deep Emerald
    draw_container(x=26.6, y=31.0, w=46.0, h=13.4,
                   bg="#F0FDF4", border="#059669",
                   title="TIER-1 EDGE FAST PATH (REAL-TIME SYSTEM 1)",
                   banner_bg="#047857", banner_txt_color="#FFFFFF",
                   banner_w=28.0)

    # Card 3.1: DINOv2 Small (Center y = 36.7)
    draw_card(27.6, 32.2, 13.6, 9.0, "#FFFFFF", "#10B981",
              title="DINOv2 Small (INT8)",
              items=[
                  (r"$\mathbf{f}_{\mathrm{dino}} \in \mathbb{R}^{384}$ (Patch 14)", "#047857", 6.9, True),
                  ("Latency: 21.1 ms on CPU", "#334155", 6.7, False),
                  ("Captures Fine Lesion Edges", "#475569", 6.4, False)
              ], title_color="#047857")

    # Card 3.2: Fast Fisher-LDA Projection
    draw_card(42.4, 32.2, 14.4, 9.0, "#FFFFFF", "#059669",
              title="Fast Fisher Projection",
              items=[
                  (r"$\mathbf{p}_{\mathrm{fast}} = \mathrm{softmax}(\mathbf{W}_{\mathrm{dino}}\mathbf{f} + \mathbf{b})$", "#065F46", 6.8, True),
                  ("Precomputed 384-D Hyperplane", "#334155", 6.6, False),
                  ("FLOPs: <10K (<0.003 ms)", "#475569", 6.4, False)
              ], title_color="#065F46", title_fs=7.2)

    draw_arrow(41.2, 36.7, 42.4, 36.7, color="#059669", lw=1.4)

    # Card 3.3: Shannon Margin Gate C(x)
    draw_card(58.0, 32.2, 13.6, 9.0, "#FEFCE8", "#CA8A04",
              title=r"Shannon-Margin Gate $\mathcal{C}(\mathbf{x})$",
              items=[
                  (r"$\mathcal{C} = \Delta p \cdot (1 - \mathcal{H}_{\mathrm{norm}})_+$", "#A16207", 6.9, True),
                  (r"Fast-Exit Cutoff: $\tau_{\mathrm{conf}} = 0.35$", "#C2410C", 6.8, True),
                  ("Filters Spurious Attn Sinks", "#475569", 6.4, False)
              ], title_color="#A16207", title_fs=7.1)

    draw_arrow(56.8, 36.7, 58.0, 36.7, color="#CA8A04", lw=1.4)

    # Wire 1: Live Query straight horizontal arrow from Card 1.1 right edge (20.8) to DINOv2 left edge (27.6)
    draw_arrow(20.8, 36.7, 27.6, 36.7, color="#2563EB", lw=1.5,
               label=r"Live Query $\mathbf{x}_q$", label_pos=0.5,
               label_color="#1D4ED8", label_bg="#EFF6FF", label_fs=6.3)

    # --- Box 4: Tier-2 Deep Escalation (Middle-Center) ---
    # Container y: 16.0 to 27.5 (h = 11.5). Banner: Royal Violet
    draw_container(x=26.6, y=16.0, w=46.0, h=11.5,
                   bg="#FAF5FF", border="#7C3AED",
                   title="TIER-2 DEEP ESCALATION (MULTIMODAL SYSTEM 2)",
                   banner_bg="#6D28D9", banner_txt_color="#FFFFFF",
                   banner_w=28.0)

    # Card 4.1: BioCLIP ViT-B (INT8) (Center y = 20.0)
    draw_card(27.6, 16.8, 13.6, 7.5, "#FFFFFF", "#8B5CF6",
              title="BioCLIP ViT-B (INT8)",
              items=[
                  (r"$\mathbf{f}_{\mathrm{bio}} \in \mathbb{R}^{512}$ (ViT-B/16)", "#6D28D9", 6.9, True),
                  ("Taxonomy Priors (TreeOfLife-10M)", "#475569", 6.5, False)
              ], title_color="#6D28D9")

    # Card 4.2: 896-D Feature Fusion
    draw_card(42.4, 16.8, 14.4, 7.5, "#FFFFFF", "#7C3AED",
              title="896-D Feature Fusion",
              items=[
                  (r"$\mathbf{z}_q = [\mathbf{f}_{\mathrm{bio}} \,\|\, \mathbf{f}_{\mathrm{dino}}]$", "#5B21B6", 6.9, True),
                  (r"Reuses Cached $\mathbf{f}_{\mathrm{dino}}$ from Tier-1", "#334155", 6.5, False)
              ], title_color="#5B21B6")

    draw_arrow(41.2, 20.5, 42.4, 20.5, color="#7C3AED", lw=1.4)

    # Card 4.3: Fisher Projection & Multimodal Blend
    draw_card(58.0, 16.8, 13.6, 7.5, "#F0F9FF", "#0284C7",
              title="Fisher & Cached Text Blend",
              items=[
                  (r"$\mathbf{p}_{\mathrm{blend}} = 0.75 \mathbf{p}_{\mathrm{fish}} + 0.25 \mathbf{p}_{\mathrm{text}}$", "#0284C7", 6.6, True),
                  (r"$\mathbf{p}_{\mathrm{text}} = \mathrm{softmax}(\gamma \mathbf{f}_{\mathrm{bio}} \mathbf{P}_{\mathrm{text}}^T)$", "#0369A1", 6.4, True),
                  ("Deep Path Latency: 42.9 ms", "#0F172A", 6.5, False)
              ], title_color="#0369A1", title_fs=7.1)

    draw_arrow(56.8, 20.5, 58.0, 20.5, color="#0284C7", lw=1.4)

    # Wire 2: Ambiguous Fallback cleanly routes from Gate Card 3.3 to BioCLIP Card 4.1!
    # Path: from Card 3.3 bottom (64.8, 32.2) -> down to y=29.25 (in open horizontal corridor)
    #       -> left along y=29.25 to x=34.4 (to left of Tier-2 banner!) -> down to Card 4.1 top (34.4, 24.45)
    draw_orthogonal_arrow([(64.8, 32.05), (64.8, 29.25), (34.4, 29.25), (34.4, 24.45)],
                          color="#7C3AED", lw=1.5, ls="--",
                          label=r"Ambiguous (11.6%): $\mathcal{C}(\mathbf{x}) < 0.35$" + "\n" + r"$\to$ Triggers BioCLIP Escalation",
                          label_seg=1, label_pos=0.5,
                          label_color="#6D28D9", label_bg="#FAF5FF", label_fs=5.9)

    # Wire 3a: Cached Text Prototypes P_text from Card 2.1 (Offline Prompts) into Tier-2
    # Starts at Card 2.1 right edge (20.8, 25.15) -> gutter (24.1) -> into Tier-2 left edge (27.6, 22.5)
    draw_orthogonal_arrow([(20.8, 25.15), (24.1, 25.15), (24.1, 22.2), (27.6, 22.2)],
                          color="#0284C7", lw=1.4, ls=":",
                          label=r"Cached $\mathbf{P}_{\mathrm{text}}^{(c)}$ (Offline)",
                          label_seg=1, label_pos=0.5,
                          label_color="#0369A1", label_bg="#F0F9FF", label_fs=5.9)

    # Wire 3b: Fisher Weights (W, b) from Calibration Card 2.4 to Tier-2
    # Starts at Card 2.4 right edge (20.8, 5.0) -> gutter (24.1) -> into Tier-2 left edge (27.6, 18.0)
    draw_orthogonal_arrow([(20.8, 5.0), (24.1, 5.0), (24.1, 18.0), (27.6, 18.0)],
                          color="#EA580C", lw=1.4, ls=":",
                          label=r"Fisher $(\mathbf{W}, \mathbf{b})$" + "\n" + r"(Zero Grads)",
                          label_seg=1, label_pos=0.5,
                          label_color="#C2410C", label_bg="#FFF7ED", label_fs=5.9)

    # --- Box 5: Soft-Anchored Recalibration Engine (Bottom-Center) ---
    # Container y: 1.6 to 12.5 (h = 10.9). Banner: Deep Ocean Cyan
    draw_container(x=26.6, y=1.6, w=46.0, h=10.9,
                   bg="#F0F9FF", border="#0284C7",
                   title="ONLINE LOOP: SOFT-ANCHORED RECALIBRATION",
                   banner_bg="#0369A1", banner_txt_color="#FFFFFF",
                   banner_w=28.0)

    # Card 5.1: On-Device FIFO Buffer (Center y = 5.2)
    draw_card(27.6, 2.2, 13.6, 6.8, "#E0F2FE", "#0284C7",
              title="On-Device FIFO Buffer",
              items=[
                  (r"Anchor Gate: $p_{\mathrm{max}} \geq 0.70$", "#0369A1", 6.8, True),
                  ("Top-20 Confident Pseudo-Anchors", "#334155", 6.5, False)
              ], title_color="#0369A1", title_fs=7.1)

    # Card 5.2: Soft-Anchoring (3x Weight)
    draw_card(42.4, 2.2, 14.4, 6.8, "#FEFCE8", "#D97706",
              title=r"Soft-Anchoring ($3\times$ Penalty)",
              items=[
                  (r"$\tilde{\mathcal{S}}_c^{\mathrm{trans}} = (\tilde{\mathcal{S}}_c \otimes 3) \cup \mathcal{Q}_{\mathrm{anc}}$", "#A16207", 6.7, True),
                  ("Safeguards Ground-Truth Centroids", "#475569", 6.5, False)
              ], title_color="#A16207")

    draw_arrow(41.2, 5.6, 42.4, 5.6, color="#D97706", lw=1.4)

    # Card 5.3: Closed-Form Update
    draw_card(58.0, 2.2, 13.6, 6.8, "#FFFFFF", "#10B981",
              title="Closed-Form Update (<22 ms)",
              items=[
                  (r"Re-estimates $\mathbf{W}^{\mathrm{trans}}$ & $\hat{\mathbf{\Sigma}}^{\mathrm{trans}}$", "#047857", 6.8, True),
                  ("Zero Gradient Retraining on CPU", "#334155", 6.5, False)
              ], title_color="#047857")

    draw_arrow(56.8, 5.6, 58.0, 5.6, color="#10B981", lw=1.4)

    # Wire 4: High-Confidence Anchors routed from Tier-2 Decision (Card 4.3) into FIFO Buffer (Card 5.1)
    # Starts at Card 4.3 bottom-left (59.2, 16.65) -> down to y=14.6 -> left along corridor to x=34.4 -> down to Card 5.1 top (34.4, 9.15)
    draw_orthogonal_arrow([(59.2, 16.65), (59.2, 14.6), (34.4, 14.6), (34.4, 9.15)],
                          color="#0284C7", lw=1.4, ls="--",
                          label="Anchor Filter\n" + r"$p_{\max} \geq 0.70$",
                          label_seg=1, label_pos=0.5,
                          label_color="#0369A1", label_bg="#F0F9FF", label_fs=5.9)

    # Wire 5: Updates Tier-2 Boundaries straight UP from Closed-Form Update into Fisher Projection
    # Runs at x=66.8 with zero collision with Wire 4 (x=59.2) and clearly separated vertically from Wire 4's line
    # Starts at Card 5.3 top (66.8, 9.15), ends at Card 4.3 bottom (66.8, 16.65).
    draw_arrow(66.8, 9.15, 66.8, 16.65, color="#059669", lw=1.5, ls=":",
               label=r"Updates Tier-2" + "\n" + r"Boundaries $\mathbf{W}^{\mathrm{trans}}$", label_pos=0.38,
               label_color="#047857", label_bg="#ECFDF5", label_fs=5.9)

    # =========================================================================
    # 3. COLUMN 3 (RIGHT): Field Diagnostic Output
    #    x = 78.4 to 98.4 (Width = 20.0, Height = 42.8)
    #    Gutter 2 on left: x = 72.6 to 78.4 (Width = 5.8)
    # =========================================================================
    draw_container(x=78.4, y=1.6, w=20.0, h=42.8,
                   bg="#ECFDF5", border="#059669",
                   title="FIELD DIAGNOSTIC OUTPUT",
                   banner_bg="#065F46", banner_txt_color="#FFFFFF", lw=1.5)

    # Card 6.1 (Top): Immediate Live Classification (Center y = 36.7)
    draw_card(79.4, 32.2, 18.0, 9.0, "#FFFFFF", "#059669",
              title="Immediate Live Diagnosis",
              items=[
                  ("Fast-Exit (Tier-1): 21.1 ms (>45 FPS)", "#047857", 6.8, True),
                  ("Deep-Path (Tier-2): 42.9 ms (23 FPS)", "#0284C7", 6.8, True),
                  ("Real-Time Foliar Pathology Output", "#334155", 6.6, False)
              ], title_color="#047857")

    # Wire 6: Fast-Exit straight horizontal arrow from Routing Gate right edge (71.6) to Live Classification left edge (79.4)
    draw_arrow(71.6, 36.7, 79.4, 36.7, color="#059669", lw=1.6,
               label="Fast-Exit (88.4%)\n" + r"$\mathcal{C}(\mathbf{x}) \geq 0.35$",
               label_pos=0.5, label_color="#047857", label_bg="#ECFDF5", label_fs=6.0)

    # Wire 7: Deep Consensus cleanly routed orthogonally (90-degree) into Immediate Classification
    # Starts at Card 4.3 right edge (71.6, 20.5), enters Card 6.1 left edge at (79.4, 34.0)
    draw_orthogonal_arrow([(71.6, 20.5), (74.8, 20.5), (74.8, 34.0), (79.4, 34.0)],
                          color="#0284C7", lw=1.5,
                          label="Deep Consensus (11.6%)\n" + r"$0.75\,\mathbf{p}_{\mathrm{fish}} + 0.25\,\mathbf{p}_{\mathrm{text}}$",
                          label_seg=1, label_pos=0.5,
                          label_color="#0369A1", label_bg="#F0F9FF", label_fs=5.9)

    # Card 6.2 (Middle): Continuous Session Adaptation
    draw_card(79.4, 18.0, 18.0, 11.2, "#FFFFFF", "#0284C7",
              title="Session Adaptation Log",
              items=[
                  ("Dynamic Hyperplane Recalibration", "#0369A1", 6.8, True),
                  (r"Applies $\mathbf{W}^{\mathrm{trans}}$ to Next Video Frames", "#334155", 6.6, False),
                  ("Autonomous Zero-Overhead Tracking", "#047857", 6.5, True)
              ], title_color="#0369A1")

    # Wire 8: Updated Hyperplanes from Recalibration cleanly routed orthogonally (90-degree) into Session Log
    draw_orthogonal_arrow([(71.6, 5.6), (77.0, 5.6), (77.0, 23.6), (79.4, 23.6)],
                          color="#059669", lw=1.5,
                          label="Updated " + r"$\mathbf{W}^{\mathrm{trans}}$" + "\n(<22 ms)",
                          label_seg=1, label_pos=0.5,
                          label_color="#047857", label_bg="#ECFDF5", label_fs=6.0)

    # Card 6.3 (Bottom): Validated Hardware Benchmark Specs
    draw_card(79.4, 2.2, 18.0, 14.0, "#064E3B", "#047857",
              title="Validated Benchmark Specs",
              items=[
                  ("Peak Accuracy: 98.72% | Macro F1: 98.41%", "#A7F3D0", 6.8, True),
                  ("PlantVillage Mean: 97.37% (30-Episodes)", "#FFFFFF", 6.8, True),
                  ("PlantDoc In-the-Wild (5-Way): 83.61%", "#A7F3D0", 6.8, True),
                  ("Calibration Reliability: ECE = 0.81%", "#E2E8F0", 6.6, False),
                  (r"Working Set Budget: $\leq$ 110 MB (APK: 22.7 MB)", "#6EE7B7", 6.8, True),
                  ("100% Offline Edge CPU Execution (>45 FPS)", "#A7F3D0", 6.7, True)
              ], title_color="#A7F3D0", title_fs=7.5)

    # Save vector PDF and high-res raster PNG
    pdf_path = PAPER_DIR / "fig_architecture.pdf"
    png_path = PAPER_DIR / "fig_architecture.png"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    plt.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()

    if WEB_FIG_DIR.exists():
        shutil.copy2(png_path, WEB_FIG_DIR / "fig_architecture.png")

    print(f"[SUCCESS] Saved publication-grade {pdf_path} and {png_path}!")


if __name__ == "__main__":
    create_architecture_diagram()
