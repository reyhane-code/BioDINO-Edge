# -*- coding: utf-8 -*-
"""
Generate the post-hoc explainability figure (frozen DINOv2 CLS attention vs.
Fisher decision attribution of the deployed Tier-2 LDA) and Expected
Calibration Error (ECE) reliability diagrams for BioDINO-Edge.
"""

import os, sys, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torchvision import transforms as T

_curr = Path(__file__).resolve().parent if "__file__" in locals() else Path(".").resolve()
PROJECT_ROOT = _curr if (_curr / "paper").exists() else _curr.parent
PAPER_DIR = PROJECT_ROOT / "paper"
DOCS_ASSETS = PROJECT_ROOT / "docs" / "assets"
CACHE_DIR = PROJECT_ROOT / "cache_artifacts"
PAPER_DIR.mkdir(exist_ok=True)
DOCS_ASSETS.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Running on device: {DEVICE}")

# -----------------------------------------------------------------------------
# 1. Visual Saliency Maps via DINOv2 Self-Attention
# -----------------------------------------------------------------------------
def generate_saliency_figures():
    """Frozen-encoder attention vs. post-hoc Fisher decision attribution.

    The encoders are never modified by the LDA. Row (b) shows the raw DINOv2
    last-block CLS attention; row (c) shows a post-hoc linear attribution of
    the deployed Tier-2 decision: L2-normalized last-block patch tokens of both
    encoders projected onto the target-vs-healthy contrast direction of the
    canonical 896-D Fisher-LDA (8-view augmented support, seed-42 split),
    averaged after bicubic upsampling, ReLU'd and min-max normalized per image.
    BioCLIP tokens pass through ln_post and the visual projection so that the
    attribution lives in the deployed 512-D feature space.
    Badges report the measured Shannon gate confidence and resulting routing.
    """
    import json
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.core_math import (
        fit_fisher_lda, compute_class_prototypes,
        compute_fisher_ensemble_probs, calculate_shannon_confidence,
    )
    from pipeline.data_engine import load_plantvillage
    from pipeline.biodino_edge_pipeline import load_backbones, extract_support_features

    print("\n[*] Loading canonical support split and deployed Fisher-LDA...")
    cdata = torch.load(CACHE_DIR / "pv_novel_embeddings.pt", map_location="cpu", weights_only=True)
    class_names = cdata["novel_classes"]
    text_protos = cdata["bio_text_protos"]
    n_classes = len(class_names)

    models = load_backbones(DEVICE)
    support_samples, _, _, _ = load_plantvillage(PROJECT_ROOT / "plantvillage", k_shot=5, seed=42)
    s_bio, s_dino, s_y_t = extract_support_features(support_samples, models, device=DEVICE)
    s_bio, s_dino, s_y_t = s_bio.cpu(), s_dino.cpu(), s_y_t.cpu()
    s_x = torch.cat([s_bio, s_dino], dim=-1).numpy()
    W, b, clf = fit_fisher_lda(s_x, s_y_t.numpy(), device="cpu")
    dino_protos = compute_class_prototypes(s_dino, s_y_t, n_classes)

    # DINOv2 last-block attention capture (frozen encoder; LDA never touches it)
    last_block = models.dino_model.blocks[-1]
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

    # BioCLIP last-block patch tokens (pre ln_post; L2-normalized below)
    captured_bio = []

    def bio_hook(module, args, output):
        captured_bio.append(output.detach().cpu())

    models.bioclip_model.visual.transformer.resblocks[-1].register_forward_hook(bio_hook)

    short_names = ["Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
                   "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
                   "Potato Early Blight", "Potato Late Blight", "Potato Healthy"]

    pv_color = PROJECT_ROOT / "plantvillage" / "plantvillage dataset" / "color"
    pd_dir = PROJECT_ROOT / "pipeline" / "plantdoc_samples"

    # 4 cases: 2 PlantVillage (lab) + 2 PlantDoc (field, cross-domain queries
    # attributed against the corresponding PlantVillage class prototypes).
    cases = [
        {"img": sorted(list((pv_color / "Apple___Cedar_apple_rust").glob("*.JPG")) + list((pv_color / "Apple___Cedar_apple_rust").glob("*.jpg")))[8],
         "title": "Apple Cedar Rust", "intended": 2, "healthy": 3},
        {"img": sorted(list((pv_color / "Apple___Black_rot").glob("*.JPG")) + list((pv_color / "Apple___Black_rot").glob("*.jpg")))[18],
         "title": "Apple Black Rot", "intended": 1, "healthy": 3},
        {"img": pd_dir / "apple_rust_leaf_10.jpg",
         "title": "Apple Cedar Rust", "intended": 2, "healthy": 3},
        {"img": pd_dir / "grape_leaf_black_rot_4.jpg",
         "title": "Grape Black Rot", "intended": 4, "healthy": 7},
    ]

    maps, diagnostics = [], []
    for c in cases:
        pil_img = Image.open(c["img"]).convert("RGB")
        rgb_np = np.array(pil_img.resize((224, 224))) / 255.0
        dino_x = models.dino_transforms(pil_img).unsqueeze(0).to(DEVICE)
        bio_x = models.bioclip_transforms(pil_img).unsqueeze(0).to(DEVICE)

        captured_attn.clear()
        captured_bio.clear()
        with torch.no_grad():
            feats = models.dino_model.forward_features(dino_x)
            f_dino = F.normalize(models.dino_model.forward_head(feats), dim=-1).cpu()
            f_bio = F.normalize(models.bioclip_model.encode_image(bio_x), dim=-1).cpu()

        # Tier-1 gate exactly as deployed: gamma=20 cosine prototypes + Shannon margin-entropy
        prob1 = F.softmax(20.0 * (f_dino @ dino_protos.t()), dim=-1)
        conf = float(calculate_shannon_confidence(prob1, n_classes).item())
        tier = 1 if conf >= 0.35 else 2

        # Deployed Tier-2 ensemble decision (0.75 LDA + 0.25 text, scale=20)
        joint_q = torch.cat([f_bio, f_dino], dim=-1)
        prob_ens, _, _ = compute_fisher_ensemble_probs(
            fisher_weights=W, fisher_biases=b, joint_features=joint_q,
            bioclip_features=f_bio, text_prototypes=text_protos,
            scale=20.0, w_lda=0.75, device="cpu")
        pred = int(prob_ens.argmax(dim=-1).item())

        # (b) Frozen DINOv2 CLS attention (head-averaged, last block)
        attn = captured_attn[0]
        cls_attn = attn[0, :, 0, 1:].mean(dim=0).reshape(16, 16).numpy()
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        attn_up = F.interpolate(torch.tensor(cls_attn)[None, None], size=(224, 224),
                                mode="bicubic", align_corners=False)[0, 0].numpy()
        attn_up = np.clip(attn_up, 0.0, 1.0)

        # (c) Post-hoc Fisher decision attribution: contrast direction of the
        # joint LDA split per encoder, projected onto L2-normalized patch tokens.
        w_contrast = clf.coef_[pred] - clf.coef_[c["healthy"]]
        w_bio, w_dino = w_contrast[:512], w_contrast[512:]
        pd_tok = F.normalize(feats[0, 1:, :], dim=-1).cpu().numpy()
        m_dino = (pd_tok @ w_dino).reshape(16, 16)
        bt = captured_bio[0]
        if bt.dim() == 3 and bt.shape[0] == 1:
            bt = bt[0]
        elif bt.dim() == 3 and bt.shape[1] == 1:
            bt = bt[:, 0]
        pb_tok = bt[1:, :]
        with torch.no_grad():
            pb_tok = models.bioclip_model.visual.ln_post(pb_tok)
            if models.bioclip_model.visual.proj is not None:
                pb_tok = pb_tok @ models.bioclip_model.visual.proj
        pb_tok = F.normalize(pb_tok, dim=-1).numpy()
        m_bio = (pb_tok @ w_bio).reshape(14, 14)
        m_dino_up = F.interpolate(torch.tensor(m_dino)[None, None].float(), size=(224, 224),
                                  mode="bicubic", align_corners=False)[0, 0].numpy()
        m_bio_up = F.interpolate(torch.tensor(m_bio)[None, None].float(), size=(224, 224),
                                 mode="bicubic", align_corners=False)[0, 0].numpy()
        fisher = np.maximum(0.5 * (m_dino_up + m_bio_up), 0.0)
        fisher = (fisher - fisher.min()) / (fisher.max() - fisher.min() + 1e-8)

        maps.append({"rgb": rgb_np, "attn": attn_up, "fisher": fisher,
                     "conf": conf, "tier": tier, "pred": pred})
        diagnostics.append({
            "image": str(c["img"]), "title": c["title"],
            "gate_confidence": round(conf, 4), "routed_tier": tier,
            "lda_predicted_class": short_names[pred],
            "intended_class": short_names[c["intended"]],
            "contrast": f"{short_names[pred]} vs {short_names[c['healthy']]}",
        })
        print(f"  [{c['title']}] C(x)={conf:.3f} -> Tier-{tier} | LDA: {short_names[pred]}")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig = plt.figure(figsize=(10.4, 7.9), dpi=300)
    gs = gridspec.GridSpec(nrows=3, ncols=4, figure=fig,
                           left=0.125, right=0.985, top=0.865, bottom=0.155,
                           wspace=0.035, hspace=0.055)
    axes = np.empty((3, 4), dtype=object)
    for r in range(3):
        for col in range(4):
            axes[r, col] = fig.add_subplot(gs[r, col])
    fig.canvas.draw()

    # Dataset regime labels (source dataset, not routing; routing is per-column)
    center_lab = (axes[0, 0].get_position().x0 + axes[0, 1].get_position().x1) / 2.0
    center_field = (axes[0, 2].get_position().x0 + axes[0, 3].get_position().x1) / 2.0
    fig.text(center_lab, 0.955, "Laboratory Benchmark (PlantVillage)", ha="center", va="center",
             fontsize=9.0, fontweight="bold", color="#1b5e20")
    fig.text(center_field, 0.955, "In-the-Wild Field Queries (PlantDoc)", ha="center", va="center",
             fontsize=9.0, fontweight="bold", color="#0d47a1")
    div_x = (axes[0, 1].get_position().x1 + axes[0, 2].get_position().x0) / 2.0
    fig.add_artist(Line2D([div_x, div_x], [axes[2, 0].get_position().y0 - 0.075, 0.925],
                          transform=fig.transFigure, color="#777777", linestyle="--",
                          linewidth=1.2, alpha=0.8))

    row_labels = [
        "(a) Input Query Leaf\n[RGB Observation]",
        "(b) Frozen DINOv2\nCLS Attention",
        "(c) Fisher Decision\nAttribution (Post-hoc)",
    ]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=9.0, fontweight="bold",
                                    labelpad=10, rotation=90, va="center", linespacing=1.18)

    im_heat = None
    for col_idx, (c, m) in enumerate(zip(cases, maps)):
        axes[0, col_idx].set_title(c["title"], fontsize=9.4, fontweight="bold", pad=6)
        axes[0, col_idx].imshow(m["rgb"])
        im_heat = axes[1, col_idx].imshow(m["attn"], cmap="turbo", vmin=0, vmax=1)
        heat = plt.get_cmap("turbo")(m["fisher"])[:, :, :3]
        axes[2, col_idx].imshow(0.52 * m["rgb"] + 0.48 * heat)
        for r in range(3):
            axes[r, col_idx].set_xticks([])
            axes[r, col_idx].set_yticks([])

        badge = rf"$\mathcal{{C}}(\mathbf{{x}})$ = {m['conf']:.2f} $\to$ Tier-{m['tier']}"
        if m["pred"] != c["intended"]:
            badge += "\n" + f"LDA: {short_names[m['pred']]}"
        axes[2, col_idx].text(0.5, -0.13, badge, transform=axes[2, col_idx].transAxes,
                              ha="center", va="top", fontsize=8.0, fontweight="bold",
                              color="#1b5e20" if m["tier"] == 1 else "#0d47a1", linespacing=1.24,
                              bbox=dict(boxstyle="round,pad=0.32",
                                        facecolor="#e8f5e9" if m["tier"] == 1 else "#e3f2fd",
                                        edgecolor="#388e3c" if m["tier"] == 1 else "#1976d2",
                                        linewidth=1.0))

    for r in range(3):
        for col in range(4):
            for spine in axes[r, col].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#333333")
                spine.set_linewidth(0.85)

    cbar_ax = fig.add_axes([0.31, 0.025, 0.38, 0.015])
    cb = fig.colorbar(im_heat, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Normalized Map Intensity [0, 1] (per-map min\u2013max)", fontsize=8.4, fontweight="bold", labelpad=3)
    cb.ax.tick_params(labelsize=7.5)

    pdf_path = PAPER_DIR / "fig_saliency.pdf"
    png_path = PAPER_DIR / "fig_saliency.png"
    docs_png = DOCS_ASSETS / "fig_saliency.png"
    web_png = PROJECT_ROOT / "web" / "public" / "figures" / "fig_saliency.png"

    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(docs_png, bbox_inches="tight", dpi=300)
    if web_png.parent.exists():
        plt.savefig(web_png, bbox_inches="tight", dpi=300)
    plt.close()

    diag_path = PAPER_DIR / "fig_saliency_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump({"protocol": "canonical seed-42 5-shot split, 8-view augmented support (N=440), "
                               "joint 896-D Fisher-LDA (lsqr, auto shrinkage), gamma=20, w_lda=0.75",
                   "cases": diagnostics}, f, indent=2, ensure_ascii=False)

    print(f"  [+] Attribution figures saved:")
    print(f"      - {pdf_path}")
    print(f"      - {png_path}")
    print(f"      - {docs_png}")
    print(f"      - {diag_path}")

# -----------------------------------------------------------------------------
# 2. Expected Calibration Error (ECE) & Reliability Diagrams
# -----------------------------------------------------------------------------
def compute_ece(probs, targets, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    probs: (N, C) numpy array of predicted posteriors
    targets: (N,) numpy array of ground truth integer labels
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_data = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        
        if np.sum(in_bin) > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(acc_in_bin - conf_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)
            bin_data.append({
                "bin_idx": i,
                "bin_lower": float(bin_boundaries[i]),
                "bin_upper": float(bin_boundaries[i + 1]),
                "count": int(np.sum(in_bin)),
                "accuracy": float(acc_in_bin),
                "confidence": float(conf_in_bin),
                "gap": float(abs_diff)
            })
        else:
            bin_data.append({
                "bin_idx": i,
                "bin_lower": float(bin_boundaries[i]),
                "bin_upper": float(bin_boundaries[i + 1]),
                "count": 0,
                "accuracy": 0.0,
                "confidence": (bin_boundaries[i] + bin_boundaries[i + 1]) / 2.0,
                "gap": 0.0
            })

    return ece * 100.0, mce * 100.0, bin_data

def generate_calibration_figures():
    print("\n[*] Loading Embedding Cache to compute exact ECE on 9,330 Test Queries...")
    cache_path = PROJECT_ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
    cdata = torch.load(cache_path, map_location="cpu", weights_only=False)

    c_bio = cdata["bio_feats"]
    c_dino = cdata["dino_feats"]
    c_labels = cdata["labels"].numpy()
    c_concat = torch.cat([c_bio, c_dino], dim=-1).numpy()
    c_text = cdata["bio_text_protos"]
    class_names = cdata["novel_classes"]
    n_classes = len(class_names)

    # Replicate standard 5-shot seed=42 split
    rng = random.Random(42)
    c_indices = [np.where(c_labels == c)[0].tolist() for c in range(n_classes)]
    s_idx, q_idx = [], []
    for c in range(n_classes):
        shuf = c_indices[c].copy()
        rng.shuffle(shuf)
        s_idx.extend(shuf[:5])
        q_idx.extend(shuf[5:])

    q_labels = c_labels[q_idx]

    # 1. Baseline: Uncalibrated Tip-Adapter / Cosine Protos
    proto_dino = np.stack([c_dino[s_idx][c_labels[s_idx] == c].mean(dim=0).numpy() for c in range(n_classes)])
    proto_dino = proto_dino / np.linalg.norm(proto_dino, axis=-1, keepdims=True)
    sims = np.dot(c_dino[q_idx].numpy(), proto_dino.T) * 15.0
    exp_sims = np.exp(sims - np.max(sims, axis=1, keepdims=True))
    probs_uncalibrated = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)

    # 2. BioDINO-Edge Initial Consensus
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf_init = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_init.fit(c_concat[s_idx], c_labels[s_idx])

    W_init = torch.tensor(clf_init.coef_, dtype=torch.float32)
    b_init = torch.tensor(clf_init.intercept_, dtype=torch.float32)
    q_concat_t = torch.tensor(c_concat[q_idx], dtype=torch.float32)
    lgt_lda = q_concat_t @ W_init.t() + b_init
    p_lda = F.softmax(lgt_lda, dim=-1)
    p_txt = F.softmax(20.0 * (c_bio[q_idx] @ c_text.t()), dim=-1)
    p_ens = 0.75 * p_lda + 0.25 * p_txt

    preds_ep = p_ens.argmax(dim=-1).numpy()
    top_confs, _ = p_ens.max(dim=-1)

    # 3. Soft-Anchored Transductive Refinement (3x support anchor weighting & 0.70 confidence threshold)
    ref_x = [c_concat[s_idx]] * 3
    ref_y = [c_labels[s_idx]] * 3
    for c in range(n_classes):
        c_cand = (preds_ep == c)
        if c_cand.sum() > 0:
            c_cand_idx = np.where(c_cand)[0]
            c_conf_vals = top_confs[c_cand_idx]
            high_conf = c_conf_vals > 0.70
            if high_conf.sum() > 0:
                c_cand_idx = c_cand_idx[high_conf.numpy()]
                c_conf_vals = c_conf_vals[high_conf]
            top_k = min(20, len(c_cand_idx))
            if top_k > 0:
                best_local_idx = c_cand_idx[torch.topk(c_conf_vals, top_k).indices.numpy()]
                best_q_idx = [q_idx[i] for i in best_local_idx]
                ref_x.append(c_concat[best_q_idx])
                ref_y.append(np.full(top_k, c))

    ref_x = np.concatenate(ref_x, axis=0)
    ref_y = np.concatenate(ref_y, axis=0)

    clf_trans = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_trans.fit(ref_x, ref_y)

    W_trans = torch.tensor(clf_trans.coef_, dtype=torch.float32)
    b_trans = torch.tensor(clf_trans.intercept_, dtype=torch.float32)
    lgt_trans = q_concat_t @ W_trans.t() + b_trans
    p_trans = 0.75 * F.softmax(lgt_trans, dim=-1) + 0.25 * p_txt
    # Apply post-hoc temperature scaling (Guo et al., 2017) with T = 0.50
    log_p = torch.log(p_trans + 1e-12)
    p_cal = F.softmax(log_p / 0.50, dim=-1)
    probs_calibrated = p_cal.numpy()

    # Compute ECE
    ece_uncal, mce_uncal, bins_uncal = compute_ece(probs_uncalibrated, q_labels, n_bins=10)
    ece_cal, mce_cal, bins_cal = compute_ece(probs_calibrated, q_labels, n_bins=10)

    print(f"  • Uncalibrated Baseline ECE : {ece_uncal:.2f}% | MCE: {mce_uncal:.2f}%")
    print(f"  • BioDINO-Edge Calibrated ECE: {ece_cal:.2f}% | MCE: {mce_cal:.2f}%")

    # Plot Reliability Diagram (2 panels: Uncalibrated vs Calibrated)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Left: Baseline
    accs_uncal = [b["accuracy"] for b in bins_uncal]
    confs_uncal = [b["confidence"] for b in bins_uncal]
    bin_centers = np.linspace(0.05, 0.95, 10)

    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    axes[0].bar(bin_centers, accs_uncal, width=0.08, color="#e74c3c", alpha=0.65, edgecolor="#c0392b", label="Outputs (Acc)")
    axes[0].bar(bin_centers, [abs(a - c) for a, c in zip(accs_uncal, confs_uncal)], bottom=np.minimum(accs_uncal, confs_uncal),
                width=0.08, color="#95a5a6", alpha=0.45, hatch="//", edgecolor="#7f8c8d", label="Calibration Gap")
    axes[0].set_title(f"(a) Uncalibrated Generic Baseline\nECE = {ece_uncal:.2f}% | MCE = {mce_uncal:.2f}%", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Confidence", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Empirical Accuracy", fontsize=10, fontweight="bold")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left", fontsize=9)

    # Right: BioDINO-Edge Calibrated
    accs_cal = [b["accuracy"] for b in bins_cal]
    confs_cal = [b["confidence"] for b in bins_cal]

    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    axes[1].bar(bin_centers, accs_cal, width=0.08, color="#2ecc71", alpha=0.70, edgecolor="#27ae60", label="BioDINO-Edge (Acc)")
    axes[1].bar(bin_centers, [abs(a - c) for a, c in zip(accs_cal, confs_cal)], bottom=np.minimum(accs_cal, confs_cal),
                width=0.08, color="#95a5a6", alpha=0.45, hatch="//", edgecolor="#7f8c8d", label="Calibration Gap")
    axes[1].set_title(f"(b) BioDINO-Edge (Calibrated & Gated)\nECE = {ece_cal:.2f}% | MCE = {mce_cal:.2f}%", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Confidence", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Empirical Accuracy", fontsize=10, fontweight="bold")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    pdf_path = PAPER_DIR / "fig_calibration.pdf"
    png_path = PAPER_DIR / "fig_calibration.png"
    docs_png = DOCS_ASSETS / "fig_calibration.png"

    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(docs_png, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"  [+] Calibration figures saved successfully:")
    print(f"      - {pdf_path}")
    print(f"      - {png_path}")
    print(f"      - {docs_png}")

    # Return numbers to update JSON
    return {
        "uncalibrated_baseline": {
            "ece_pct": float(round(ece_uncal, 2)),
            "mce_pct": float(round(mce_uncal, 2))
        },
        "biodino_edge_calibrated": {
            "ece_pct": float(round(ece_cal, 2)),
            "mce_pct": float(round(mce_cal, 2)),
            "bin_data": bins_cal
        }
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliency-only", action="store_true",
                        help="Regenerate only the attribution figure (skip the legacy ECE panel).")
    args = parser.parse_args()
    generate_saliency_figures()
    if args.saliency_only:
        sys.exit(0)
    calib_stats = generate_calibration_figures()

    # Update biodino_edge_results.json
    res_path = PROJECT_ROOT / "biodino_edge_results.json"
    if res_path.exists():
        import json
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "plantvillage" in data:
            data["plantvillage"]["calibration"] = calib_stats
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[*] Updated {res_path} with ECE calibration metrics.")
