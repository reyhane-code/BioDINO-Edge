# -*- coding: utf-8 -*-
"""Regenerate Figure 2: row-normalized PlantVillage confusion matrix.

Reads the canonical single-run confusion matrix produced by
experiments/transductive_validation/compute_canonical_single_run.py and renders
it as a publication-quality vector figure for the EAAI manuscript.

Scientific constraints
----------------------
- The counts, the 11 diagnostic classes, and their ordering are taken verbatim
  from canonical_single_run.json; nothing is regenerated or approximated.
- Cells are row-normalized so the diagonal equals per-class recall (%).
- The color scale is 0-100 %, matching the in-cell percentage annotations.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "experiments" / "transductive_validation" / "canonical_single_run.json"
PAPER_DIR = PROJECT_ROOT / "paper"

# Class ordering of the canonical PlantVillage split (must not be reordered).
SHORT_NAMES = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
]

EXPECTED_TOTAL = 9330
EXPECTED_CROSS_SPECIES = 2  # 0.021 % of 9,330 queries
ANNOTATION_THRESHOLD = 0.0  # annotate every nonzero cell, however small


def load_matrix():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    cm = np.asarray(data["single_run"]["confusion_matrix"], dtype=float)
    return data, cm


def validate(data, cm):
    """Fail loudly if the source matrix no longer matches the reported numbers."""
    assert cm.shape == (len(SHORT_NAMES), len(SHORT_NAMES)), f"unexpected shape {cm.shape}"
    assert int(cm.sum()) == EXPECTED_TOTAL, f"total queries {int(cm.sum())} != {EXPECTED_TOTAL}"

    recall = 100.0 * np.diag(cm) / cm.sum(axis=1)
    reported = np.array([c["recall"] for c in data["single_run"]["per_class"]]) * 100.0
    assert np.allclose(recall, reported, atol=1e-6), "diagonal != reported per-class recall"

    host = [entry["name"].split("___")[0] for entry in data["single_run"]["per_class"]]
    cross = 0
    for i in range(len(SHORT_NAMES)):
        for j in range(len(SHORT_NAMES)):
            if host[i] != host[j]:
                cross += int(cm[i, j])
    assert cross == EXPECTED_CROSS_SPECIES, f"cross-species count {cross} != {EXPECTED_CROSS_SPECIES}"

    return recall


def build_figure(pct):
    n = pct.shape[0]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "axes.linewidth": 0.6,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.15, 5.25), constrained_layout=True)

    # pcolormesh keeps the cell field as true vector graphics in the PDF.
    mesh = ax.pcolormesh(
        pct, cmap="Blues", vmin=0.0, vmax=100.0,
        edgecolors="white", linewidth=0.6, shading="flat",
    )

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ticks = np.arange(n) + 0.5
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(SHORT_NAMES)
    ax.tick_params(axis="x", which="both", length=0, pad=4)
    ax.tick_params(axis="y", which="both", length=0, pad=5)

    ax.set_xlabel("Predicted class", labelpad=6)
    ax.set_ylabel("True class", labelpad=6)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#444444")

    for i in range(n):
        for j in range(n):
            value = pct[i, j]
            if value <= ANNOTATION_THRESHOLD:
                continue
            ax.text(
                j + 0.5, i + 0.5, f"{value:.1f}",
                ha="center", va="center", fontsize=7.8,
                color="white" if value > 55.0 else "#1a1a1a",
            )

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.025, aspect=28)
    cbar.set_label("Row-normalized recall (%)", fontsize=10.0, labelpad=5)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    cbar.ax.tick_params(labelsize=8.8, length=3, width=0.6)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("#444444")

    return fig


def main():
    data, cm = load_matrix()
    recall = validate(data, cm)
    pct = 100.0 * cm / cm.sum(axis=1, keepdims=True)

    fig = build_figure(pct)
    pdf_path = PAPER_DIR / "fig_confusion.pdf"
    png_path = PAPER_DIR / "fig_confusion.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)

    print(f"[ok] matrix {cm.shape}, {int(cm.sum())} queries, diagonal recall "
          f"{recall.min():.2f}-{recall.max():.2f}%")
    print(f"[ok] cross-species confusion: {EXPECTED_CROSS_SPECIES} "
          f"({100.0 * EXPECTED_CROSS_SPECIES / cm.sum():.3f}%)")
    print(f"[ok] wrote {pdf_path}")
    print(f"[ok] wrote {png_path}")


if __name__ == "__main__":
    main()
