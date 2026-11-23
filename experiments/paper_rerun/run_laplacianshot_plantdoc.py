# -*- coding: utf-8 -*-
"""LaplacianShot on PlantDoc: the dataset the paper actually deploys on.

Why this file exists. The 11-class PlantVillage comparison is a controlled
laboratory benchmark. The paper's claim is about field deployment, and the field
benchmark is PlantDoc. A reviewer will ask why the strongest transductive
baseline appears only on the laboratory dataset. This closes that gap.

Two regimes, mirroring Table 5 of the paper:
  27-way  all 27 classes, 5 support leaves/class from the training split,
          252 field queries (single run, the paper's stress-test protocol)
  5-way   50 Monte Carlo episodes over the test split

Both methods see the identical support set and the identical 896-D features.

Output: experiments/paper_rerun/results/laplacianshot_plantdoc_2026-11-18.json
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "pipeline")):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quantization_accuracy"))
from run_laplacianshot_faithful import fisher_anchor, laplacianshot  # noqa: E402
from run_quantization_accuracy import (  # noqa: E402
    N_VIEWS, aug_pipeline, build_sessions, build_transforms, norm_stats,
    run_onnx_companion, run_onnx_tier1,
)
from core_math import compute_class_prototypes  # noqa: E402
from data_engine import load_plantdoc  # noqa: E402
import open_clip  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
DATE = "2026-11-18"
SCALE, W_LDA, W_TEXT = 20.0, 0.75, 0.20
LAM = 2.0
N_EP_5WAY = 50


def main():
    t0 = time.time()
    sup, qry, classes, prompts = load_plantdoc(ROOT / "plantdoc", k_shot=5, seed=42)
    C = len(classes)
    print(f"[PD] support={len(sup)} query={len(qry)} classes={C}", flush=True)

    dino_tf, bioclip_tf, dino_cfg = build_transforms()
    dino_aug = aug_pipeline(dino_cfg["mean"], dino_cfg["std"])
    bio_aug = aug_pipeline(*norm_stats(bioclip_tf))

    s_dino, s_bio, s_y = [], [], []
    for i, (p, y) in enumerate(sup):
        img = Image.open(p).convert("RGB")
        d_views = [np.asarray(dino_tf(img), dtype=np.float32)]
        b_views = [np.asarray(bioclip_tf(img), dtype=np.float32)]
        torch.manual_seed(42 + 10_000 + i)
        for _ in range(N_VIEWS - 1):
            d_views.append(np.asarray(dino_aug(img), dtype=np.float32))
            b_views.append(np.asarray(bio_aug(img), dtype=np.float32))
        s_dino.extend(d_views); s_bio.extend(b_views)
        s_y.extend([y] * N_VIEWS)
    s_dino, s_bio, s_y = np.stack(s_dino), np.stack(s_bio), np.array(s_y)

    q_dino = np.stack([np.asarray(dino_tf(Image.open(p).convert("RGB")), np.float32) for p, _ in qry])
    q_bio = np.stack([np.asarray(bioclip_tf(Image.open(p).convert("RGB")), np.float32) for p, _ in qry])
    q_y = np.array([y for _, y in qry])
    print(f"[PD] support views={len(s_dino)} queries={len(q_dino)}", flush=True)

    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    text_protos = []
    with torch.no_grad():
        for c in classes:
            f = F.normalize(bioclip_model.encode_text(tokenizer(prompts[c])), dim=-1)
            text_protos.append(F.normalize(f.mean(dim=0), dim=-1))
    text_protos = torch.stack(text_protos)

    t1, t2 = build_sessions("int8")

    def feats(dx, bx):
        do, bo = [], []
        for i in range(0, len(dx), 64):
            _, _, f = run_onnx_tier1(t1, dx[i:i + 64])
            do.append(f)
            _, fb = run_onnx_companion(t2, bx[i:i + 64], f)
            bo.append(fb)
        return np.concatenate(do), np.concatenate(bo)

    sd, sb = feats(s_dino, s_bio)
    qd, qb = feats(q_dino, q_bio)
    sj = np.concatenate([sb, sd], axis=-1)
    qj = np.concatenate([qb, qd], axis=-1)

    # Deployed PlantDoc constants (pipeline/biodino_edge_pipeline.py:574-576):
    # support_weight=1, max_anchors=2, conf_threshold=None, because 252 queries
    # over 27 classes starve a fixed threshold.
    PD = dict(support_weight=1, max_anchors=2, conf_threshold=None)

    # ---- 27-way single run ----
    p_lap = laplacianshot(sj, s_y, qj, C, lam=LAM)
    p_our = fisher_anchor(sj, s_y, qj, qb, text_protos, C, **PD)
    r27 = {
        "laplacianshot": {"acc": round(float(100 * accuracy_score(q_y, p_lap)), 2),
                          "f1": round(float(100 * f1_score(q_y, p_lap, average="macro", zero_division=0)), 2)},
        "ours": {"acc": round(float(100 * accuracy_score(q_y, p_our)), 2),
                 "f1": round(float(100 * f1_score(q_y, p_our, average="macro", zero_division=0)), 2)},
        "n_queries": int(len(q_y)), "n_classes": int(C)}
    print(f"[27-way] Lap {r27['laplacianshot']['acc']:.2f}  ours {r27['ours']['acc']:.2f}", flush=True)

    # ---- 5-way, 50 episodes over the test split ----
    idx_by_class = [np.where(q_y == c)[0].tolist() for c in range(C)]
    classes_with_data = [c for c in range(C) if len(idx_by_class[c]) >= 2]
    lap5, our5 = [], []
    rng = random.Random(42)
    for ep in range(N_EP_5WAY):
        pick = rng.sample(classes_with_data, 5)
        qi = np.concatenate([idx_by_class[c] for c in pick])
        qy = q_y[qi]
        # relabel to 0..4 and reuse the global support (all 27 classes present)
        remap = {c: i for i, c in enumerate(pick)}
        qy_local = np.array([remap[v] for v in qy])
        # restrict support to the 5 sampled classes
        mask = np.isin(s_y, pick)
        s_sub = sj[mask]
        sy_local = np.array([remap[v] for v in s_y[mask]])
        tp = text_protos[pick]
        lap5.append(100 * accuracy_score(qy_local,
                                         laplacianshot(s_sub, sy_local, qj[qi], 5, lam=LAM)))
        our5.append(100 * accuracy_score(qy_local,
                                         fisher_anchor(s_sub, sy_local, qj[qi], qb[qi], tp, 5, **PD)))
        if (ep + 1) % 10 == 0:
            print(f"  5way ep {ep+1}/{N_EP_5WAY} ({time.time()-t0:.0f}s)", flush=True)

    lap5, our5 = np.array(lap5), np.array(our5)
    d5 = our5 - lap5
    t5, p5 = stats.ttest_rel(our5, lap5)
    r5 = {"episodes": N_EP_5WAY,
          "laplacianshot": {"acc": round(float(lap5.mean()), 2), "std": round(float(lap5.std(ddof=1)), 2)},
          "ours": {"acc": round(float(our5.mean()), 2), "std": round(float(our5.std(ddof=1)), 2)},
          "delta_pp": round(float(d5.mean()), 2), "p_value": float(p5),
          "wins": int((d5 > 0).sum()), "losses": int((d5 < 0).sum())}
    print(f"[5-way] Lap {r5['laplacianshot']['acc']:.2f}  ours {r5['ours']['acc']:.2f}  "
          f"delta {r5['delta_pp']:+.2f} p={p5:.2e} W/L {r5['wins']}/{r5['losses']}", flush=True)

    payload = {"date": DATE, "dataset": "PlantDoc (int8 deployed binaries)",
               "lambda": LAM, "regime_27way": r27, "regime_5way": r5,
               "runtime_seconds": round(time.time() - t0, 1)}
    pth = OUT / f"laplacianshot_plantdoc_{DATE}.json"
    pth.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {pth}")


if __name__ == "__main__":
    main()
