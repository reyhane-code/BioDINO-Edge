# -*- coding: utf-8 -*-
"""Faithful LaplacianShot on the 5-way and 11-class regimes, replacing the
soft-propagation stand-in used in the first fair-baseline run.

Also re-runs BD-CSPN and TIM as published variants for completeness? No: those
two were implemented correctly enough as prototype rectification and information
maximisation. Only LaplacianShot was substituted, so only it is redone here.

Output: experiments/paper_rerun/results/laplacianshot_faithful_regimes_2026-11-18.json
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "pipeline")):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_laplacianshot_faithful import fisher_anchor, laplacianshot  # noqa: E402

CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
OUT = Path(__file__).resolve().parent / "results"
DATE = "2026-11-18"
SEED, K_SHOT = 42, 5
LAM = 2.0
N_EP_11, N_EP_5W = 10, 20


def main():
    t0 = time.time()
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    bio, dino = data["bio_feats"].numpy(), data["dino_feats"].numpy()
    labels = data["labels"].numpy()
    text_protos = data["bio_text_protos"]
    C = len(data["novel_classes"])
    joint = np.concatenate([bio, dino], axis=-1)
    class_idx = [np.where(labels == c)[0].tolist() for c in range(C)]

    out = {"date": DATE, "lambda": LAM,
           "implementation": "binary-assignment majorise-minimise (Ziko et al. ICML 2020)"}

    # ---- 5-way 5-shot ----
    lap5, ours5, f1_lap, f1_ours = [], [], [], []
    rng = random.Random(SEED)
    for ep in range(N_EP_5W):
        sampled = rng.sample(range(C), 5)
        cmap = {o: l for l, o in enumerate(sampled)}
        si, qi, sy, qy = [], [], [], []
        for o in sampled:
            sh = class_idx[o].copy()
            rng.shuffle(sh)
            si.extend(sh[:K_SHOT]); qi.extend(sh[K_SHOT:])
            sy.extend([cmap[o]] * K_SHOT)
            qy.extend([cmap[o]] * (len(sh) - K_SHOT))
        si, qi = np.array(si), np.array(qi)
        sy, qy = np.array(sy), np.array(qy)
        s, q = joint[si], joint[qi]
        p_lap = laplacianshot(s, sy, q, 5, lam=LAM)
        p_our = fisher_anchor(s, sy, q, bio[qi], text_protos[sampled], 5)
        lap5.append(100 * accuracy_score(qy, p_lap))
        ours5.append(100 * accuracy_score(qy, p_our))
        f1_lap.append(100 * f1_score(qy, p_lap, average="macro", zero_division=0))
        f1_ours.append(100 * f1_score(qy, p_our, average="macro", zero_division=0))
    lap5, ours5 = np.array(lap5), np.array(ours5)
    t5, p5 = stats.ttest_rel(ours5, lap5)
    d5 = ours5 - lap5
    out["regime_5way_5shot"] = {
        "episodes": N_EP_5W,
        "laplacianshot": {"acc": round(float(lap5.mean()), 2), "std": round(float(lap5.std(ddof=1)), 2),
                          "f1": round(float(np.mean(f1_lap)), 2)},
        "ours": {"acc": round(float(ours5.mean()), 2), "std": round(float(ours5.std(ddof=1)), 2),
                 "f1": round(float(np.mean(f1_ours)), 2)},
        "delta_pp": round(float(d5.mean()), 2), "p_value": float(p5),
        "wins": int((d5 > 0).sum()), "losses": int((d5 < 0).sum())}
    print(f"[5-way] Laplacian {np.mean(lap5):.2f}  ours {np.mean(ours5):.2f}  "
          f"delta {np.mean(d5):+.2f} p={p5:.2e} W/L {int((d5>0).sum())}/{int((d5<0).sum())}",
          flush=True)

    # ---- 11-class 5-shot ----
    lap11, ours11, f1l, f1o = [], [], [], []
    for ep in range(N_EP_11):
        rng = random.Random(SEED + ep * 17)
        si, qi = [], []
        for c in range(C):
            sh = class_idx[c].copy(); rng.shuffle(sh)
            si.extend(sh[:K_SHOT]); qi.extend(sh[K_SHOT:])
        si, qi = np.array(si), np.array(qi)
        s, sy = joint[si], labels[si]
        q, qy = joint[qi], labels[qi]
        p_lap = laplacianshot(s, sy, q, C, lam=LAM)
        p_our = fisher_anchor(s, sy, q, bio[qi], text_protos, C)
        lap11.append(100 * accuracy_score(qy, p_lap))
        ours11.append(100 * accuracy_score(qy, p_our))
        f1l.append(100 * f1_score(qy, p_lap, average="macro", zero_division=0))
        f1o.append(100 * f1_score(qy, p_our, average="macro", zero_division=0))
    lap11, ours11 = np.array(lap11), np.array(ours11)
    t11, p11 = stats.ttest_rel(ours11, lap11)
    d11 = ours11 - lap11
    out["regime_11class_5shot"] = {
        "episodes": N_EP_11,
        "laplacianshot": {"acc": round(float(lap11.mean()), 2), "std": round(float(lap11.std(ddof=1)), 2),
                          "f1": round(float(np.mean(f1l)), 2)},
        "ours": {"acc": round(float(ours11.mean()), 2), "std": round(float(ours11.std(ddof=1)), 2),
                 "f1": round(float(np.mean(f1o)), 2)},
        "delta_pp": round(float(d11.mean()), 2), "p_value": float(p11),
        "wins": int((d11 > 0).sum()), "losses": int((d11 < 0).sum())}
    print(f"[11c] Laplacian {np.mean(lap11):.2f}  ours {np.mean(ours11):.2f}  "
          f"delta {np.mean(d11):+.2f} p={p11:.2e} W/L {int((d11>0).sum())}/{int((d11<0).sum())}",
          flush=True)

    out["runtime_seconds"] = round(time.time() - t0, 1)
    pth = OUT / f"laplacianshot_faithful_regimes_{DATE}.json"
    pth.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] {pth}")


if __name__ == "__main__":
    main()
