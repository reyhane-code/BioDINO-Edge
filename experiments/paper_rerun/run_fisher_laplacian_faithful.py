# -*- coding: utf-8 -*-
"""Does the regularized Fisher projection help the FAITHFUL LaplacianShot?

Replaces the earlier soft-propagation version. Uses the published binary
majorise-minimise solver (run_laplacianshot_faithful.laplacianshot) and asks
whether changing the feature space it operates on helps or hurts.

Variants
  raw       faithful LaplacianShot on 896-D (the baseline as published)
  fisher    same solver, but the query graph and prototypes are built in the
            Fisher-projected subspace
  both      affinity = average of raw and projected cosine affinities

Output: experiments/paper_rerun/results/fisher_laplacian_faithful_2026-11-18.json
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "pipeline")):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_laplacianshot_faithful import laplacianshot  # noqa: E402
from core_math import fit_fisher_lda  # noqa: E402

CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
OUT = Path(__file__).resolve().parent / "results"
DATE = "2026-11-18"
SEED, K_SHOT, N_EP, LAM = 42, 5, 10, 2.0


def project(x, W):
    return (torch.tensor(x, dtype=torch.float32) @ W.T).numpy()


def main():
    t0 = time.time()
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    bio, dino = data["bio_feats"].numpy(), data["dino_feats"].numpy()
    labels = data["labels"].numpy()
    C = len(data["novel_classes"])
    joint = np.concatenate([bio, dino], axis=-1)
    class_idx = [np.where(labels == c)[0].tolist() for c in range(C)]
    print(f"[*] C={C}, lambda={LAM}", flush=True)

    acc = {"raw": [], "fisher": [], "both": []}
    for ep in range(N_EP):
        rng = random.Random(SEED + ep * 17)
        si, qi = [], []
        for c in range(C):
            sh = class_idx[c].copy()
            rng.shuffle(sh)
            si.extend(sh[:K_SHOT]); qi.extend(sh[K_SHOT:])
        si, qi = np.array(si), np.array(qi)
        s, sy = joint[si], labels[si]
        q, qy = joint[qi], labels[qi]

        acc["raw"].append(100 * accuracy_score(
            qy, laplacianshot(s, sy, q, C, lam=LAM)))

        W, b, _ = fit_fisher_lda(s, sy, device="cpu")
        sp, qp = project(s, W), project(q, W)
        acc["fisher"].append(100 * accuracy_score(
            qy, laplacianshot(sp, sy, qp, C, lam=LAM)))

        # concat raw and projected features: graph sees both geometries
        sc = np.concatenate([s / np.linalg.norm(s, axis=1, keepdims=True),
                             sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-12)],
                            axis=1)
        qc = np.concatenate([q / np.linalg.norm(q, axis=1, keepdims=True),
                             qp / (np.linalg.norm(qp, axis=1, keepdims=True) + 1e-12)],
                            axis=1)
        acc["both"].append(100 * accuracy_score(
            qy, laplacianshot(sc, sy, qc, C, lam=LAM)))
        print(f"  ep {ep+1}/{N_EP} ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for k, v in acc.items():
        arr = np.array(v)
        summary[k] = {"mean": round(float(arr.mean()), 2),
                      "std": round(float(arr.std(ddof=1)), 2)}
        print(f"  {k:8s} {arr.mean():6.2f} +/- {arr.std(ddof=1):5.2f}", flush=True)

    payload = {"date": DATE, "lambda": LAM, "episodes": N_EP,
               "implementation": "faithful binary majorise-minimise",
               "summary": summary, "per_episode": acc}
    pth = OUT / f"fisher_laplacian_faithful_{DATE}.json"
    pth.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {pth}")


if __name__ == "__main__":
    main()
