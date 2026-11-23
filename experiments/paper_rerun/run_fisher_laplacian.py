# -*- coding: utf-8 -*-
"""Does the regularized Fisher projection help transductive label propagation?

Motivation. The fair-baseline run showed LaplacianShot (label propagation on raw
896-D cosine kNN) is 0.5 pp ABOVE BioDINO-Edge transductive. That raises the
question the paper actually needs to answer: is the discriminative projection
useful at all in the transductive regime, or is propagation doing the work?

This script runs label propagation in the Fisher-projected subspace and compares
it against propagation in raw 896-D space on the SAME episodes. If projection
helps propagation, the Fisher term has transductive value even though our
particular anchor-recalibration rule is not the strongest propagator.

Variants
  aa      raw 896-D cosine affinity (LaplacianShot as reported)
  fisher  affinity computed after projection through the fitted Fisher-LDA W
  both    sum of the two affinity matrices (cosine + projected cosine)

Output: experiments/paper_rerun/results/fisher_laplacian_2026-11-18.json
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "pipeline")):
    if str(p) not in sys.path:
        sys.path.insert(0, p)

from core_math import fit_fisher_lda  # noqa: E402

CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
OUT = Path(__file__).resolve().parent / "results"
DATE = "2026-11-18"
SEED, K_SHOT = 42, 5
N_EP_11, N_EP_5WAY = 30, 50
ALPHA, KNN, ITERS = 0.99, 10, 200


def l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def knn_affinity(x, k=KNN):
    n = x.shape[0]
    k = min(k, n - 1)
    rows, cols, vals = [], [], []
    for start in range(0, n, 512):
        blk = x[start:start + 512] @ x.T
        for i in range(blk.shape[0]):
            blk[i, start + i] = -np.inf
        idx = np.argpartition(-blk, k, axis=1)[:, :k]
        for i, row_idx in enumerate(idx):
            r = start + i
            rows.extend([r] * len(row_idx))
            cols.extend(row_idx.tolist())
            vals.extend(np.clip(blk[i, row_idx], 0.0, None).tolist())
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A.maximum(A.T)


def propagate(A, n_sup, C, y_sup, alpha=ALPHA, iters=ITERS):
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    dinv = 1.0 / np.sqrt(d)
    S = A.multiply(dinv[:, None]).multiply(dinv[None, :]).tocsr()
    Y = np.zeros((A.shape[0], C))
    Y[np.arange(n_sup), y_sup] = 1.0
    Y = csr_matrix(Y)
    Fy = Y.copy()
    for _ in range(iters):
        Fy = alpha * (S @ Fy) + (1 - alpha) * Y
    return np.asarray(Fy[n_sup:].todense()).argmax(-1)


def variants(s_joint, s_y, q_joint, C):
    allx = np.concatenate([s_joint, q_joint], axis=0)
    n_sup = len(s_y)
    out = {}

    A_raw = knn_affinity(l2norm(allx))
    out["aa"] = propagate(A_raw, n_sup, C, s_y)

    W, b, _ = fit_fisher_lda(s_joint, s_y, device="cpu")
    proj = (torch.tensor(allx, dtype=torch.float32) @ W.T).numpy()
    A_prj = knn_affinity(l2norm(proj))
    out["fisher"] = propagate(A_prj, n_sup, C, s_y)

    out["both"] = propagate((A_raw + A_prj).tocsr(), n_sup, C, s_y)
    return out


METHODS = ["aa", "fisher", "both"]


def main():
    t0 = time.time()
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    bio, dino = data["bio_feats"].numpy(), data["dino_feats"].numpy()
    labels = data["labels"].numpy()
    C = len(data["novel_classes"])
    joint = np.concatenate([bio, dino], axis=-1)
    class_idx = [np.where(labels == c)[0].tolist() for c in range(C)]
    print(f"[*] {len(labels)} images, C={C}", flush=True)

    def one_episode(si, qi, sy, qy):
        preds = variants(joint[si], sy, joint[qi], len(np.unique(sy)))
        return {m: dict(acc=float(100 * accuracy_score(qy, p)),
                        f1=float(100 * f1_score(qy, p, average="macro", zero_division=0)))
                for m, p in preds.items()}

    ep11 = []
    for ep in range(N_EP_11):
        rng = random.Random(SEED + ep * 17)
        si, qi = [], []
        for c in range(C):
            sh = class_idx[c].copy()
            rng.shuffle(sh)
            si.extend(sh[:K_SHOT])
            qi.extend(sh[K_SHOT:])
        si, qi = np.array(si), np.array(qi)
        ep11.append(one_episode(si, qi, labels[si], labels[qi]))
    print(f"[R1] {N_EP_11} episodes in {time.time()-t0:.0f}s", flush=True)

    ep5 = []
    rng = random.Random(SEED)
    for _ in range(N_EP_5WAY):
        sampled = rng.sample(range(C), 5)
        cmap = {o: l for l, o in enumerate(sampled)}
        si, qi, sy, qy = [], [], [], []
        for o in sampled:
            sh = class_idx[o].copy()
            rng.shuffle(sh)
            si.extend(sh[:K_SHOT])
            qi.extend(sh[K_SHOT:])
            sy.extend([cmap[o]] * K_SHOT)
            qy.extend([cmap[o]] * (len(sh) - K_SHOT))
        ep5.append(one_episode(np.array(si), np.array(qi), np.array(sy), np.array(qy)))
    print(f"[R2] {N_EP_5WAY} episodes in {time.time()-t0:.0f}s", flush=True)

    def summ(eps):
        r = {}
        for m in METHODS:
            a = np.array([e[m]["acc"] for e in eps], float)
            r[m] = dict(acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)))
        return r

    payload = {"date": DATE,
               "protocol": {"seed": SEED, "k_shot": K_SHOT, "alpha": ALPHA,
                            "knn": KNN, "iters": ITERS,
                            "note": "affinity over pooled support+query; Fisher variant "
                                    "projects through the fitted regularized Fisher-LDA W"},
               "regime_11class_5shot": {"summary": summ(ep11), "per_episode": ep11},
               "regime_5way_5shot": {"summary": summ(ep5), "per_episode": ep5},
               "runtime_seconds": round(time.time() - t0, 1)}
    p = OUT / f"fisher_laplacian_{DATE}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {p}")
    for k in ("regime_11class_5shot", "regime_5way_5shot"):
        print(f"\n== {k} ==")
        for m, s in payload[k]["summary"].items():
            print(f"  {m:8s} {s['acc_mean']:6.2f} +/- {s['acc_std']:5.2f}")


if __name__ == "__main__":
    main()
