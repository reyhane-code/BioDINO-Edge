# -*- coding: utf-8 -*-
"""Accuracy of transductive methods as a function of how much of the query set
is available at decision time.

Why this matters. LaplacianShot propagates labels over a kNN graph built from the
pooled support+query set, so it needs the whole query batch before it can label
anything. A phone diagnoses one leaf at a time and discards it (or buffers only
confident anchors). If propagation degrades when it sees a small batch, that is a
deployment-relevant weakness of the stronger-in-batch method; if it does not,
the accuracy gap is simply real and must be reported as such.

Protocol. Same 30 canonical 11-class episodes and the same cached features as
run_fair_baselines.py. For each batch size B we draw REPS random query subsets of
size B and label each using ONLY the support set plus that subset. B = "all"
reproduces the full-batch result. Both methods see exactly the same subsets, so
the comparison is paired. This measures how each method's accuracy depends on how
much of the query set is visible at decision time.

Output: experiments/paper_rerun/results/batchsize_2026-11-18.json
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "pipeline")):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core_math import (  # noqa: E402
    compute_fisher_ensemble_probs, fit_fisher_lda, refine_transductive,
)

CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
OUT = Path(__file__).resolve().parent / "results"
DATE = "2026-11-18"
SEED, K_SHOT, N_EP = 42, 5, 10
SCALE, W_LDA, ALPHA, KNN, ITERS = 20.0, 0.75, 0.99, 10, 200
BATCHES = [25, 50, 100, 250, 500, 1000, None]
REPS = 3


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


def propagate(s, sy, q, C):
    allx = np.concatenate([s, q], axis=0)
    A = knn_affinity(l2norm(allx))
    dg = np.asarray(A.sum(1)).ravel()
    dg[dg == 0] = 1.0
    dinv = 1.0 / np.sqrt(dg)
    S = A.multiply(dinv[:, None]).multiply(dinv[None, :]).tocsr()
    Y = csr_matrix((np.ones(len(sy)), (np.arange(len(sy)), sy)),
                   shape=(allx.shape[0], C))
    F = Y.copy()
    for _ in range(ITERS):
        F = ALPHA * (S @ F) + (1 - ALPHA) * Y
    return np.asarray(F[len(sy):].todense()).argmax(-1)


def fisher_anchor(s, sy, q, q_bio, text_protos, C):
    qt = torch.tensor(q, dtype=torch.float32)
    W, b, _ = fit_fisher_lda(s, sy, device="cpu")
    prob, _, txt = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=qt,
        bioclip_features=torch.tensor(q_bio, dtype=torch.float32),
        text_prototypes=text_protos, scale=SCALE, w_lda=W_LDA, device="cpu")
    rp, _, _ = refine_transductive(
        support_features=s, support_labels=sy, query_features=qt,
        initial_preds=prob.argmax(-1).numpy(), confidences=prob.max(dim=-1).values,
        text_prob=txt, n_classes=C, support_weight=3, max_anchors=20,
        conf_threshold=0.70, device="cpu", w_lda=W_LDA)
    return rp


def main():
    t0 = time.time()
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    bio, dino = data["bio_feats"].numpy(), data["dino_feats"].numpy()
    labels = data["labels"].numpy()
    text_protos = data["bio_text_protos"]
    C = len(data["novel_classes"])
    joint = np.concatenate([bio, dino], axis=-1)
    class_idx = [np.where(labels == c)[0].tolist() for c in range(C)]
    print(f"[*] {len(labels)} images, C={C}", flush=True)
    print(f"[*] batches={BATCHES} reps={REPS}", flush=True)

    res = {str(b): {"lap": [], "fish": []} for b in BATCHES}
    for ep in range(N_EP):
        rng = random.Random(SEED + ep * 17)
        si, qi = [], []
        for c in range(C):
            sh = class_idx[c].copy()
            rng.shuffle(sh)
            si.extend(sh[:K_SHOT])
            qi.extend(sh[K_SHOT:])
        si, qi = np.array(si), np.array(qi)
        s, sy = joint[si], labels[si]
        q, qy, qb = joint[qi], labels[qi], bio[qi]

        for B in BATCHES:
            key = str(B)
            n_rep = 1 if B is None else REPS
            for r in range(n_rep):
                if B is None or B >= len(q):
                    idx = np.arange(len(q))
                else:
                    idx = np.array(random.Random(SEED + ep * 17 + r * 991)
                                   .sample(range(len(q)), B))
                sub_q, sub_y, sub_b = q[idx], qy[idx], qb[idx]
                res[key]["lap"].append(float(100 * accuracy_score(
                    sub_y, propagate(s, sy, sub_q, C))))
                res[key]["fish"].append(float(100 * accuracy_score(
                    sub_y, fisher_anchor(s, sy, sub_q, sub_b, text_protos, C))))
        print(f"  ep {ep+1}/{N_EP} ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for B in BATCHES:
        key = str(B)
        lap = np.array(res[key]["lap"])
        fish = np.array(res[key]["fish"])
        summary[key] = {
            "batch_size": "all" if B is None else B,
            "laplacianshot_acc": round(float(lap.mean()), 2),
            "laplacianshot_std": round(float(lap.std(ddof=1)), 2),
            "fisher_anchor_acc": round(float(fish.mean()), 2),
            "fisher_anchor_std": round(float(fish.std(ddof=1)), 2),
            "delta_fish_minus_lap": round(float((fish - lap).mean()), 2),
        }
        print(f"  B={summary[key]['batch_size']!s:>4}  Laplacian {lap.mean():6.2f}  "
              f"Fisher+anchor {fish.mean():6.2f}  delta {(fish-lap).mean():+6.2f}", flush=True)

    payload = {"date": DATE,
               "protocol": {"episodes": N_EP, "k_shot": K_SHOT, "seed": SEED,
                            "alpha": ALPHA, "knn": KNN, "iters": ITERS,
                            "note": "each chunk labelled from support + that chunk only"},
               "summary": summary,
               "per_episode": {k: res[k] for k in res},
               "runtime_seconds": round(time.time() - t0, 1)}
    p = OUT / f"batchsize_{DATE}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] {p}")


if __name__ == "__main__":
    main()
