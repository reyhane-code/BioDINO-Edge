# -*- coding: utf-8 -*-
"""Faithful LaplacianShot, per Ziko et al., ICML 2020 (arXiv:2006.15486).

Why this file exists. The first fair-baseline run used soft label propagation
(F <- a*S@F + (1-a)*Y), which is Zhou-style label propagation, NOT LaplacianShot.
The published method minimises a quadratic BINARY assignment with a pairwise
Laplacian term, solved by majorize-minimise. This script implements that, so the
comparison in the paper is against the real algorithm.

Objective (Eq. 3 of the paper):
    min_{Y in {0,1}^{n x C}, Y 1 = 1}  <Y, D>  -  (lambda/2) <Y, W Y>
where D[i,c] is the unary cost of query i to class prototype c and W is the
pairwise affinity over QUERY samples only.

Majorise-minimise update (Algorithm 1):
    y_i <- onehot( argmin_c [ D[i,c] - lambda * (W Y)[i,c] ] )
iterated to convergence. The graph is over queries, so a single query has an
empty graph and the method degenerates to nearest-prototype, which is exactly the
structural batch dependence the paper needs to quantify.

Output: experiments/paper_rerun/results/laplacianshot_faithful_2026-11-18.json
"""
import json
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

from core_math import fit_fisher_lda  # noqa: E402

CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
OUT = Path(__file__).resolve().parent / "results"
DATE = "2026-11-18"


def l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def query_graph(q, k):
    """Symmetric kNN cosine affinity over the query set only (paper: W over Q)."""
    x = l2norm(q)
    n = x.shape[0]
    if n < 2:
        return csr_matrix((n, n))
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
    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return W.maximum(W.T)


def laplacianshot(s, sy, q, C, lam=0.75, k=10, iters=50):
    """Binary-assignment LaplacianShot by majorise-minimise."""
    protos = l2norm(np.stack([l2norm(s)[sy == c].mean(0) for c in range(C)]))
    qn = l2norm(q)
    # unary cost: negative cosine similarity (lower = closer)
    D = -(qn @ protos.T)
    n = len(q)
    Y = np.eye(C)[D.argmin(1)]                      # nearest-prototype init
    if n < 2:
        return Y.argmax(1)
    W = query_graph(q, k)
    obj_prev = np.inf
    for _ in range(iters):
        WY = np.asarray(W @ Y)                      # (n, C)
        score = D - lam * WY
        new = np.eye(C)[score.argmin(1)]
        obj = float((D * new).sum() - 0.5 * lam * (new * WY).sum())
        Y = new
        if abs(obj_prev - obj) < 1e-7:
            break
        obj_prev = obj
    return Y.argmax(1)


def fisher_anchor(s, sy, q, q_bio, text_protos, C,
                  support_weight=3, max_anchors=20, conf_threshold=0.70):
    from core_math import compute_fisher_ensemble_probs, refine_transductive
    qt = torch.tensor(q, dtype=torch.float32)
    W, b, _ = fit_fisher_lda(s, sy, device="cpu")
    prob, _, txt = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=qt,
        bioclip_features=torch.tensor(q_bio, dtype=torch.float32),
        text_prototypes=text_protos, scale=20.0, w_lda=0.75, device="cpu")
    rp, _, _ = refine_transductive(
        support_features=s, support_labels=sy, query_features=qt,
        initial_preds=prob.argmax(-1).numpy(), confidences=prob.max(dim=-1).values,
        text_prob=txt, n_classes=C, support_weight=support_weight,
        max_anchors=max_anchors, conf_threshold=conf_threshold,
        device="cpu", w_lda=0.75)
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

    import random
    K_SHOT, SEED = 5, 42

    # --- lambda sweep on 5 episodes, so we do not handicap the baseline ---
    print("[sweep] lambda over 5 episodes (11-class):", flush=True)
    best = None
    sweep = {}
    for lam in (0.1, 0.3, 0.5, 0.75, 1.0, 2.0):
        accs = []
        for ep in range(5):
            rng = random.Random(SEED + ep * 17)
            si, qi = [], []
            for c in range(C):
                sh = class_idx[c].copy(); rng.shuffle(sh)
                si.extend(sh[:K_SHOT]); qi.extend(sh[K_SHOT:])
            si, qi = np.array(si), np.array(qi)
            p = laplacianshot(joint[si], labels[si], joint[qi], C, lam=lam)
            accs.append(100 * accuracy_score(labels[qi], p))
        sweep[lam] = round(float(np.mean(accs)), 2)
        print(f"  lambda={lam:<5} {np.mean(accs):6.2f}", flush=True)
        if best is None or np.mean(accs) > sweep[best]:
            best = lam
    print(f"[sweep] best lambda = {best}", flush=True)

    # --- full comparison at the best lambda ---
    N_EP = 10
    ep11 = []
    for ep in range(N_EP):
        rng = random.Random(SEED + ep * 17)
        si, qi = [], []
        for c in range(C):
            sh = class_idx[c].copy(); rng.shuffle(sh)
            si.extend(sh[:K_SHOT]); qi.extend(sh[K_SHOT:])
        si, qi = np.array(si), np.array(qi)
        s, sy = joint[si], labels[si]
        q, qy, qb = joint[qi], labels[qi], bio[qi]
        ep11.append({
            "lap": float(100 * accuracy_score(qy, laplacianshot(s, sy, q, C, lam=best))),
            "fish": float(100 * accuracy_score(qy, fisher_anchor(s, sy, q, qb, text_protos, C))),
        })
        print(f"  ep {ep+1}/{N_EP} ({time.time()-t0:.0f}s)", flush=True)

    lap = np.array([e["lap"] for e in ep11])
    fish = np.array([e["fish"] for e in ep11])
    d = fish - lap
    from scipy import stats
    t, p = stats.ttest_rel(fish, lap)
    print(f"\n11-class: LaplacianShot {lap.mean():.2f}  ours {fish.mean():.2f}  "
          f"delta {d.mean():+.2f}  p={p:.2e}  W/L/T {int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}",
          flush=True)

    payload = {"date": DATE,
               "implementation": "binary-assignment majorise-minimise, Ziko et al. ICML 2020",
               "lambda_sweep_5ep": sweep, "lambda_used": best,
               "regime_11class_5shot": {
                   "episodes": N_EP,
                   "laplacianshot_mean": round(float(lap.mean()), 2),
                   "laplacianshot_std": round(float(lap.std(ddof=1)), 2),
                   "ours_mean": round(float(fish.mean()), 2),
                   "ours_std": round(float(fish.std(ddof=1)), 2),
                   "delta_pp": round(float(d.mean()), 2),
                   "paired_t": round(float(t), 2), "p_value": float(p),
                   "wins": int((d > 0).sum()), "losses": int((d < 0).sum())},
               "per_episode": ep11,
               "runtime_seconds": round(time.time() - t0, 1)}
    pth = OUT / f"laplacianshot_faithful_{DATE}.json"
    pth.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {pth}")


if __name__ == "__main__":
    main()
