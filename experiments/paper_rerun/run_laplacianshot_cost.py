# -*- coding: utf-8 -*-
"""What does LaplacianShot actually cost, and what is its operating assumption?

The fair-baseline run showed LaplacianShot is 0.53 pp more accurate than
BioDINO-Edge transductive on identical features. This script measures the three
things that decide whether that matters on an edge device:

  1. Parameter count / model size   -- neither method adds a backbone, so this is
     about whether either needs extra stored state.
  2. Algebraic cost at the deployed batch size (9330 queries, 11 classes).
  3. How the cost scales with query-set size, because label propagation needs
     the WHOLE query set at once (transductive batch), whereas anchor
     recalibration is incremental.

Timings are desktop CPU (this machine) and are indicative only; they are NOT
device numbers and must never be quoted as such. The device measurements live in
experiments/edge_device_benchmark/results/benchmark_report.json.

Output: experiments/paper_rerun/results/laplacianshot_cost_2026-11-18.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "pipeline")):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core_math import fit_fisher_lda, refine_transductive  # noqa: E402

CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
OUT = Path(__file__).resolve().parent / "results"
DATE = "2026-11-18"
KNN, ALPHA, ITERS = 10, 0.99, 200


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


def time_laplacian(n_sup, n_qry, C, d, repeats=3):
    rng = np.random.default_rng(0)
    s = l2norm(rng.normal(size=(n_sup, d)).astype(np.float32))
    q = l2norm(rng.normal(size=(n_qry, d)).astype(np.float32))
    sy = rng.integers(0, C, n_sup)
    allx = np.concatenate([s, q], axis=0)
    ts, nnz = [], 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        A = knn_affinity(allx)
        dg = np.asarray(A.sum(1)).ravel()
        dg[dg == 0] = 1.0
        dinv = 1.0 / np.sqrt(dg)
        S = A.multiply(dinv[:, None]).multiply(dinv[None, :]).tocsr()
        Y = csr_matrix((np.ones(n_sup), (np.arange(n_sup), sy)), shape=(allx.shape[0], C))
        F = Y.copy()
        for _ in range(ITERS):
            F = ALPHA * (S @ F) + (1 - ALPHA) * Y
        ts.append(time.perf_counter() - t0)
        nnz = A.nnz
    return float(np.median(ts)), int(nnz)


def main():
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    bio, dino = data["bio_feats"].numpy(), data["dino_feats"].numpy()
    joint = np.concatenate([bio, dino], axis=-1)
    d = joint.shape[1]
    print(f"[*] d={d}", flush=True)

    out = {"date": DATE,
           "note": "desktop CPU timings, indicative only; NOT device numbers",
           "processor": "see platform.uname", "regimes": []}

    # 1. Cost of the two algebra paths at deployed scale
    s55, sy55 = joint[:55], np.arange(55) % 11
    for n_q, C in ((9330, 11), (252, 27), (500, 27), (2000, 27)):
        n_sup = 55 if C == 11 else 135
        s = joint[:n_sup]
        sy = np.arange(n_sup) % C

        t0 = time.perf_counter()
        for _ in range(3):
            W, b, _ = fit_fisher_lda(s, sy, device="cpu")
            qf = torch.tensor(joint[:n_q], dtype=torch.float32)
            _ = refine_transductive(
                support_features=s, support_labels=sy, query_features=qf,
                initial_preds=np.zeros(n_q, int), confidences=torch.ones(n_q),
                text_prob=None, n_classes=C, support_weight=3, max_anchors=20,
                conf_threshold=0.70, device="cpu", w_lda=0.75)
        t_fisher = (time.perf_counter() - t0) / 3

        t_lap, nnz = time_laplacian(n_sup, n_q, C, d, repeats=3)
        out["regimes"].append({
            "n_support": int(n_sup), "n_query": int(n_q), "n_classes": int(C),
            "fisher_anchor_seconds": round(t_fisher, 4),
            "laplacianshot_seconds": round(t_lap, 4),
            "laplacianshot_graph_nnz": nnz,
            "ratio_lap_over_fisher": round(t_lap / t_fisher, 2),
        })
        print(f"[C={C} nq={n_q}] fisher+anchor {t_fisher*1000:8.2f} ms  "
              f"LaplacianShot {t_lap*1000:8.2f} ms  ratio {t_lap/t_fisher:5.2f}x", flush=True)

    # 2. Scaling of the propagation graph
    print("\nscaling with query-set size (C=11, d=896):", flush=True)
    scaling = []
    for n_q in (100, 250, 500, 1000, 2000, 4000, 9330):
        t_lap, nnz = time_laplacian(55, n_q, 11, d, repeats=2)
        scaling.append({"n_query": n_q, "seconds": round(t_lap, 4), "nnz": nnz})
        print(f"  nq={n_q:5d}  {t_lap*1000:9.2f} ms  nnz={nnz}", flush=True)
    out["scaling"] = scaling

    # 3. Parameter accounting
    lda = fit_fisher_lda(joint[:55], np.arange(55) % 11, device="cpu")[0]
    out["parameters"] = {
        "backbone_params": "identical (frozen DINOv2-S + BioCLIP-B, shared by both)",
        "fisher_W_plus_b": int(lda.numel() + 11),
        "laplacianshot_stored_params": 0,
        "laplacianshot_runtime_state": "query-set kNN graph, O(n*k) edges",
        "fisher_runtime_state": "C x d projection, O(C*d) floats",
    }

    p = OUT / f"laplacianshot_cost_{DATE}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[OK] {p}")


if __name__ == "__main__":
    main()
