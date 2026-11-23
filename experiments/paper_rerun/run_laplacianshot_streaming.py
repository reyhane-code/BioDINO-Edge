# -*- coding: utf-8 -*-
"""Can LaplacianShot be deployed incrementally? The decisive test.

LaplacianShot's graph is over the QUERY set, so it cannot label a leaf until a
batch exists. The only field-deployable form keeps a buffer of previously seen
queries and propagates over support + buffer + the current chunk. If that works,
the batch-size argument in the paper collapses and LaplacianShot is simply better.

Strategies compared on identical episodes and identical query orderings, using
the faithful binary-assignment implementation (run_laplacianshot_faithful.py):

  ours            closed-form Fisher recalibration, as shipped
  lap-batch       faithful LaplacianShot over the whole query set (lab protocol)
  lap-chunk-B     propagation over support + current chunk of B (frozen batch)
  lap-grow-B      propagation over support + ALL queries seen so far, scoring the
                  newest chunk of B. This is the most favourable realistic
                  implementation of the baseline: its graph accumulates.

Cost control: each episode contributes a fixed random subset of STOP_AT queries
so that the growing-buffer variant stays tractable.

Output: experiments/paper_rerun/results/laplacianshot_streaming_2026-11-18.json
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "pipeline")):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_laplacianshot_faithful import fisher_anchor, laplacianshot  # noqa: E402

CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
OUT = Path(__file__).resolve().parent / "results"
DATE = "2026-11-18"
SEED, K_SHOT, N_EP = 42, 5, 8
STOP_AT = 600          # queries per episode used by the streaming variants
LAM = 2.0
CHUNKS = [25, 50, 100, 200]


def main():
    t0 = time.time()
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    bio, dino = data["bio_feats"].numpy(), data["dino_feats"].numpy()
    labels = data["labels"].numpy()
    text_protos = data["bio_text_protos"]
    C = len(data["novel_classes"])
    joint = np.concatenate([bio, dino], axis=-1)
    class_idx = [np.where(labels == c)[0].tolist() for c in range(C)]
    print(f"[*] {len(labels)} images, C={C}, stop_at={STOP_AT}, lambda={LAM}", flush=True)

    keys = ["ours", "lap_batch", "lap_single"]
    for B in CHUNKS:
        keys += [f"lap_chunk_{B}", f"lap_grow_{B}"]
    acc = {k: [] for k in keys}

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
        q_all, y_all, b_all = joint[qi], labels[qi], bio[qi]

        sub = np.array(random.Random(SEED + ep * 17 + 7).sample(range(len(q_all)),
                                                                min(STOP_AT, len(q_all))))
        q, qy, qb = q_all[sub], y_all[sub], b_all[sub]

        acc["ours"].append(float(100 * accuracy_score(
            qy, fisher_anchor(s, sy, q, qb, text_protos, C))))
        acc["lap_batch"].append(float(100 * accuracy_score(
            qy, laplacianshot(s, sy, q, C, lam=LAM))))
        # one query at a time: empty graph, degenerates to nearest prototype
        acc["lap_single"].append(float(100 * accuracy_score(
            qy, np.concatenate([laplacianshot(s, sy, q[i:i + 1], C, lam=LAM)
                                for i in range(len(q))]))))

        for B in CHUNKS:
            # frozen chunk batches
            preds = np.concatenate([
                laplacianshot(s, sy, q[i:i + B], C, lam=LAM)
                for i in range(0, len(q), B)])
            acc[f"lap_chunk_{B}"].append(float(100 * accuracy_score(qy, preds)))

            # growing buffer: graph over all queries seen so far
            preds = []
            for i in range(0, len(q), B):
                pool = q[:i + B]
                gp = laplacianshot(s, sy, pool, C, lam=LAM)
                preds.append(gp[i:])
            acc[f"lap_grow_{B}"].append(
                float(100 * accuracy_score(qy, np.concatenate(preds))))
        print(f"  ep {ep+1}/{N_EP} ({time.time()-t0:.0f}s)", flush=True)

    summary = {k: {"mean": round(float(np.mean(v)), 2),
                   "std": round(float(np.std(v, ddof=1)), 2)} for k, v in acc.items()}
    for k in keys:
        print(f"  {k:16s} {summary[k]['mean']:6.2f} +/- {summary[k]['std']:5.2f}", flush=True)

    ours = np.array(acc["ours"])
    payload = {"date": DATE, "lambda": LAM, "episodes": N_EP,
               "queries_per_episode": STOP_AT,
               "implementation": "binary-assignment majorise-minimise (Ziko et al. 2020)",
               "summary": summary, "comparisons": {}}
    for k in keys:
        if k == "ours":
            continue
        other = np.array(acc[k])
        d = ours - other
        t, p = stats.ttest_rel(ours, other)
        payload["comparisons"][k] = {"ours_minus_them_pp": round(float(d.mean()), 2),
                                     "p_value": float(p),
                                     "wins": int((d > 0).sum()),
                                     "losses": int((d < 0).sum())}
    payload["per_episode"] = acc
    pth = OUT / f"laplacianshot_streaming_{DATE}.json"
    pth.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nours minus each variant:")
    for k, v in payload["comparisons"].items():
        print(f"  {k:16s} {v['ours_minus_them_pp']:+7.2f} pp  p={v['p_value']:.2e}  "
              f"W/L {v['wins']}/{v['losses']}", flush=True)
    print(f"\n[OK] {pth}")


if __name__ == "__main__":
    main()
