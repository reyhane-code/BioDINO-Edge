# -*- coding: utf-8 -*-
"""Faithful LaplacianShot on the canonical episodes, so the paper's table is coherent.

The first fair-baseline run used soft label propagation (Zhou-style) as a stand-in
for LaplacianShot. The published method minimises a quadratic BINARY assignment
(Ziko et al., ICML 2020), which turns out to be stronger. This script re-runs it
with the faithful solver on exactly the episodes used elsewhere in the paper:
30 canonical 11-class episodes (seed 42 + 17e) and 50 five-way episodes (seed 42).

Lambda is selected on a DISJOINT seed stream (42 + 1000 + 17e) so the reported
number is not tuned on its own test episodes.

Output: experiments/paper_rerun/results/laplacianshot_canonical_2026-11-18.json
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
N_EP_11, N_EP_5W = 30, 50
N_TUNE = 5


def split_11(rng, C, class_idx):
    si, qi = [], []
    for c in range(C):
        sh = class_idx[c].copy()
        rng.shuffle(sh)
        si.extend(sh[:K_SHOT])
        qi.extend(sh[K_SHOT:])
    return np.array(si), np.array(qi)


def main():
    t0 = time.time()
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    bio, dino = data["bio_feats"].numpy(), data["dino_feats"].numpy()
    labels = data["labels"].numpy()
    text_protos = data["bio_text_protos"]
    C = len(data["novel_classes"])
    joint = np.concatenate([bio, dino], axis=-1)
    class_idx = [np.where(labels == c)[0].tolist() for c in range(C)]
    print(f"[*] C={C}, tuning on seed stream 42+1000+17e", flush=True)

    # ---- lambda selection on a DISJOINT seed stream ----
    sweep = {}
    for lam in (0.5, 1.0, 2.0, 4.0):
        accs = []
        for ep in range(N_TUNE):
            rng = random.Random(SEED + 1000 + ep * 17)
            si, qi = split_11(rng, C, class_idx)
            p = laplacianshot(joint[si], labels[si], joint[qi], C, lam=lam)
            accs.append(100 * accuracy_score(labels[qi], p))
        sweep[str(lam)] = round(float(np.mean(accs)), 2)
        print(f"  lambda={lam:<5} {np.mean(accs):6.2f}", flush=True)
    LAM = max(sweep, key=lambda k: sweep[k])
    LAM = float(LAM)
    print(f"[*] lambda = {LAM}", flush=True)

    # ---- 11-class, canonical 30 episodes ----
    ep11 = []
    for ep in range(N_EP_11):
        rng = random.Random(SEED + ep * 17)
        si, qi = split_11(rng, C, class_idx)
        s, sy = joint[si], labels[si]
        q, qy = joint[qi], labels[qi]
        pl = laplacianshot(s, sy, q, C, lam=LAM)
        po = fisher_anchor(s, sy, q, bio[qi], text_protos, C)
        ep11.append({"lap_acc": float(100 * accuracy_score(qy, pl)),
                     "lap_f1": float(100 * f1_score(qy, pl, average="macro", zero_division=0)),
                     "our_acc": float(100 * accuracy_score(qy, po)),
                     "our_f1": float(100 * f1_score(qy, po, average="macro", zero_division=0))})
        print(f"  11c ep {ep+1}/{N_EP_11} ({time.time()-t0:.0f}s)", flush=True)

    la = np.array([e["lap_acc"] for e in ep11])
    ou = np.array([e["our_acc"] for e in ep11])
    d = ou - la
    t, p = stats.ttest_rel(ou, la)
    out = {"date": DATE, "lambda": LAM, "lambda_sweep_disjoint": sweep,
           "implementation": "binary-assignment majorise-minimise (Ziko et al. ICML 2020)",
           "regime_11class_5shot": {
               "episodes": N_EP_11,
               "laplacianshot": {"acc": round(float(la.mean()), 2),
                                 "std": round(float(la.std(ddof=1)), 2),
                                 "f1": round(float(np.mean([e["lap_f1"] for e in ep11])), 2)},
               "ours": {"acc": round(float(ou.mean()), 2),
                        "std": round(float(ou.std(ddof=1)), 2),
                        "f1": round(float(np.mean([e["our_f1"] for e in ep11])), 2)},
               "delta_pp": round(float(d.mean()), 2), "p_value": float(p),
               "wins": int((d > 0).sum()), "losses": int((d < 0).sum())},
           "per_episode_11": ep11}

    # ---- 5-way 5-shot, canonical 50 episodes ----
    ep5 = []
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
        pl = laplacianshot(s, sy, q, 5, lam=LAM)
        po = fisher_anchor(s, sy, q, bio[qi], text_protos[sampled], 5)
        ep5.append({"lap_acc": float(100 * accuracy_score(qy, pl)),
                    "lap_f1": float(100 * f1_score(qy, pl, average="macro", zero_division=0)),
                    "our_acc": float(100 * accuracy_score(qy, po)),
                    "our_f1": float(100 * f1_score(qy, po, average="macro", zero_division=0))})
        print(f"  5way ep {ep+1}/{N_EP_5W} ({time.time()-t0:.0f}s)", flush=True)

    la5 = np.array([e["lap_acc"] for e in ep5])
    ou5 = np.array([e["our_acc"] for e in ep5])
    d5 = ou5 - la5
    t5, p5 = stats.ttest_rel(ou5, la5)
    out["regime_5way_5shot"] = {
        "episodes": N_EP_5W,
        "laplacianshot": {"acc": round(float(la5.mean()), 2),
                          "std": round(float(la5.std(ddof=1)), 2),
                          "f1": round(float(np.mean([e["lap_f1"] for e in ep5])), 2)},
        "ours": {"acc": round(float(ou5.mean()), 2),
                 "std": round(float(ou5.std(ddof=1)), 2),
                 "f1": round(float(np.mean([e["our_f1"] for e in ep5])), 2)},
        "delta_pp": round(float(d5.mean()), 2), "p_value": float(p5),
        "wins": int((d5 > 0).sum()), "losses": int((d5 < 0).sum())}
    out["per_episode_5way"] = ep5
    out["runtime_seconds"] = round(time.time() - t0, 1)

    pth = OUT / f"laplacianshot_canonical_{DATE}.json"
    pth.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for k in ("regime_11class_5shot", "regime_5way_5shot"):
        b = out[k]
        print(f"\n{k}: Laplacian {b['laplacianshot']}  ours {b['ours']}")
        print(f"  delta {b['delta_pp']:+} p={b['p_value']:.2e} W/L {b['wins']}/{b['losses']}")
    print(f"[OK] {pth}")


if __name__ == "__main__":
    main()
