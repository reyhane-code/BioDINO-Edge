# -*- coding: utf-8 -*-
"""
Controlled, reproducible baseline comparison (reviewer response: fair comparison).
=================================================================================
Addresses the reviewer concern that Table I mixed literature values with
author-supplied numbers under non-identical protocols. Here EVERY baseline is
re-run by us on the IDENTICAL frozen BioCLIP+DINOv2 embeddings, the IDENTICAL
support/query splits, seeds, K-shot, and evaluation protocol as BioDINO-Edge.

Two regimes on the PlantVillage 11-class novel subset (9,385 images):
  R1. 11-class simultaneous, 30 Monte-Carlo episodes (canonical: seed 42+ep*17,
      K=5 support/class, query = all remaining class images). This is the exact
      protocol of run_routing_validation.py / run_transductive_validation.py.
  R2. 5-way K-shot episodic sweep (K=1,5,10,15), 50 episodes each (seed 42,
      sample 5 classes/episode, K support/class, query = all remaining images of
      the sampled classes). This matches the standard N-way K-shot regimes used
      by VSF (1/5-shot) and AgriFewNet (1/5/10/15-shot), so our numbers can be
      compared in-regime (same regime, different random splits).

All methods share the same frozen features (no backbone is retrained), so the
comparison isolates the adaptation algorithm, not the encoder.

Baselines (training-free, CPU, correctly referenced):
  - Zero-Shot CLIP (BioCLIP text prompts)      [radford2021clip, stevens2024bioclip]
  - Tip-Adapter (training-free cache adapter)  [zhang2022tip]  alpha=1.0, beta=5.5
  - Prototypical Networks (Euclidean)          [snell2017prototypical]
  - SimpleShot (L2-norm + cosine centroid)     [wang2019simpleshot]
Ours:
  - BioDINO-Edge Inductive  (Ledoit-Wolf Fisher-LDA + BioCLIP text late ensemble)
  - BioDINO-Edge Transductive (+ closed-form pseudo-anchor recalibration)

Output: controlled_baseline_results.json (mean/std/peak acc + macro F1 per method).
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import LedoitWolf
import scipy.linalg as la

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(ROOT), str(ROOT / "pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.core_math import (  # noqa: E402
    fit_fisher_lda, compute_fisher_ensemble_probs, refine_transductive,
)

CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
HERE = Path(__file__).resolve().parent
SEED, K_SHOT = 42, 5
SCALE, W_LDA = 20.0, 0.75
N_EP_11, N_EP_5WAY = 30, 50
TIP_ALPHA, TIP_BETA = 1.0, 5.5


def l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


# ------------------------------------------------------------------ methods
def m_zero_shot(q_bio, text_protos):
    logits = SCALE * (q_bio @ text_protos.numpy().T)
    return logits.argmax(-1)


def m_tip_adapter(s_bio, s_y, q_bio, text_protos, C):
    keys = l2norm(s_bio)
    vals = np.eye(C)[s_y]
    sim = l2norm(q_bio) @ keys.T
    adapter = np.exp(-TIP_BETA * (1.0 - sim)) @ vals
    clip_logits = SCALE * (q_bio @ text_protos.numpy().T)
    logits = TIP_ALPHA * adapter + clip_logits
    return logits.argmax(-1)


def m_protonet(s_joint, s_y, q_joint, C):
    fn = l2norm(s_joint)
    protos = l2norm(np.stack([fn[s_y == c].mean(0) for c in range(C)]))
    q = l2norm(q_joint)
    dist = ((q[:, None, :] - protos[None, :, :]) ** 2).sum(-1)
    return (-dist).argmax(-1)


def m_simpleshot(s_joint, s_y, q_joint, C):
    fn = l2norm(s_joint)
    protos = l2norm(np.stack([fn[s_y == c].mean(0) for c in range(C)]))
    q = l2norm(q_joint)
    return (SCALE * (q @ protos.T)).argmax(-1)


def fit_fisher_robust(X, y):
    """Canonical sklearn LW-Fisher when N>C; manual Ledoit-Wolf regularized
    Fisher (the paper's closed-form singularity resolution) when N<=C, e.g.
    5-way 1-shot where raw support N=5 equals C=5. Shrinkage makes Sigma PD."""
    n = X.shape[0]
    C = len(np.unique(y))
    if n > C:
        W, b, _ = fit_fisher_lda(X, y, device="cpu")
        return W, b
    classes = np.unique(y)
    means = np.stack([X[y == c].mean(0) for c in classes])
    Xc = X - means[np.searchsorted(classes, y)]
    S = LedoitWolf().fit(Xc).covariance_
    p = X.shape[1]
    # At 1-shot (one sample/class) the within-class scatter Xc is exactly zero,
    # so Ledoit-Wolf's scaled-identity target collapses to 0 (singular). Add a
    # data-driven ridge floor (mean support feature variance) to guarantee a PD
    # covariance; Fisher then reduces to regularized nearest-centroid, the
    # correct degenerate limit. When S is non-degenerate the ridge is trace(S)/p
    # (small relative to S), preserving the Ledoit-Wolf estimate.
    ridge = max(float(np.trace(S)) / p, float(np.var(X, axis=0).mean()), 1e-6)
    Sigma = S + ridge * np.eye(p)
    L = la.cholesky(Sigma, lower=True)
    SiM = la.cho_solve((L, True), means.T)
    prior = np.array([(y == c).mean() for c in classes])
    b = -0.5 * np.einsum("cd,dc->c", means, SiM) + np.log(prior + 1e-12)
    return torch.tensor(SiM.T, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


def m_biodino_ind(s_joint, s_y, q_joint, q_bio, text_protos):
    qt = torch.tensor(q_joint, dtype=torch.float32)
    qb = torch.tensor(q_bio, dtype=torch.float32)
    W, b = fit_fisher_robust(s_joint, s_y)
    prob, _, txt = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=qt,
        bioclip_features=qb, text_prototypes=text_protos,
        scale=SCALE, w_lda=W_LDA, device="cpu")
    return prob.argmax(-1).numpy(), prob, txt, qt


def m_biodino_trans(s_joint, s_y, q_joint, q_bio, text_protos, C):
    pred, prob, txt, qt = m_biodino_ind(s_joint, s_y, q_joint, q_bio, text_protos)
    rp, _, _ = refine_transductive(
        support_features=s_joint, support_labels=s_y, query_features=qt,
        initial_preds=pred, confidences=prob.max(dim=-1).values,
        text_prob=txt, n_classes=C, support_weight=3, max_anchors=20,
        conf_threshold=0.70, device="cpu", w_lda=W_LDA)
    return rp


METHODS = ["zero_shot_clip", "tip_adapter", "protonet", "simpleshot",
           "biodino_inductive", "biodino_transductive"]


def run_episode(si, qi, s_y, q_y, C, bio, dino, joint, text_protos):
    s_bio, s_dino, s_joint = bio[si], dino[si], joint[si]
    q_bio, q_dino, q_joint = bio[qi], dino[qi], joint[qi]
    preds = {
        "zero_shot_clip": m_zero_shot(q_bio, text_protos),
        "tip_adapter": m_tip_adapter(s_bio, s_y, q_bio, text_protos, C),
        "protonet": m_protonet(s_joint, s_y, q_joint, C),
        "simpleshot": m_simpleshot(s_joint, s_y, q_joint, C),
        "biodino_inductive": m_biodino_ind(s_joint, s_y, q_joint, q_bio, text_protos)[0],
        "biodino_transductive": m_biodino_trans(s_joint, s_y, q_joint, q_bio, text_protos, C),
    }
    out = {}
    for m, p in preds.items():
        out[m] = dict(acc=float(100 * accuracy_score(q_y, p)),
                      f1=float(100 * f1_score(q_y, p, average="macro", zero_division=0)))
    return out


def summarize(ep_list):
    res = {}
    for m in METHODS:
        accs = [e[m]["acc"] for e in ep_list]
        f1s = [e[m]["f1"] for e in ep_list]
        res[m] = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                      acc_peak=float(np.max(accs)), f1_mean=float(np.mean(f1s)),
                      f1_std=float(np.std(f1s)),
                      ci95_low=float(np.mean(accs) - 1.96 * np.std(accs) / np.sqrt(len(accs))),
                      ci95_high=float(np.mean(accs) + 1.96 * np.std(accs) / np.sqrt(len(accs))))
    return res


def run_5way(K, n_ep, bio, dino, joint, text_protos, class_idx, C):
    eps = []
    rng = random.Random(SEED)
    for ep in range(n_ep):
        sampled = rng.sample(range(C), 5)
        cmap = {orig: loc for loc, orig in enumerate(sampled)}
        si, qi, sy, qy = [], [], [], []
        for orig in sampled:
            sh = class_idx[orig].copy()
            rng.shuffle(sh)
            si.extend(sh[:K])
            qi.extend(sh[K:])
            sy.extend([cmap[orig]] * K)
            qy.extend([cmap[orig]] * (len(sh) - K))
        si, qi = np.array(si), np.array(qi)
        sy, qy = np.array(sy), np.array(qy)
        eps.append(run_episode(si, qi, sy, qy, 5, bio, dino, joint, text_protos[sampled]))
    return summarize(eps), eps


def main():
    t0 = time.time()
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    bio, dino = data["bio_feats"].numpy(), data["dino_feats"].numpy()
    labels = data["labels"].numpy()
    text_protos = data["bio_text_protos"]
    classes = data["novel_classes"]
    C = len(classes)
    joint = np.concatenate([bio, dino], axis=-1)
    class_idx = [np.where(labels == c)[0].tolist() for c in range(C)]
    print(f"[*] Loaded {len(labels)} images, {C} classes, joint dim {joint.shape[1]}", flush=True)

    # ---- R1: 11-class simultaneous, canonical 30 episodes ----
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
        ep11.append(run_episode(si, qi, labels[si], labels[qi], C,
                                bio, dino, joint, text_protos))
    r11 = summarize(ep11)
    print(f"[R1] 11-class 30-ep done in {time.time()-t0:.1f}s", flush=True)
    for m in METHODS:
        print(f"     {m:22s} acc {r11[m]['acc_mean']:.2f} +/- {r11[m]['acc_std']:.2f} "
              f"(peak {r11[m]['acc_peak']:.2f}) | F1 {r11[m]['f1_mean']:.2f}", flush=True)

    # ---- R2: 5-way K-shot episodic sweep (matches VSF / AgriFewNet regimes) ----
    K_GRID = [1, 5, 10, 15]
    r5_all, ep5_all = {}, {}
    for K in K_GRID:
        r5, ep5 = run_5way(K, N_EP_5WAY, bio, dino, joint, text_protos, class_idx, C)
        r5_all[K] = r5
        ep5_all[K] = ep5
        print(f"[R2] 5-way {K}-shot {N_EP_5WAY}-ep done in {time.time()-t0:.1f}s", flush=True)
        for m in METHODS:
            print(f"     {m:22s} acc {r5[m]['acc_mean']:.2f} +/- {r5[m]['acc_std']:.2f} "
                  f"(peak {r5[m]['acc_peak']:.2f}) | F1 {r5[m]['f1_mean']:.2f}", flush=True)

    payload = dict(
        dataset="PlantVillage 11-class novel subset",
        n_images=int(len(labels)), n_classes=C, class_names=classes,
        provenance="All baselines re-run by us on identical frozen BioCLIP+DINOv2 "
                   "embeddings, identical splits/seeds/K-shot/protocol as BioDINO-Edge.",
        backbones=dict(dinov2="timm vit_small_patch14_dinov2 (ViT-S/14, 384-D)",
                       bioclip="open_clip hf-hub:imageomics/bioclip (ViT-B/16, 512-D)"),
        protocol=dict(seed=SEED, k_shot=K_SHOT, scale=SCALE, w_lda=W_LDA,
                      episode_seed_11="42 + ep*17", n_ep_11=N_EP_11,
                      n_ep_5way=N_EP_5WAY, tip_alpha=TIP_ALPHA, tip_beta=TIP_BETA,
                      transductive=dict(support_weight=3, max_anchors=20, conf_threshold=0.70)),
        regime_11class=dict(n_episodes=N_EP_11, summary=r11, per_episode=ep11),
        regime_5way={f"{K}shot": dict(n_episodes=N_EP_5WAY, summary=r5_all[K], per_episode=ep5_all[K]) for K in K_GRID},
        k_grid=K_GRID,
        runtime_seconds=round(time.time() - t0, 1),
    )
    out = HERE / "controlled_baseline_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] wrote {out} ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()