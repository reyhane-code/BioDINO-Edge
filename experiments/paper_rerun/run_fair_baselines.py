# -*- coding: utf-8 -*-
"""
Fair-baseline experiment: closes the two comparison gaps reviewers will find.

GAP 1. In the paper, zero-shot BioCLIP and Tip-Adapter only ever see the 512-D
BioCLIP branch, while Prototypical Networks and BioDINO-Edge see the 896-D
BioCLIP+DINOv2 concatenation. Any gain could therefore be attributed to the
extra encoder rather than to the adaptation rule. We re-run Tip-Adapter and two
plain linear probes on the SAME 896-D features.

GAP 2. The paper compares its transductive mode only against inductive
baselines and admits it in Section 7.2. We re-run three transductive FSL
methods on the same frozen features and the same episodes.

Everything here is training-free or closed-form on cached features: no backbone
is run, no gradient touches a backbone. Feature matrices come from
cache_artifacts/pv_novel_embeddings.pt (the exact cache used by
run_controlled_baselines.py), so support/query indices and seeds match the
published numbers episode for episode.

Methods
  Inductive
    tip_adapter_512      published configuration (BioCLIP cache + text logits)
    tip_adapter_896      same rule, cache built from 896-D joint features
    ridge_896            closed-form RidgeClassifier on 896-D
    logistic_896         multinomial logistic regression on 896-D
  Transductive (same features, same episodes)
    laplacianshot        Laplacian label propagation (Ziko et al., 2020)
    tim                  transductive information maximization (Boudiaf et al., 2020)
    bdcspn               prototype rectification from confident pseudo-labels
                         (Liu et al., 2021, BD-CSPN)

Outputs per-episode accuracies so paired tests can be recomputed exactly:
  experiments/paper_rerun/results/fair_baselines_2026-11-18.json
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core_math import (  # noqa: E402
    compute_fisher_ensemble_probs, fit_fisher_lda, refine_transductive,
)

CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
DATE = "2026-11-18"

SEED, K_SHOT = 42, 5
SCALE, W_LDA = 20.0, 0.75
N_EP_11, N_EP_5WAY = 30, 50
TIP_ALPHA, TIP_BETA = 1.0, 5.5


def l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# --------------------------------------------------------------- inductive
def m_tip_adapter(s_cache, s_y, q_cache, q_bio, text_protos, C):
    keys = l2norm(s_cache)
    vals = np.eye(C)[s_y]
    sim = l2norm(q_cache) @ keys.T
    adapter = np.exp(-TIP_BETA * (1.0 - sim)) @ vals
    clip_logits = SCALE * (q_bio @ text_protos.numpy().T)
    return (TIP_ALPHA * adapter + clip_logits).argmax(-1)


def m_ridge(s_joint, s_y, q_joint):
    clf = RidgeClassifier(alpha=1.0).fit(s_joint, s_y)
    return clf.predict(q_joint)


def m_logistic(s_joint, s_y, q_joint):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(s_joint, s_y)
    return clf.predict(q_joint)


# ------------------------------------------------------------- transductive
def _affinity(s_all, k=10):
    """Symmetric kNN cosine affinity over pooled support+query rows (sparse).

    Returns a CSR matrix in similarity space so propagation costs O(nnz * C)
    instead of O(n^2 * C); the dense variant is intractable at n ~ 9.4k.
    """
    from scipy.sparse import csr_matrix

    x = l2norm(s_all)
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
            vals.extend(blk[i, row_idx].tolist())
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)
    return A


def m_laplacianshot(s_joint, s_y, q_joint, C, alpha=0.99, k=10, iters=200):
    """LaplacianShot: Laplacian-regularised label propagation over support+query."""
    allx = np.concatenate([s_joint, q_joint], axis=0)
    A = _affinity(allx, k=k)
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    dinv = 1.0 / np.sqrt(d)
    S = (A.multiply(dinv[:, None])).multiply(dinv[None, :]).tocsr()
    Y = np.zeros((len(allx), C))
    Y[np.arange(len(s_y)), s_y] = 1.0
    Y = csr_matrix(Y)
    Fy = Y.copy()
    for _ in range(iters):
        Fy = alpha * (S @ Fy) + (1 - alpha) * Y
    return np.asarray(Fy[len(s_y):].todense()).argmax(-1)


def m_tim(s_joint, s_y, q_joint, C, beta=1.0, steps=150, lr=0.5):
    """TIM: transductive information maximization on frozen-feature logits.

    Minimises  -<p, sim>/n + beta*H(p) - H(mean p), i.e. confident predictions
    with a balanced marginal, without touching the encoder.
    """
    protos = l2norm(np.stack([l2norm(s_joint)[s_y == c].mean(0) for c in range(C)]))
    sim = SCALE * (l2norm(q_joint) @ protos.T)
    W = torch.tensor(sim, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    n = len(sim)
    for _ in range(steps):
        opt.zero_grad()
        p = F.softmax(W, dim=-1)
        ent = -(p * torch.log(p + 1e-8)).sum(-1).mean()
        marginal = p.mean(0)
        ment = -(marginal * torch.log(marginal + 1e-8)).sum()
        loss = -(p * torch.tensor(sim, dtype=torch.float32)).sum() / n + beta * ent - ment
        loss.backward()
        opt.step()
    with torch.no_grad():
        return F.softmax(W, dim=-1).argmax(-1).numpy()


def m_bdcspn(s_joint, s_y, q_joint, C, topk=None):
    """BD-CSPN: rectifies prototypes with confident pseudo-label statistics."""
    fn_s, fn_q = l2norm(s_joint), l2norm(q_joint)
    protos = l2norm(np.stack([fn_s[s_y == c].mean(0) for c in range(C)]))
    sim = fn_q @ protos.T
    pred = sim.argmax(-1)
    conf = sim.max(-1)
    new = protos.copy()
    for c in range(C):
        idx = np.where(pred == c)[0]
        if len(idx) == 0:
            continue
        if topk is not None and len(idx) > topk:
            idx = idx[np.argsort(-conf[idx])[:topk]]
        mu = fn_q[idx].mean(0)
        w = len(idx) / (len(idx) + 5.0)
        new[c] = l2norm((1 - w) * protos[c] + w * l2norm(mu))
    return (fn_q @ new.T).argmax(-1)


# ----------------------------------------------------------------- ours
def m_biodino(s_joint, s_y, q_joint, q_bio, text_protos, C, transductive):
    qt = torch.tensor(q_joint, dtype=torch.float32)
    W, b, _ = fit_fisher_lda(s_joint, s_y, device="cpu")
    prob, _, txt = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=qt,
        bioclip_features=torch.tensor(q_bio, dtype=torch.float32),
        text_prototypes=text_protos, scale=SCALE, w_lda=W_LDA, device="cpu")
    if not transductive:
        return prob.argmax(-1).numpy()
    rp, _, _ = refine_transductive(
        support_features=s_joint, support_labels=s_y, query_features=qt,
        initial_preds=prob.argmax(-1).numpy(), confidences=prob.max(dim=-1).values,
        text_prob=txt, n_classes=C, support_weight=3, max_anchors=20,
        conf_threshold=0.70, device="cpu", w_lda=W_LDA)
    return rp


METHODS = ["tip_adapter_512", "tip_adapter_896", "ridge_896", "logistic_896",
           "protonet_896", "biodino_inductive", "laplacianshot", "tim", "bdcspn",
           "biodino_transductive"]


def m_protonet(s_joint, s_y, q_joint, C):
    fn = l2norm(s_joint)
    protos = l2norm(np.stack([fn[s_y == c].mean(0) for c in range(C)]))
    return (l2norm(q_joint) @ protos.T).argmax(-1)


def run_episode(si, qi, s_y, q_y, C, bio, dino, joint, text_protos):
    s_bio, s_joint = bio[si], joint[si]
    q_bio, q_joint = bio[qi], joint[qi]
    preds = {
        "tip_adapter_512": m_tip_adapter(s_bio, s_y, q_bio, q_bio, text_protos, C),
        "tip_adapter_896": m_tip_adapter(s_joint, s_y, q_joint, q_bio, text_protos, C),
        "ridge_896": m_ridge(s_joint, s_y, q_joint),
        "logistic_896": m_logistic(s_joint, s_y, q_joint),
        "protonet_896": m_protonet(s_joint, s_y, q_joint, C),
        "biodino_inductive": m_biodino(s_joint, s_y, q_joint, q_bio, text_protos, C, False),
        "laplacianshot": m_laplacianshot(s_joint, s_y, q_joint, C),
        "tim": m_tim(s_joint, s_y, q_joint, C),
        "bdcspn": m_bdcspn(s_joint, s_y, q_joint, C),
        "biodino_transductive": m_biodino(s_joint, s_y, q_joint, q_bio, text_protos, C, True),
    }
    return {m: dict(acc=float(100 * accuracy_score(q_y, p)),
                    f1=float(100 * f1_score(q_y, p, average="macro", zero_division=0)))
            for m, p in preds.items()}


def summarize(eps):
    out = {}
    for m in METHODS:
        a = np.array([e[m]["acc"] for e in eps], float)
        f = np.array([e[m]["f1"] for e in eps], float)
        out[m] = dict(acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)),
                      acc_peak=float(a.max()), f1_mean=float(f.mean()),
                      f1_std=float(f.std(ddof=1)))
    return out


def main():
    t0 = time.time()
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    bio = data["bio_feats"].numpy()
    dino = data["dino_feats"].numpy()
    labels = data["labels"].numpy()
    text_protos = data["bio_text_protos"]
    C = len(data["novel_classes"])
    joint = np.concatenate([bio, dino], axis=-1)
    class_idx = [np.where(labels == c)[0].tolist() for c in range(C)]
    print(f"[*] {len(labels)} images, C={C}, joint={joint.shape[1]}-D", flush=True)

    # R1: 11-class simultaneous, canonical 30 episodes (seed 42 + ep*17)
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
    print(f"[R1] 11-class 5-shot x{N_EP_11} in {time.time()-t0:.1f}s", flush=True)

    # R2: 5-way 5-shot, 50 episodes, single random.Random(42) stream
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
        ep5.append(run_episode(np.array(si), np.array(qi), np.array(sy), np.array(qy),
                               5, bio, dino, joint, text_protos[sampled]))
    print(f"[R2] 5-way 5-shot x{N_EP_5WAY} in {time.time()-t0:.1f}s", flush=True)

    payload = {
        "date": DATE,
        "purpose": "fair baselines: 896-D inductive comparators and transductive FSL methods",
        "source_cache": str(CACHE.relative_to(ROOT)),
        "protocol": {"seed": SEED, "k_shot": K_SHOT, "scale": SCALE, "w_lda": W_LDA,
                     "episode_seed_11": "42 + ep*17", "n_ep_11": N_EP_11,
                     "n_ep_5way": N_EP_5WAY, "tip_alpha": TIP_ALPHA, "tip_beta": TIP_BETA,
                     "transductive": {"support_weight": 3, "max_anchors": 20,
                                      "conf_threshold": 0.70},
                     "laplacianshot": {"alpha": 0.99, "knn": 10, "iters": 200},
                     "tim": {"beta": 1.0, "steps": 150, "lr": 0.5},
                     "bdcspn": {"support_prior": 5.0}},
        "regime_11class_5shot": {"summary": summarize(ep11), "per_episode": ep11},
        "regime_5way_5shot": {"summary": summarize(ep5), "per_episode": ep5},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    p = OUT / f"fair_baselines_{DATE}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {p}")

    for name, blk in (("11-class 5-shot", payload["regime_11class_5shot"]),
                      ("5-way 5-shot", payload["regime_5way_5shot"])):
        print(f"\n== {name} ==")
        for m in METHODS:
            s = blk["summary"][m]
            print(f"  {m:22s} {s['acc_mean']:6.2f} +/- {s['acc_std']:5.2f}  F1 {s['f1_mean']:6.2f}")


if __name__ == "__main__":
    main()
