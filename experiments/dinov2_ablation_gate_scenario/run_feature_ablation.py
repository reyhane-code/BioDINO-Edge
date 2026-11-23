# -*- coding: utf-8 -*-
"""
DINOv2 Feature-Contribution Ablation for the Regularized Fisher-LDA.
====================================================================
Question: how much does DINOv2 actually contribute to the Fisher classifier?

We re-run the EXACT 30-episode benchmark protocol (seed 42 + ep*17, raw 5-shot
support, cached embeddings, Ledoit-Wolf Fisher + soft-anchored transductive
recalibration with support_weight=3 / max_anchors=20 / conf_threshold=0.70) on
three feature spaces:

  fused : [BioCLIP(512) || DINOv2(384)] = 896-D   (the paper's deployed system)
  bio   : BioCLIP only                  = 512-D   (DINOv2 REMOVED from Fisher)
  dino  : DINOv2 only                   = 384-D   (BioCLIP REMOVED from Fisher)

For each space we report two operating modes:
  pure_lda : w_lda = 1.00  (Fisher only, no text blend) -> isolates the raw
             discriminative power of each feature space.
  ensemble : w_lda = 0.75  (+ 0.25 BioCLIP taxonomic text) -> the deployed system.

The text branch ALWAYS uses BioCLIP image features + the same text prototypes,
so the ONLY thing that varies is the Fisher input. The `fused/ensemble` row must
reproduce the published 97.41% transductive mean (internal sanity check).

DINOv2's marginal contribution = fused - bio (paired over the same 30 episodes).

Outputs: feature_ablation_results.json (this directory).
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats
from sklearn.metrics import accuracy_score, f1_score

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT), str(_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.core_math import (  # noqa: E402
    fit_fisher_lda, compute_fisher_ensemble_probs, refine_transductive,
)

CACHE = _ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
OUT = Path(__file__).resolve().parent / "feature_ablation_results.json"
N_EPISODES, SEED, K_SHOT = 30, 42, 5
SCALE = 20.0
REFINE = dict(support_weight=3, max_anchors=20, conf_threshold=0.70)
MODES = {"pure_lda": 1.00, "ensemble": 0.75}


def build_episodes():
    data = torch.load(CACHE, map_location="cpu")
    bio, dino = data["bio_feats"], data["dino_feats"]
    labels = data["labels"].numpy()
    text_protos = data["bio_text_protos"]
    class_names = data["novel_classes"]
    n_classes = len(class_names)
    spaces = {
        "fused": torch.cat([bio, dino], dim=-1).numpy(),
        "bio": bio.numpy(),
        "dino": dino.numpy(),
    }
    class_indices = [np.where(labels == c)[0].tolist() for c in range(n_classes)]
    episodes = []
    for ep in range(N_EPISODES):
        rng = random.Random(SEED + ep * 17)
        si, qi = [], []
        for c in range(n_classes):
            sh = class_indices[c].copy()
            rng.shuffle(sh)
            si.extend(sh[:K_SHOT])
            qi.extend(sh[K_SHOT:])
        episodes.append(dict(ep=ep, si=si, qi=qi, sy=labels[si], y=labels[qi],
                             q_bio=bio[qi], text_protos=text_protos,
                             n_classes=n_classes))
    return episodes, spaces, class_names


def run_space(ep, space_x, w_lda):
    """Inductive fit + transductive refinement on one feature space."""
    s_x, s_y = space_x[ep["si"]], ep["sy"]
    q_t = torch.tensor(space_x[ep["qi"]], dtype=torch.float32)
    W, b, _ = fit_fisher_lda(s_x, s_y, device="cpu")
    prob, _, text_prob = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=q_t,
        bioclip_features=ep["q_bio"], text_prototypes=ep["text_protos"],
        scale=SCALE, w_lda=w_lda, device="cpu")
    ind_pred = prob.argmax(dim=-1).numpy()
    ind_conf = prob.max(dim=-1).values
    ref_pred, _, _ = refine_transductive(
        support_features=s_x, support_labels=s_y, query_features=q_t,
        initial_preds=ind_pred, confidences=ind_conf, text_prob=text_prob,
        n_classes=ep["n_classes"], device="cpu", w_lda=w_lda, **REFINE)
    return ind_pred, ref_pred


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    t, p = sstats.ttest_rel(a, b)
    sd = diff.std(ddof=1)
    return dict(mean_diff_pp=float(diff.mean()),
                paired_t=float(t), paired_p=float(p),
                wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
                ties=int((diff == 0).sum()),
                cohens_dz=float(diff.mean() / sd) if sd > 0 else float("inf"))


def main():
    t0 = time.time()
    episodes, spaces, class_names = build_episodes()
    print(f"[*] {len(episodes)} episodes, {len(episodes[0]['y'])} queries each, "
          f"{len(class_names)} classes ({time.time()-t0:.1f}s)", flush=True)

    results = {}
    per_ep = {}
    for name, space_x in spaces.items():
        results[name] = {}
        per_ep[name] = {}
        for mode, w in MODES.items():
            ia, if1, ta, tf1 = [], [], [], []
            for ep in episodes:
                ip, rp = run_space(ep, space_x, w)
                ia.append(accuracy_score(ep["y"], ip) * 100)
                if1.append(f1_score(ep["y"], ip, average="macro", zero_division=0) * 100)
                ta.append(accuracy_score(ep["y"], rp) * 100)
                tf1.append(f1_score(ep["y"], rp, average="macro", zero_division=0) * 100)
            results[name][mode] = dict(
                inductive_acc_mean=float(np.mean(ia)), inductive_acc_std=float(np.std(ia)),
                transductive_acc_mean=float(np.mean(ta)), transductive_acc_std=float(np.std(ta)),
                inductive_f1_mean=float(np.mean(if1)), transductive_f1_mean=float(np.mean(tf1)))
            per_ep[name][mode] = dict(inductive_acc=ia, transductive_acc=ta)
            print(f"[{name:5s}/{mode:8s}] ind {np.mean(ia):6.2f}+/-{np.std(ia):.2f} | "
                  f"trans {np.mean(ta):6.2f}+/-{np.std(ta):.2f} | "
                  f"transF1 {np.mean(tf1):6.2f}  ({time.time()-t0:.0f}s)", flush=True)

    # DINOv2 marginal contribution = fused - bio ; BioCLIP contribution = fused - dino
    contributions = {}
    for mode in MODES:
        contributions[mode] = dict(
            dinov2_added_to_fisher=paired(per_ep["fused"][mode]["transductive_acc"],
                                          per_ep["bio"][mode]["transductive_acc"]),
            bioclip_added_to_fisher=paired(per_ep["fused"][mode]["transductive_acc"],
                                           per_ep["dino"][mode]["transductive_acc"]),
            dinov2_inductive=paired(per_ep["fused"][mode]["inductive_acc"],
                                    per_ep["bio"][mode]["inductive_acc"]))
    print("\n[CONTRIBUTION] DINOv2 added to Fisher (fused - bio), transductive:", flush=True)
    for mode in MODES:
        c = contributions[mode]["dinov2_added_to_fisher"]
        print(f"    {mode:8s}: {c['mean_diff_pp']:+.2f} pp | t={c['paired_t']:.2f} "
              f"p={c['paired_p']:.3e} | W/L/T={c['wins']}/{c['losses']}/{c['ties']} "
              f"dz={c['cohens_dz']:.2f}", flush=True)

    out = dict(protocol=dict(seed=SEED, episodes=N_EPISODES, k_shot=K_SHOT, scale=SCALE,
                             modes=MODES, refine=REFINE, cache=str(CACHE),
                             feature_dims={k: int(v.shape[1]) for k, v in spaces.items()}),
               results=results, contributions=contributions, per_episode=per_ep)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[SUCCESS] saved {OUT.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
