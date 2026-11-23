# -*- coding: utf-8 -*-
"""
Augmentation-dependence sensitivity study (reviewer response experiment).
================================================================================
Addresses the reviewer concern that the 440 augmented support vectors
(55 leaves x 8 views) are NOT statistically independent: multiple views of
the same source image are correlated pseudo-replicates, which violates the
i.i.d. assumption behind the analytical Ledoit-Wolf shrinkage intensity and
inflates the nominal N-C degrees of freedom.

Two one-factor-at-a-time sweeps over the canonical 30-episode protocol
(seed 42 + ep*17, stratified 5-shot splits, cached query embeddings):

  A. View-count sweep (30 episodes): v in {1, 2, 4, 8} views per support
     image, full augmentation pipeline. v=1 is the canonical (unaugmented)
     view; v=8 reproduces the paper's N=440 configuration. Views are nested
     prefixes of ONE seeded 8-view extraction per episode.
  B. Augmentation-type sweep (first --type-episodes episodes, 8 views):
     geometric-only (flips + rotation) vs photometric-only (ColorJitter)
     vs the full pipeline (condition views8 from sweep A).
  C. Replicate-8 control (30 episodes, no encoding cost): the canonical
     embedding tiled 8x per image (N=440 pure pseudo-replicates, zero
     within-image diversity) - isolates the effect of replication alone on
     the Ledoit-Wolf intensity.

Per condition and episode we report:
  - Ledoit-Wolf shrinkage intensity alpha (mean/min/max over the 11 classes).
    sklearn 1.9.1's lsqr solver applies LW per class in standardized space
    (StandardScaler -> ledoit_wolf -> rescale) and pools with class priors;
    the fitted object exposes covariance_ but not the intensities, so we
    replicate that exact path here to expose them.
  - condition number of the pooled within-class covariance (eig ratio of
    clf.covariance_, the matrix actually solved by lstsq)
  - inductive accuracy / macro-F1 (Fisher-LDA + text ensemble, 0.75/0.25)
  - transductive accuracy / macro-F1 (3x support / 20 anchors / tau=0.70)

Augmentation RNG is re-seeded per (episode, pass) for reproducibility.
Outputs: results/augmentation_sensitivity.json (incremental per-episode save).

Run with the project venv:
  .venv\\Scripts\\python.exe experiments\\augmentation_sensitivity\\run_augmentation_sensitivity.py
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.covariance import ledoit_wolf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.core_math import (  # noqa: E402
    fit_fisher_lda, compute_fisher_ensemble_probs, refine_transductive,
)
from pipeline.data_engine import extract_norm_stats, build_augmentation_pipeline  # noqa: E402
from pipeline.biodino_edge_pipeline import load_backbones  # noqa: E402

CACHE = _PROJECT_ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
HERE = Path(__file__).resolve().parent
SEED = 42
K_SHOT = 5
SCALE = 20.0
W_LDA = 0.75
TRANS_DEFAULTS = dict(support_weight=3, max_anchors=20, conf_threshold=0.70)


def sample_episode(class_indices, n_classes, ep):
    """Replicates benchmark_engine.run_plantvillage_montecarlo episode splits."""
    rng = random.Random(SEED + ep * 17)
    support_indices, query_indices = [], []
    for c in range(n_classes):
        shuffled = class_indices[c].copy()
        rng.shuffle(shuffled)
        support_indices.extend(shuffled[:K_SHOT])
        query_indices.extend(shuffled[K_SHOT:])
    return support_indices, query_indices


def build_type_pipeline(mean, std, kind):
    """Augmentation variants: 'full' (paper), 'geo' (flips+rotation), 'photo' (jitter)."""
    if kind == "full":
        return build_augmentation_pipeline(mean, std)
    if kind == "geo":
        ops = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15)]
    elif kind == "photo":
        ops = [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
    else:
        raise ValueError(f"unknown augmentation kind: {kind}")
    return T.Compose([T.Resize((224, 224))] + ops + [T.ToTensor(), T.Normalize(mean=mean, std=std)])


def extract_views(support_samples, models, device, n_views, kind, seed):
    """Seeded multi-view support extraction.

    View 0 is always the canonical (unaugmented) transform, views 1..n_views-1
    are stochastic passes of the selected pipeline - mirrors
    biodino_edge_pipeline.extract_support_features (which uses n_views=8,
    kind='full'). Returns (bio, dino, labels) with row order
    [img0_view0, img0_view1, ..., img1_view0, ...].
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    bio_mean, bio_std = extract_norm_stats(models.bioclip_transforms)
    aug_bio = build_type_pipeline(bio_mean, bio_std, kind)
    aug_dino = build_type_pipeline(models.dino_config["mean"], models.dino_config["std"], kind)

    bio_list, dino_list, lab_list = [], [], []
    with torch.no_grad():
        for image_path, class_label in support_samples:
            image = Image.open(image_path).convert("RGB")
            bio_views = [models.bioclip_transforms(image)] + [aug_bio(image) for _ in range(n_views - 1)]
            dino_views = [models.dino_transforms(image)] + [aug_dino(image) for _ in range(n_views - 1)]
            bio_list.append(F.normalize(models.bioclip_model.encode_image(torch.stack(bio_views).to(device)), dim=-1))
            dino_list.append(F.normalize(models.dino_model(torch.stack(dino_views).to(device)), dim=-1))
            lab_list.append(torch.full((n_views,), class_label, dtype=torch.long))
    return torch.cat(bio_list, dim=0), torch.cat(dino_list, dim=0), torch.cat(lab_list, dim=0)


def evaluate_condition(s_x, s_y, q_x, q_y, q_t, q_bio, text_protos, n_classes):
    """Fit Phase-0 Fisher-LDA, record covariance diagnostics, run both modes."""
    W, b, clf = fit_fisher_lda(s_x, s_y, device="cpu")

    # Replicate sklearn 1.9.1 _class_cov/_cov(shrinkage='auto') to expose the
    # per-class Ledoit-Wolf intensities (not stored on the fitted estimator).
    alphas = []
    for c in np.unique(s_y):
        Xs = StandardScaler().fit_transform(s_x[s_y == c])
        _, alpha = ledoit_wolf(Xs)
        alphas.append(float(alpha))
    cov = np.asarray(clf.covariance_, dtype=np.float64)
    eig = np.linalg.eigvalsh(cov)
    cond_number = float(eig[-1] / max(eig[0], 1e-300))

    prob, _, text_prob = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=q_t,
        bioclip_features=q_bio, text_prototypes=text_protos,
        scale=SCALE, w_lda=W_LDA, device="cpu")
    ind_pred = prob.argmax(dim=-1).numpy()
    ind_conf = prob.max(dim=-1).values

    tr_pred, _, _ = refine_transductive(
        support_features=s_x, support_labels=s_y, query_features=q_t,
        initial_preds=ind_pred, confidences=ind_conf, text_prob=text_prob,
        n_classes=n_classes, device="cpu", w_lda=W_LDA, **TRANS_DEFAULTS)

    return dict(
        n_support=int(len(s_y)),
        lw_alpha=float(np.mean(alphas)),
        lw_alpha_min=float(np.min(alphas)),
        lw_alpha_max=float(np.max(alphas)),
        cov_cond=cond_number,
        ind_acc_pct=float(accuracy_score(q_y, ind_pred) * 100),
        ind_f1_pct=float(f1_score(q_y, ind_pred, average="macro", zero_division=0) * 100),
        tr_acc_pct=float(accuracy_score(q_y, tr_pred) * 100),
        tr_f1_pct=float(f1_score(q_y, tr_pred, average="macro", zero_division=0) * 100),
    )


def aggregate(records):
    """Mean +/- population std (ddof=0), matching the paper's convention."""
    keys = ["lw_alpha", "lw_alpha_min", "lw_alpha_max", "cov_cond",
            "ind_acc_pct", "ind_f1_pct", "tr_acc_pct", "tr_f1_pct"]
    out = {"n_episodes": len(records)}
    for k in keys:
        vals = np.asarray([r[k] for r in records], dtype=np.float64)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std(ddof=0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=30, help="episodes for the view-count sweep")
    ap.add_argument("--type-episodes", type=int, default=10, help="episodes for the geo/photo type sweep")
    ap.add_argument("--views", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke", action="store_true", help="1 episode, views 1/8 + geo/photo, with sanity checks")
    ap.add_argument("--out", type=str, default=str(HERE / "results" / "augmentation_sensitivity.json"))
    args = ap.parse_args()

    if args.smoke:
        args.episodes, args.type_episodes, args.views = 1, 1, [1, 8]

    t_start = time.time()
    device = torch.device(args.device)
    data = torch.load(CACHE, map_location="cpu", weights_only=True)
    bio, dino = data["bio_feats"], data["dino_feats"]
    labels = data["labels"].numpy()
    paths = data["paths"]
    class_names = data["novel_classes"]
    n_classes = len(class_names)
    joint = torch.cat([bio, dino], dim=-1).numpy()
    text_protos = data["bio_text_protos"]
    class_indices = [np.where(labels == c)[0].tolist() for c in range(n_classes)]

    print(f"[*] Cache loaded: {len(labels)} embeddings, {n_classes} classes", flush=True)
    models = load_backbones(device)
    print(f"[*] Backbones ready on {device} ({time.time() - t_start:.1f}s)", flush=True)

    conditions = {}   # name -> list of per-episode records
    ep_meta = []      # per-episode diagnostics (canonical-view fidelity, timings)

    def record(name, ep, rec):
        conditions.setdefault(name, []).append(dict(ep=ep, **rec))

    for ep in range(args.episodes):
        t_ep = time.time()
        support_indices, query_indices = sample_episode(class_indices, n_classes, ep)
        s_y_cached = labels[support_indices]
        support_samples = [(paths[i], int(labels[i])) for i in support_indices]
        q_x, q_y = joint[query_indices], labels[query_indices]
        q_t = torch.tensor(q_x, dtype=torch.float32)
        q_bio = bio[query_indices]
        eval_args = (q_x, q_y, q_t, q_bio, text_protos, n_classes)

        # ---- A. view-count sweep: ONE seeded 8-view full extraction, nested prefixes ----
        n_max = max(args.views)
        bio8, dino8, lab8 = extract_views(support_samples, models, device, n_max, "full", seed=SEED + ep * 17)
        joint8 = torch.cat([bio8, dino8], dim=-1).numpy()
        lab8 = lab8.numpy()
        assert np.array_equal(lab8.reshape(len(support_samples), n_max)[:, 0], s_y_cached), \
            "episode support labels disagree with cache labels"

        # Sanity: canonical extracted view must reproduce the cached embedding
        canon = joint8.reshape(len(support_samples), n_max, -1)[:, 0, :]
        canon_err = float(np.abs(canon - joint[support_indices]).max())
        if ep == 0:
            print(f"[check] canonical-view vs cache max abs diff: {canon_err:.3e}", flush=True)
            assert canon_err < 5e-3, "canonical extraction does not match cached embeddings"

        for v in args.views:
            sel = joint8.reshape(len(support_samples), n_max, -1)[:, :v, :].reshape(-1, joint8.shape[1])
            sel_y = lab8.reshape(len(support_samples), n_max)[:, :v].reshape(-1)
            record(f"views{v}", ep, evaluate_condition(sel, sel_y, *eval_args))

        # ---- C. replicate-8 control: canonical embedding tiled 8x (no diversity) ----
        rep = np.repeat(joint[support_indices], 8, axis=0)
        rep_y = np.repeat(s_y_cached, 8)
        record("replicate8", ep, evaluate_condition(rep, rep_y, *eval_args))

        # ---- B. type sweep: separate seeded 8-view extractions (geo / photo) ----
        if ep < args.type_episodes:
            for kind in ("geo", "photo"):
                bk, dk, lk = extract_views(support_samples, models, device, 8, kind, seed=SEED + ep * 17 + 1000 * (1 if kind == "geo" else 2))
                record(f"{kind}8", ep, evaluate_condition(
                    torch.cat([bk, dk], dim=-1).numpy(), lk.numpy(), *eval_args))

        ep_meta.append(dict(ep=ep, canonical_view_max_abs_diff=canon_err,
                            episode_seconds=round(time.time() - t_ep, 1)))
        print(f"[ep {ep + 1}/{args.episodes}] views8 ind/tr acc: "
              f"{conditions['views8'][-1]['ind_acc_pct']:.2f}/{conditions['views8'][-1]['tr_acc_pct']:.2f} | "
              f"alpha={conditions['views8'][-1]['lw_alpha']:.4f} | {time.time() - t_ep:.1f}s "
              f"(total {time.time() - t_start:.0f}s)", flush=True)

        # Incremental save so partial runs are never lost
        partial = dict(
            protocol=dict(seed=SEED, seed_rule="42 + ep*17", k_shot=K_SHOT, scale=SCALE, w_lda=W_LDA,
                          transductive=TRANS_DEFAULTS, episodes_requested=args.episodes,
                          episodes_completed=ep + 1, type_episodes=min(args.type_episodes, ep + 1),
                          views=args.views, device=str(device), cache=str(CACHE),
                          augmentation_rng="random/numpy/torch re-seeded per (episode, pass)"),
            conditions={k: dict(per_episode=v, aggregate=aggregate(v)) for k, v in conditions.items()},
            episode_meta=ep_meta,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    result = partial
    result["protocol"]["episodes_completed"] = args.episodes
    result["total_seconds"] = round(time.time() - t_start, 1)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== AGGREGATE (mean +/- population std) ===", flush=True)
    for name, agg in ((k, v["aggregate"]) for k, v in result["conditions"].items()):
        print(f"{name:>11} (n={agg['n_episodes']:2d}): "
              f"alpha={agg['lw_alpha_mean']:.4f}+/-{agg['lw_alpha_std']:.4f} | "
              f"cond={agg['cov_cond_mean']:.3e} | "
              f"ind={agg['ind_acc_pct_mean']:.2f}+/-{agg['ind_acc_pct_std']:.2f} | "
              f"tr={agg['tr_acc_pct_mean']:.2f}+/-{agg['tr_acc_pct_std']:.2f}", flush=True)
    print(f"\n[SUCCESS] saved {out_path} (total {result['total_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()