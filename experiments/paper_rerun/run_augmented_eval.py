# -*- coding: utf-8 -*-
"""
Method-consistent re-evaluation of BioDINO-Edge.

Manuscript Section 3.3.1 specifies 8-view augmented support expansion
(K=5 -> N = 40C augmented views per episode). The stored benchmark engine
(pipeline/benchmark_engine.py) instead consumes raw cached embeddings
(one view per support leaf), so its 30-episode means are not produced by the
described pipeline. This script evaluates the pipeline as described:

  * each episode draws 5 support leaves per class at random,
    expands them to 8 augmented views, and scores the remaining images
  * Fisher-LDA with Ledoit-Wolf shrinkage + text-prior ensemble
  * transductive recalibration at the deployed constants (3x, 0.70, 20)

Query features are extracted once (single canonical view per image) and reused
across episodes, so episodes are paired by construction.

Output: experiments/paper_rerun/results/augmented_eval_<date>.json
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from biodino_edge_pipeline import build_text_prototypes, load_backbones  # noqa: E402
from data_engine import (  # noqa: E402
    build_augmentation_pipeline, extract_norm_stats, load_plantdoc, load_plantvillage,
)
from core_math import (  # noqa: E402
    calculate_shannon_confidence, compute_class_prototypes,
    compute_fisher_ensemble_probs, fit_fisher_lda, refine_transductive,
)

DATE = "2026-11-17"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
N_VIEWS = 8
SEED = 42
CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"


class Extractor:
    def __init__(self, models, device):
        self.models = models
        self.device = device
        self.aug_bio = build_augmentation_pipeline(*extract_norm_stats(models.bioclip_transforms))
        self.aug_dino = build_augmentation_pipeline(models.dino_config["mean"],
                                                   models.dino_config["std"])

    def single(self, paths, tag=""):
        bio_out, dino_out = [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(paths), 64):
                chunk = paths[i:i + 64]
                imgs = [Image.open(p).convert("RGB") for p in chunk]
                b = torch.stack([self.models.bioclip_transforms(im) for im in imgs]).to(self.device)
                d = torch.stack([self.models.dino_transforms(im) for im in imgs]).to(self.device)
                bio_out.append(F.normalize(self.models.bioclip_model.encode_image(b), dim=-1).cpu())
                dino_out.append(F.normalize(self.models.dino_model(d), dim=-1).cpu())
        print(f"    [{tag}] {len(paths)} queries in {time.time()-t0:.0f}s", flush=True)
        return torch.cat(bio_out).numpy(), torch.cat(dino_out).numpy()

    def views(self, paths, n_views=N_VIEWS, tag=""):
        bio_out, dino_out = [], []
        t0 = time.time()
        with torch.no_grad():
            for i, p in enumerate(paths):
                # deterministic per-image stream, matching the deployed pipeline
                torch.manual_seed(SEED + 10_000 + i)
                img = Image.open(p).convert("RGB")
                b_views = [self.models.bioclip_transforms(img)]
                d_views = [self.models.dino_transforms(img)]
                for _ in range(n_views - 1):
                    b_views.append(self.aug_bio(img))
                    d_views.append(self.aug_dino(img))
                bio_out.append(F.normalize(self.models.bioclip_model.encode_image(
                    torch.stack(b_views).to(self.device)), dim=-1).cpu())
                dino_out.append(F.normalize(self.models.dino_model(
                    torch.stack(d_views).to(self.device)), dim=-1).cpu())
        print(f"    [{tag}] {len(paths)} leaves x{n_views} views in {time.time()-t0:.0f}s", flush=True)
        return torch.cat(bio_out).numpy(), torch.cat(dino_out).numpy()


def score(sup_bio, sup_dino, sup_y, q_bio, q_dino, n_classes, text_protos,
          support_weight=3, max_anchors=20, conf_threshold=0.70,
          scale=20.0, w_lda=0.75, tau=0.35):
    s_joint = np.concatenate([sup_bio, sup_dino], axis=-1)
    W, b, _ = fit_fisher_lda(s_joint, sup_y, device="cpu")

    q_bio_t, q_dino_t = torch.tensor(q_bio), torch.tensor(q_dino)
    dino_protos = compute_class_prototypes(torch.tensor(sup_dino), torch.tensor(sup_y), n_classes)
    bio_vis = compute_class_prototypes(torch.tensor(sup_bio), torch.tensor(sup_y), n_classes)
    bio_protos = F.normalize(0.80 * bio_vis + 0.20 * text_protos, dim=-1)
    prob1 = F.softmax(scale * (q_dino_t @ dino_protos.t()), dim=-1)
    conf1 = calculate_shannon_confidence(prob1, n_classes)
    pred1 = prob1.argmax(-1).numpy()

    dino_logits = scale * (q_dino_t @ dino_protos.t())
    bio_logits = scale * (q_bio_t @ bio_protos.t())
    text_logits = scale * (q_bio_t @ text_protos.t())

    joint = torch.cat([q_bio_t, q_dino_t], dim=-1)
    ens, _, text_p = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=joint,
        bioclip_features=q_bio_t, text_prototypes=text_protos,
        scale=scale, w_lda=w_lda, device="cpu")
    pred_ind = ens.argmax(-1).numpy()

    deep = (conf1 < tau).numpy()
    pred_casc = pred1.copy()
    pred_casc[deep] = pred_ind[deep]

    pred_tr, _, _ = refine_transductive(
        support_features=s_joint, support_labels=sup_y, query_features=joint,
        initial_preds=pred_ind, confidences=ens.max(-1).values, text_prob=text_p,
        n_classes=n_classes, support_weight=support_weight, max_anchors=max_anchors,
        conf_threshold=conf_threshold, device="cpu", w_lda=w_lda)
    return {"tier1": pred1, "cascaded": pred_casc, "inductive": pred_ind,
            "transductive": pred_tr, "text_only": text_logits.argmax(-1).numpy(),
            "dino_only": dino_logits.argmax(-1).numpy(),
            "bio_only": bio_logits.argmax(-1).numpy(),
            "static_blend": (0.65 * bio_logits + 0.35 * dino_logits).argmax(-1).numpy(),
            "fast_exit_pct": float((~deep).mean() * 100)}


def m(pred, y):
    return {"acc": float(accuracy_score(y, pred) * 100),
            "f1": float(f1_score(y, pred, average="macro", zero_division=0) * 100)}


def summarize(a):
    a = np.asarray(a, float)
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "std": float(s),
            "ci95": [float(a.mean() - 1.96 * s / np.sqrt(len(a))),
                     float(a.mean() + 1.96 * s / np.sqrt(len(a)))],
            "peak": float(a.max()), "n": int(len(a))}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device={device}", flush=True)
    models = load_backbones(device)
    ex = Extractor(models, device)
    out = {"date": DATE, "protocol": {
        "episodes_11class": 30, "episodes_5way": 50, "n_views": N_VIEWS, "seed": SEED,
        "scale": 20.0, "w_lda": 0.75, "tau": 0.35,
        "transductive": {"support_weight": 3, "max_anchors": 20, "conf_threshold": 0.70},
        "note": "support expanded to 8 augmented views per leaf, as in Section 3.3.1"}}

    # ---------- PlantVillage ----------
    cache = torch.load(CACHE, map_location="cpu")
    classes = cache["novel_classes"]
    n_classes = len(classes)
    text_protos = cache["bio_text_protos"]
    all_bio = cache["bio_feats"].numpy()
    all_dino = cache["dino_feats"].numpy()
    all_y = cache["labels"].numpy()
    paths = cache["paths"]
    print(f"[PV] {len(paths)} images, {n_classes} classes", flush=True)

    ind, tr, t1, casc = [], [], [], []
    for ep in range(30):
        rng = random.Random(SEED + ep * 17)
        sup_idx, qry_idx = [], []
        for c in range(n_classes):
            c_idx = np.where(all_y == c)[0].tolist()
            rng.shuffle(c_idx)
            sup_idx.extend(c_idx[:5])
            qry_idx.extend(c_idx[5:])
        sup_paths = [paths[i] for i in sup_idx]
        sb, sd = ex.views(sup_paths, N_VIEWS, f"PV-ep{ep}-support")
        sy = np.repeat(all_y[sup_idx], N_VIEWS)
        res = score(sb, sd, sy, all_bio[qry_idx], all_dino[qry_idx], n_classes, text_protos)
        for key, bucket in (("inductive", ind), ("transductive", tr),
                             ("tier1", t1), ("cascaded", casc)):
            bucket.append(m(res[key], all_y[qry_idx])["acc"])
        print(f"  [PV] ep{ep:02d} inductive={ind[-1]:.2f} transductive={tr[-1]:.2f}", flush=True)

    out["plantvillage"] = {"11class_inductive": summarize(ind),
                           "11class_transductive": summarize(tr),
                           "11class_tier1": summarize(t1),
                           "11class_cascaded": summarize(casc)}

    # 5-way K-shot
    for K in (1, 5):
        rng = random.Random(SEED)
        a_ind, a_tr = [], []
        for ep in range(50):
            sampled = rng.sample(range(n_classes), 5)
            mp = {o: l for l, o in enumerate(sampled)}
            sup_idx, qry_idx = [], []
            for c in sampled:
                c_idx = np.where(all_y == c)[0].tolist()
                random.Random(1000 * K + ep).shuffle(c_idx)
                sup_idx.extend(c_idx[:K])
                qry_idx.extend(c_idx[K:])
            sb, sd = ex.views([paths[i] for i in sup_idx], N_VIEWS, f"PV-5way-K{K}-ep{ep}")
            sy = np.array([mp[y] for y in all_y[sup_idx]])
            sy = np.repeat(sy, N_VIEWS)
            res = score(sb, sd, sy, all_bio[qry_idx], all_dino[qry_idx], 5,
                        text_protos[sampled])
            qy = np.array([mp[y] for y in all_y[qry_idx]])
            a_ind.append(m(res["inductive"], qy)["acc"])
            a_tr.append(m(res["transductive"], qy)["acc"])
        out["plantvillage"][f"5way_{K}shot_inductive"] = summarize(a_ind)
        out["plantvillage"][f"5way_{K}shot_transductive"] = summarize(a_tr)
        print(f"[PV] 5way K={K}: ind={np.mean(a_ind):.2f} tr={np.mean(a_tr):.2f}", flush=True)

    # ---------- PlantDoc ----------
    psup, pqry, pclasses, pprompts = load_plantdoc(ROOT / "plantdoc", k_shot=5, seed=SEED)
    pn = len(pclasses)
    ptext = build_text_prototypes(pclasses, pprompts, models.bioclip_model,
                                  models.tokenizer, device)
    sb, sd = ex.views([p for p, _ in psup], N_VIEWS, "PD-27way-support")
    psy = np.repeat(np.array([y for _, y in psup]), N_VIEWS)
    pqb, pqd = ex.single([p for p, _ in pqry], "PD-query")
    pqy = np.array([y for _, y in pqry])
    res = score(sb, sd, psy, pqb, pqd, pn, ptext, support_weight=1, max_anchors=2,
                conf_threshold=None)
    pqy_np = np.array(pqy)
    out["plantdoc"] = {"27way_" + k: m(v, pqy_np) for k, v in res.items()
                       if k != "fast_exit_pct"}
    print(f"[PD] 27way " + "  ".join(
        f"{k}={out['plantdoc']['27way_'+k]['acc']:.2f}"
        for k in ("text_only", "dino_only", "bio_only", "static_blend",
                  "inductive", "transductive")), flush=True)

    # 5-way 5-shot on PlantDoc
    srng = random.Random(SEED)
    a_ind, a_tr, a_f1, a_peak = [], [], [], []
    for ep in range(50):
        sampled = srng.sample(range(pn), 5)
        mp = {o: l for l, o in enumerate(sampled)}
        ep_sup = [(p, mp[y]) for p, y in psup if y in sampled]
        ep_qry = [(p, mp[y]) for p, y in pqry if y in sampled]
        if not ep_qry:
            continue
        eb, ed = ex.views([p for p, _ in ep_sup], N_VIEWS, f"PD-5way-ep{ep}")
        ey = np.repeat(np.array([y for _, y in ep_sup]), N_VIEWS)
        qb2, qd2 = ex.single([p for p, _ in ep_qry], f"PD-5way-ep{ep}-q")
        qy2 = np.array([y for _, y in ep_qry])
        r = score(eb, ed, ey, qb2, qd2, 5, ptext[sampled],
                  support_weight=2, max_anchors=2, conf_threshold=None)
        a_ind.append(m(r["inductive"], qy2)["acc"])
        a_tr.append(m(r["transductive"], qy2)["acc"])
        a_f1.append(m(r["transductive"], qy2)["f1"])
        a_peak.append(m(r["transductive"], qy2)["acc"])
    out["plantdoc"]["5way_5shot_inductive"] = summarize(a_ind)
    out["plantdoc"]["5way_5shot_transductive"] = summarize(a_tr)
    out["plantdoc"]["5way_5shot_macro_f1"] = summarize(a_f1)

    p = OUT / f"augmented_eval_{DATE}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[OK] {p}\n{json.dumps(out, indent=2)}")


if __name__ == "__main__":
    main()
