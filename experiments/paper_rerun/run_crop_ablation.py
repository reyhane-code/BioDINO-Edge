# -*- coding: utf-8 -*-
"""
Lesion-cropping ablation on the canonical pipeline.

Compares whole-leaf input against a tight center crop (60% of the shorter side),
which mimics attention-rollout lesion cropping. Both support and query images are
processed identically (8-view augmented support, deployed constants), so the only
variable is host context.

Only pipeline/ code and the PlantDoc/PlantVillage datasets are used.

Output: experiments/paper_rerun/results/crop_ablation_2026-11-17.json
"""

import json
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
    build_augmentation_pipeline, extract_norm_stats, load_plantdoc,
)
from core_math import (  # noqa: E402
    compute_class_prototypes, compute_fisher_ensemble_probs, fit_fisher_lda,
    refine_transductive,
)

DATE = "2026-11-17"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
N_VIEWS = 8
CROP = 0.60


def center_crop(img, frac=CROP):
    w, h = img.size
    side = int(min(w, h) * frac)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def extract_single(models, device, paths, crop=False):
    """One canonical view per query image (no augmentation)."""
    bio_out, dino_out = [], []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            imgs = []
            for p in paths[i:i + 64]:
                im = Image.open(p).convert("RGB")
                imgs.append(center_crop(im) if crop else im)
            b = torch.stack([models.bioclip_transforms(im) for im in imgs]).to(device)
            d = torch.stack([models.dino_transforms(im) for im in imgs]).to(device)
            bio_out.append(F.normalize(models.bioclip_model.encode_image(b), dim=-1).cpu())
            dino_out.append(F.normalize(models.dino_model(d), dim=-1).cpu())
    return torch.cat(bio_out).numpy(), torch.cat(dino_out).numpy()


def extract(models, device, paths, crop=False):
    aug_bio = build_augmentation_pipeline(*extract_norm_stats(models.bioclip_transforms))
    aug_dino = build_augmentation_pipeline(models.dino_config["mean"], models.dino_config["std"])
    bio_out, dino_out = [], []
    with torch.no_grad():
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            if crop:
                img = center_crop(img)
            b = [models.bioclip_transforms(img)]
            d = [models.dino_transforms(img)]
            torch.manual_seed(10_000 + i)
            for _ in range(N_VIEWS - 1):
                b.append(aug_bio(img))
                d.append(aug_dino(img))
            bio_out.append(F.normalize(models.bioclip_model.encode_image(
                torch.stack(b).to(device)), dim=-1).cpu())
            dino_out.append(F.normalize(models.dino_model(
                torch.stack(d).to(device)), dim=-1).cpu())
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(paths)}", flush=True)
    return torch.cat(bio_out).numpy(), torch.cat(dino_out).numpy()


def evaluate(sb, sd, sy, qb, qd, qy, n_classes, text_protos):
    sj = np.concatenate([sb, sd], axis=-1)
    W, b, _ = fit_fisher_lda(sj, sy, device="cpu")
    qb_t, qd_t = torch.tensor(qb), torch.tensor(qd)
    joint = torch.cat([qb_t, qd_t], dim=-1)
    ens, _, text_p = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=joint,
        bioclip_features=qb_t, text_prototypes=text_protos,
        scale=20.0, w_lda=0.75, device="cpu")
    ind = ens.argmax(-1).numpy()
    tr, _, _ = refine_transductive(
        support_features=sj, support_labels=sy, query_features=joint,
        initial_preds=ind, confidences=ens.max(-1).values, text_prob=text_p,
        n_classes=n_classes, support_weight=1, max_anchors=2, conf_threshold=None,
        device="cpu", w_lda=0.75)
    return {"inductive": {"acc": float(accuracy_score(qy, ind) * 100),
                          "f1": float(f1_score(qy, ind, average="macro", zero_division=0) * 100)},
            "transductive": {"acc": float(accuracy_score(qy, tr) * 100),
                             "f1": float(f1_score(qy, tr, average="macro", zero_division=0) * 100)}}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_backbones(device)
    sup, qry, classes, prompts = load_plantdoc(ROOT / "plantdoc", k_shot=5, seed=42)
    n = len(classes)
    text_protos = build_text_prototypes(classes, prompts, models.bioclip_model,
                                        models.tokenizer, device)
    qy = np.array([y for _, y in qry])
    out = {"date": DATE, "protocol": {"dataset": "PlantDoc 27-way 5-shot",
                                      "n_views": N_VIEWS, "crop_fraction": CROP},
           "conditions": {}}
    for name, crop in (("full_leaf", False), ("center_crop_60", True)):
        t0 = time.time()
        print(f"[*] {name}", flush=True)
        sb, sd = extract(models, device, [p for p, _ in sup], crop)
        qb, qd = extract_single(models, device, [p for p, _ in qry], crop)
        res = evaluate(sb, sd, np.repeat(np.array([y for _, y in sup]), N_VIEWS),
                        qb, qd, qy, n, text_protos)
        out["conditions"][name] = res
        print(f"[*] {name}: ind={res['inductive']['acc']:.2f} "
              f"trans={res['transductive']['acc']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    p = OUT / f"crop_ablation_{DATE}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] {p}\n{json.dumps(out, indent=2)}")


if __name__ == "__main__":
    main()
