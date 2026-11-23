# -*- coding: utf-8 -*-
"""One harness, both methods, PlantDoc 27-way: removes the 51.19/50.79 split.

Why this file exists. The paper reports 51.19% for transductive BioDINO-Edge on
PlantDoc 27-way in one table (from the ladder run) and 50.79% in another (from the
fair-comparison harness). Two numbers for the same quantity is exactly what a
reviewer flags, and the earlier footnote only explained the 5-way pair, not this
one. This script computes BOTH methods inside the ladder's code path so the
table has a single source and a single set of numbers.

Output: experiments/quantization_accuracy/results/plantdoc_ladder_plus_lap_2026-11-18.json
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "pipeline", HERE.parent / "paper_rerun"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_quantization_accuracy as R  # noqa: E402
from run_laplacianshot_faithful import laplacianshot  # noqa: E402
from core_math import (  # noqa: E402
    compute_class_prototypes, compute_fisher_ensemble_probs, fit_fisher_lda,
    refine_transductive,
)
from data_engine import load_plantdoc  # noqa: E402
import open_clip  # noqa: E402

DATE = "2026-11-18"
W_TEXT, W_BASE, SCALE, W_LDA = 0.20, 0.65, 20.0, 0.75
LAM = 2.0


def main():
    t0 = time.time()
    sup, qry, classes, prompts = load_plantdoc(ROOT / "plantdoc", k_shot=5, seed=42)
    n_classes = len(classes)
    print(f"[PD] support={len(sup)} query={len(qry)} classes={n_classes}", flush=True)

    dino_tf, bioclip_tf, dino_cfg = R.build_transforms()
    dino_aug = R.aug_pipeline(dino_cfg["mean"], dino_cfg["std"])
    bio_aug = R.aug_pipeline(*R.norm_stats(bioclip_tf))

    s_dino, s_bio, s_y = [], [], []
    for i, (p, y) in enumerate(sup):
        img = Image.open(p).convert("RGB")
        d_views = [np.asarray(dino_tf(img), dtype=np.float32)]
        b_views = [np.asarray(bioclip_tf(img), dtype=np.float32)]
        torch.manual_seed(42 + 10_000 + i)
        for _ in range(R.N_VIEWS - 1):
            d_views.append(np.asarray(dino_aug(img), dtype=np.float32))
            b_views.append(np.asarray(bio_aug(img), dtype=np.float32))
        s_dino.extend(d_views)
        s_bio.extend(b_views)
        s_y.extend([y] * R.N_VIEWS)
    s_dino, s_bio, s_y = np.stack(s_dino), np.stack(s_bio), np.array(s_y)

    q_dino = np.stack([np.asarray(dino_tf(Image.open(p).convert("RGB")), np.float32) for p, _ in qry])
    q_bio = np.stack([np.asarray(bioclip_tf(Image.open(p).convert("RGB")), np.float32) for p, _ in qry])
    q_y = np.array([y for _, y in qry])

    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    text_protos = []
    with torch.no_grad():
        for c in classes:
            tok = tokenizer(prompts[c])
            f = F.normalize(bioclip_model.encode_text(tok), dim=-1)
            text_protos.append(F.normalize(f.mean(dim=0), dim=-1))
    text_protos = torch.stack(text_protos)

    t1, t2 = R.build_sessions("int8")

    def feats(dx, bx):
        do, bo = [], []
        for i in range(0, len(dx), 64):
            _, _, f = R.run_onnx_tier1(t1, dx[i:i + 64])
            do.append(f)
            _, fb = R.run_onnx_companion(t2, bx[i:i + 64], f)
            bo.append(fb)
        return np.concatenate(do), np.concatenate(bo)

    sd, sb = feats(s_dino, s_bio)
    qd, qb = feats(q_dino, q_bio)

    sy_t = torch.tensor(s_y)
    dino_protos = compute_class_prototypes(torch.tensor(sd), sy_t, n_classes)
    bio_vis = compute_class_prototypes(torch.tensor(sb), sy_t, n_classes)
    bio_protos = F.normalize((1 - W_TEXT) * bio_vis + W_TEXT * text_protos, dim=-1)

    sj = np.concatenate([sb, sd], axis=-1)
    W, b, _ = fit_fisher_lda(sj, s_y, device="cpu")

    qd_t, qb_t = torch.tensor(qd), torch.tensor(qb)
    dino_logits = SCALE * (qd_t @ dino_protos.t())
    bio_logits = SCALE * (qb_t @ bio_protos.t())
    text_logits = SCALE * (qb_t @ text_protos.t())

    joint = torch.cat([qb_t, qd_t], dim=-1)
    ens, _, text_p = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=joint,
        bioclip_features=qb_t, text_prototypes=text_protos,
        scale=SCALE, w_lda=W_LDA, device="cpu")
    fish = ens.argmax(-1).numpy()
    trans, _, _ = refine_transductive(
        support_features=sj, support_labels=s_y, query_features=joint,
        initial_preds=fish, confidences=ens.max(-1).values, text_prob=text_p,
        n_classes=n_classes, support_weight=1, max_anchors=2, conf_threshold=None,
        device="cpu", w_lda=W_LDA)
    lap = laplacianshot(sj, s_y, joint.numpy(), n_classes, lam=LAM)

    def row(p):
        return {"acc": round(float(accuracy_score(q_y, p) * 100), 2),
                "f1": round(float(f1_score(q_y, p, average="macro", zero_division=0) * 100), 2)}

    out = {"date": DATE, "dataset": "PlantDoc 27-way 5-shot, INT8, single harness",
           "protocol": {"n_views": R.N_VIEWS, "seed": 42, "scale": SCALE,
                        "w_lda": W_LDA, "w_text": W_TEXT, "w_base": W_BASE,
                        "n_queries": int(len(q_y)), "lambda": LAM,
                        "transductive": {"support_weight": 1, "max_anchors": 2,
                                         "conf_threshold": None}},
           "rows": {
               "text_only": row(text_logits.argmax(-1).numpy()),
               "dino_only": row(dino_logits.argmax(-1).numpy()),
               "bio_only": row(bio_logits.argmax(-1).numpy()),
               "static_blend": row((W_BASE * bio_logits + (1 - W_BASE) * dino_logits).argmax(-1).numpy()),
               "fisher_inductive": row(fish),
               "transductive_ours": row(trans),
               "laplacianshot": row(lap),
           }}
    for k, v in out["rows"].items():
        print(f"  {k:20s} {v['acc']:6.2f}  F1 {v['f1']:6.2f}", flush=True)

    pth = HERE / "results" / f"plantdoc_ladder_plus_lap_{DATE}.json"
    pth.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] {pth} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
