# -*- coding: utf-8 -*-
"""
PlantDoc component ladder + FP32/INT8 fidelity, one consistent protocol.

Single script, single split (seed-42, 5 support leaves/class x 8 augmented
views), single set of hyperparameters, so every row in the resulting table is
comparable. Each precision is evaluated end-to-end from its own ONNX exports.

Rows
  (1) text prototypes only            : argmax of 20 * f_bio @ P_text^T
  (2) DINOv2 fast path alone           : argmax of 20 * f_dino @ P_dino^T
  (3) BioCLIP deep path alone          : argmax of 20 * f_bio @ P_bio^T
  (4) static logit blend               : 0.65 * bio + 0.35 * dino
  (5) regularized Fisher (inductive)   : 0.75 * p_fisher + 0.25 * p_text
  (6) + transductive recalibration

Output: experiments/quantization_accuracy/results/plantdoc_ladder_2026-11-17.json
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
for p in (ROOT, ROOT / "pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_quantization_accuracy as R  # noqa: E402
from core_math import (  # noqa: E402
    compute_class_prototypes, compute_fisher_ensemble_probs, fit_fisher_lda,
    refine_transductive,
)
from data_engine import (  # noqa: E402
    build_augmentation_pipeline, extract_norm_stats, load_plantdoc,
)
import open_clip  # noqa: E402

DATE = "2026-11-17"
W_TEXT, W_BASE, SCALE, W_LDA = 0.20, 0.65, 20.0, 0.75


def rows(y, preds):
    out = {}
    for name, p in preds.items():
        out[name] = {"acc": float(accuracy_score(y, p) * 100),
                     "f1": float(f1_score(y, p, average="macro", zero_division=0) * 100)}
    return out


def main():
    t_all = time.time()
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
    print(f"[PD] support views={len(s_dino)} queries={len(q_dino)}", flush=True)

    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    text_protos = []
    with torch.no_grad():
        for c in classes:
            tok = tokenizer(prompts[c])
            f = F.normalize(bioclip_model.encode_text(tok), dim=-1)
            text_protos.append(F.normalize(f.mean(dim=0), dim=-1))
    text_protos = torch.stack(text_protos)

    out = {"date": DATE,
           "protocol": {"dataset": "PlantDoc 27-way 5-shot", "n_views": R.N_VIEWS,
                        "seed": 42, "scale": SCALE, "w_lda": W_LDA, "w_text": W_TEXT,
                        "w_base": W_BASE, "n_queries": int(len(q_y)),
                        "transductive": {"support_weight": 1, "max_anchors": 2,
                                         "conf_threshold": None}},
           "precision": {}}

    for precision in ("fp32", "int8"):
        t0 = time.time()
        t1, t2 = R.build_sessions(precision)

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

        preds = {
            "text_only": text_logits.argmax(-1).numpy(),
            "dino_only": dino_logits.argmax(-1).numpy(),
            "bio_only": bio_logits.argmax(-1).numpy(),
            "static_blend": (W_BASE * bio_logits + (1 - W_BASE) * dino_logits).argmax(-1).numpy(),
            "fisher_inductive": fish,
            "transductive": trans,
        }
        out["precision"][precision] = rows(q_y, preds)
        print(f"[PD:{precision}] " + "  ".join(
            f"{k}={v['acc']:.2f}" for k, v in out["precision"][precision].items())
            + f"  ({time.time()-t0:.0f}s)", flush=True)

    p = HERE / "results" / f"plantdoc_ladder_{DATE}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] {p} (total {time.time()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
