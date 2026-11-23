# -*- coding: utf-8 -*-
"""INT8 vs FP32 accuracy audit on PlantDoc (27 classes, 5-shot)."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_quantization_accuracy as R  # noqa: E402
from pipeline.core_math import (  # noqa: E402
    calculate_shannon_confidence,
    compute_class_prototypes,
    compute_fisher_ensemble_probs,
    fit_fisher_lda,
    refine_transductive,
)
from pipeline.data_engine import load_plantdoc  # noqa: E402
from pipeline.taxonomic_prompts import PLANTDOC_TAXONOMIC_PROMPTS as PROMPTS  # noqa: E402
import open_clip  # noqa: E402

DATE = "2026-11-17"
HEALTHY = None  # PlantDoc has healthy classes by name


def healthy_idx(classes):
    return [i for i, c in enumerate(classes) if c.lower().rstrip("_").endswith("leaf")]


if __name__ == "__main__":
    t0 = time.time()
    sup, qry, classes, _ = load_plantdoc(ROOT / "plantdoc", k_shot=5, seed=42)
    n_classes = len(classes)
    print(f"[PD] support={len(sup)} query={len(qry)} classes={n_classes}")

    dino_tf, bioclip_tf, dino_cfg = R.build_transforms()
    dino_aug = R.aug_pipeline(dino_cfg["mean"], dino_cfg["std"])
    bio_aug = R.aug_pipeline(*R.norm_stats(bioclip_tf))

    s_dino, s_bio, s_y = [], [], []
    for i, (p, y) in enumerate(sup):
        img = Image.open(p).convert("RGB")
        d_views = [np.asarray(dino_tf(img), dtype=np.float32)]
        b_views = [np.asarray(bioclip_tf(img), dtype=np.float32)]
        torch.manual_seed(10_000 + i)
        for _ in range(R.N_VIEWS - 1):
            d_views.append(np.asarray(dino_aug(img), dtype=np.float32))
            b_views.append(np.asarray(bio_aug(img), dtype=np.float32))
        s_dino.extend(d_views); s_bio.extend(b_views); s_y.extend([y] * R.N_VIEWS)
    s_dino, s_bio, s_y = np.stack(s_dino), np.stack(s_bio), np.array(s_y)

    q_dino = np.stack([np.asarray(dino_tf(Image.open(p).convert("RGB")), dtype=np.float32) for p, _ in qry])
    q_bio = np.stack([np.asarray(bioclip_tf(Image.open(p).convert("RGB")), dtype=np.float32) for p, _ in qry])
    q_y = np.array([y for _, y in qry])
    print(f"[PD] support views={s_dino.shape[0]} query={q_dino.shape[0]}")

    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    text_protos = []
    with torch.no_grad():
        for c in classes:
            tok = tokenizer(PROMPTS[c])
            f = F.normalize(bioclip_model.encode_text(tok), dim=-1)
            text_protos.append(F.normalize(f.mean(dim=0), dim=-1))
    text_protos = torch.stack(text_protos)

    out = {"date": DATE, "protocol": {"k_shot": 5, "n_views": R.N_VIEWS, "seed": 42,
                                       "scale": R.SCALE, "w_lda": R.W_LDA, "tau": R.TAU,
                                       "providers": "CPUExecutionProvider"}, "plantdoc": {}}

    for precision in ("fp32", "int8"):
        t1, t2 = R.build_sessions(precision)
        ts = time.time()
        def feats(dx, bx):
            do, bo = [], []
            for i in range(0, len(dx), 64):
                _, _, f = R.run_onnx_tier1(t1, dx[i:i + 64]); do.append(f)
                _, fb = R.run_onnx_companion(t2, bx[i:i + 64], f); bo.append(fb)
            return np.concatenate(do), np.concatenate(bo)
        sd, sb = feats(s_dino, s_bio)
        qd, qb = feats(q_dino, q_bio)

        s_labels = torch.tensor(s_y)
        dino_protos = compute_class_prototypes(torch.tensor(sd), s_labels, n_classes)
        bio_vis = compute_class_prototypes(torch.tensor(sb), s_labels, n_classes)
        bio_protos = F.normalize(0.80 * bio_vis + 0.20 * text_protos, dim=-1)

        sj = np.concatenate([sb, sd], axis=-1)
        W, b, _ = fit_fisher_lda(sj, s_y, device="cpu")

        qd_t, qb_t = torch.tensor(qd), torch.tensor(qb)
        prob1 = F.softmax(R.SCALE * (qd_t @ dino_protos.t()), dim=-1)
        conf1 = calculate_shannon_confidence(prob1, n_classes)
        pred1 = prob1.argmax(-1).numpy()
        pred_bio = (R.SCALE * (qb_t @ bio_protos.t())).argmax(-1).numpy()
        joint = torch.cat([qb_t, qd_t], dim=-1)
        ens, _, text_p = compute_fisher_ensemble_probs(
            fisher_weights=W, fisher_biases=b, joint_features=joint, bioclip_features=qb_t,
            text_prototypes=text_protos, scale=R.SCALE, w_lda=R.W_LDA, device="cpu")
        pred_fish = ens.argmax(-1).numpy()
        trans, _, _ = refine_transductive(
            support_features=sj, support_labels=s_y, query_features=joint,
            initial_preds=pred_fish, confidences=ens.max(-1).values, text_prob=text_p,
            n_classes=n_classes, support_weight=2, max_anchors=2, conf_threshold=None,
            device="cpu", w_lda=R.W_LDA)

        def line(pred):
            return {"accuracy": float(accuracy_score(q_y, pred) * 100),
                    "macro_f1": float(f1_score(q_y, pred, average="macro", zero_division=0) * 100)}
        out["plantdoc"][precision] = {
            "tier1": line(pred1), "bioclip_only": line(pred_bio),
            "fisher": line(pred_fish), "transductive": line(trans),
            "feature_seconds": round(time.time() - ts, 1)}
        print(f"[PD] {precision}: tier1={out['plantdoc'][precision]['tier1']['accuracy']:.2f} "
              f"fisher={out['plantdoc'][precision]['fisher']['accuracy']:.2f} "
              f"trans={out['plantdoc'][precision]['transductive']['accuracy']:.2f}")

    R.OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = R.OUT_DIR / f"quantization_plantdoc_{DATE}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] {p} ({time.time()-t0:.0f}s)")
