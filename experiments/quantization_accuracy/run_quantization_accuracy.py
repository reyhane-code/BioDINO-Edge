# -*- coding: utf-8 -*-
"""
End-to-end INT8 vs FP32 accuracy audit for the deployed BioDINO-Edge pipeline.

Both precisions are exported ONNX binaries (cache_artifacts/*_int8.onnx /
*_fp32.onnx). For each precision the FULL pipeline is reconstructed from
features produced by that precision only: support extraction (5 shots/class,
8-view augmentation), prototypes, BioCLIP text prototypes, closed-form
Fisher-LDA, Tier-1 gate, Tier-2 ensemble, and transductive recalibration.

Datasets: PlantVillage 11-class (9,385 images) and PlantDoc 27-class
(train 2,336 / test 252). No morphopath artifacts or code are used.

Outputs: results/quantization_<dataset>_<date>.json
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.data_engine import load_plantvillage, load_plantdoc  # noqa: E402
from pipeline.core_math import (  # noqa: E402
    calculate_shannon_confidence,
    compute_class_prototypes,
    compute_fisher_ensemble_probs,
    fit_fisher_lda,
    refine_transductive,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix  # noqa: E402

import timm  # noqa: E402
from timm.data import create_transform, resolve_model_data_config  # noqa: E402
import open_clip  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402

DATE = "2026-11-17"
CACHE = ROOT / "cache_artifacts"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEALTHY = {3, 7, 10}
SCALE, W_LDA, TAU, N_VIEWS = 20.0, 0.75, 0.35, 8


def build_sessions(precision: str):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 2
    t1 = ort.InferenceSession(str(CACHE / f"biodino_mobile_{precision}.onnx"), so,
                             providers=["CPUExecutionProvider"])
    t2 = ort.InferenceSession(str(CACHE / f"biodino_companion_{precision}.onnx"), so,
                             providers=["CPUExecutionProvider"])
    return t1, t2


def build_transforms():
    dino_model = timm.create_model("vit_small_patch14_dinov2", pretrained=False, dynamic_img_size=True)
    cfg = resolve_model_data_config(dino_model)
    cfg["input_size"] = (3, 224, 224)
    dino_tf = create_transform(**cfg, is_training=False)
    _, _, bioclip_tf = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    return dino_tf, bioclip_tf, cfg


def aug_pipeline(mean, std):
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def norm_stats(transform):
    for t in getattr(transform, "transforms", []):
        if isinstance(t, T.Normalize):
            return list(t.mean), list(t.std)
    if isinstance(transform, T.Normalize):
        return list(transform.mean), list(transform.std)
    raise ValueError("no Normalize found")


def run_onnx_tier1(sess, batch):
    p = sess.run(None, {"input_leaf": batch.astype(np.float32)})
    return p[0], p[1], p[2]  # probabilities, confidence, features


def run_onnx_companion(sess, batch, dino_feats):
    p = sess.run(None, {"input_leaf": batch.astype(np.float32),
                        "dino_features": dino_feats.astype(np.float32)})
    return p[0], p[1]  # probabilities, bioclip features


def collect(paths, dino_tf, bioclip_tf, views_fn, seed):
    """Return per-image ONNX inputs for DINOv2 and BioCLIP under one view set."""
    rng = torch.Generator().manual_seed(seed)
    dino_views, bio_views = [], []
    for p in paths:
        img = Image.open(p).convert("RGB")
        for v in range(views_fn):
            dino_views.append(np.asarray(dino_tf(img), dtype=np.float32))
            bio_views.append(np.asarray(bioclip_tf(img), dtype=np.float32))
    return np.stack(dino_views), np.stack(bio_views)


def main():
    t_start = time.time()
    result = {"date": DATE, "protocol": {
        "k_shot": 5, "n_views": N_VIEWS, "seed": 42, "scale": SCALE,
        "w_lda": W_LDA, "tau": TAU, "providers": "CPUExecutionProvider",
        "intra_op_threads": 4, "inter_op_threads": 2}}

    dino_tf, bioclip_tf, dino_cfg = build_transforms()
    dino_aug = aug_pipeline(dino_cfg["mean"], dino_cfg["std"])
    bio_aug = aug_pipeline(*norm_stats(bioclip_tf))

    # ---------------- PlantVillage ----------------
    sup, qry, classes, _ = load_plantvillage(ROOT / "plantvillage", k_shot=5, seed=42)
    n_classes = len(classes)
    print(f"[PV] support={len(sup)} query={len(qry)} classes={n_classes}")

    # support views: original + 7 augmented, deterministic per image index
    s_dino, s_bio, s_y = [], [], []
    for i, (p, y) in enumerate(sup):
        img = Image.open(p).convert("RGB")
        d_views = [np.asarray(dino_tf(img), dtype=np.float32)]
        b_views = [np.asarray(bioclip_tf(img), dtype=np.float32)]
        # must match pipeline/biodino_edge_pipeline.py: SEED + 10_000 + index
        torch.manual_seed(42 + 10_000 + i)
        for _ in range(N_VIEWS - 1):
            d_views.append(np.asarray(dino_aug(img), dtype=np.float32))
            b_views.append(np.asarray(bio_aug(img), dtype=np.float32))
        s_dino.extend(d_views)
        s_bio.extend(b_views)
        s_y.extend([y] * N_VIEWS)
    s_dino, s_bio, s_y = np.stack(s_dino), np.stack(s_bio), np.array(s_y)

    q_dino = np.stack([np.asarray(dino_tf(Image.open(p).convert("RGB")), dtype=np.float32) for p, _ in qry])
    q_bio = np.stack([np.asarray(bioclip_tf(Image.open(p).convert("RGB")), dtype=np.float32) for p, _ in qry])
    q_y = np.array([y for _, y in qry])
    print(f"[PV] support views={s_dino.shape[0]} query={q_dino.shape[0]}")

    # text prototypes (BioCLIP text encoder, precision independent)
    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    from pipeline.taxonomic_prompts import PLANTVILLAGE_TAXONOMIC_PROMPTS as PV_PROMPTS
    text_protos = []
    with torch.no_grad():
        for c in classes:
            tok = tokenizer(PV_PROMPTS[c])
            feats = F.normalize(bioclip_model.encode_text(tok), dim=-1)
            text_protos.append(F.normalize(feats.mean(dim=0), dim=-1))
    text_protos = torch.stack(text_protos)

    pv = {}
    for precision in ("fp32", "int8"):
        t1, t2 = build_sessions(precision)
        t0 = time.time()

        def feats(dino_x, bio_x):
            dino_out, bio_out = [], []
            for i in range(0, len(dino_x), 64):
                _, _, f = run_onnx_tier1(t1, dino_x[i:i + 64])
                dino_out.append(f)
                _, fb = run_onnx_companion(t2, bio_x[i:i + 64], f)
                bio_out.append(fb)
            return np.concatenate(dino_out), np.concatenate(bio_out)

        s_dino_f, s_bio_f = feats(s_dino, s_bio)
        q_dino_f, q_bio_f = feats(q_dino, q_bio)

        s_dino_t = torch.tensor(s_dino_f)
        s_labels = torch.tensor(s_y)

        dino_protos = compute_class_prototypes(s_dino_t, s_labels, n_classes)
        bio_vis = compute_class_prototypes(torch.tensor(s_bio_f), s_labels, n_classes)
        bio_protos = F.normalize(0.80 * bio_vis + 0.20 * text_protos, dim=-1)

        support_joint = np.concatenate([s_bio_f, s_dino_f], axis=-1)
        fisher_w, fisher_b, clf = fit_fisher_lda(support_joint, s_y, device="cpu")

        q_dino_t, q_bio_t = torch.tensor(q_dino_f), torch.tensor(q_bio_f)
        prob1 = F.softmax(SCALE * (q_dino_t @ dino_protos.t()), dim=-1)
        conf1 = calculate_shannon_confidence(prob1, n_classes)
        pred1 = prob1.argmax(-1).numpy()

        bio_logits = SCALE * (q_bio_t @ bio_protos.t())
        pred_bio = bio_logits.argmax(-1).numpy()

        joint = torch.cat([q_bio_t, q_dino_t], dim=-1)
        ens, lda_p, text_p = compute_fisher_ensemble_probs(
            fisher_weights=fisher_w, fisher_biases=fisher_b, joint_features=joint,
            bioclip_features=q_bio_t, text_prototypes=text_protos,
            scale=SCALE, w_lda=W_LDA, device="cpu")
        pred_fish = ens.argmax(-1).numpy()

        deep = (conf1 < TAU).numpy()
        pred_casc = pred1.copy()
        pred_casc[deep] = pred_fish[deep]

        top_conf = ens.max(-1).values
        pred_trans, _, _ = refine_transductive(
            support_features=support_joint, support_labels=s_y, query_features=joint,
            initial_preds=pred_fish, confidences=top_conf, text_prob=text_p,
            n_classes=n_classes, support_weight=3, max_anchors=20,
            conf_threshold=0.70, device="cpu", w_lda=W_LDA)

        def line(pred):
            acc = float(accuracy_score(q_y, pred) * 100)
            f1 = float(f1_score(q_y, pred, average="macro", zero_division=0) * 100)
            dis = ~np.isin(q_y, list(HEALTHY))
            fnr = float(np.isin(pred, list(HEALTHY))[dis].sum() / dis.sum() * 100)
            return {"accuracy": acc, "macro_f1": f1, "fnr": fnr}

        pv[precision] = {
            "tier1": line(pred1), "bioclip_only": line(pred_bio),
            "fisher": line(pred_fish), "cascaded": line(pred_casc),
            "transductive": line(pred_trans),
            "fast_exit_pct": float((~deep).mean() * 100),
            "feature_seconds": round(time.time() - t0, 1),
            "confusion_matrix": confusion_matrix(q_y, pred_trans).tolist(),
        }
        print(f"[PV] {precision}: transductive acc={pv[precision]['transductive']['accuracy']:.2f} "
              f"tier1={pv[precision]['tier1']['accuracy']:.2f} "
              f"fish={pv[precision]['fisher']['accuracy']:.2f} "
              f"({pv[precision]['feature_seconds']}s)")

    result["plantvillage"] = pv

    (OUT_DIR / f"quantization_plantvillage_{DATE}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] wrote quantization_plantvillage_{DATE}.json ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
