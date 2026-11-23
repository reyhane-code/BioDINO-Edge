# -*- coding: utf-8 -*-
"""
Canonical single-run metrics for the paper (current code generation).
================================================================================
Replicates pipeline/data_engine.load_plantvillage fixed split (shared RNG seed
42 over path-sorted class lists) on the cached embeddings, then reports:
  - single-run Fisher-LDA (inductive) accuracy / macro-F1
  - single-run transductive refined accuracy / macro-F1 (3x / 20 / 0.70)
  - refined confusion matrix, per-class precision/recall/F1/support
  - macro averages, clinical FNR (diseased->healthy), cross-species confusion
Also reports peak episodic macro-F1 from the paired 30-episode validation run.
Outputs: canonical_single_run.json (this directory).
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.core_math import fit_fisher_lda, compute_fisher_ensemble_probs, refine_transductive, compute_class_prototypes  # noqa: E402
from pipeline.data_engine import find_healthy_indices, load_plantvillage  # noqa: E402
from pipeline.biodino_edge_pipeline import load_backbones, extract_support_features  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = _PROJECT_ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"


def main():
    data = torch.load(CACHE, map_location="cpu")
    bio, dino = data["bio_feats"], data["dino_feats"]
    labels = data["labels"].numpy()
    paths = data["paths"]
    class_names = data["novel_classes"]
    joint = torch.cat([bio, dino], dim=-1).numpy()
    text_protos = data["bio_text_protos"]
    n_classes = len(class_names)

    # Match the original pipeline exactly: GPU device + call order
    # (load_backbones -> load_plantvillage -> extract_support_features).
    # Module import of biodino_edge_pipeline already sets seed 42 (random/np/torch).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_backbones(device)
    support_samples, query_samples, _, _ = load_plantvillage(
        _PROJECT_ROOT / "plantvillage", k_shot=5, seed=42)
    # Cache stores a mix of absolute/relative paths; UUID basenames are unique
    path_to_idx = {os.path.basename(p): i for i, p in enumerate(paths)}
    assert len(path_to_idx) == len(paths), "duplicate basenames in cache paths"
    query_indices = [path_to_idx[os.path.basename(p)] for p, _ in query_samples]

    # Phase-0 protocol: 8-view augmented support embeddings (N = 440)
    s_bio, s_dino, s_y_t = extract_support_features(support_samples, models, device=device)
    s_bio, s_dino = s_bio.cpu(), s_dino.cpu()
    s_x = torch.cat([s_bio, s_dino], dim=-1).numpy()
    s_y = s_y_t.cpu().numpy() if isinstance(s_y_t, torch.Tensor) else np.asarray(s_y_t)
    q_x, q_y = joint[query_indices], labels[query_indices]
    q_t = torch.tensor(q_x, dtype=torch.float32)

    W, b, _ = fit_fisher_lda(s_x, s_y, device="cpu")
    prob, _, text_prob = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=q_t,
        bioclip_features=bio[query_indices], text_prototypes=text_protos,
        scale=20.0, w_lda=0.75, device="cpu")
    ind_pred = prob.argmax(dim=-1).numpy()
    ind_acc = accuracy_score(q_y, ind_pred) * 100
    ind_f1 = f1_score(q_y, ind_pred, average="macro", zero_division=0) * 100

    tr_pred, _, _ = refine_transductive(
        support_features=s_x, support_labels=s_y, query_features=q_t,
        initial_preds=ind_pred, confidences=prob.max(dim=-1).values,
        text_prob=text_prob, n_classes=n_classes, support_weight=3,
        max_anchors=20, conf_threshold=0.70, device="cpu", w_lda=0.75)
    tr_acc = accuracy_score(q_y, tr_pred) * 100
    tr_f1 = f1_score(q_y, tr_pred, average="macro", zero_division=0) * 100

    # ---- Full single-run ablation ladder (Fast-Cache mode: scale=20, w_text=0.2, w_base=0.65) ----
    s_y_cpu = s_y_t.cpu()
    dino_proto_aug = compute_class_prototypes(s_dino, s_y_cpu, n_classes)
    bio_proto_aug = F.normalize(
        0.8 * compute_class_prototypes(s_bio, s_y_cpu, n_classes) + 0.2 * text_protos, dim=-1)
    # Raw 5-shot support prototypes (no augmentation) from the same cached embeddings
    support_indices = [path_to_idx[os.path.basename(p)] for p, _ in support_samples]
    raw_y = torch.tensor(labels[support_indices], dtype=torch.long)
    dino_proto_raw = compute_class_prototypes(dino[support_indices], raw_y, n_classes)
    bio_proto_raw = F.normalize(
        0.8 * compute_class_prototypes(bio[support_indices], raw_y, n_classes) + 0.2 * text_protos, dim=-1)

    q_bio_t, q_dino_t = bio[query_indices], dino[query_indices]

    def _ladder(dino_proto, bio_proto):
        dl = 20.0 * (q_dino_t @ dino_proto.t())
        bl = 20.0 * (q_bio_t @ bio_proto.t())
        return (dl.argmax(-1).numpy(), bl.argmax(-1).numpy(),
                (0.65 * bl + 0.35 * dl).argmax(-1).numpy())

    dino_pred, bio_pred, cons_aug_pred = _ladder(dino_proto_aug, bio_proto_aug)
    _, _, cons_raw_pred = _ladder(dino_proto_raw, bio_proto_raw)
    text_pred = (20.0 * (q_bio_t @ text_protos.t())).argmax(-1).numpy()

    def _accf1(p):
        return dict(acc=float(accuracy_score(q_y, p) * 100),
                    f1=float(f1_score(q_y, p, average="macro", zero_division=0) * 100))

    ladder = dict(fast_dino=_accf1(dino_pred), deep_bio=_accf1(bio_pred),
                  text_only=_accf1(text_pred), consensus_raw5=_accf1(cons_raw_pred),
                  consensus_aug=_accf1(cons_aug_pred))

    cm = confusion_matrix(q_y, tr_pred, labels=list(range(n_classes)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        q_y, tr_pred, labels=list(range(n_classes)), zero_division=0)

    healthy = find_healthy_indices(class_names)
    diseased = ~np.isin(q_y, healthy)
    fn = int(np.isin(tr_pred, healthy)[diseased].sum())
    fnr = 100.0 * fn / int(diseased.sum())
    species = [n.split("___")[0] for n in class_names]
    cross = int((np.array(species)[q_y] != np.array(species)[tr_pred]).sum())

    val = json.loads((HERE / "transductive_validation_results.json").read_text())
    p = val["paired_30ep"]

    out = dict(
        single_run=dict(inductive_acc=ind_acc, inductive_f1=ind_f1,
                        transductive_acc=tr_acc, transductive_f1=tr_f1,
                        confusion_matrix=cm.tolist(),
                        per_class=[dict(name=class_names[c], precision=float(prec[c]),
                                        recall=float(rec[c]), f1=float(f1[c]), support=int(sup[c]))
                                   for c in range(n_classes)],
                        macro_precision=float(prec.mean()), macro_recall=float(rec.mean()),
                        macro_f1=float(f1.mean()),
                        fn_count=fn, diseased_count=int(diseased.sum()), fnr_pct=fnr,
                        cross_species_count=cross, cross_species_pct=100.0 * cross / len(q_y)),
        ablation_ladder=ladder,
        episodic_peaks=dict(transductive_acc_peak=max(p["transductive_acc"]),
                            transductive_f1_peak=max(p["transductive_f1"]),
                            inductive_acc_peak=max(p["inductive_acc"])))
    (HERE / "canonical_single_run.json").write_text(json.dumps(out, indent=2))

    print(f"single-run Fisher: acc {ind_acc:.4f} | f1 {ind_f1:.4f}")
    print(f"single-run refined: acc {tr_acc:.4f} | f1 {tr_f1:.4f}")
    print(f"macro P/R/F1: {prec.mean():.3f}/{rec.mean():.3f}/{f1.mean():.3f}")
    print(f"FNR: {fn}/{int(diseased.sum())} = {fnr:.3f}% | cross-species: {cross} = {100.0 * cross / len(q_y):.3f}%")
    for k, v in ladder.items():
        print(f"ladder {k}: acc {v['acc']:.4f} | f1 {v['f1']:.4f}")
    for c in range(n_classes):
        print(f"  {class_names[c]}: P {prec[c]:.3f} R {rec[c]:.3f} F1 {f1[c]:.3f} n {int(sup[c])}")
    print(f"episodic peaks: tr acc {max(p['transductive_acc']):.2f} | tr f1 {max(p['transductive_f1']):.2f}")


if __name__ == "__main__":
    main()