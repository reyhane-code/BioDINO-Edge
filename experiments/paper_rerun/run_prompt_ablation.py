# -*- coding: utf-8 -*-
"""
Prompt-design ablation: botanical templates vs generic class names vs syntactically
negated healthy descriptions.

Only the BioCLIP text encoder is executed; image features come from the cached,
model-parallel embeddings of the canonical seed-42 split (cache_artifacts/
pv_novel_embeddings.pt), so the comparison isolates the text prototype.

Reported per condition: text-prototype similarity to the disease prototypes, and
end-to-end accuracy / macro F1 / false-negative disease escape rate of the
deployed ensemble (0.75 Fisher + 0.25 text) and of the transductive variant.

Output: experiments/paper_rerun/results/prompt_ablation_2026-11-17.json
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core_math import (  # noqa: E402
    calculate_shannon_confidence, compute_class_prototypes,
    compute_fisher_ensemble_probs, fit_fisher_lda, refine_transductive,
)
from taxonomic_prompts import PLANTVILLAGE_TAXONOMIC_PROMPTS  # noqa: E402
from data_engine import find_healthy_indices  # noqa: E402

DATE = "2026-11-17"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
SEED = 42

# --- Alternative prompt sets for the three healthy classes -------------------
DISPLAY = {
    "Apple___healthy": "apple",
    "Grape___healthy": "grape",
    "Potato___healthy": "potato",
}

NEGATED_HEALTHY = {
    "Apple___healthy": [
        "A healthy apple leaf without any fungal lesions or necrotic spots.",
        "Apple foliage with no signs of scab, rot, or rust disease.",
        "Apple leaf that does not show any disease symptoms or lesions.",
    ],
    "Grape___healthy": [
        "A healthy grape leaf without any fungal lesions or necrotic spots.",
        "Grape foliage with no signs of black rot, esca, or leaf blight.",
        "Grape leaf that does not show any disease symptoms or lesions.",
    ],
    "Potato___healthy": [
        "A healthy potato leaf without any fungal lesions or necrotic spots.",
        "Potato foliage with no signs of early or late blight disease.",
        "Potato leaf that does not show any disease symptoms or lesions.",
    ],
}


def generic(name):
    plant = name.split("___")[0].lower()
    return [f"A photo of a {plant} leaf.",
            f"A close-up photograph of a {plant} plant leaf.",
            f"A {plant} leaf captured in the field."]


def encode(model, tokenizer, prompts, device):
    with torch.no_grad():
        tok = tokenizer(prompts).to(device)
        f = F.normalize(model.encode_text(tok), dim=-1)
    return F.normalize(f.mean(dim=0), dim=-1)


def build(model, tokenizer, classes, device, mode):
    protos = []
    for c in classes:
        if mode == "botanical":
            prompts = PLANTVILLAGE_TAXONOMIC_PROMPTS[c]
        elif mode == "generic":
            prompts = generic(c)
        elif mode == "negated_healthy":
            prompts = NEGATED_HEALTHY.get(c, PLANTVILLAGE_TAXONOMIC_PROMPTS[c])
        else:
            raise ValueError(mode)
        protos.append(encode(model, tokenizer, prompts, device))
    return torch.stack(protos)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(CACHE, map_location="cpu")
    classes = cache["novel_classes"]
    n_classes = len(classes)
    bio, dino, labels = cache["bio_feats"], cache["dino_feats"], cache["labels"].numpy()
    joint = torch.cat([bio, dino], dim=-1).numpy()
    healthy = find_healthy_indices(classes)

    model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    model = model.to(device).eval()

    rng = random.Random(SEED)
    sup, qry = [], []
    for c in range(n_classes):
        idx = np.where(labels == c)[0].tolist()
        rng.shuffle(idx)
        sup.extend(idx[:5])
        qry.extend(idx[5:])
    sup, qry = np.array(sup), np.array(qry)
    q_y = labels[qry]

    W, b, _ = fit_fisher_lda(joint[sup], labels[sup], device="cpu")
    q_joint = torch.tensor(joint[qry])

    out = {"date": DATE, "protocol": {"split": "canonical seed-42 5-shot, cached features",
                                       "n_queries": int(len(qry)), "w_lda": 0.75, "scale": 20.0},
           "conditions": {}}

    protos = {mode: build(model, tokenizer, classes, device, mode)
              for mode in ("botanical", "generic", "negated_healthy")}

    # text-vs-disease-prototype geometry
    diseased = [i for i in range(n_classes) if i not in healthy]
    for mode, P in protos.items():
        H = P[healthy]
        D = P[diseased]
        S = H @ D.t()
        out["conditions"].setdefault(mode, {})["healthy_vs_disease_cosine"] = {
            "mean": float(S.mean()), "min": float(S.min()), "max": float(S.max())}

    for mode, P in protos.items():
        ens, _, text_p = compute_fisher_ensemble_probs(
            fisher_weights=W, fisher_biases=b, joint_features=q_joint,
            bioclip_features=bio[qry], text_prototypes=P, scale=20.0, w_lda=0.75, device="cpu")
        pred_ind = ens.argmax(-1).numpy()
        pred_tr, _, _ = refine_transductive(
            support_features=joint[sup], support_labels=labels[sup], query_features=q_joint,
            initial_preds=pred_ind, confidences=ens.max(-1).values, text_prob=text_p,
            n_classes=n_classes, support_weight=3, max_anchors=20, conf_threshold=0.70,
            device="cpu", w_lda=0.75)

        def stats(pred):
            dis = ~np.isin(q_y, healthy)
            fnr = float(np.isin(pred, healthy)[dis].sum() / dis.sum() * 100)
            return {"acc": float(accuracy_score(q_y, pred) * 100),
                    "f1": float(f1_score(q_y, pred, average="macro", zero_division=0) * 100),
                    "fnr": fnr}

        base = F.softmax(20.0 * (bio[qry].cpu() @ P.cpu().t()), dim=-1)
        conf1 = calculate_shannon_confidence(base, n_classes)
        pred_fast = base.argmax(-1).numpy()
        deep = (conf1 < 0.35).numpy()
        pred_casc = pred_fast.copy()
        pred_casc[deep] = pred_ind[deep]

        out["conditions"][mode].update({
            "text_only": stats(pred_fast),
            "ensemble_inductive": stats(pred_ind),
            "cascaded": stats(pred_casc),
            "transductive": stats(pred_tr),
        })
        print(f"{mode:16s} text-only {stats(pred_fast)['acc']:6.2f} | "
              f"ens {stats(pred_ind)['acc']:6.2f} FNR {stats(pred_ind)['fnr']:.3f} | "
              f"trans {stats(pred_tr)['acc']:6.2f} FNR {stats(pred_tr)['fnr']:.3f}", flush=True)

    p = OUT / f"prompt_ablation_{DATE}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] {p}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
