# -*- coding: utf-8 -*-
"""Which constant explains 84.22 (pipeline) vs 85.01 (fair harness)?

The paper footnote currently claims the gap comes from restricting episodes to
classes with >=2 test leaves. That filter is a no-op (min is 4), so the claim is
wrong and needs a real cause. The pipeline's run_plantdoc_episodic uses
support_weight=2, while the comparison harness used support_weight=1.
This isolates that single constant on the cached features.
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
from sklearn.metrics import accuracy_score

ROOT = Path(r"E:\work\research-few-shot")
for p in (ROOT, ROOT / "pipeline", ROOT / "experiments/quantization_accuracy",
          ROOT / "experiments/paper_rerun"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_quantization_accuracy as R
from run_laplacianshot_faithful import fisher_anchor
from data_engine import load_plantdoc
import open_clip

OUT = ROOT / "experiments/paper_rerun/results"
DATE = "2026-11-18"


def main():
    t0 = time.time()
    sup, qry, classes, prompts = load_plantdoc(ROOT / "plantdoc", k_shot=5, seed=42)
    C = len(classes)
    dino_tf, bioclip_tf, dino_cfg = R.build_transforms()
    dino_aug = R.aug_pipeline(dino_cfg["mean"], dino_cfg["std"])
    bio_aug = R.aug_pipeline(*R.norm_stats(bioclip_tf))

    s_dino, s_bio, s_y = [], [], []
    for i, (p, y) in enumerate(sup):
        img = Image.open(p).convert("RGB")
        dv = [np.asarray(dino_tf(img), dtype=np.float32)]
        bv = [np.asarray(bioclip_tf(img), dtype=np.float32)]
        torch.manual_seed(42 + 10_000 + i)
        for _ in range(R.N_VIEWS - 1):
            dv.append(np.asarray(dino_aug(img), dtype=np.float32))
            bv.append(np.asarray(bio_aug(img), dtype=np.float32))
        s_dino.extend(dv); s_bio.extend(bv); s_y.extend([y] * R.N_VIEWS)
    s_dino, s_bio, s_y = np.stack(s_dino), np.stack(s_bio), np.array(s_y)

    q_dino = np.stack([np.asarray(dino_tf(Image.open(p).convert("RGB")), np.float32) for p, _ in qry])
    q_bio = np.stack([np.asarray(bioclip_tf(Image.open(p).convert("RGB")), np.float32) for p, _ in qry])
    q_y = np.array([y for _, y in qry])

    m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    tps = []
    with torch.no_grad():
        for c in classes:
            f = F.normalize(m.encode_text(tok(prompts[c])), dim=-1)
            tps.append(F.normalize(f.mean(0), dim=-1))
    tps = torch.stack(tps)

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
    sj = np.concatenate([sb, sd], axis=-1)
    qj = np.concatenate([qb, qd], axis=-1)

    idx_by_class = [np.where(q_y == c)[0].tolist() for c in range(C)]
    print("min test images per class:", min(len(v) for v in idx_by_class), flush=True)

    results = {}
    for w_sup in (1, 2, 3):
        accs = []
        rng = random.Random(42)
        for ep in range(50):
            pick = rng.sample(range(C), 5)
            qi = np.concatenate([idx_by_class[c] for c in pick])
            remap = {c: i for i, c in enumerate(pick)}
            qy = np.array([remap[v] for v in q_y[qi]])
            mask = np.isin(s_y, pick)
            accs.append(100 * accuracy_score(qy, fisher_anchor(
                sj[mask], np.array([remap[v] for v in s_y[mask]]),
                qj[qi], qb[qi], tps[pick], 5,
                support_weight=w_sup, max_anchors=2, conf_threshold=None)))
        results[f"support_weight_{w_sup}"] = round(float(np.mean(accs)), 2)
        print(f"  w_sup={w_sup}: {np.mean(accs):.2f}", flush=True)

    p = OUT / f"plantdoc_wsup_diagnosis_{DATE}.json"
    p.write_text(json.dumps({"date": DATE, "results": results,
                             "note": "isolates support_weight; episode filter is a no-op since min test images/class = 4"},
                            indent=2), encoding="utf-8")
    print(f"[OK] {p}")


if __name__ == "__main__":
    main()
