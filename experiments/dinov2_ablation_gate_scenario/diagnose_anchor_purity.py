# -*- coding: utf-8 -*-
"""
Diagnostic: why does pure-LDA transductive refinement collapse while the
text-blended ensemble improves?

Hypothesis: pure Fisher-LDA softmax is overconfident, so the conf>=0.70 anchor
gate admits many WRONG pseudo-labels (low purity) -> manifold drift on re-fit.
The 0.25 BioCLIP text blend softens/calibrates confidence -> purer anchors.

We measure, over the same 30 episodes, for w_lda=1.00 (pure) vs 0.75 (ensemble):
  - anchor purity at conf_threshold=0.70 (max_anchors=20/class)
  - number of anchors admitted
  - mean top-1 confidence and fraction of queries above 0.70
No re-fit is needed for purity; this is the inductive anchor selection only.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT), str(_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from pipeline.core_math import fit_fisher_lda, compute_fisher_ensemble_probs  # noqa: E402

CACHE = _ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
SEED, K_SHOT, N_EP, NC = 42, 5, 30, 11


def main():
    d = torch.load(CACHE, map_location="cpu")
    bio, dino = d["bio_feats"], d["dino_feats"]
    lab = d["labels"].numpy()
    joint = torch.cat([bio, dino], -1).numpy()
    tp = d["bio_text_protos"]
    ci = [np.where(lab == c)[0].tolist() for c in range(NC)]
    stats = {1.0: dict(pur=[], n=[], cf=[], frac=[]),
             0.75: dict(pur=[], n=[], cf=[], frac=[])}
    for ep in range(N_EP):
        rng = random.Random(SEED + ep * 17)
        si, qi = [], []
        for c in range(NC):
            sh = ci[c].copy(); rng.shuffle(sh)
            si += sh[:K_SHOT]; qi += sh[K_SHOT:]
        sx, sy, y = joint[si], lab[si], lab[qi]
        qt = torch.tensor(joint[qi], dtype=torch.float32)
        W, b, _ = fit_fisher_lda(sx, sy)
        for w in (1.0, 0.75):
            p, _, _ = compute_fisher_ensemble_probs(
                fisher_weights=W, fisher_biases=b, joint_features=qt,
                bioclip_features=bio[qi], text_prototypes=tp, scale=20.0, w_lda=w)
            pred = p.argmax(-1).numpy()
            conf = p.max(-1).values.numpy()
            tot = bad = 0
            for c in range(NC):
                idx = np.where(pred == c)[0]; cf = conf[idx]
                keep = cf >= 0.70; idx, cf = idx[keep], cf[keep]
                k = min(20, len(idx))
                if k:
                    sel = idx[np.argsort(-cf)[:k]]
                    tot += len(sel); bad += int((y[sel] != c).sum())
            s = stats[w]
            s["pur"].append(100 * (1 - bad / tot) if tot else 100.0)
            s["n"].append(tot)
            s["cf"].append(float(conf.mean()))
            s["frac"].append(float((conf >= 0.70).mean() * 100))
    for w, name in [(1.0, "pure_lda "), (0.75, "ensemble ")]:
        s = stats[w]
        print(f"{name} w_lda={w}: anchor purity {np.mean(s['pur']):.2f}% | "
              f"anchors/ep {np.mean(s['n']):.0f} | mean top1 conf {np.mean(s['cf']):.3f} | "
              f"%queries conf>=0.70: {np.mean(s['frac']):.1f}%", flush=True)


if __name__ == "__main__":
    main()
