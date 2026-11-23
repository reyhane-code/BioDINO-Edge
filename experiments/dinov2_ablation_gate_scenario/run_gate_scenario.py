# -*- coding: utf-8 -*-
"""
Gate -> BioCLIP Scenario  vs.  the Deployed Two-Tier Cascade.
=============================================================
Tests the proposed simplification:

    Use DINOv2 ONLY as a confidence gate (reject low-confidence inputs), then let
    BioCLIP + Fisher-LDA classify EVERY accepted query. The DINOv2 class label is
    NEVER used as the final answer.

Compared against the paper's two-tier cascade under the SAME 30-episode benchmark
protocol (seed 42 + ep*17, raw 5-shot, cached embeddings, transductive refinement
support_weight=3 / max_anchors=20 / conf_threshold=0.70).

Systems:
  A. cascade      : DINOv2 label on confident fast-exit (C(x)>=0.35), BioCLIP-Fisher
                    refined on the escalated rest.  (the published system)
  B. bioclip_all  : BioCLIP-Fisher refined on ALL queries, no gate, no DINOv2 label.
  C. gate_bioclip : DINOv2 Shannon gate C(x)>=tau accepts -> BioCLIP-Fisher refined
                    classifies accepted; low-confidence rejected (abstain).
  D. conf_abstain : BioCLIP-Fisher refined on all, abstain when refined conf < tau.

Latency model (edge constants, same as routing_calibration_validation):
  T1 = 21.12 ms (DINOv2 gate only), T2 = 64.02 ms (DINOv2 + BioCLIP + Fisher).
  A: f*T1 + (1-f)*T2  (f = fast-exit fraction)
  B: T2               (BioCLIP on everything)
  C: accept*T2 + reject*T1   (rejected queries stop at the cheap DINOv2 gate)
  D: T2               (abstention is post-hoc, BioCLIP already ran)

NOTE ON BLUR / OOD: the PlantVillage cache holds only clean leaf images (no blur or
out-of-distribution labels), so blur / non-leaf rejection cannot be measured here.
That role is handled at deployment by the foliage OOD gate already implemented in
pipeline/export_edge_models.py. This script measures the confidence-rejection part
of the proposal, which is what the cache can validate.

Outputs: gate_scenario_results.json (this directory).
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT), str(_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.core_math import (  # noqa: E402
    fit_fisher_lda, compute_fisher_ensemble_probs, refine_transductive,
    compute_class_prototypes, calculate_shannon_confidence,
)
from pipeline.data_engine import find_healthy_indices  # noqa: E402

CACHE = _ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
HERE = Path(__file__).resolve().parent
OUT = HERE / "gate_scenario_results.json"
PUBLISHED = _ROOT / "experiments" / "routing_calibration_validation" / "routing_validation_results.json"
N_EPISODES, SEED, K_SHOT = 30, 42, 5
SCALE, W_LDA, TAU = 20.0, 0.75, 0.35
# Measured on-device per-path costs (Samsung SM-S721B, INT8 CPU); see edge_device_benchmark.
T1_MS, T2_MS = 59.78, 164.52
REFINE = dict(support_weight=3, max_anchors=20, conf_threshold=0.70)
GATE_GRID = [0.0, 0.02, 0.05, 0.08, 0.10, 0.13, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
CONF_GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]


def build_episodes():
    data = torch.load(CACHE, map_location="cpu")
    bio, dino = data["bio_feats"], data["dino_feats"]
    labels = data["labels"].numpy()
    text_protos = data["bio_text_protos"]
    class_names = data["novel_classes"]
    n_classes = len(class_names)
    healthy = find_healthy_indices(class_names)
    joint = torch.cat([bio, dino], dim=-1).numpy()
    class_indices = [np.where(labels == c)[0].tolist() for c in range(n_classes)]
    episodes = []
    for ep in range(N_EPISODES):
        rng = random.Random(SEED + ep * 17)
        si, qi = [], []
        for c in range(n_classes):
            sh = class_indices[c].copy()
            rng.shuffle(sh)
            si.extend(sh[:K_SHOT])
            qi.extend(sh[K_SHOT:])
        episodes.append(dict(ep=ep, si=si, qi=qi, sy=labels[si], y=labels[qi],
                             s_x=joint[si], q_x=joint[qi], q_bio=bio[qi],
                             q_dino=dino[qi], s_dino=dino[si],
                             text_protos=text_protos, n_classes=n_classes))
    return episodes, healthy, class_names


def metrics(pred, y, healthy, mask):
    """Accuracy / macro-F1 / FNR on the accepted subset defined by mask."""
    cov = float(100 * mask.mean())
    if mask.sum() == 0:
        return dict(coverage_pct=cov, acc_pct=None, f1_pct=None, fnr_pct=None)
    p, yy = pred[mask], y[mask]
    dis = ~np.isin(yy, healthy)
    fnr = float(100 * np.isin(p[dis], healthy).mean()) if dis.any() else 0.0
    return dict(coverage_pct=cov,
                acc_pct=float(100 * (p == yy).mean()),
                f1_pct=float(100 * f1_score(yy, p, average="macro", zero_division=0)),
                fnr_pct=fnr)


def risk_coverage(scores, correct):
    order = np.argsort(-np.asarray(scores))
    cs = np.asarray(correct)[order]
    k = np.arange(1, len(cs) + 1)
    risks = 1.0 - np.cumsum(cs) / k
    cov = k / len(cs)
    return cov, risks, float(100 * risks.mean())


def acc_at_coverage(cov, risks, targets=(0.90, 0.95, 0.99)):
    out = {}
    for t in targets:
        j = int(np.argmin(np.abs(cov - t)))
        out[f"acc_at_{int(t*100)}pct_cov"] = float(100 * (1 - risks[j]))
    return out


def agg(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals))) if vals else None
    return out


def main():
    t0 = time.time()
    episodes, healthy, class_names = build_episodes()
    print(f"[*] {len(episodes)} episodes, {len(episodes[0]['y'])} queries each "
          f"({time.time()-t0:.1f}s)", flush=True)

    A_rows, B_rows = [], []
    C_rows = {tau: [] for tau in GATE_GRID}
    D_rows = {tau: [] for tau in CONF_GRID}
    aurc_gate, aurc_conf = [], []
    op_gate, op_conf = [], []
    mkeys = ["coverage_pct", "acc_pct", "f1_pct", "fnr_pct"]

    for ep in episodes:
        y, n_classes = ep["y"], ep["n_classes"]
        # --- DINOv2 Tier-1 gate (cheap, runs before BioCLIP) ---
        dp = compute_class_prototypes(ep["s_dino"], torch.tensor(ep["sy"]), n_classes)
        tl = SCALE * (ep["q_dino"] @ dp.t())
        tp = F.softmax(tl, dim=-1)
        c_score = calculate_shannon_confidence(tp, n_classes).numpy()
        t1_pred = tl.argmax(dim=-1).numpy()
        # --- BioCLIP + Fisher refined (the strong classifier) ---
        q_t = torch.tensor(ep["q_x"], dtype=torch.float32)
        W, b, _ = fit_fisher_lda(ep["s_x"], ep["sy"], device="cpu")
        prob, _, text_prob = compute_fisher_ensemble_probs(
            fisher_weights=W, fisher_biases=b, joint_features=q_t,
            bioclip_features=ep["q_bio"], text_prototypes=ep["text_protos"],
            scale=SCALE, w_lda=W_LDA, device="cpu")
        ref_pred, ref_prob, _ = refine_transductive(
            support_features=ep["s_x"], support_labels=ep["sy"], query_features=q_t,
            initial_preds=prob.argmax(dim=-1).numpy(),
            confidences=prob.max(dim=-1).values, text_prob=text_prob,
            n_classes=n_classes, device="cpu", w_lda=W_LDA, **REFINE)
        ref_conf = ref_prob.max(dim=-1).values.numpy()

        # --- A: published cascade (DINOv2 label on accepted, BioCLIP on rest) ---
        acc_mask = c_score >= TAU
        casc = t1_pred.copy()
        casc[~acc_mask] = ref_pred[~acc_mask]
        mA = metrics(casc, y, healthy, np.ones(len(y), dtype=bool))
        f = float(acc_mask.mean())
        mA["fast_exit_pct"] = float(100 * f)
        mA["latency_ms"] = float(f * T1_MS + (1 - f) * T2_MS)
        A_rows.append(mA)

        # --- B: BioCLIP-Fisher on ALL queries (no DINOv2 label, no gate) ---
        mB = metrics(ref_pred, y, healthy, np.ones(len(y), dtype=bool))
        mB["latency_ms"] = float(T2_MS)
        B_rows.append(mB)

        # --- C: DINOv2 gate accepts -> BioCLIP classifies accepted (abstain rest) ---
        for tau in GATE_GRID:
            mask = c_score >= tau
            m = metrics(ref_pred, y, healthy, mask)
            af = float(mask.mean())
            m["latency_ms"] = float(af * T2_MS + (1 - af) * T1_MS)
            C_rows[tau].append(m)

        # --- D: BioCLIP on all, abstain when refined confidence < tau ---
        for tau in CONF_GRID:
            mask = ref_conf >= tau
            m = metrics(ref_pred, y, healthy, mask)
            m["latency_ms"] = float(T2_MS)
            D_rows[tau].append(m)

        # --- risk-coverage of each gate against BioCLIP correctness ---
        correct = (ref_pred == y)
        cg, rg, ag = risk_coverage(c_score, correct)
        cc, rc, ac = risk_coverage(ref_conf, correct)
        aurc_gate.append(ag)
        aurc_conf.append(ac)
        op_gate.append(acc_at_coverage(cg, rg))
        op_conf.append(acc_at_coverage(cc, rc))

    A = agg(A_rows, mkeys + ["fast_exit_pct", "latency_ms"])
    B = agg(B_rows, mkeys + ["latency_ms"])
    C = {str(tau): agg(C_rows[tau], mkeys + ["latency_ms"]) for tau in GATE_GRID}
    D = {str(tau): agg(D_rows[tau], mkeys + ["latency_ms"]) for tau in CONF_GRID}

    def avg_op(rows):
        keys = rows[0].keys()
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    published = None
    if PUBLISHED.exists():
        pj = json.loads(PUBLISHED.read_text())
        published = pj["episodes_30"]["refined"]["cascaded_acc_pct"]["mean"]

    print("\n[A] published cascade   : acc {:.2f} | fast-exit {:.2f}% | FNR {:.3f}% | {:.2f} ms".format(
        A["acc_pct"]["mean"], A["fast_exit_pct"]["mean"], A["fnr_pct"]["mean"], A["latency_ms"]["mean"]), flush=True)
    print("[B] BioCLIP-on-all      : acc {:.2f} | coverage 100% | FNR {:.3f}% | {:.2f} ms".format(
        B["acc_pct"]["mean"], B["fnr_pct"]["mean"], B["latency_ms"]["mean"]), flush=True)
    if published is not None:
        print(f"    (published routing_validation cascade acc = {published:.2f}%)", flush=True)
    print("\n[C] gate(C(x)) -> BioCLIP, abstain low-confidence:", flush=True)
    for tau in GATE_GRID:
        m = C[str(tau)]
        if m["acc_pct"]:
            print(f"    tau={tau:.2f}: cov {m['coverage_pct']['mean']:6.2f}% | "
                  f"acc {m['acc_pct']['mean']:6.2f}% | FNR {m['fnr_pct']['mean']:.3f}% | "
                  f"{m['latency_ms']['mean']:.2f} ms", flush=True)
    print("\n[D] BioCLIP + abstain on refined confidence:", flush=True)
    for tau in CONF_GRID:
        m = D[str(tau)]
        if m["acc_pct"]:
            print(f"    tau={tau:.3f}: cov {m['coverage_pct']['mean']:6.2f}% | "
                  f"acc {m['acc_pct']['mean']:6.2f}% | FNR {m['fnr_pct']['mean']:.3f}%", flush=True)
    print(f"\nAURC (gate C(x)) = {np.mean(aurc_gate):.3f}% | AURC (refined conf) = {np.mean(aurc_conf):.3f}%", flush=True)
    print(f"Gate C(x) acc@90/95/99% cov: {avg_op(op_gate)}", flush=True)
    print(f"Refined-conf acc@90/95/99% cov: {avg_op(op_conf)}", flush=True)

    out = dict(protocol=dict(seed=SEED, episodes=N_EPISODES, k_shot=K_SHOT, scale=SCALE,
                             w_lda=W_LDA, tau=TAU, T1_MS=T1_MS, T2_MS=T2_MS, refine=REFINE,
                             cache=str(CACHE),
                             note="PlantVillage cache = clean leaves only; blur/OOD rejection "
                                  "not measurable here (foliage gate lives in export_edge_models.py)"),
               A_cascade=A, B_bioclip_all=B,
               C_gate_bioclip=C, D_conf_abstain=D,
               risk_coverage=dict(auc_gate_mean_pct=float(np.mean(aurc_gate)),
                                  auc_refined_conf_mean_pct=float(np.mean(aurc_conf)),
                                  gate_operating_points=avg_op(op_gate),
                                  conf_operating_points=avg_op(op_conf)),
               published_cascade_acc_pct=published)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[SUCCESS] saved {OUT.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
