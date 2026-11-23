# -*- coding: utf-8 -*-
"""
Routing Reliability & Calibration Validation Suite (reviewer response experiments).
================================================================================
Addresses reviewer concerns on the uncertainty-based two-tier routing mechanism:
  A. Tier-separated accuracy: Tier-1 fast-exit accepted vs. Tier-2 escalated
     queries, on the canonical single run AND over the 30 benchmark episodes.
  B. Reliability audit: per-bin counts / accuracy / confidence / gap, ECE & MCE
     (equal-width and equal-mass bins) for Tier-1 gate probabilities and Tier-2
     consensus probabilities, before vs. after temperature scaling.
  C. Temperature scaling provenance: scalar T fitted by NLL on an INDEPENDENT
     stratified 50% calibration split, evaluated on the held-out 50%; compared
     against the paper's hardcoded power-scaling T=0.50 and an oracle fit.
  D. Risk-Coverage: selective-risk curves (AURC) for Tier-1 max-prob, the
     Shannon gate score C(x), and Tier-2 refined confidence; gate operating point.
  E. Routing threshold: tau sensitivity sweep + independent tuning on episodes
     0-14 with evaluation on episodes 15-29 (tests the "calibrated on a
     validation split" claim).

Protocol: single run replicates Phase-0 (8-view augmented support, N=440,
closed-form Ledoit-Wolf Fisher); episodes replicate benchmark_engine exactly
(seed 42 + ep*17, raw 5-shot support, cached embeddings).
Outputs: routing_validation_results.json + fig_reliability / fig_risk_coverage /
fig_tau_sensitivity (PDF+PNG) in this directory.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.core_math import (  # noqa: E402
    fit_fisher_lda, compute_fisher_ensemble_probs, refine_transductive,
    compute_class_prototypes, calculate_shannon_confidence,
)
from pipeline.data_engine import find_healthy_indices, load_plantvillage  # noqa: E402
from pipeline.biodino_edge_pipeline import load_backbones, extract_support_features  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = _PROJECT_ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
N_EPISODES, SEED, K_SHOT = 30, 42, 5
SCALE, W_LDA, TAU = 20.0, 0.75, 0.35
# Per-path edge costs measured on the physical Android testbed (Samsung SM-S721B,
# Exynos 2400e, ONNX Runtime 1.18.0 INT8, 4-thread CPU, NNAPI off); see
# experiments/edge_device_benchmark/results/benchmark_report.json (mean over 100 iters).
# T1 = Tier-1 fast-exit end-to-end; T2 = escalated Tier-2 end-to-end (T1 + incremental BioCLIP).
T1_MS, T2_MS = 59.78, 164.52
TAU_GRID = [round(0.05 * i, 2) for i in range(1, 15)]  # 0.05 .. 0.70


# ---------------------------------------------------------------- helpers
def bin_stats(conf, correct, n_bins=10, adaptive=False):
    """ECE / MCE with per-bin counts. adaptive=True -> equal-mass bins."""
    conf, correct = np.asarray(conf), np.asarray(correct)
    if adaptive:
        edges = np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    bins, ece, mce, mce30 = [], 0.0, 0.0, 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append(dict(lower=float(lo), upper=float(hi), count=0,
                             accuracy=None, confidence=None, gap=None))
            continue
        acc = float(correct[mask].mean())
        cf = float(conf[mask].mean())
        gap = abs(acc - cf)
        ece += gap * n / len(conf)
        mce = max(mce, gap)
        if n >= 30:
            mce30 = max(mce30, gap)
        bins.append(dict(lower=float(lo), upper=float(hi), count=n,
                         accuracy=acc, confidence=cf, gap=gap))
    return dict(ece_pct=100 * ece, mce_pct=100 * mce, mce_ge30_pct=100 * mce30, bins=bins)


def fit_temperature(logits, y, mask, power_probs=None, blend_text=None, w=W_LDA):
    """Scalar T minimizing NLL on the masked (calibration) subset.
    power_probs -> softmax(log(p)/T); blend_text -> w*softmax(logits/T)+(1-w)*blend_text.
    Returns (T, probs_at) where probs_at applies to the FULL set."""
    yt = torch.as_tensor(y, dtype=torch.long)

    def probs_at(T):
        if power_probs is not None:
            return F.softmax(torch.log(power_probs + 1e-12) / T, dim=-1)
        p = F.softmax(logits / T, dim=-1)
        if blend_text is not None:
            p = w * p + (1.0 - w) * blend_text
        return p

    def nll(T):
        p = probs_at(T)[mask]
        return F.nll_loss(torch.log(p + 1e-12), yt[mask]).item()

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x), probs_at


def ts_logits(logits, T):
    return F.softmax(logits / T, dim=-1)


def risk_coverage(scores, correct):
    """Selective risk vs coverage (rank by score, descending). AURC = mean risk."""
    order = np.argsort(-np.asarray(scores))
    cs = np.asarray(correct)[order]
    k = np.arange(1, len(cs) + 1)
    risks = 1.0 - np.cumsum(cs) / k
    cov = k / len(cs)
    return dict(coverage=cov.tolist(), risk=risks.tolist(),
                aurc_pct=float(100 * risks.mean()),
                final_acc_pct=float(100 * cs.mean()))


def tier_report(c_score, t1_pred, t2_pred, y, healthy, tau):
    """Tier-separated metrics at routing threshold tau."""
    acc_mask = c_score >= tau
    esc_mask = ~acc_mask
    n = len(y)
    diseased = ~np.isin(y, healthy)
    out = dict(tau=float(tau), fast_exit_pct=float(100 * acc_mask.mean()),
               n_accepted=int(acc_mask.sum()), n_escalated=int(esc_mask.sum()))
    out["tier1_acc_pct"] = float(100 * (t1_pred[acc_mask] == y[acc_mask]).mean()) if acc_mask.any() else None
    out["tier2_acc_pct"] = float(100 * (t2_pred[esc_mask] == y[esc_mask]).mean()) if esc_mask.any() else None
    casc = t1_pred.copy()
    casc[esc_mask] = t2_pred[esc_mask]
    out["cascaded_acc_pct"] = float(100 * (casc == y).mean())
    out["cascaded_f1_pct"] = float(100 * f1_score(y, casc, average="macro", zero_division=0))
    dm = acc_mask & diseased
    out["tier1_fnr_pct"] = float(100 * np.isin(t1_pred[dm], healthy).mean()) if dm.any() else 0.0
    em = esc_mask & diseased
    out["tier2_fnr_pct"] = float(100 * np.isin(t2_pred[em], healthy).mean()) if em.any() else 0.0
    f = acc_mask.mean()
    out["blended_latency_ms"] = float(f * T1_MS + (1 - f) * T2_MS)
    return out


def summarize(reps, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in reps if r[k] is not None]
        out[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)))
    return out


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    data = torch.load(CACHE, map_location="cpu")
    bio, dino = data["bio_feats"], data["dino_feats"]
    labels = data["labels"].numpy()
    paths = data["paths"]
    class_names = data["novel_classes"]
    joint = torch.cat([bio, dino], dim=-1).numpy()
    text_protos = data["bio_text_protos"]
    n_classes = len(class_names)
    healthy = find_healthy_indices(class_names)
    path_to_idx = {os.path.basename(p): i for i, p in enumerate(paths)}
    assert len(path_to_idx) == len(paths)

    # ================= Part A: canonical single run (augmented support) =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_backbones(device)
    support_samples, query_samples, _, _ = load_plantvillage(
        _PROJECT_ROOT / "plantvillage", k_shot=K_SHOT, seed=SEED)
    qi = [path_to_idx[os.path.basename(p)] for p, _ in query_samples]
    s_bio, s_dino, s_y_t = extract_support_features(support_samples, models, device=device)
    s_bio, s_dino, s_y_t = s_bio.cpu(), s_dino.cpu(), s_y_t.cpu()
    s_x = torch.cat([s_bio, s_dino], dim=-1).numpy()
    s_y = s_y_t.numpy()
    q_x, q_y = joint[qi], labels[qi]
    q_t = torch.tensor(q_x, dtype=torch.float32)
    q_bio, q_dino = bio[qi], dino[qi]

    dino_proto = compute_class_prototypes(s_dino, s_y_t, n_classes)
    t1_logits = SCALE * (q_dino @ dino_proto.t())
    t1_prob = F.softmax(t1_logits, dim=-1)
    c_score = calculate_shannon_confidence(t1_prob, n_classes).numpy()
    t1_pred = t1_logits.argmax(dim=-1).numpy()

    W, b, _ = fit_fisher_lda(s_x, s_y, device="cpu")
    ens_prob, _, text_prob = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=q_t,
        bioclip_features=q_bio, text_prototypes=text_protos,
        scale=SCALE, w_lda=W_LDA, device="cpu")
    ind_pred = ens_prob.argmax(dim=-1).numpy()
    lda_logits = q_t @ W.t() + b

    ref_pred, ref_prob, (Wr, br) = refine_transductive(
        support_features=s_x, support_labels=s_y, query_features=q_t,
        initial_preds=ind_pred, confidences=ens_prob.max(dim=-1).values,
        text_prob=text_prob, n_classes=n_classes, support_weight=3,
        max_anchors=20, conf_threshold=0.70, device="cpu", w_lda=W_LDA)
    ref_logits = q_t @ Wr.t() + br

    single = dict(
        n_queries=len(q_y),
        tier1_all_acc_pct=float(100 * (t1_pred == q_y).mean()),
        fisher_all_acc_pct=float(100 * (ind_pred == q_y).mean()),
        refined_all_acc_pct=float(100 * (ref_pred == q_y).mean()),
        routing_inductive=tier_report(c_score, t1_pred, ind_pred, q_y, healthy, TAU),
        routing_refined=tier_report(c_score, t1_pred, ref_pred, q_y, healthy, TAU),
    )
    print(f"[A] single-run: T1-all {single['tier1_all_acc_pct']:.4f} | "
          f"Fisher-all {single['fisher_all_acc_pct']:.4f} | refined-all {single['refined_all_acc_pct']:.4f}", flush=True)
    print(f"[A] routing(refined) @tau={TAU}: {json.dumps(single['routing_refined'], indent=1)}", flush=True)

    # Gate-rule diagnostics: which acceptance rule could yield the paper's 88.4%?
    t1_maxp = t1_prob.max(dim=-1).values.numpy()
    top2 = np.sort(t1_prob.numpy(), axis=1)[:, -2:]
    t1_margin = top2[:, 1] - top2[:, 0]
    gate_diag = {}
    for rule, mask in [("shannon_C_ge_0.35", c_score >= 0.35),
                       ("maxprob_ge_0.35", t1_maxp >= 0.35),
                       ("maxprob_ge_0.50", t1_maxp >= 0.50),
                       ("margin_ge_0.35", t1_margin >= 0.35)]:
        gate_diag[rule] = dict(
            coverage_pct=float(100 * mask.mean()),
            accepted_acc_pct=float(100 * (t1_pred[mask] == q_y[mask]).mean()) if mask.any() else None)
    single["gate_rule_diagnostics"] = gate_diag
    single["c_score_quantiles"] = {str(q): float(np.quantile(c_score, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]}
    print(f"[A] gate-rule diagnostics: {json.dumps(gate_diag)}", flush=True)
    print(f"[A] C(x) quantiles: {single['c_score_quantiles']}", flush=True)

    # ================= Part B: 30-episode tier-separated routing ============
    class_indices = [np.where(labels == c)[0].tolist() for c in range(n_classes)]
    ep_reps_ind, ep_reps_ref, ep_store = [], [], []
    for ep in range(N_EPISODES):
        rng = random.Random(SEED + ep * 17)
        si, qi_ep = [], []
        for c in range(n_classes):
            sh = class_indices[c].copy()
            rng.shuffle(sh)
            si.extend(sh[:K_SHOT])
            qi_ep.extend(sh[K_SHOT:])
        sx, sy = joint[si], labels[si]
        qx, qy = joint[qi_ep], labels[qi_ep]
        qt = torch.tensor(qx, dtype=torch.float32)
        dp = compute_class_prototypes(dino[si], torch.tensor(labels[si]), n_classes)
        tl = SCALE * (dino[qi_ep] @ dp.t())
        tp = F.softmax(tl, dim=-1)
        cs = calculate_shannon_confidence(tp, n_classes).numpy()
        t1p = tl.argmax(dim=-1).numpy()
        We, be, _ = fit_fisher_lda(sx, sy, device="cpu")
        ep_prob, _, tp_txt = compute_fisher_ensemble_probs(
            fisher_weights=We, fisher_biases=be, joint_features=qt,
            bioclip_features=bio[qi_ep], text_prototypes=text_protos,
            scale=SCALE, w_lda=W_LDA, device="cpu")
        ip = ep_prob.argmax(dim=-1).numpy()
        rp, _, _ = refine_transductive(
            support_features=sx, support_labels=sy, query_features=qt,
            initial_preds=ip, confidences=ep_prob.max(dim=-1).values,
            text_prob=tp_txt, n_classes=n_classes, support_weight=3,
            max_anchors=20, conf_threshold=0.70, device="cpu", w_lda=W_LDA)
        ep_reps_ind.append(tier_report(cs, t1p, ip, qy, healthy, TAU))
        ep_reps_ref.append(tier_report(cs, t1p, rp, qy, healthy, TAU))
        ep_store.append(dict(c=cs, t1=t1p, ref=rp, y=qy))
    keys = ["fast_exit_pct", "tier1_acc_pct", "tier2_acc_pct", "cascaded_acc_pct",
            "cascaded_f1_pct", "tier1_fnr_pct", "tier2_fnr_pct", "blended_latency_ms"]
    episodes = dict(inductive=summarize(ep_reps_ind, keys),
                    refined=summarize(ep_reps_ref, keys),
                    per_episode_refined=ep_reps_ref)
    print(f"[B] 30-ep refined routing: fast-exit {episodes['refined']['fast_exit_pct']['mean']:.2f}% | "
          f"T1 acc {episodes['refined']['tier1_acc_pct']['mean']:.2f} | "
          f"T2 acc {episodes['refined']['tier2_acc_pct']['mean']:.2f} | "
          f"casc {episodes['refined']['cascaded_acc_pct']['mean']:.2f}", flush=True)

    # ================= Part C: calibration audit (single run) ===============
    # Independent stratified 50/50 calibration/test split of the query set
    rng = np.random.RandomState(SEED)
    cal_mask = np.zeros(len(q_y), dtype=bool)
    for c in range(n_classes):
        idx = np.where(q_y == c)[0]
        rng.shuffle(idx)
        cal_mask[idx[: len(idx) // 2]] = True
    test_mask = ~cal_mask

    def audit(prob, logits, name, power=False, blend=None):
        conf = prob.max(dim=-1).values.numpy()
        pred = prob.argmax(dim=-1).numpy()
        correct = (pred == q_y).astype(float)
        full = bin_stats(conf, correct)
        full_ad = bin_stats(conf, correct, adaptive=True)
        T, probs_at = fit_temperature(
            logits, q_y, cal_mask,
            power_probs=prob if power else None, blend_text=blend)
        p_ts = probs_at(T)
        conf_ts = p_ts.max(dim=-1).values.numpy()
        corr_ts = (p_ts.argmax(dim=-1).numpy() == q_y).astype(float)
        held = bin_stats(conf_ts[test_mask], corr_ts[test_mask])
        held_raw = bin_stats(conf[test_mask], correct[test_mask])
        T_or, _ = fit_temperature(
            logits, q_y, test_mask,
            power_probs=prob if power else None, blend_text=blend)
        print(f"[C] {name}: ECE {full['ece_pct']:.2f} MCE {full['mce_pct']:.2f} | "
              f"T_fit={T:.3f} (oracle {T_or:.3f}) | held-out ECE {held_raw['ece_pct']:.2f}->{held['ece_pct']:.2f} "
              f"MCE {held_raw['mce_pct']:.2f}->{held['mce_pct']:.2f}", flush=True)
        return dict(full_equal_width=full, full_equal_mass=full_ad,
                    T_fitted_calib=T, T_oracle_test=T_or,
                    heldout_before=held_raw, heldout_after=held,
                    ts_full_equal_width=bin_stats(conf_ts, corr_ts))

    calib = dict(
        n_calib=int(cal_mask.sum()), n_test=int(test_mask.sum()),
        tier1=audit(t1_prob, t1_logits, "Tier-1 (gate input)"),
        tier2_inductive=audit(ens_prob, lda_logits, "Tier-2 inductive (logit TS)", blend=text_prob),
        tier2_refined=audit(ref_prob, ref_logits, "Tier-2 refined (logit TS)", blend=text_prob),
        tier2_refined_power=audit(ref_prob, ref_logits, "Tier-2 refined (power TS, paper style)", power=True),
    )
    # Paper's hardcoded T=0.50 power scaling, for direct comparison
    p_paper = F.softmax(torch.log(ref_prob + 1e-12) / 0.50, dim=-1)
    conf_pp = p_paper.max(dim=-1).values.numpy()
    corr_pp = (p_paper.argmax(dim=-1).numpy() == q_y).astype(float)
    calib["paper_T0.50_power"] = dict(
        full_equal_width=bin_stats(conf_pp, corr_pp),
        heldout=bin_stats(conf_pp[test_mask], corr_pp[test_mask]))
    print(f"[C] paper T=0.50 power: ECE {calib['paper_T0.50_power']['full_equal_width']['ece_pct']:.2f} "
          f"MCE {calib['paper_T0.50_power']['full_equal_width']['mce_pct']:.2f}", flush=True)

    # ================= Part D: risk-coverage ================================
    rc = dict(
        tier1_maxprob=risk_coverage(t1_prob.max(dim=-1).values.numpy(), (t1_pred == q_y)),
        gate_score=risk_coverage(c_score, (t1_pred == q_y)),
        refined_conf=risk_coverage(ref_prob.max(dim=-1).values.numpy(), (ref_pred == q_y)),
    )
    gate_pt = single["routing_refined"]
    rc["gate_operating_point"] = dict(
        coverage_pct=gate_pt["fast_exit_pct"],
        risk_pct=100 - gate_pt["tier1_acc_pct"])
    for k in ["tier1_maxprob", "gate_score", "refined_conf"]:
        cov = np.array(rc[k]["coverage"]); risk = np.array(rc[k]["risk"])
        j = int(np.argmin(np.abs(cov - 0.884)))
        rc[k]["risk_at_88.4pct_coverage"] = float(100 * risk[j])
        print(f"[D] {k}: AURC {rc[k]['aurc_pct']:.3f}% | risk@88.4% cov {100*risk[j]:.3f}%", flush=True)

    # ================= Part E: tau sweep + independent tuning ===============
    sweep_single = []
    for tau in TAU_GRID:
        r = tier_report(c_score, t1_pred, ref_pred, q_y, healthy, tau)
        sweep_single.append(r)
    # episode-level tuning: episodes 0-14 tune, 15-29 test
    tune_rows, test_rows = [], []
    for tau in TAU_GRID:
        tr = [tier_report(e["c"], e["t1"], e["ref"], e["y"], healthy, tau) for e in ep_store[:15]]
        te = [tier_report(e["c"], e["t1"], e["ref"], e["y"], healthy, tau) for e in ep_store[15:]]
        tune_rows.append(dict(tau=tau, **summarize(tr, ["cascaded_acc_pct", "fast_exit_pct", "tier1_acc_pct"])))
        test_rows.append(dict(tau=tau, **summarize(te, ["cascaded_acc_pct", "fast_exit_pct", "tier1_acc_pct", "blended_latency_ms"])))
    best = max(tune_rows, key=lambda r: (r["cascaded_acc_pct"]["mean"], r["fast_exit_pct"]["mean"]))
    tau_star = best["tau"]
    at_035 = next(r for r in test_rows if abs(r["tau"] - 0.35) < 1e-9)
    at_star = next(r for r in test_rows if abs(r["tau"] - tau_star) < 1e-9)
    tuning = dict(tau_grid=TAU_GRID, tune_episodes="0-14", test_episodes="15-29",
                  tune_curve=tune_rows, test_curve=test_rows,
                  tau_star=tau_star,
                  test_at_tau_star=at_star, test_at_0p35=at_035)
    print(f"[E] tau* tuned on eps 0-14 = {tau_star} | test casc acc: tau* "
          f"{at_star['cascaded_acc_pct']['mean']:.2f} vs 0.35 {at_035['cascaded_acc_pct']['mean']:.2f}", flush=True)

    # ================= figures ==============================================
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})

    def reliability_panel(ax, stats, title, color):
        bins = [b for b in stats["bins"] if b["count"] > 0]
        centers = [(b["lower"] + b["upper"]) / 2 for b in bins]
        accs = [b["accuracy"] for b in bins]
        gaps = [b["gap"] for b in bins]
        widths = [0.8 * (b["upper"] - b["lower"]) for b in bins]
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
        ax.bar(centers, accs, width=widths, color=color, alpha=0.7, edgecolor="black", lw=0.5, label="Accuracy")
        ax.bar(centers, gaps, bottom=np.minimum(accs, [b["confidence"] for b in bins]),
               width=widths, color="#95a5a6", alpha=0.5, hatch="//", edgecolor="#7f8c8d", lw=0.4, label="Gap")
        for b, cx in zip(bins, centers):
            ax.text(cx, 0.02, f"n={b['count']}", ha="center", fontsize=6.5, rotation=90)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"{title}\nECE={stats['ece_pct']:.2f}%  MCE={stats['mce_pct']:.2f}%", fontsize=9.5)
        ax.grid(True, ls=":", alpha=0.5); ax.legend(loc="upper left", fontsize=7.5)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    reliability_panel(axes[0, 0], calib["tier1"]["full_equal_width"],
                      "(a) Tier-1 gate probabilities (uncalibrated)", "#e74c3c")
    reliability_panel(axes[0, 1], calib["tier1"]["ts_full_equal_width"],
                      f"(b) Tier-1 after temperature scaling (T={calib['tier1']['T_fitted_calib']:.2f}, held-out fit)", "#2ecc71")
    reliability_panel(axes[1, 0], calib["tier2_refined"]["full_equal_width"],
                      "(c) Tier-2 refined consensus (uncalibrated)", "#e74c3c")
    reliability_panel(axes[1, 1], calib["paper_T0.50_power"]["full_equal_width"],
                      "(d) Tier-2 after power-transform T=0.50 (bin collapse)", "#f39c12")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(HERE / f"fig_reliability.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for k, lbl, col in [("tier1_maxprob", "Tier-1 max-prob", "#e74c3c"),
                        ("gate_score", "Shannon gate score C(x)", "#2980b9"),
                        ("refined_conf", "Tier-2 refined confidence", "#27ae60")]:
        ax.plot(rc[k]["coverage"], np.array(rc[k]["risk"]) * 100, color=col, lw=1.6,
                label=f"{lbl} (AURC={rc[k]['aurc_pct']:.2f}%)")
    gp = rc["gate_operating_point"]
    ax.scatter([gp["coverage_pct"] / 100], [gp["risk_pct"]], color="black", zorder=5, marker="D",
               label=f"Gate @tau=0.35 (cov={gp['coverage_pct']:.1f}%, risk={gp['risk_pct']:.2f}%)")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (%)")
    ax.set_title("Risk-Coverage: uncertainty-based routing (PlantVillage single run)")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(HERE / f"fig_risk_coverage.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6.2, 4.4))
    taus = [r["tau"] for r in sweep_single]
    ax1.plot(taus, [r["fast_exit_pct"] for r in sweep_single], "o-", color="#2980b9", label="Fast-exit rate")
    ax1.set_xlabel(r"Routing threshold $\tau$"); ax1.set_ylabel("Fast-exit rate (%)", color="#2980b9")
    ax2 = ax1.twinx()
    ax2.plot(taus, [r["cascaded_acc_pct"] for r in sweep_single], "s-", color="#27ae60", label="Cascaded accuracy")
    ax2.plot(taus, [r["tier1_acc_pct"] for r in sweep_single], "^--", color="#e74c3c", label="Tier-1 accepted accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax1.axvline(0.35, color="gray", ls="--", lw=1)
    ax1.text(0.355, 5, r"$\tau=0.35$ (paper)", fontsize=8, color="gray")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.set_title(r"Routing threshold sensitivity (single run, refined Tier-2)")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(HERE / f"fig_tau_sensitivity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ================= persist ==============================================
    out = dict(protocol=dict(seed=SEED, episodes=N_EPISODES, k_shot=K_SHOT, scale=SCALE,
                             w_lda=W_LDA, tau=TAU, cache=str(CACHE),
                             single_run_support="8-view augmented (N=440)",
                             episode_support="raw 5-shot (benchmark protocol)"),
               single_run=single, episodes_30=episodes, calibration=calib,
               risk_coverage=rc, tau_sensitivity=dict(single_run=sweep_single, **tuning))
    (HERE / "routing_validation_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[SUCCESS] saved routing_validation_results.json + 3 figures ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()