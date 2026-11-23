# -*- coding: utf-8 -*-
"""
Transductive Recalibration Validation Suite (reviewer response experiments).
================================================================================
Addresses four reviewer concerns on the soft-anchored transductive module:
  A. Paired statistical significance of Inductive vs. Transductive gains over
     the SAME 30 Monte Carlo episodes (paired t-test, Wilcoxon, sign test).
  B. One-factor-at-a-time sensitivity sweep: support_weight, conf_threshold,
     max_anchors (buffer capacity per class).
  C. Pseudo-label (anchor) purity vs. confidence threshold + coverage.
  D. Sequential streaming simulation with FIFO anchor buffer: online accuracy
     per chunk, buffer purity drift, and error accumulation over updates.

Protocol replicates pipeline/benchmark_engine.run_plantvillage_montecarlo
exactly (seed 42 + ep*17, stratified 5-shot splits, cached embeddings).
Outputs: transductive_validation_results.json (this directory).
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats
from sklearn.metrics import accuracy_score, f1_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.core_math import fit_fisher_lda, compute_fisher_ensemble_probs  # noqa: E402
from pipeline.data_engine import find_healthy_indices  # noqa: E402

CACHE = _PROJECT_ROOT / "cache_artifacts" / "pv_novel_embeddings.pt"
OUT = Path(__file__).resolve().parent / "transductive_validation_results.json"
N_EPISODES = 30
SEED = 42
K_SHOT = 5
SCALE = 20.0
W_LDA = 0.75
DEFAULTS = dict(support_weight=3, max_anchors=20, conf_threshold=0.70)


def load_episodes():
    """Replicates the benchmark episode splits and inductive forward pass."""
    data = torch.load(CACHE, map_location="cpu")
    bio = data["bio_feats"]
    dino = data["dino_feats"]
    labels = data["labels"].numpy()
    joint = torch.cat([bio, dino], dim=-1).numpy()
    text_protos = data["bio_text_protos"]
    class_names = data["novel_classes"]
    n_classes = len(class_names)
    class_indices = [np.where(labels == c)[0].tolist() for c in range(n_classes)]

    episodes = []
    for ep in range(N_EPISODES):
        rng = random.Random(SEED + ep * 17)
        support_indices, query_indices = [], []
        for c in range(n_classes):
            shuffled = class_indices[c].copy()
            rng.shuffle(shuffled)
            support_indices.extend(shuffled[:K_SHOT])
            query_indices.extend(shuffled[K_SHOT:])
        s_x, s_y = joint[support_indices], labels[support_indices]
        q_x, q_y = joint[query_indices], labels[query_indices]
        q_t = torch.tensor(q_x, dtype=torch.float32)
        W, b, _ = fit_fisher_lda(s_x, s_y, device="cpu")
        prob, _, text_prob = compute_fisher_ensemble_probs(
            fisher_weights=W, fisher_biases=b, joint_features=q_t,
            bioclip_features=bio[query_indices], text_prototypes=text_protos,
            scale=SCALE, w_lda=W_LDA, device="cpu")
        pred = prob.argmax(dim=-1).numpy()
        conf = prob.max(dim=-1).values
        episodes.append(dict(ep=ep, s_x=s_x, s_y=s_y, q_x=q_x, q_y=q_y,
                             q_t=q_t, q_bio=bio[query_indices], text_protos=text_protos,
                             ind_pred=pred, ind_conf=conf, ind_prob=prob, text_prob=text_prob,
                             n_classes=n_classes))
    return episodes, class_names


def select_anchors(pred, conf, n_classes, max_anchors, conf_threshold):
    """Mirrors core_math.refine_transductive anchor selection (top-k per class)."""
    sel = {}
    for c in range(n_classes):
        idx = np.where(pred == c)[0]
        if len(idx) == 0:
            continue
        cf = conf[idx]
        if conf_threshold is not None:
            keep = (cf >= conf_threshold).numpy()
            idx, cf = idx[keep], cf[keep]
        k = min(max_anchors, len(idx))
        if k > 0:
            top = torch.topk(cf, k).indices.numpy()
            sel[c] = idx[top]
    return sel


def run_transductive(ep, support_weight, max_anchors, conf_threshold):
    """Closed-form refinement identical to core_math.refine_transductive."""
    feats = [ep["s_x"]] * support_weight
    labs = [ep["s_y"]] * support_weight
    sel = select_anchors(ep["ind_pred"], ep["ind_conf"], ep["n_classes"],
                         max_anchors, conf_threshold)
    for c, idx in sel.items():
        feats.append(ep["q_x"][idx])
        labs.append(np.full(len(idx), c))
    W, b, _ = fit_fisher_lda(np.concatenate(feats), np.concatenate(labs), device="cpu")
    prob, _, _ = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=ep["q_t"],
        text_prob=ep["text_prob"], w_lda=W_LDA, device="cpu")
    pred = prob.argmax(dim=-1).numpy()
    return pred, sel


def paired_stats(ind, trans):
    ind, trans = np.asarray(ind), np.asarray(trans)
    diff = trans - ind
    n = len(diff)
    t, p_t = sstats.ttest_rel(trans, ind)
    try:
        w, p_w = sstats.wilcoxon(trans, ind)
    except ValueError:
        w, p_w = float("nan"), float("nan")
    sd = diff.std(ddof=1)
    ci = sstats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(n_episodes=n, mean_inductive=float(ind.mean()), mean_transductive=float(trans.mean()),
                mean_gain_pp=float(diff.mean()), std_gain_pp=float(sd),
                ci95_gain_pp=[float(diff.mean() - ci), float(diff.mean() + ci)],
                paired_t=float(t), paired_p=float(p_t), wilcoxon_W=float(w), wilcoxon_p=float(p_w),
                wins=int((diff > 0).sum()), losses=int((diff < 0).sum()), ties=int((diff == 0).sum()),
                cohens_dz=float(diff.mean() / sd) if sd > 0 else float("inf"))


def streaming_simulation(ep, stream_seed, conf_threshold, n_chunks=10,
                         support_weight=3, max_anchors=20):
    """FIFO anchor buffer over a sequential query stream (scouting pass)."""
    rng = np.random.RandomState(stream_seed)
    order = rng.permutation(len(ep["q_y"]))
    chunks = np.array_split(order, n_chunks)
    W, b, _ = fit_fisher_lda(ep["s_x"], ep["s_y"], device="cpu")
    buffers = {c: [] for c in range(ep["n_classes"])}
    history, cum_wrong = [], 0
    for t, chunk in enumerate(chunks):
        prob, _, _ = compute_fisher_ensemble_probs(
            fisher_weights=W, fisher_biases=b, joint_features=ep["q_t"][chunk],
            bioclip_features=ep["q_bio"][chunk], text_prototypes=ep["text_protos"],
            scale=SCALE, w_lda=W_LDA, device="cpu")
        pred = prob.argmax(dim=-1).numpy()
        conf = prob.max(dim=-1).values
        online_acc = float((pred == ep["q_y"][chunk]).mean() * 100)
        admitted = wrong = 0
        for c in range(ep["n_classes"]):
            mask = pred == c
            idx, cf = chunk[mask], conf[mask]
            keep = (cf >= conf_threshold).numpy()
            idx = idx[keep]
            for i in idx:
                buffers[c].append(int(i))
                admitted += 1
                if ep["q_y"][i] != c:
                    wrong += 1
                if len(buffers[c]) > max_anchors:
                    buffers[c].pop(0)
        cum_wrong += wrong
        tot = sum(len(v) for v in buffers.values())
        bad = sum(1 for c, v in buffers.items() for i in v if ep["q_y"][i] != c)
        purity = 1.0 - bad / tot if tot else 1.0
        feats = [ep["s_x"]] * support_weight + [ep["q_x"][v] for v in buffers.values() if v]
        labs = [ep["s_y"]] * support_weight + [np.full(len(v), c) for c, v in buffers.items() if v]
        W, b, _ = fit_fisher_lda(np.concatenate(feats), np.concatenate(labs), device="cpu")
        history.append(dict(chunk=t, online_acc_pct=online_acc, buffer_size=tot,
                            buffer_purity_pct=purity * 100, admitted=admitted,
                            admitted_wrong=wrong, cumulative_wrong=cum_wrong))
    prob, _, _ = compute_fisher_ensemble_probs(
        fisher_weights=W, fisher_biases=b, joint_features=ep["q_t"],
        bioclip_features=ep["q_bio"], text_prototypes=ep["text_protos"],
        scale=SCALE, w_lda=W_LDA, device="cpu")
    final_acc = float((prob.argmax(dim=-1).numpy() == ep["q_y"]).mean() * 100)
    return history, final_acc


def main():
    t0 = time.time()
    episodes, class_names = load_episodes()
    print(f"[*] Loaded {len(episodes)} episodes ({len(episodes[0]['q_y'])} queries each) "
          f"in {time.time() - t0:.1f}s", flush=True)

    # ---- Part A: paired 30-episode comparison + anchor purity ----
    ind_acc, ind_f1, tr_acc, tr_f1, purity, n_anch = [], [], [], [], [], []
    for ep in episodes:
        ia = accuracy_score(ep["q_y"], ep["ind_pred"]) * 100
        if1 = f1_score(ep["q_y"], ep["ind_pred"], average="macro", zero_division=0) * 100
        pred, sel = run_transductive(ep, **DEFAULTS)
        ta = accuracy_score(ep["q_y"], pred) * 100
        tf1 = f1_score(ep["q_y"], pred, average="macro", zero_division=0) * 100
        tot = sum(len(v) for v in sel.values())
        bad = sum(1 for c, v in sel.items() for i in v if ep["q_y"][i] != c)
        ind_acc.append(ia); ind_f1.append(if1); tr_acc.append(ta); tr_f1.append(tf1)
        purity.append(100.0 * (1 - bad / tot) if tot else 100.0)
        n_anch.append(tot / ep["n_classes"])
    stats = paired_stats(ind_acc, tr_acc)
    print("\n[A] PAIRED 30-EPISODE INDUCTIVE vs TRANSDUCTIVE", flush=True)
    print(f"    Inductive  mean acc: {stats['mean_inductive']:.2f} | Transductive mean acc: {stats['mean_transductive']:.2f}", flush=True)
    print(f"    Gain: {stats['mean_gain_pp']:.2f} pp  CI95 [{stats['ci95_gain_pp'][0]:.2f}, {stats['ci95_gain_pp'][1]:.2f}]", flush=True)
    print(f"    paired t={stats['paired_t']:.3f} p={stats['paired_p']:.3e} | Wilcoxon p={stats['wilcoxon_p']:.3e} | "
          f"W/L/T={stats['wins']}/{stats['losses']}/{stats['ties']} | dz={stats['cohens_dz']:.2f}", flush=True)
    print(f"    Anchor purity @0.70: {np.mean(purity):.2f}% +/- {np.std(purity):.2f} | anchors/class: {np.mean(n_anch):.1f}", flush=True)

    # ---- Part C: purity / coverage vs confidence threshold ----
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    purity_curve = {}
    for th in thresholds:
        pur, cov = [], []
        for ep in episodes:
            sel = select_anchors(ep["ind_pred"], ep["ind_conf"], ep["n_classes"], 20, th)
            tot = sum(len(v) for v in sel.values())
            bad = sum(1 for c, v in sel.items() for i in v if ep["q_y"][i] != c)
            pur.append(100.0 * (1 - bad / tot) if tot else 100.0)
            cov.append(100.0 * tot / len(ep["q_y"]))
        purity_curve[str(th)] = dict(purity_mean=float(np.mean(pur)), purity_std=float(np.std(pur)),
                                     coverage_mean_pct=float(np.mean(cov)))
    print("\n[C] ANCHOR PURITY / COVERAGE vs THRESHOLD", flush=True)
    for th, v in purity_curve.items():
        print(f"    tau={th}: purity {v['purity_mean']:.2f}% | coverage {v['coverage_mean_pct']:.2f}%", flush=True)

    # ---- Part B: OFAT sensitivity sweep (same 30 episodes) ----
    sweep = {"support_weight": [1, 2, 3, 5, 10],
             "conf_threshold": [0.50, 0.60, 0.70, 0.80, 0.90],
             "max_anchors": [5, 10, 20, 40, 80]}
    sensitivity = {}
    for param, values in sweep.items():
        sensitivity[param] = {}
        for val in values:
            cfg = dict(DEFAULTS)
            cfg[param] = val
            accs = []
            for ep in episodes:
                pred, _ = run_transductive(ep, **cfg)
                accs.append(accuracy_score(ep["q_y"], pred) * 100)
            sensitivity[param][str(val)] = dict(mean_acc=float(np.mean(accs)), std_acc=float(np.std(accs)),
                                                per_episode=accs)
            print(f"[B] {param}={val}: acc {np.mean(accs):.2f} +/- {np.std(accs):.2f} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    # ---- Part D: sequential FIFO streaming (error accumulation) ----
    streaming = {}
    for sseed in [42, 7, 2024]:
        streaming[str(sseed)] = {}
        for th in [0.70, 0.90]:
            hist, final_acc = streaming_simulation(episodes[0], sseed, th)
            streaming[str(sseed)][str(th)] = dict(chunks=hist, final_full_query_acc=final_acc)
            accs = [h["online_acc_pct"] for h in hist]
            purs = [h["buffer_purity_pct"] for h in hist]
            print(f"[D] stream seed={sseed} tau={th}: online acc first/last chunk "
                  f"{accs[0]:.2f}/{accs[-1]:.2f} | purity first/last {purs[0]:.2f}/{purs[-1]:.2f} | "
                  f"final {final_acc:.2f} | cum wrong anchors {hist[-1]['cumulative_wrong']}", flush=True)

    result = dict(protocol=dict(seed=SEED, episodes=N_EPISODES, k_shot=K_SHOT, scale=SCALE,
                                w_lda=W_LDA, defaults=DEFAULTS, cache=str(CACHE)),
                  paired_30ep=dict(inductive_acc=ind_acc, inductive_f1=ind_f1,
                                   transductive_acc=tr_acc, transductive_f1=tr_f1,
                                   anchor_purity_pct=purity, anchors_per_class=n_anch, stats=stats),
                  purity_vs_threshold=purity_curve, sensitivity=sensitivity, streaming=streaming)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\n[SUCCESS] Results saved to {OUT} (total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()