# -*- coding: utf-8 -*-
"""Paired significance tests of BioDINO-Edge against every re-run baseline.

Reads experiments/baseline_comparison/controlled_baseline_results.json, which
stores per-episode accuracies for all six methods on identical splits.
Reports, for each baseline and each regime, the mean paired gain, 95% CI,
paired t-test, Wilcoxon signed-rank, win count, and Cohen's dz.

Source data are existing artifacts; no new model runs.
Output: experiments/paper_rerun/results/paired_baselines_2026-11-17.json
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments/baseline_comparison/controlled_baseline_results.json"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
DATE = "2026-11-17"

METHODS = ["zero_shot_clip", "tip_adapter", "protonet", "simpleshot"]


def accs(per_episode, method, field="acc"):
    return np.array([ep[method][field] for ep in per_episode], dtype=float)


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    t, p = stats.ttest_rel(a, b)
    w, pw = stats.wilcoxon(a, b)
    return {
        "n_episodes": int(n),
        "mean_gain_pp": float(d.mean()),
        "std_gain_pp": float(d.std(ddof=1)),
        "ci95_pp": [float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(n)),
                    float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(n))],
        "paired_t": float(t), "p_value": float(p),
        "wilcoxon_p": float(pw),
        "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
        "cohens_dz": float(d.mean() / d.std(ddof=1)),
        "baseline_mean": float(np.mean(b)), "ours_mean": float(np.mean(a)),
    }


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    out = {"date": DATE, "source": str(SRC.relative_to(ROOT)), "regimes": {}}

    ep11 = data["regime_11class"]["per_episode"]
    ours = accs(ep11, "biodino_transductive")
    ours_ind = accs(ep11, "biodino_inductive")
    out["regimes"]["11class_5shot"] = {
        "biodino_transductive": {"mean": float(ours.mean()), "std": float(ours.std(ddof=1)),
                                 "n": len(ours)},
        "vs_baselines": {m: paired(ours, accs(ep11, m)) for m in METHODS},
        "inductive_vs_baselines": {m: paired(ours_ind, accs(ep11, m)) for m in METHODS},
    }

    for k in ("1shot", "5shot", "10shot", "15shot"):
        ep = data["regime_5way"][k]["per_episode"]
        ours5 = accs(ep, "biodino_transductive")
        out["regimes"][f"5way_{k}"] = {
            "biodino_transductive": {"mean": float(ours5.mean()), "std": float(ours5.std(ddof=1)),
                                     "n": len(ours5)},
            "vs_baselines": {m: paired(ours5, accs(ep, m)) for m in METHODS},
        }

    p = OUT / f"paired_baselines_{DATE}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] {p}")

    for regime, r in out["regimes"].items():
        print(f"\n== {regime}: ours={r['biodino_transductive']['mean']:.2f}%")
        for m, s in r["vs_baselines"].items():
            print(f"  vs {m:14s} {s['mean_gain_pp']:+6.2f} pp  CI[{s['ci95_pp'][0]:+.2f},{s['ci95_pp'][1]:+.2f}]"
                  f"  t={s['paired_t']:6.2f} p={s['p_value']:.2e}  wilcoxon p={s['wilcoxon_p']:.2e}"
                  f"  W/L/T={s['wins']}/{s['losses']}/{s['ties']}  dz={s['cohens_dz']:.2f}")


if __name__ == "__main__":
    main()
