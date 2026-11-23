# -*- coding: utf-8 -*-
"""Paired tests for the fair-baseline run.

Compares BioDINO-Edge against the 896-D inductive comparators and the
transductive FSL methods on the SAME episodes, so the paired difference is
free of episode-difficulty variance.

Output: experiments/paper_rerun/results/fair_baselines_paired_2026-11-18.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent / "results" / "fair_baselines_2026-11-18.json"
OUT = Path(__file__).resolve().parent / "results" / "fair_baselines_paired_2026-11-18.json"
DATE = "2026-11-18"


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    t, p = stats.ttest_rel(a, b)
    try:
        _, pw = stats.wilcoxon(a, b)
    except ValueError:
        pw = float("nan")
    se = d.std(ddof=1) / np.sqrt(n)
    return {
        "n": int(n), "mean_gain_pp": float(d.mean()),
        "ci95_pp": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
        "paired_t": float(t), "p_value": float(p), "wilcoxon_p": float(pw),
        "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
        "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("inf"),
        "baseline_mean": float(b.mean()), "ours_mean": float(a.mean()),
    }


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    out = {"date": DATE, "source": str(SRC.relative_to(ROOT)), "comparisons": {}}

    for regime in ("regime_11class_5shot", "regime_5way_5shot"):
        eps = data[regime]["per_episode"]
        ours = [e["biodino_transductive"]["acc"] for e in eps]
        ours_ind = [e["biodino_inductive"]["acc"] for e in eps]
        block = {"ours_transductive_vs": {}, "ours_inductive_vs": {}}
        for m in data[regime]["summary"]:
            if m.startswith("biodino"):
                continue
            base = [e[m]["acc"] for e in eps]
            block["ours_transductive_vs"][m] = paired(ours, base)
            block["ours_inductive_vs"][m] = paired(ours_ind, base)
        block["transductive_vs_inductive"] = paired(ours, ours_ind)
        out["comparisons"][regime] = block

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] {OUT}\n")

    for regime, block in out["comparisons"].items():
        print(f"== {regime} ==")
        print(f"  ours transductive = {block['transductive_vs_inductive']['ours_mean']:.2f}%")
        for m, s in block["ours_transductive_vs"].items():
            verdict = "OURS BETTER" if s["mean_gain_pp"] > 0 else "BASELINE BETTER"
            print(f"  vs {m:22s} {s['mean_gain_pp']:+6.2f} pp  p={s['p_value']:.2e} "
                  f"W/L/T={s['wins']}/{s['losses']}/{s['ties']}  {verdict}")
        print()


if __name__ == "__main__":
    main()
