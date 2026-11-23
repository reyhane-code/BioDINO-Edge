# -*- coding: utf-8 -*-
"""
Generate the LaTeX table body + prose numbers for the augmentation-sensitivity
table (tab:aug_sensitivity) directly from results/augmentation_sensitivity.json.
Run after the sweep completes:
  .venv\\Scripts\\python.exe experiments\\augmentation_sensitivity\\make_table.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "results" / "augmentation_sensitivity.json").read_text(encoding="utf-8"))
CONDS = DATA["conditions"]
N_TYPE = DATA["protocol"]["type_episodes"]


def agg(records):
    keys = ["lw_alpha", "cov_cond", "ind_acc_pct", "tr_acc_pct", "ind_f1_pct", "tr_f1_pct"]
    out = {"n": len(records)}
    for k in keys:
        v = np.asarray([r[k] for r in records], dtype=np.float64)
        out[k + "_mean"] = float(v.mean())
        out[k + "_std"] = float(v.std(ddof=0))
    return out


def sci(x):
    """Format a condition number as mantissa x 10^exp LaTeX."""
    if x <= 0:
        return "--"
    e = int(np.floor(np.log10(x)))
    m = x / 10**e
    return f"{m:.1f}$\\times10^{{{e}}}$"


def row(label, records, n_support):
    a = agg(records)
    return (f"{label} & {n_support} & "
            f"{a['lw_alpha_mean']:.3f}$\\pm${a['lw_alpha_std']:.3f} & "
            f"{sci(a['cov_cond_mean'])} & "
            f"{a['ind_acc_pct_mean']:.2f}$\\pm${a['ind_acc_pct_std']:.2f} & "
            f"{a['tr_acc_pct_mean']:.2f}$\\pm${a['tr_acc_pct_std']:.2f} \\\\")


def main():
    print("% ---- Panel A: view-count sweep (all episodes) ----")
    for v in (1, 2, 4, 8):
        recs = CONDS[f"views{v}"]["per_episode"]
        label = f"$v={v}$" + (" (deployed)" if v == 8 else "")
        print(row(label, recs, 55 * v))
    print(f"% ---- Panel B: augmentation type (first {N_TYPE} episodes, 8 views) ----")
    for kind, label in (("geo8", "Geometric only"), ("photo8", "Photometric only")):
        recs = CONDS[kind]["per_episode"]
        print(row(label, recs, 440))
    full_subset = [r for r in CONDS["views8"]["per_episode"] if r["ep"] < N_TYPE]
    print(row("Full pipeline", full_subset, 440))
    print("% ---- Panel C: pure-replication control (all episodes) ----")
    print(row("Replicate-8 (no diversity)", CONDS["replicate8"]["per_episode"], 440))

    print("\n% ---- prose numbers ----")
    a1 = agg(CONDS["views1"]["per_episode"])
    a8 = agg(CONDS["views8"]["per_episode"])
    ar = agg(CONDS["replicate8"]["per_episode"])
    ag = agg(CONDS["geo8"]["per_episode"])
    ap = agg(CONDS["photo8"]["per_episode"])
    af = agg(full_subset)
    print(f"episodes: view-sweep n={a1['n']}, type-sweep n={ag['n']}")
    print(f"views1: alpha {a1['lw_alpha_mean']:.3f} cond {a1['cov_cond_mean']:.3e} "
          f"ind {a1['ind_acc_pct_mean']:.2f}+/-{a1['ind_acc_pct_std']:.2f} tr {a1['tr_acc_pct_mean']:.2f}+/-{a1['tr_acc_pct_std']:.2f}")
    print(f"views8: alpha {a8['lw_alpha_mean']:.3f} cond {a8['cov_cond_mean']:.3e} "
          f"ind {a8['ind_acc_pct_mean']:.2f}+/-{a8['ind_acc_pct_std']:.2f} tr {a8['tr_acc_pct_mean']:.2f}+/-{a8['tr_acc_pct_std']:.2f}")
    print(f"replicate8: alpha {ar['lw_alpha_mean']:.3f} cond {ar['cov_cond_mean']:.3e} "
          f"ind {ar['ind_acc_pct_mean']:.2f} tr {ar['tr_acc_pct_mean']:.2f}")
    print(f"alpha ratio views8/replicate8: {a8['lw_alpha_mean']/ar['lw_alpha_mean']:.2f}")
    print(f"alpha ratio views1/views8: {a1['lw_alpha_mean']/a8['lw_alpha_mean']:.2f}")
    print(f"type sweep (n={ag['n']}): geo ind {ag['ind_acc_pct_mean']:.2f} tr {ag['tr_acc_pct_mean']:.2f} alpha {ag['lw_alpha_mean']:.3f} | "
          f"photo ind {ap['ind_acc_pct_mean']:.2f} tr {ap['tr_acc_pct_mean']:.2f} alpha {ap['lw_alpha_mean']:.3f} | "
          f"full ind {af['ind_acc_pct_mean']:.2f} tr {af['tr_acc_pct_mean']:.2f} alpha {af['lw_alpha_mean']:.3f}")
    inds = [agg(CONDS[f"views{v}"]["per_episode"])["ind_acc_pct_mean"] for v in (1, 2, 4, 8)]
    trs = [agg(CONDS[f"views{v}"]["per_episode"])["tr_acc_pct_mean"] for v in (1, 2, 4, 8)]
    print(f"view-sweep ind range: {min(inds):.2f}--{max(inds):.2f} | tr range: {min(trs):.2f}--{max(trs):.2f}")
    print(f"views8 F1: ind {a8['ind_f1_pct_mean']:.2f}+/-{a8['ind_f1_pct_std']:.2f} tr {a8['tr_f1_pct_mean']:.2f}+/-{a8['tr_f1_pct_std']:.2f}")


if __name__ == "__main__":
    main()