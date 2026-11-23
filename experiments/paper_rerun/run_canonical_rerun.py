# -*- coding: utf-8 -*-
"""
Re-run the canonical BioDINO-Edge benchmarks and write them to a DATED results
file. The existing result JSONs are never overwritten.

Datasets: PlantVillage (11 classes) and PlantDoc (27 classes), both through the
shared pipeline (pipeline/biodino_edge_pipeline.py: run_benchmark).

Output: experiments/paper_rerun/results/<dataset>_2026-11-17.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.biodino_edge_pipeline import (  # noqa: E402
    EdgeConfig,
    load_backbones,
    run_benchmark,
)
from pipeline.data_engine import load_plantdoc, load_plantvillage  # noqa: E402

DATE = "2026-11-17"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device={device}")
    models = load_backbones(device)
    cfg = EdgeConfig(seed=42, k_shot=5)

    runs = {}
    for name, loader, root in (
        ("plantvillage", load_plantvillage, ROOT / "plantvillage"),
        ("plantdoc", load_plantdoc, ROOT / "plantdoc"),
    ):
        support, query, classes, prompts = loader(root, cfg.k_shot, cfg.seed)
        print(f"[*] {name}: support={len(support)} query={len(query)} classes={len(classes)}")
        res = run_benchmark(
            dataset_name=name,
            support_samples=support,
            query_samples=query,
            class_names=classes,
            taxonomic_prompts=prompts,
            models=models,
            cfg=cfg,
            run_latency_bench=False,
            use_cache=(name == "plantvillage"),
        )
        runs[name] = res
        p = OUT / f"{name}_{DATE}.json"
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[OK] {p}")

    print(json.dumps({k: {m: v for m, v in r["metrics"].items()
                           if isinstance(v, (int, float))} for k, r in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
