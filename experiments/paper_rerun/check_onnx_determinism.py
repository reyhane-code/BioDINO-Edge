# -*- coding: utf-8 -*-
"""Is the INT8 ONNX path bit-deterministic across sessions?

The paper carries three values for the same PlantDoc 27-way transductive
configuration: 51.19 (FP32 ladder), 51.59 (INT8 ladder) and 50.79 (INT8 inside
the comparison harness). Either the INT8 session is not reproducible, or
something upstream differs. This script feeds one fixed tensor through two
independently created INT8 sessions and reports the max absolute deviation.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "pipeline", HERE.parent / "quantization_accuracy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_quantization_accuracy as R  # noqa: E402

rng = np.random.default_rng(0)
x = rng.random((8, 3, 224, 224)).astype(np.float32)

outs = []
for prec in ("fp32", "int8"):
    outs = []
    for trial in range(3):
        t1, _ = R.build_sessions(prec)
        _, _, f = R.run_onnx_tier1(t1, x)
        outs.append(np.asarray(f))
    base = outs[0]
    devs = [float(np.abs(o - base).max()) for o in outs[1:]]
    print(f"{prec}: max deviation {max(devs):.3e}  uniques {len({o.tobytes() for o in outs})}/3",
          flush=True)
