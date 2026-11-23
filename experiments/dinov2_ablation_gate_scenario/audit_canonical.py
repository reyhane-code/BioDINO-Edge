"""Dump every scalar from the canonical validation JSONs for a paper audit.

Large arrays (per-episode lists, risk-coverage curves) are summarized by length
instead of expanded. Output: canonical_audit.txt next to this script.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SOURCES = [
    ROOT / "routing_calibration_validation" / "routing_validation_results.json",
    ROOT / "transductive_validation" / "transductive_validation_results.json",
    ROOT / "transductive_validation" / "canonical_single_run.json",
]


def walk(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, list):
        if len(obj) > 12:
            out.append(f"{path}: list[{len(obj)}]")
        else:
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", out)
    else:
        if isinstance(obj, float):
            out.append(f"{path} = {obj:.6f}")
        else:
            out.append(f"{path} = {obj}")


lines = []
for src in SOURCES:
    lines.append("=" * 90)
    lines.append(str(src))
    lines.append("=" * 90)
    walk(json.loads(src.read_text(encoding="utf-8")), "", lines)

out_path = HERE / "canonical_audit.txt"
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {len(lines)} lines -> {out_path}")