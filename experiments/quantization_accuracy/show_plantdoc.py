import json, pathlib
p = pathlib.Path(r"E:\work\research-few-shot\experiments\quantization_accuracy\results\quantization_plantdoc_2026-11-17.json")
d = json.loads(p.read_text(encoding="utf-8"))
pd = d["plantdoc"]
for prec in ("fp32", "int8"):
    print(f"== {prec} ({pd[prec]['feature_seconds']}s)")
    for stage in ("tier1", "bioclip_only", "fisher", "transductive"):
        v = pd[prec][stage]
        print(f"  {stage:12s} acc={v['accuracy']:6.2f} f1={v['macro_f1']:6.2f}")
print("== DELTAS")
for stage in ("tier1", "bioclip_only", "fisher", "transductive"):
    a, b = pd["fp32"][stage], pd["int8"][stage]
    print(f"  {stage:12s} dAcc={b['accuracy']-a['accuracy']:+.2f} dF1={b['macro_f1']-a['macro_f1']:+.2f}")
