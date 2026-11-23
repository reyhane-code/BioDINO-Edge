import json, pathlib
p = pathlib.Path(r"E:\work\research-few-shot\experiments\quantization_accuracy\results\quantization_plantvillage_2026-11-17.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("date", d["date"], "| protocol", d["protocol"])
pv = d["plantvillage"]
for prec in ("fp32", "int8"):
    print(f"\n== {prec} (features {pv[prec]['feature_seconds']}s, fast-exit {pv[prec]['fast_exit_pct']:.2f}%)")
    for stage in ("tier1", "bioclip_only", "fisher", "cascaded", "transductive"):
        v = pv[prec][stage]
        print(f"  {stage:12s} acc={v['accuracy']:6.2f}  f1={v['macro_f1']:6.2f}  fnr={v['fnr']:.3f}")
print("\n== DELTAS (int8 - fp32)")
for stage in ("tier1", "bioclip_only", "fisher", "cascaded", "transductive"):
    a = pv[prec]["tier1"] if False else pv["fp32"][stage]
    b = pv["int8"][stage]
    print(f"  {stage:12s} dAcc={b['accuracy']-a['accuracy']:+.2f}  dF1={b['macro_f1']-a['macro_f1']:+.2f}  dFNR={b['fnr']-a['fnr']:+.3f}")
