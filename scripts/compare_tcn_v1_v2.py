from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
v1=ROOT/"artifacts"/"experiments"/"tcn_v1_160samples"/"test_metrics.json"
v2=ROOT/"artifacts"/"experiments"/"tcn_v2_targeted"/"test_metrics.json"
for p in (v1,v2):
    if not p.is_file(): raise FileNotFoundError(p)
a=json.loads(v1.read_text(encoding="utf-8"))
b=json.loads(v2.read_text(encoding="utf-8"))
print("\nTCN V1 vs V2 — SAME LOCKED TEST")
print("="*64)
print(f"V1 accuracy={a['test_accuracy']:.4f} macro_f1={a['test_macro_f1']:.4f}")
print(f"V2 accuracy={b['test_accuracy']:.4f} macro_f1={b['test_macro_f1']:.4f}")
print(f"Δ accuracy={b['test_accuracy']-a['test_accuracy']:+.4f}")
print(f"Δ macro_f1={b['test_macro_f1']-a['test_macro_f1']:+.4f}")
print("="*64)
