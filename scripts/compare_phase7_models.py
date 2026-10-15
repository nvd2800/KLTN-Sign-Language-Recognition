from pathlib import Path
import json,yaml
ROOT=Path(__file__).resolve().parents[1]
def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p
raw=yaml.safe_load(rp("configs/phase7_temporal.yaml").read_text(encoding="utf-8"))
arts=raw["artifacts"]
print("\nPHASE 7 MODEL COMPARISON")
print("="*62)
for model in ("gru","tcn"):
    p=rp(arts[f"{model}_dir"])/"test_metrics.json"
    if not p.is_file():
        print(f"{model.upper():4}: metrics not found")
        continue
    m=json.loads(p.read_text(encoding="utf-8"))
    print(f"{model.upper():4}: accuracy={m['test_accuracy']:.4f}  macro_f1={m['test_macro_f1']:.4f}")
print("="*62)
