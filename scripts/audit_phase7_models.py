from pathlib import Path
import json,yaml

ROOT=Path(__file__).resolve().parents[1]
def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

raw=yaml.safe_load(rp("configs/phase7_temporal.yaml").read_text(encoding="utf-8"))
arts=raw["artifacts"]
errors=[]

for model in ("gru","tcn"):
    d=rp(arts[f"{model}_dir"])
    for name in ("model.keras","history.json","training_config.json","test_metrics.json","confusion_matrix.csv"):
        p=d/name
        if not p.is_file():
            errors.append(f"{model}:missing:{name}")

print("\nPHASE 7 MODEL ARTIFACT AUDIT")
print("="*62)
print(f"Errors : {len(errors)}")
for e in errors: print(" -",e)
print(f"PHASE 7 MODELS : {'PASS' if not errors else 'FAIL'}")
print("="*62)
raise SystemExit(0 if not errors else 1)
