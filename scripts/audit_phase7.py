from pathlib import Path
import json,sys
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]
def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p
raw=yaml.safe_load(rp("configs/phase7_temporal.yaml").read_text(encoding="utf-8"))
ds=raw["dataset_output"]
errors=[]
shapes={}
for split,key in [("train","train_npz"),("val","val_npz"),("test","test_npz")]:
    p=rp(ds[key])
    if not p.is_file():
        errors.append(f"missing:{p}"); continue
    with np.load(p,allow_pickle=False) as d:
        X=d["X"]; M=d["mask"]; y=d["y"]
        shapes[split]={"X":list(X.shape),"mask":list(M.shape),"y":list(y.shape)}
        if X.ndim!=3 or X.shape[1]!=32: errors.append(f"{split}:bad_X_shape:{X.shape}")
        if M.shape!=X.shape[:2]: errors.append(f"{split}:bad_mask_shape:{M.shape}")
        if y.shape!=(X.shape[0],): errors.append(f"{split}:bad_y_shape:{y.shape}")
        if not np.isfinite(X).all(): errors.append(f"{split}:non_finite")
        if np.any(X[~M] != 0): errors.append(f"{split}:padding_not_zero")
labels=rp(ds["labels_json"])
stats=rp(ds["preprocess_stats_npz"])
if not labels.is_file(): errors.append("missing_labels_json")
if not stats.is_file(): errors.append("missing_preprocess_stats")
print("\nPHASE 7 DATA AUDIT")
print("="*60)
for k,v in shapes.items(): print(f"{k.upper():5}: {v}")
print(f"Errors : {len(errors)}")
for e in errors[:20]: print(" -",e)
print(f"PHASE 7 DATA : {'PASS' if not errors else 'FAIL'}")
print("="*60)
raise SystemExit(0 if not errors else 1)
