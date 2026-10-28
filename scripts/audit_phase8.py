from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[1]

def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/phase8_unknown.yaml")
    args=ap.parse_args()
    cfg=yaml.safe_load(rp(args.config).read_text(encoding="utf-8"))

    errors=[]
    mode_path=rp(cfg["unknown_data"]["manifest_csv"]).parent/"unknown_manifest_mode.json"
    if mode_path.is_file() and json.loads(mode_path.read_text(encoding="utf-8")).get("mode")=="exploratory_val_only":
        errors.append("EXPLORATORY MODE: official Phase08 audit cannot PASS without independent UNKNOWN TEST")

    required=[
        rp(cfg["unknown_data"]["manifest_csv"]),
        rp(cfg["unknown_data"]["val_npz"]),
        rp(cfg["unknown_data"]["test_npz"]),
        rp(cfg["output"]["calibrated_json"]),
        rp(cfg["output"]["validation_report_json"]),
        rp(cfg["output"]["test_report_json"]),
    ]

    for p in required:
        if not p.is_file():
            errors.append(f"Missing: {p}")

    if not errors:
        with np.load(rp(cfg["unknown_data"]["val_npz"]),allow_pickle=False) as d:
            val_n=len(d["X"])
        with np.load(rp(cfg["unknown_data"]["test_npz"]),allow_pickle=False) as d:
            test_n=len(d["X"])
        if val_n<=0: errors.append("UNKNOWN VAL is empty")
        if test_n<=0: errors.append("UNKNOWN TEST is empty")

        cal=json.loads(rp(cfg["output"]["calibrated_json"]).read_text(encoding="utf-8"))
        thr=float(cal["threshold"])
        allowed={float(x) for x in cfg["thresholds"]["candidates"]}
        if thr not in allowed:
            errors.append(f"Threshold {thr} not in configured candidates")

        report=json.loads(rp(cfg["output"]["test_report_json"]).read_text(encoding="utf-8"))
        if abs(float(report["threshold"])-thr)>1e-9:
            errors.append("Test report threshold differs from calibrated threshold")

    print("\nPHASE 08 ARTIFACT AUDIT")
    print("="*64)
    print("Errors :",len(errors))
    for e in errors: print("-",e)
    print("PHASE 8 :", "PASS" if not errors else "FAIL")
    print("="*64)

    raise SystemExit(0 if not errors else 1)

if __name__=="__main__":
    main()
