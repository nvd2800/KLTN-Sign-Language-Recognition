from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

def rows(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(description="Audit Phase 5")
    ap.add_argument("--config",default="configs/feature_phase5.yaml")
    a=ap.parse_args()

    raw=yaml.safe_load(rp(a.config).read_text(encoding="utf-8")) or {}
    inp=raw.get("input",{}) or {}; out=raw.get("output",{}) or {}
    p4=rows(rp(inp.get("phase4_manifest","dataset/processed/features_phase4/feature_manifest.csv")))
    p5=rows(rp(out.get("manifest_csv","dataset/processed/features_phase5/feature_manifest.csv")))
    selected=[r for r in p4 if (r.get("phase4_status") or "").upper()=="PASS"]

    qmap={r["sample_id"]:r for r in selected}
    fmap={r["sample_id"]:r for r in p5}

    missing=sorted(set(qmap)-set(fmap))
    extra=sorted(set(fmap)-set(qmap))
    failed=[]
    split_mismatch=[]
    feature_errors=[]

    required_shapes={
        "position126":126,
        "angles30":30,
        "distances9":9,
        "velocity126":126,
        "acceleration126":126,
        "ablation_position_velocity252":252,
        "ablation_position_velocity_acceleration378":378,
        "full_features417":417,
    }

    for sid,row in fmap.items():
        if row.get("phase5_status")!="PASS":
            failed.append(sid)

        if sid in qmap and row.get("split","")!=qmap[sid].get("split",""):
            split_mismatch.append(sid)

        fp=row.get("features_path","")
        if not fp:
            feature_errors.append(f"{sid}:missing_feature_path")
            continue

        p=Path(fp)
        if not p.is_absolute():
            p=ROOT/p

        if not p.is_file():
            feature_errors.append(f"{sid}:features_file_missing")
            continue

        try:
            with np.load(p,allow_pickle=False) as d:
                T=None
                for name,dim in required_shapes.items():
                    if name not in d.files:
                        feature_errors.append(f"{sid}:missing:{name}")
                        continue
                    arr=np.asarray(d[name])
                    if arr.ndim!=2 or arr.shape[1]!=dim:
                        feature_errors.append(f"{sid}:bad_shape:{name}:{arr.shape}")
                    if T is None and arr.ndim==2:
                        T=arr.shape[0]
                    if not np.isfinite(arr).all():
                        feature_errors.append(f"{sid}:non_finite:{name}")

                # First frame velocity must be zero.
                if "velocity126" in d.files:
                    v=np.asarray(d["velocity126"])
                    if len(v) and not np.allclose(v[0],0.0):
                        feature_errors.append(f"{sid}:velocity_first_frame_nonzero")

                # First two frames acceleration should be zero by definition.
                if "acceleration126" in d.files:
                    acc=np.asarray(d["acceleration126"])
                    if len(acc)>0 and not np.allclose(acc[0],0.0):
                        feature_errors.append(f"{sid}:acceleration_frame0_nonzero")
        except Exception as e:
            feature_errors.append(f"{sid}:{type(e).__name__}:{e}")

    passed=(
        not missing
        and not extra
        and not failed
        and not split_mismatch
        and not feature_errors
    )

    report={
        "phase":5,
        "selected_phase4_samples":len(selected),
        "phase5_rows":len(p5),
        "pass_rows":sum(r.get("phase5_status")=="PASS" for r in p5),
        "fail_rows":sum(r.get("phase5_status")!="PASS" for r in p5),
        "missing":missing,
        "extra":extra,
        "split_mismatch":split_mismatch,
        "feature_errors":feature_errors,
        "phase5_pass":passed,
    }

    apath=rp(out.get("audit_json","dataset/processed/features_phase5/feature_audit.json"))
    apath.parent.mkdir(parents=True,exist_ok=True)
    apath.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    print("\nPHASE 5 FEATURE AUDIT")
    print("="*64)
    print(f"Selected Phase-4 : {len(selected)}")
    print(f"Phase-5 rows     : {len(p5)}")
    print(f"PASS rows        : {report['pass_rows']}")
    print(f"FAIL rows        : {report['fail_rows']}")
    print(f"Missing          : {len(missing)}")
    print(f"Extra            : {len(extra)}")
    print(f"Split mismatch   : {len(split_mismatch)}")
    print(f"Feature errors   : {len(feature_errors)}")
    print(f"PHASE 5          : {'PASS' if passed else 'FAIL'}")
    print(f"Report           : {apath}")
    print("="*64)
    raise SystemExit(0 if passed else 1)

if __name__=="__main__":
    main()
