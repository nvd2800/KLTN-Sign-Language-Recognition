from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]

def rp(x):
    p=Path(x)
    return p if p.is_absolute() else ROOT/p

def rows(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(description="Audit Phase 6")
    ap.add_argument("--config",default="configs/feature_phase6.yaml")
    a=ap.parse_args()

    raw=yaml.safe_load(rp(a.config).read_text(encoding="utf-8")) or {}
    inp=raw.get("input",{}) or {}
    out=raw.get("output",{}) or {}
    audit_cfg=raw.get("audit",{}) or {}

    p5=rows(rp(inp.get("phase5_manifest","dataset/processed/features_phase5/feature_manifest.csv")))
    p6=rows(rp(out.get("manifest_csv","dataset/processed/features_phase6/feature_manifest.csv")))

    selected=[r for r in p5 if (r.get("phase5_status") or "").upper()=="PASS"]

    p5map={r["sample_id"]:r for r in selected}
    p6map={r["sample_id"]:r for r in p6}

    missing=sorted(set(p5map)-set(p6map))
    extra=sorted(set(p6map)-set(p5map))
    failed=[]
    split_mismatch=[]
    feature_errors=[]

    pose_ratios=[]
    face_ratios=[]

    for sid,row in p6map.items():
        if row.get("phase6_status")!="PASS":
            failed.append(sid)

        if sid in p5map and row.get("split","") != p5map[sid].get("split",""):
            split_mismatch.append(sid)

        try:
            if row.get("pose_valid_ratio","") != "":
                pose_ratios.append(float(row["pose_valid_ratio"]))
            if row.get("face_valid_ratio","") != "":
                face_ratios.append(float(row["face_valid_ratio"]))
        except ValueError:
            pass

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
                checks={
                    "hands_full417":417,
                    "pose27":27,
                    "face9":9,
                    "ablation_hands417":417,
                    "ablation_hands_pose444":444,
                    "ablation_hands_pose_face453":453,
                }

                T=None
                for name,dim in checks.items():
                    if name not in d.files:
                        feature_errors.append(f"{sid}:missing:{name}")
                        continue

                    arr=np.asarray(d[name])

                    if arr.ndim != 2 or arr.shape[1] != dim:
                        feature_errors.append(f"{sid}:bad_shape:{name}:{arr.shape}")

                    if T is None and arr.ndim==2:
                        T=arr.shape[0]

                    if not np.isfinite(arr).all():
                        feature_errors.append(f"{sid}:non_finite:{name}")

                for name in ("pose_valid","face_valid"):
                    if name not in d.files:
                        feature_errors.append(f"{sid}:missing:{name}")
                    elif T is not None and np.asarray(d[name]).shape != (T,):
                        feature_errors.append(f"{sid}:bad_shape:{name}:{np.asarray(d[name]).shape}")

        except Exception as e:
            feature_errors.append(f"{sid}:{type(e).__name__}:{e}")

    mean_pose=float(np.mean(pose_ratios)) if pose_ratios else 0.0
    mean_face=float(np.mean(face_ratios)) if face_ratios else 0.0

    min_pose=float(audit_cfg.get("min_pose_valid_ratio",0.0))
    min_face=float(audit_cfg.get("min_face_valid_ratio",0.0))

    coverage_errors=[]
    if mean_pose < min_pose:
        coverage_errors.append(f"pose_coverage:{mean_pose:.4f}<{min_pose:.4f}")
    if mean_face < min_face:
        coverage_errors.append(f"face_coverage:{mean_face:.4f}<{min_face:.4f}")

    passed=(
        not missing
        and not extra
        and not failed
        and not split_mismatch
        and not feature_errors
        and not coverage_errors
    )

    report={
        "phase":6,
        "selected_phase5_samples":len(selected),
        "phase6_rows":len(p6),
        "pass_rows":sum(r.get("phase6_status")=="PASS" for r in p6),
        "fail_rows":sum(r.get("phase6_status")!="PASS" for r in p6),
        "missing":missing,
        "extra":extra,
        "split_mismatch":split_mismatch,
        "feature_errors":feature_errors,
        "coverage_errors":coverage_errors,
        "mean_pose_valid_ratio":mean_pose,
        "mean_face_valid_ratio":mean_face,
        "phase6_pass":passed,
    }

    apath=rp(out.get("audit_json","dataset/processed/features_phase6/feature_audit.json"))
    apath.parent.mkdir(parents=True,exist_ok=True)
    apath.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    print("\nPHASE 6 POSE + FACE FEATURE AUDIT")
    print("="*68)
    print(f"Selected Phase-5 : {len(selected)}")
    print(f"Phase-6 rows     : {len(p6)}")
    print(f"PASS rows        : {report['pass_rows']}")
    print(f"FAIL rows        : {report['fail_rows']}")
    print(f"Missing          : {len(missing)}")
    print(f"Extra            : {len(extra)}")
    print(f"Split mismatch   : {len(split_mismatch)}")
    print(f"Feature errors   : {len(feature_errors)}")
    print(f"Pose coverage    : {mean_pose:.4f}")
    print(f"Face coverage    : {mean_face:.4f}")
    print(f"Coverage errors  : {len(coverage_errors)}")
    print(f"PHASE 6          : {'PASS' if passed else 'FAIL'}")
    print(f"Report           : {apath}")
    print("="*68)

    raise SystemExit(0 if passed else 1)


if __name__=="__main__":
    main()
