from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from core.feature_phase6 import (
    load_phase6_config,
    build_phase6_features,
    inspect_phase6_arrays,
)


def rp(x):
    p=Path(x)
    return p if p.is_absolute() else ROOT/p


def read_csv(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def optional_array(d,names):
    for n in names:
        if n in d.files:
            return np.asarray(d[n])
    return None


def load_raw_npz(row,raw_root):
    sp=(row.get("sample_path") or "").strip()
    candidates=[]
    if sp:
        candidates.append(rp(sp))
    candidates.append(
        raw_root
        /(row.get("sign") or "")
        /(row.get("subject_id") or "")
        /(row.get("session_id") or "")
        /(row.get("sample_id") or "")
    )
    for p in candidates:
        if p.is_dir() and (p/"landmarks.npz").is_file():
            return p/"landmarks.npz"
    return None


def main():
    ap=argparse.ArgumentParser(description="Phase 6 - compact Pose + Face features")
    ap.add_argument("--config",default="configs/feature_phase6.yaml")
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()

    cfg,raw=load_phase6_config(rp(a.config))
    inp=raw.get("input",{}) or {}
    out=raw.get("output",{}) or {}
    sel=raw.get("selection",{}) or {}

    p5_manifest=rp(inp.get("phase5_manifest","dataset/processed/features_phase5/feature_manifest.csv"))
    raw_root=rp(inp.get("raw_root","dataset/raw"))
    output_root=rp(out.get("root","dataset/processed/features_phase6"))
    out_manifest=rp(out.get("manifest_csv","dataset/processed/features_phase6/feature_manifest.csv"))
    out_summary=rp(out.get("summary_json","dataset/processed/features_phase6/feature_summary.json"))

    required={str(x).upper() for x in sel.get("require_phase5_status",["PASS"])}

    rows=read_csv(p5_manifest)
    selected=[r for r in rows if (r.get("phase5_status") or "").upper() in required]
    if a.limit>0:
        selected=selected[:a.limit]

    output_root.mkdir(parents=True,exist_ok=True)
    records=[]
    counts=Counter()
    pose_cov=[]
    face_cov=[]

    for i,row in enumerate(selected,1):
        sid=(row.get("sample_id") or "").strip()
        try:
            p5_path=Path((row.get("features_path") or "").strip())
            if not p5_path.is_absolute():
                p5_path=ROOT/p5_path

            if not p5_path.is_file():
                raise FileNotFoundError(f"Phase5 features missing: {p5_path}")

            with np.load(p5_path,allow_pickle=False) as d:
                hands=np.asarray(d["full_features417"],dtype=np.float32)

            raw_npz=load_raw_npz(row,raw_root)
            pose_xyz=pose_present=face_xyz=face_present=None

            if raw_npz is not None:
                with np.load(raw_npz,allow_pickle=False) as d:
                    pose_xyz=optional_array(
                        d,
                        ("pose","pose_landmarks","pose_xyz")
                    )
                    face_xyz=optional_array(
                        d,
                        ("face","face_landmarks","face_xyz")
                    )
                    pp=optional_array(d,("pose_present",))
                    fp=optional_array(d,("face_present",))

                    if pp is not None:
                        pose_present=np.asarray(pp).reshape(-1).astype(bool)
                    if fp is not None:
                        face_present=np.asarray(fp).reshape(-1).astype(bool)

            arrays=build_phase6_features(
                phase5_full417=hands,
                pose_xyz=pose_xyz,
                pose_present=pose_present,
                face_xyz=face_xyz,
                face_present=face_present,
                cfg=cfg,
            )

            audit=inspect_phase6_arrays(arrays)
            status="PASS" if audit["pass"] else "FAIL"
            counts[status]+=1
            pose_cov.append(audit["pose_valid_ratio"])
            face_cov.append(audit["face_valid_ratio"])

            sample_out=output_root/sid
            sample_out.mkdir(parents=True,exist_ok=True)

            fpath=sample_out/"features_v3.npz"
            np.savez_compressed(fpath,**arrays)

            meta={
                "phase":6,
                "feature_version":raw.get("feature_version","v1-pose-face-compact"),
                "sample_id":sid,
                "sign":row.get("sign",""),
                "subject_id":row.get("subject_id",""),
                "session_id":row.get("session_id",""),
                "split":row.get("split",""),
                "phase5_status":row.get("phase5_status",""),
                "feature_dimensions":{
                    "hands":417,
                    "pose":27,
                    "face":9,
                    "hands_pose":444,
                    "hands_pose_face":453,
                },
                "pose_definition":{
                    "points":[
                        "nose","left_shoulder","right_shoulder",
                        "left_elbow","right_elbow",
                        "left_wrist","right_wrist"
                    ],
                    "normalization":"shoulder-center origin + shoulder-width scale",
                    "relations":[
                        "left/right wrist to nose",
                        "left/right wrist to shoulder",
                        "left/right wrist to chest"
                    ],
                },
                "face_definition":{
                    "features":[
                        "mouth_open","mouth_width",
                        "left_eye_open","right_eye_open",
                        "left_brow_height","right_brow_height",
                        "head_yaw_proxy","head_pitch_proxy","head_roll_proxy"
                    ],
                    "normalization":"inter-ocular outer-corner distance",
                    "note":"orientation values are lightweight proxies, not full solvePnP head pose",
                },
                "audit":audit,
            }

            mpath=sample_out/"feature_metadata.json"
            mpath.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")

            records.append({
                "sample_id":sid,
                "sign":row.get("sign",""),
                "subject_id":row.get("subject_id",""),
                "session_id":row.get("session_id",""),
                "split":row.get("split",""),
                "phase5_status":row.get("phase5_status",""),
                "phase6_status":status,
                "frame_count":audit["frame_count"],
                "pose_valid_ratio":f'{audit["pose_valid_ratio"]:.6f}',
                "face_valid_ratio":f'{audit["face_valid_ratio"]:.6f}',
                "hands_dim":audit["hands_dim"],
                "pose_dim":audit["pose_dim"],
                "face_dim":audit["face_dim"],
                "hands_pose_dim":audit["hands_pose_dim"],
                "hands_pose_face_dim":audit["hands_pose_face_dim"],
                "issues":";".join(audit["issues"]),
                "features_path":str(fpath),
                "metadata_path":str(mpath),
            })

            print(
                f"[{i:>3}/{len(selected)}] {status:<4} {sid} "
                f"pose={audit['pose_valid_ratio']:.2f} "
                f"face={audit['face_valid_ratio']:.2f}"
            )

        except Exception as e:
            counts["FAIL"]+=1
            records.append({
                "sample_id":sid,
                "sign":row.get("sign",""),
                "subject_id":row.get("subject_id",""),
                "session_id":row.get("session_id",""),
                "split":row.get("split",""),
                "phase5_status":row.get("phase5_status",""),
                "phase6_status":"FAIL",
                "frame_count":"",
                "pose_valid_ratio":"",
                "face_valid_ratio":"",
                "hands_dim":"",
                "pose_dim":"",
                "face_dim":"",
                "hands_pose_dim":"",
                "hands_pose_face_dim":"",
                "issues":f"{type(e).__name__}:{e}",
                "features_path":"",
                "metadata_path":"",
            })
            print(f"[{i:>3}/{len(selected)}] FAIL {sid}: {type(e).__name__}: {e}")

    fields=list(records[0].keys()) if records else []
    out_manifest.parent.mkdir(parents=True,exist_ok=True)
    with out_manifest.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    summary={
        "phase":6,
        "selected_samples":len(selected),
        "processed_rows":len(records),
        "pass":counts["PASS"],
        "fail":counts["FAIL"],
        "mean_pose_valid_ratio":float(np.mean(pose_cov)) if pose_cov else 0.0,
        "mean_face_valid_ratio":float(np.mean(face_cov)) if face_cov else 0.0,
        "feature_sets":{
            "hands_only":417,
            "hands_pose":444,
            "hands_pose_face":453,
        },
        "notes":{
            "missing_pose":"zero + pose_valid=false",
            "missing_face":"zero + face_valid=false",
            "orientation":"yaw/pitch/roll are lightweight proxies",
        },
    }
    out_summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    print("\nPHASE 6 POSE + FACE FEATURE EXTRACTION FINISHED")
    print("="*70)
    print(f"Selected         : {len(selected)}")
    print(f"PASS             : {counts['PASS']}")
    print(f"FAIL             : {counts['FAIL']}")
    print(f"Mean pose cover  : {summary['mean_pose_valid_ratio']:.4f}")
    print(f"Mean face cover  : {summary['mean_face_valid_ratio']:.4f}")
    print(f"Manifest         : {out_manifest}")
    print(f"Summary          : {out_summary}")
    print("="*70)


if __name__=="__main__":
    main()
