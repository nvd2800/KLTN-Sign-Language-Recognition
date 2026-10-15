from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from core.feature_phase5 import (
    load_phase5_config, build_phase5_features, inspect_phase5_arrays
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

def load_raw_sample(row, raw_root):
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
        if p.is_dir():
            npz=p/"landmarks.npz"
            if npz.is_file():
                return p,npz
    return None,None

def load_phase4_npz(row):
    p=Path((row.get("features_path") or "").strip())
    if not p.is_absolute():
        p=ROOT/p
    if not p.is_file():
        raise FileNotFoundError(f"Phase4 features missing: {p}")
    return p

def main():
    ap=argparse.ArgumentParser(description="Phase 5 - geometric and motion feature extraction")
    ap.add_argument("--config",default="configs/feature_phase5.yaml")
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()

    cfg,raw=load_phase5_config(rp(a.config))
    inp=raw.get("input",{}) or {}
    out=raw.get("output",{}) or {}
    sel=raw.get("selection",{}) or {}
    groups=raw.get("groups",{}) or {}

    p4_manifest=rp(inp.get("phase4_manifest","dataset/processed/features_phase4/feature_manifest.csv"))
    raw_root=rp(inp.get("raw_root","dataset/raw"))
    output_root=rp(out.get("root","dataset/processed/features_phase5"))
    out_manifest=rp(out.get("manifest_csv","dataset/processed/features_phase5/feature_manifest.csv"))
    out_summary=rp(out.get("summary_json","dataset/processed/features_phase5/feature_summary.json"))

    require={str(x).upper() for x in sel.get("require_phase4_status",["PASS"])}

    rows=read_csv(p4_manifest)
    selected=[r for r in rows if (r.get("phase4_status") or "").upper() in require]
    if a.limit>0:
        selected=selected[:a.limit]

    output_root.mkdir(parents=True,exist_ok=True)
    records=[]
    counts=Counter()
    issue_counts=Counter()

    for i,row in enumerate(selected,1):
        sid=(row.get("sample_id") or "").strip()
        try:
            p4_path=load_phase4_npz(row)

            with np.load(p4_path,allow_pickle=False) as d:
                p4={
                    "normalized_xyz126":np.asarray(d["normalized_xyz126"],dtype=np.float32),
                    "left_normalized_xyz":np.asarray(d["left_normalized_xyz"],dtype=np.float32),
                    "right_normalized_xyz":np.asarray(d["right_normalized_xyz"],dtype=np.float32),
                    "hand_presence2":np.asarray(d["hand_presence2"],dtype=bool),
                    "scale2":np.asarray(d["scale2"],dtype=np.float32),
                    "timestamps":np.asarray(d["timestamps"],dtype=np.float64),
                }

            sample_dir,raw_npz=load_raw_sample(row,raw_root)
            raw_left=raw_right=pose_xyz=pose_present=None

            if raw_npz is not None:
                with np.load(raw_npz,allow_pickle=False) as d:
                    if "left_hand" in d.files:
                        raw_left=np.asarray(d["left_hand"],dtype=np.float32)
                    if "right_hand" in d.files:
                        raw_right=np.asarray(d["right_hand"],dtype=np.float32)

                    pose_xyz=optional_array(
                        d,
                        ("pose","pose_landmarks","pose_xyz")
                    )
                    pp=optional_array(
                        d,
                        ("pose_present",)
                    )
                    if pp is not None:
                        pose_present=np.asarray(pp).reshape(-1).astype(bool)

            arrays=build_phase5_features(
                phase4_arrays=p4,
                raw_left=raw_left,
                raw_right=raw_right,
                pose_xyz=pose_xyz,
                pose_present=pose_present,
                cfg=cfg,
            )

            # Configurable groups: arrays are always computed for reproducibility,
            # but enabled group metadata records the selected experimental setup.
            enabled_groups=[
                name for name in ("position","angles","distances","velocity","acceleration")
                if bool(groups.get(name,True))
            ]

            audit=inspect_phase5_arrays(arrays)
            status="PASS" if audit["pass"] else "FAIL"
            counts[status]+=1
            for issue in audit["issues"]:
                issue_counts[issue.split(":")[0]]+=1

            sample_out=output_root/sid
            sample_out.mkdir(parents=True,exist_ok=True)
            fpath=sample_out/"features_v2.npz"
            np.savez_compressed(fpath,**arrays)

            metadata={
                "phase":5,
                "feature_version":raw.get("feature_version","v2-geometric-motion"),
                "sample_id":sid,
                "sign":row.get("sign",""),
                "subject_id":row.get("subject_id",""),
                "session_id":row.get("session_id",""),
                "split":row.get("split",""),
                "phase4_status":row.get("phase4_status",""),
                "enabled_groups":enabled_groups,
                "feature_dimensions":{
                    "position":126,
                    "angles":30,
                    "distances":9,
                    "velocity":126,
                    "acceleration":126,
                    "full":417,
                    "ablation_position":126,
                    "ablation_position_velocity":252,
                    "ablation_position_velocity_acceleration":378,
                },
                "distance_order":[
                    "left_thumb_index",
                    "right_thumb_index",
                    "left_right_wrist",
                    "left_hand_face",
                    "right_hand_face",
                    "left_hand_shoulder",
                    "right_hand_shoulder",
                    "left_hand_chest",
                    "right_hand_chest",
                ],
                "angle_definition":"15 joint angles per hand, normalized by pi",
                "motion_definition":(
                    "frame difference" if not cfg.use_time_delta
                    else "time-normalized derivative"
                ),
                "audit":audit,
            }
            mpath=sample_out/"feature_metadata.json"
            mpath.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")

            records.append({
                "sample_id":sid,
                "sign":row.get("sign",""),
                "subject_id":row.get("subject_id",""),
                "session_id":row.get("session_id",""),
                "split":row.get("split",""),
                "phase4_status":row.get("phase4_status",""),
                "phase5_status":status,
                "frame_count":audit["frame_count"],
                "full_feature_dim":audit["full_feature_dim"],
                "angle_valid_ratio":f'{audit["angle_valid_ratio"]:.6f}',
                "distance_valid_ratio":f'{audit["distance_valid_ratio"]:.6f}',
                "velocity_valid_ratio":f'{audit["velocity_valid_ratio"]:.6f}',
                "acceleration_valid_ratio":f'{audit["acceleration_valid_ratio"]:.6f}',
                "issues":";".join(audit["issues"]),
                "features_path":str(fpath),
                "metadata_path":str(mpath),
            })

            print(f"[{i:>3}/{len(selected)}] {status:<4} {sid}")

        except Exception as e:
            counts["FAIL"]+=1
            issue_counts[type(e).__name__]+=1
            records.append({
                "sample_id":sid,
                "sign":row.get("sign",""),
                "subject_id":row.get("subject_id",""),
                "session_id":row.get("session_id",""),
                "split":row.get("split",""),
                "phase4_status":row.get("phase4_status",""),
                "phase5_status":"FAIL",
                "frame_count":"",
                "full_feature_dim":"",
                "angle_valid_ratio":"",
                "distance_valid_ratio":"",
                "velocity_valid_ratio":"",
                "acceleration_valid_ratio":"",
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
        "phase":5,
        "selected_samples":len(selected),
        "processed_rows":len(records),
        "pass":counts["PASS"],
        "fail":counts["FAIL"],
        "issue_counts":dict(issue_counts),
        "feature_sets":{
            "position":126,
            "position_velocity":252,
            "position_velocity_acceleration":378,
            "full_position_angle_distance_velocity_acceleration":417,
        },
        "notes":{
            "body_distance_features":"zero+mask=false when pose landmarks are unavailable",
            "missing_motion":"velocity/acceleration only valid across consecutive present frames",
        },
    }
    out_summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    print("\nPHASE 5 FEATURE EXTRACTION FINISHED")
    print("="*68)
    print(f"Selected : {len(selected)}")
    print(f"PASS     : {counts['PASS']}")
    print(f"FAIL     : {counts['FAIL']}")
    print(f"Manifest : {out_manifest}")
    print(f"Summary  : {out_summary}")
    print("="*68)

if __name__=="__main__":
    main()
