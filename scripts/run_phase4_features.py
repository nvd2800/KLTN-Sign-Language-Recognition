from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from core.feature_phase4 import (
    load_phase4_config, build_phase4_features,
    inspect_feature_arrays, save_feature_sample
)

def args():
    p=argparse.ArgumentParser(description="Phase 4 - XYZ feature normalization")
    p.add_argument("--config",default="configs/feature_phase4.yaml")
    p.add_argument("--limit",type=int,default=0)
    return p.parse_args()

def rp(x): 
    p=Path(x)
    return p if p.is_absolute() else ROOT/p

def read_csv(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def mask_from_npz(d,names,T,required=True):
    for n in names:
        if n in d.files:
            a=np.asarray(d[n]).reshape(-1)
            if len(a)!=T: raise ValueError(f"{n} length {len(a)} != {T}")
            return a.astype(bool)
    if required: raise ValueError("missing mask: "+"/".join(names))
    return np.zeros(T,dtype=bool)

def find_sample_dir(row,raw_root):
    sp=(row.get("sample_path") or "").strip()
    if sp:
        p=rp(sp)
        if p.is_dir(): return p
    sign=(row.get("sign") or "").strip()
    subject=(row.get("subject_id") or "").strip()
    session=(row.get("session_id") or "").strip()
    sid=(row.get("sample_id") or "").strip()
    p=raw_root/sign/subject/session/sid
    if p.is_dir(): return p
    raise FileNotFoundError(f"sample folder not found: {p}")

def main():
    a=args()
    cfg,raw=load_phase4_config(rp(a.config))
    inp=raw.get("input",{}) or {}; sel=raw.get("selection",{}) or {}; out=raw.get("output",{}) or {}
    manifest=rp(inp.get("quality_manifest","dataset/processed/quality/quality_manifest.csv"))
    raw_root=rp(inp.get("raw_root","dataset/raw"))
    output_root=rp(out.get("root","dataset/processed/features_phase4"))
    output_root.mkdir(parents=True,exist_ok=True)
    out_manifest=rp(out.get("manifest_csv","dataset/processed/features_phase4/feature_manifest.csv"))
    out_summary=rp(out.get("summary_json","dataset/processed/features_phase4/feature_summary.json"))

    include={str(x).upper() for x in sel.get("include_statuses",["ACCEPT","REVIEW"])}
    exclude={str(x).upper() for x in sel.get("exclude_statuses",["REJECT"])}

    rows=read_csv(manifest)
    selected=[r for r in rows if (r.get("status") or "").upper() in include and (r.get("status") or "").upper() not in exclude]
    if a.limit>0: selected=selected[:a.limit]

    records=[]; counts=Counter(); by_split=Counter(); by_sign=Counter()
    issue_counts=Counter()

    for i,row in enumerate(selected,1):
        sid=(row.get("sample_id") or "").strip()
        status=(row.get("status") or "").upper().strip()
        try:
            sample_dir=find_sample_dir(row,raw_root)
            npz_path=sample_dir/"landmarks.npz"
            if not npz_path.is_file(): raise FileNotFoundError(str(npz_path))

            with np.load(npz_path,allow_pickle=False) as d:
                ts=np.asarray(d["timestamps"],dtype=np.float64)
                T=len(ts)
                left=np.asarray(d["left_hand"],dtype=np.float32)
                right=np.asarray(d["right_hand"],dtype=np.float32)
                lp=mask_from_npz(d,("left_present","left_hand_present"),T)
                rp_=mask_from_npz(d,("right_present","right_hand_present"),T)
                pose=mask_from_npz(d,("pose_present",),T,False)
                face=mask_from_npz(d,("face_present",),T,False)

            arrays=build_phase4_features(left,right,lp,rp_,cfg)
            arrays["timestamps"]=ts.astype(np.float64)
            arrays["pose_present"]=pose.astype(np.bool_)
            arrays["face_present"]=face.astype(np.bool_)

            audit=inspect_feature_arrays(arrays,cfg)
            phase4_status="PASS" if audit["pass"] else "FAIL"
            counts[phase4_status]+=1
            for issue in audit["issues"]: issue_counts[issue.split(":")[0]]+=1

            meta={
                "phase":4,
                "feature_version":raw.get("feature_version","v1"),
                "sample_id":sid,
                "sign":row.get("sign",""),
                "subject_id":row.get("subject_id",""),
                "session_id":row.get("session_id",""),
                "split":row.get("split",""),
                "phase3_status":status,
                "source_landmarks":str(npz_path),
                "normalization":{
                    "translation_origin":"wrist landmark 0",
                    "scale_reference":"wrist -> middle MCP landmark 9",
                    "left_right_order":"LEFT 63 then RIGHT 63",
                    "feature_dim":126,
                },
                "audit":audit,
            }
            sample_out=output_root/sid
            fpath,mpath=save_feature_sample(sample_out,arrays,meta)

            rec={
                "sample_id":sid,
                "sign":row.get("sign",""),
                "subject_id":row.get("subject_id",""),
                "session_id":row.get("session_id",""),
                "split":row.get("split",""),
                "phase3_status":status,
                "phase4_status":phase4_status,
                "frame_count":audit["frame_count"],
                "feature_dim":audit["feature_dim"],
                "any_hand_valid_ratio":f'{audit["any_hand_valid_ratio"]:.6f}',
                "left_valid_ratio":f'{audit["left_valid_ratio"]:.6f}',
                "right_valid_ratio":f'{audit["right_valid_ratio"]:.6f}',
                "invalid_scale_hand_frames":audit["invalid_scale_hand_frames"],
                "max_abs_normalized":f'{audit["max_abs_normalized"]:.6f}',
                "max_wrist_abs":f'{audit["max_wrist_abs"]:.8f}',
                "issues":";".join(audit["issues"]),
                "features_path":str(fpath),
                "metadata_path":str(mpath),
            }
            records.append(rec); by_split[row.get("split","")]+=1; by_sign[row.get("sign","")]+=1
            print(f"[{i:>3}/{len(selected)}] {phase4_status:<4} {sid}")

        except Exception as e:
            counts["FAIL"]+=1
            issue_counts[type(e).__name__]+=1
            records.append({
                "sample_id":sid,"sign":row.get("sign",""),
                "subject_id":row.get("subject_id",""),"session_id":row.get("session_id",""),
                "split":row.get("split",""),"phase3_status":status,
                "phase4_status":"FAIL","frame_count":"","feature_dim":"",
                "any_hand_valid_ratio":"","left_valid_ratio":"","right_valid_ratio":"",
                "invalid_scale_hand_frames":"","max_abs_normalized":"","max_wrist_abs":"",
                "issues":f"{type(e).__name__}:{e}","features_path":"","metadata_path":"",
            })
            print(f"[{i:>3}/{len(selected)}] FAIL {sid}: {type(e).__name__}: {e}")

    fields=list(records[0].keys()) if records else [
        "sample_id","sign","subject_id","session_id","split","phase3_status","phase4_status",
        "frame_count","feature_dim","any_hand_valid_ratio","left_valid_ratio","right_valid_ratio",
        "invalid_scale_hand_frames","max_abs_normalized","max_wrist_abs","issues","features_path","metadata_path"
    ]
    out_manifest.parent.mkdir(parents=True,exist_ok=True)
    with out_manifest.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)

    summary={
        "phase":4,
        "feature_version":raw.get("feature_version","v1"),
        "quality_manifest_rows":len(rows),
        "selected_samples":len(selected),
        "processed_rows":len(records),
        "pass":counts["PASS"],
        "fail":counts["FAIL"],
        "selection":{"include_statuses":sorted(include),"exclude_statuses":sorted(exclude)},
        "by_split":dict(by_split),"by_sign":dict(by_sign),"issue_counts":dict(issue_counts),
        "feature_definition":{
            "per_hand":"21 landmarks x XYZ = 63",
            "two_hands":"126 features/frame",
            "translation":"subtract wrist landmark 0 independently per hand",
            "scale":"divide by Euclidean distance wrist -> middle MCP landmark 9",
            "missing":"zero vector only together with explicit presence masks",
        },
    }
    out_summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    print("\nPHASE 4 FEATURE NORMALIZATION FINISHED")
    print("="*64)
    print(f"Selected : {len(selected)}")
    print(f"PASS     : {counts['PASS']}")
    print(f"FAIL     : {counts['FAIL']}")
    print(f"Manifest : {out_manifest}")
    print(f"Summary  : {out_summary}")
    print("="*64)

if __name__=="__main__":
    main()
