from __future__ import annotations
import json,sys
from collections import Counter
from pathlib import Path
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from core.preprocess_phase7 import (
    Phase7Config, read_csv, split_rows, validate_split_subject_leakage,
    label_mapping_from_rows, encode_labels, build_fixed_split,
    fit_standardizer_train_only, apply_standardizer, FEATURE_ARRAYS
)

def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

def main():
    raw=yaml.safe_load(rp("configs/phase7_v2_temporal.yaml").read_text(encoding="utf-8"))
    seq=raw["sequence"]; feat=raw["feature"]; out=raw["dataset_output"]
    cfg=Phase7Config(
        target_frames=int(seq.get("target_frames",32)),
        short_sequence_policy="pad",
        standardize_epsilon=float(seq.get("standardize_epsilon",1e-6)),
        active_feature=str(feat.get("active","hands_pose_face453")),
    )

    rows=read_csv(rp(raw["input"]["phase6_manifest"]))
    rows=[r for r in rows if (r.get("phase6_status") or "").upper()=="PASS"]
    split_map=split_rows(rows)

    # Keep exactly the V1 locked TEST set for fair V1-vs-V2 comparison.
    lock_path=ROOT/"artifacts"/"experiments"/"tcn_v1_160samples"/"locked_test_sample_ids.json"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"{lock_path} missing. Re-run: python scripts/freeze_tcn_v1.py --force"
        )
    locked_ids=set(json.loads(lock_path.read_text(encoding="utf-8"))["sample_ids"])
    current_test={r["sample_id"]:r for r in split_map["test"]}
    missing_locked=sorted(locked_ids-set(current_test))
    if missing_locked:
        raise RuntimeError(f"Locked V1 TEST samples missing from Phase06: {missing_locked}")
    split_map["test"]=[current_test[sid] for sid in sorted(locked_ids)]

    leak=validate_split_subject_leakage(split_map)
    if not leak["pass"]:
        raise RuntimeError(f"Signer leakage: {leak['leakage']}")

    labels,label_to_id=label_mapping_from_rows(rows)
    train_labels={r["sign"] for r in split_map["train"]}
    missing_train=sorted(set(labels)-train_labels)
    if missing_train:
        raise RuntimeError(f"Classes missing from TRAIN: {missing_train}")

    built={}
    for split in ("train","val","test"):
        X,M,S,ids=build_fixed_split(split_map[split],ROOT,cfg)
        y=encode_labels(split_map[split],label_to_id)
        built[split]=[X,M,S,ids,y]

    mean,std=fit_standardizer_train_only(
        built["train"][0],built["train"][1],cfg.standardize_epsilon
    )
    for split in built:
        built[split][0]=apply_standardizer(
            built[split][0],built[split][1],mean,std
        )

    for split,key in [("train","train_npz"),("val","val_npz"),("test","test_npz")]:
        X,M,S,ids,y=built[split]
        p=rp(out[key]); p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(
            p,X=X.astype(np.float32),mask=M.astype(bool),y=y.astype(np.int64),
            sample_ids=np.asarray(ids),source_index=S.astype(np.int32)
        )

    rp(out["labels_json"]).write_text(
        json.dumps({"labels":labels,"label_to_id":label_to_id},ensure_ascii=False,indent=2),
        encoding="utf-8"
    )
    np.savez_compressed(
        rp(out["preprocess_stats_npz"]),
        mean=mean,std=std,
        target_frames=np.asarray([cfg.target_frames],np.int32),
        feature_dim=np.asarray([FEATURE_ARRAYS[cfg.active_feature][1]],np.int32),
        feature_set=np.asarray([cfg.active_feature]),
    )
    summary={
        "experiment":raw.get("experiment"),
        "feature_set":cfg.active_feature,
        "target_frames":cfg.target_frames,
        "classes":labels,
        "counts":{k:len(v) for k,v in split_map.items()},
        "subjects":leak["subjects"],
        "signer_leakage":leak["leakage"],
        "standardizer_fit_on":"TRAIN only",
        "class_counts":{
            s:dict(Counter(r["sign"] for r in split_map[s])) for s in split_map
        },
    }
    rp(out["summary_json"]).write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    print("\nTCN V2 DATA PREP")
    print("="*64)
    print("TRAIN          :",len(split_map["train"]))
    print("VAL            :",len(split_map["val"]))
    print("TEST           :",len(split_map["test"]))
    print("Signer leakage :",leak["leakage_count"])
    print("Scaler         : TRAIN only")
    print("Status         : PASS")
    print("="*64)

if __name__=="__main__":
    main()
