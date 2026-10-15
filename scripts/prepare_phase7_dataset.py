from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from core.preprocess_phase7 import (
    load_phase7_config, read_csv, split_rows, validate_split_subject_leakage,
    label_mapping_from_rows, encode_labels, build_fixed_split,
    fit_standardizer_train_only, apply_standardizer, FEATURE_ARRAYS
)

def rp(x):
    p=Path(x)
    return p if p.is_absolute() else ROOT/p

def save_npz(path,X,mask,y,sample_ids,source_index):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        mask=mask.astype(np.bool_),
        y=y.astype(np.int64),
        sample_ids=np.asarray(sample_ids),
        source_index=source_index.astype(np.int32),
    )

def main():
    cfg,raw=load_phase7_config(rp("configs/phase7_temporal.yaml"))
    inp=raw["input"]; out=raw["dataset_output"]
    rows=read_csv(rp(inp["phase6_manifest"]))
    rows=[r for r in rows if (r.get("phase6_status") or "").upper()=="PASS"]

    split_map=split_rows(rows)
    leak=validate_split_subject_leakage(split_map)
    if not leak["pass"]:
        raise RuntimeError(f"Signer leakage detected: {leak['leakage']}")

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

    for split in ("train","val","test"):
        built[split][0]=apply_standardizer(
            built[split][0],built[split][1],mean,std
        )

    outputs={
        "train":rp(out["train_npz"]),
        "val":rp(out["val_npz"]),
        "test":rp(out["test_npz"]),
    }
    for split,path in outputs.items():
        X,M,S,ids,y=built[split]
        save_npz(path,X,M,y,ids,S)

    labels_path=rp(out["labels_json"])
    labels_path.parent.mkdir(parents=True,exist_ok=True)
    labels_path.write_text(json.dumps({
        "labels":labels,
        "label_to_id":label_to_id,
    },ensure_ascii=False,indent=2),encoding="utf-8")

    np.savez_compressed(
        rp(out["preprocess_stats_npz"]),
        mean=mean,std=std,
        target_frames=np.asarray([cfg.target_frames],dtype=np.int32),
        feature_dim=np.asarray([FEATURE_ARRAYS[cfg.active_feature][1]],dtype=np.int32),
        feature_set=np.asarray([cfg.active_feature]),
    )

    summary={
        "phase":7,
        "feature_set":cfg.active_feature,
        "feature_dim":FEATURE_ARRAYS[cfg.active_feature][1],
        "target_frames":cfg.target_frames,
        "classes":labels,
        "num_classes":len(labels),
        "counts":{k:len(v) for k,v in split_map.items()},
        "subjects":leak["subjects"],
        "signer_leakage":leak["leakage"],
        "standardizer_fit_on":"TRAIN only",
        "class_counts":{
            split:dict(Counter(r["sign"] for r in split_map[split]))
            for split in ("train","val","test")
        },
    }
    rp(out["summary_json"]).write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    print("\nPHASE 7 DATA PREPARATION")
    print("="*64)
    print(f"Feature set    : {cfg.active_feature}")
    print(f"Feature dim    : {FEATURE_ARRAYS[cfg.active_feature][1]}")
    print(f"Target frames  : {cfg.target_frames}")
    print(f"TRAIN          : {len(split_map['train'])}")
    print(f"VAL            : {len(split_map['val'])}")
    print(f"TEST           : {len(split_map['test'])}")
    print(f"Classes        : {len(labels)} -> {labels}")
    print(f"Signer leakage : {leak['leakage_count']}")
    print("Standardizer   : FIT ON TRAIN ONLY")
    print("PREPARE        : PASS")
    print("="*64)

if __name__=="__main__":
    main()
