from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from core.preprocess_phase7 import uniform_resample_or_pad

FEATURE_KEY="ablation_hands_pose_face453"

def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

def load_cfg(path):
    return yaml.safe_load(rp(path).read_text(encoding="utf-8"))

def read_manifest(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    required={"sample_id","category","split","feature_path"}
    missing=required-set(rows[0].keys()) if rows else required
    if missing: raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return rows

def load_feature(path):
    with np.load(path,allow_pickle=False) as d:
        if FEATURE_KEY not in d:
            raise KeyError(f"{path}: missing array {FEATURE_KEY}")
        x=np.asarray(d[FEATURE_KEY],np.float32)
    if x.ndim!=2 or x.shape[1]!=453:
        raise ValueError(f"{path}: expected [T,453], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: NaN/Inf")
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/phase8_unknown.yaml")
    args=ap.parse_args()
    cfg=load_cfg(args.config)

    manifest=rp(cfg["unknown_data"]["manifest_csv"])
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{manifest} not found.\n"
            "Copy dataset_templates/unknown_manifest_template.csv to this path and fill it."
        )
    rows=read_manifest(manifest)

    stats_path=rp(cfg["model"]["preprocess_stats"])
    with np.load(stats_path,allow_pickle=False) as d:
        mean=np.asarray(d["mean"],np.float32)
        std=np.asarray(d["std"],np.float32)
        target_frames=int(np.asarray(d["target_frames"]).reshape(-1)[0])
        feature_dim=int(np.asarray(d["feature_dim"]).reshape(-1)[0])

    if feature_dim!=453:
        raise RuntimeError(f"Expected Phase07 scaler feature_dim=453, got {feature_dim}")

    allowed={"val","test"}
    bad=sorted({(r.get("split") or "").strip().lower() for r in rows}-allowed)
    if bad: raise ValueError(f"UNKNOWN split must be val/test only; got {bad}")

    out_by_split={}
    for split in ("val","test"):
        selected=[r for r in rows if r["split"].strip().lower()==split]
        X=[]; masks=[]; ids=[]; categories=[]; subjects=[]
        for r in selected:
            p=rp(r["feature_path"])
            x=load_feature(p)
            fixed,mask,_=uniform_resample_or_pad(x,target_frames)
            z=np.zeros_like(fixed,dtype=np.float32)
            z[mask]=(fixed[mask]-mean)/std
            z[~mask]=0.0
            X.append(z); masks.append(mask)
            ids.append(r["sample_id"])
            categories.append(r["category"])
            subjects.append(r.get("subject_id",""))
        if X:
            X=np.stack(X).astype(np.float32)
            masks=np.stack(masks).astype(bool)
        else:
            X=np.zeros((0,target_frames,453),np.float32)
            masks=np.zeros((0,target_frames),bool)
        out_path=rp(cfg["unknown_data"][f"{split}_npz"])
        out_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(
            out_path,X=X,mask=masks,
            sample_ids=np.asarray(ids),
            categories=np.asarray(categories),
            subject_ids=np.asarray(subjects),
        )
        out_by_split[split]=len(ids)
        print(f"{split.upper():4} UNKNOWN: {len(ids)} -> {out_path}")

    summary={
        "manifest":str(manifest),
        "feature_key":FEATURE_KEY,
        "target_frames":target_frames,
        "feature_dim":453,
        "counts":out_by_split,
        "standardizer":"frozen Phase07 TRAIN stats only",
    }
    summary_path=rp(cfg["output"]["dir"])/"unknown_dataset_summary.json"
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("PHASE08 UNKNOWN PREP: PASS")

if __name__=="__main__":
    main()
