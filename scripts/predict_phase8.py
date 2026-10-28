from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.unknown_phase8 import apply_unknown_threshold

def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("npz",help="NPZ containing X with shape [N,32,453]")
    ap.add_argument("--config",default="configs/phase8_unknown.yaml")
    args=ap.parse_args()

    cfg=yaml.safe_load(rp(args.config).read_text(encoding="utf-8"))
    cal=json.loads(rp(cfg["output"]["calibrated_json"]).read_text(encoding="utf-8"))
    thr=float(cal["threshold"])

    labels_path=rp(cfg["model"]["labels_json"])
    exp_labels=rp(cfg["model"]["experiment_dir"])/"labels.json"
    if exp_labels.is_file(): labels_path=exp_labels
    labels=json.loads(labels_path.read_text(encoding="utf-8"))["labels"]

    model_path=rp(cfg["model"]["experiment_dir"])/"model.keras"
    if not model_path.is_file(): model_path=rp(cfg["model"]["fallback_model"])

    with np.load(rp(args.npz),allow_pickle=False) as d:
        X=np.asarray(d["X"],np.float32)

    import tensorflow as tf
    model=tf.keras.models.load_model(model_path)
    probs=model.predict(X,verbose=0)
    pred,conf=apply_unknown_threshold(probs,labels,thr)

    for i,(lab,c) in enumerate(zip(pred,conf)):
        print(f"{i:04d}  {lab:14} confidence={c:.4f}")

if __name__=="__main__":
    main()
