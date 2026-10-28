from __future__ import annotations
import argparse,csv,json,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.unknown_phase8 import threshold_metrics,choose_threshold

def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

def load_npz(path):
    with np.load(path,allow_pickle=False) as d:
        return {k:np.asarray(d[k]) for k in d.files}

def resolve_model(cfg):
    exp=rp(cfg["model"]["experiment_dir"])/"model.keras"
    if exp.is_file(): return exp
    fb=rp(cfg["model"]["fallback_model"])
    if fb.is_file(): return fb
    raise FileNotFoundError(f"No TCN model found at {exp} or {fb}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/phase8_unknown.yaml")
    args=ap.parse_args()
    cfg=yaml.safe_load(rp(args.config).read_text(encoding="utf-8"))

    mode_path=rp(cfg["unknown_data"]["manifest_csv"]).parent/"unknown_manifest_mode.json"
    if mode_path.is_file():
        mode=json.loads(mode_path.read_text(encoding="utf-8")).get("mode")
        if mode=="exploratory_val_only":
            raise RuntimeError("QUICK TEST is not an independent Phase08 evaluation. Run scripts/calibrate_phase8_dev.py instead.")

    import tensorflow as tf
    model=tf.keras.models.load_model(resolve_model(cfg))

    known=load_npz(rp(cfg["known_data"]["val_npz"]))
    unknown=load_npz(rp(cfg["unknown_data"]["val_npz"]))
    if len(unknown["X"])==0:
        raise RuntimeError("UNKNOWN validation set is empty. Collect/prepare UNKNOWN first.")

    pk=model.predict(known["X"],verbose=0)
    pu=model.predict(unknown["X"],verbose=0)
    ck=np.max(pk,axis=1)
    cu=np.max(pu,axis=1)

    rows=[threshold_metrics(ck,cu,float(t)) for t in cfg["thresholds"]["candidates"]]
    best=choose_threshold(rows)

    out_dir=rp(cfg["output"]["dir"]); out_dir.mkdir(parents=True,exist_ok=True)
    csv_path=rp(cfg["output"]["calibration_csv"])
    with csv_path.open("w",encoding="utf-8-sig",newline="") as f:
        fields=list(asdict(rows[0]).keys())
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(asdict(r))

    calibrated={
        "threshold":best.threshold,
        "selection_metric":"balanced_accept_reject",
        "validation":{
            **asdict(best),
            "known_n":int(len(ck)),
            "unknown_n":int(len(cu)),
            "known_mean_conf":float(np.mean(ck)),
            "unknown_mean_conf":float(np.mean(cu)),
        },
        "candidate_thresholds":[float(x) for x in cfg["thresholds"]["candidates"]],
        "selection_rule":"maximize balanced_accept_reject; tie: lower FAR, lower FRR, higher threshold",
    }
    rp(cfg["output"]["calibrated_json"]).write_text(
        json.dumps(calibrated,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    rp(cfg["output"]["validation_report_json"]).write_text(
        json.dumps({
            "threshold_sweep":[asdict(r) for r in rows],
            "selected":calibrated,
        },ensure_ascii=False,indent=2),encoding="utf-8"
    )

    print("\nPHASE 08 THRESHOLD CALIBRATION")
    print("="*72)
    for r in rows:
        print(
            f"thr={r.threshold:.2f} "
            f"known_accept={r.known_accept_rate:.3f} "
            f"unknown_reject={r.unknown_reject_rate:.3f} "
            f"FAR={r.false_accept_rate:.3f} "
            f"FRR={r.false_reject_rate:.3f} "
            f"balanced={r.balanced_accept_reject:.3f}"
        )
    print("-"*72)
    print(f"SELECTED THRESHOLD = {best.threshold:.2f}")
    print("="*72)

if __name__=="__main__":
    main()
