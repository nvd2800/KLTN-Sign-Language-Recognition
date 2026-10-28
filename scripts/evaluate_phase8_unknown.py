from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.unknown_phase8 import known_classification_with_rejection,unknown_rejection_metrics

def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p
def load(path):
    with np.load(path,allow_pickle=False) as d:
        return {k:np.asarray(d[k]) for k in d.files}
def model_path(cfg):
    a=rp(cfg["model"]["experiment_dir"])/"model.keras"
    if a.is_file(): return a
    b=rp(cfg["model"]["fallback_model"])
    if b.is_file(): return b
    raise FileNotFoundError("TCN model not found")

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

    cal=json.loads(rp(cfg["output"]["calibrated_json"]).read_text(encoding="utf-8"))
    threshold=float(cal["threshold"])

    known=load(rp(cfg["known_data"]["test_npz"]))
    unknown=load(rp(cfg["unknown_data"]["test_npz"]))
    if len(unknown["X"])==0:
        raise RuntimeError("UNKNOWN test set is empty.")

    import tensorflow as tf
    model=tf.keras.models.load_model(model_path(cfg))
    pk=model.predict(known["X"],verbose=0)
    pu=model.predict(unknown["X"],verbose=0)

    known_metrics=known_classification_with_rejection(known["y"],pk,threshold)
    unknown_metrics=unknown_rejection_metrics(pu,threshold)

    report={
        "threshold":threshold,
        "threshold_source":"validation calibration only",
        "known_test":known_metrics,
        "unknown_test":unknown_metrics,
        "balanced_known_accept_unknown_reject":0.5*(
            known_metrics["accept_rate"]+unknown_metrics["unknown_reject_rate"]
        ),
    }
    rp(cfg["output"]["test_report_json"]).write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    print("\nPHASE 08 UNKNOWN/OOV TEST")
    print("="*68)
    print(f"Threshold              : {threshold:.2f}")
    print(f"Known accept rate      : {known_metrics['accept_rate']:.4f}")
    print(f"Known false reject FRR : {known_metrics['false_reject_rate']:.4f}")
    print(f"Known accepted accuracy: {known_metrics['accuracy_on_accepted']:.4f}")
    print(f"UNKNOWN reject rate    : {unknown_metrics['unknown_reject_rate']:.4f}")
    print(f"UNKNOWN false accept FAR: {unknown_metrics['false_accept_rate']:.4f}")
    print(f"Balanced score         : {report['balanced_known_accept_unknown_reject']:.4f}")
    print("="*68)

if __name__=="__main__":
    main()
