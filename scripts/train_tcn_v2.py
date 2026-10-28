from __future__ import annotations
import json,random,sys
from pathlib import Path
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.models_phase7 import build_tcn_model

def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

def load(path):
    with np.load(path,allow_pickle=False) as d:
        return np.asarray(d["X"],np.float32),np.asarray(d["y"],np.int64)

def main():
    raw=yaml.safe_load(rp("configs/phase7_v2_temporal.yaml").read_text(encoding="utf-8"))
    ds=raw["dataset_output"]; tr=raw["training"]; tc=raw["tcn"]
    seed=int(tr.get("seed",42))
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

    Xtr,ytr=load(rp(ds["train_npz"]))
    Xv,yv=load(rp(ds["val_npz"]))
    labels=json.loads(rp(ds["labels_json"]).read_text(encoding="utf-8"))["labels"]
    model=build_tcn_model(
        Xtr.shape[1],Xtr.shape[2],len(labels),
        filters=int(tc.get("filters",128)),
        kernel_size=int(tc.get("kernel_size",3)),
        dilations=tuple(tc.get("dilations",[1,2,4,8])),
        dense_units=int(tc.get("dense_units",64)),
        dropout=float(tc.get("dropout",0.3)),
        learning_rate=float(tr.get("learning_rate",0.001)),
    )
    out=rp(raw["artifacts"]["dir"]); out.mkdir(parents=True,exist_ok=True)
    best=out/"model.keras"
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            str(best),monitor="val_loss",save_best_only=True,mode="min",verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",patience=int(tr.get("patience",15)),
            restore_best_weights=True,mode="min",verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",patience=int(tr.get("reduce_lr_patience",7)),
            factor=float(tr.get("reduce_lr_factor",0.5)),
            min_lr=float(tr.get("min_learning_rate",1e-6)),mode="min",verbose=1
        )
    ]
    hist=model.fit(
        Xtr,ytr,validation_data=(Xv,yv),
        epochs=int(tr.get("epochs",120)),batch_size=int(tr.get("batch_size",16)),
        callbacks=callbacks,shuffle=True,verbose=2
    )
    (out/"history.json").write_text(
        json.dumps({k:[float(x) for x in v] for k,v in hist.history.items()},indent=2),
        encoding="utf-8"
    )
    (out/"training_config.json").write_text(
        json.dumps(raw,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    # copy exact v2 preprocessing next to model
    import shutil
    shutil.copy2(rp(ds["preprocess_stats_npz"]),out/"preprocess_stats.npz")
    shutil.copy2(rp(ds["labels_json"]),out/"labels.json")
    shutil.copy2(rp(ds["summary_json"]),out/"dataset_summary.json")
    print("\nTCN V2 BEST MODEL:",best)

if __name__=="__main__":
    main()
