from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np, yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.models_phase7 import build_model

def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

def load_split(path):
    with np.load(path,allow_pickle=False) as d:
        return np.asarray(d["X"],np.float32),np.asarray(d["y"],np.int64)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=["gru","tcn"],required=True)
    a=ap.parse_args()
    raw=yaml.safe_load(rp("configs/phase7_temporal.yaml").read_text(encoding="utf-8"))
    tr=raw["training"]; ds=raw["dataset_output"]; arts=raw["artifacts"]
    seed=int(tr.get("seed",42))
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

    Xtr,ytr=load_split(rp(ds["train_npz"]))
    Xv,yv=load_split(rp(ds["val_npz"]))
    labels=json.loads(rp(ds["labels_json"]).read_text(encoding="utf-8"))["labels"]

    model=build_model(a.model,Xtr.shape[1],Xtr.shape[2],len(labels),raw)
    outdir=rp(arts[f"{a.model}_dir"]); outdir.mkdir(parents=True,exist_ok=True)
    best=outdir/"model.keras"

    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            str(best),monitor=tr.get("monitor","val_loss"),
            save_best_only=True,mode="min",verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=tr.get("monitor","val_loss"),
            patience=int(tr.get("patience",15)),
            restore_best_weights=True,mode="min",verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=tr.get("monitor","val_loss"),
            patience=int(tr.get("reduce_lr_patience",7)),
            factor=float(tr.get("reduce_lr_factor",0.5)),
            min_lr=float(tr.get("min_learning_rate",1e-6)),
            mode="min",verbose=1
        )
    ]

    history=model.fit(
        Xtr,ytr,
        validation_data=(Xv,yv),
        epochs=int(tr.get("epochs",120)),
        batch_size=int(tr.get("batch_size",16)),
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    (outdir/"history.json").write_text(
        json.dumps({k:[float(x) for x in v] for k,v in history.history.items()},indent=2),
        encoding="utf-8"
    )
    (outdir/"training_config.json").write_text(
        json.dumps(raw,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print(f"\nBEST MODEL: {best}")

if __name__=="__main__":
    main()
