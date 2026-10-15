from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,yaml
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=["gru","tcn"],required=True)
    a=ap.parse_args()
    raw=yaml.safe_load(rp("configs/phase7_temporal.yaml").read_text(encoding="utf-8"))
    ds=raw["dataset_output"]; arts=raw["artifacts"]
    with np.load(rp(ds["test_npz"]),allow_pickle=False) as d:
        X=np.asarray(d["X"],np.float32); y=np.asarray(d["y"],np.int64)
    labels=json.loads(rp(ds["labels_json"]).read_text(encoding="utf-8"))["labels"]

    import tensorflow as tf
    model_path=rp(arts[f"{a.model}_dir"])/"model.keras"
    model=tf.keras.models.load_model(model_path)
    probs=model.predict(X,verbose=0)
    pred=np.argmax(probs,axis=1)

    metrics={
        "test_accuracy":float(accuracy_score(y,pred)),
        "test_macro_f1":float(f1_score(y,pred,average="macro")),
        "classification_report":classification_report(
            y,pred,labels=list(range(len(labels))),target_names=labels,
            output_dict=True,zero_division=0
        ),
    }
    outdir=model_path.parent
    (outdir/"test_metrics.json").write_text(
        json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    cm=confusion_matrix(y,pred,labels=list(range(len(labels))))
    with (outdir/"confusion_matrix.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f); w.writerow(["true\\pred"]+labels)
        for name,row in zip(labels,cm): w.writerow([name]+row.tolist())

    print(f"\n{a.model.upper()} UNSEEN-SIGNER TEST")
    print("="*56)
    print(f"Accuracy : {metrics['test_accuracy']:.4f}")
    print(f"Macro F1 : {metrics['test_macro_f1']:.4f}")
    print(f"Model    : {model_path}")
    print("="*56)

if __name__=="__main__":
    main()
