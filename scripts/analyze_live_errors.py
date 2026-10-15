from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--log",default="logs/phase7_live_tcn_v1.csv")
    ap.add_argument("--out",default="logs/phase7_live_error_summary.json")
    ap.add_argument("--min-extra-per-error",type=int,default=10)
    args=ap.parse_args()

    path=ROOT/args.log
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r",encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))

    total=Counter()
    correct=Counter()
    false_neg=Counter()
    confusions=Counter()

    for r in rows:
        gt=r["ground_truth"]; pred=r["prediction"]
        total[gt]+=1
        if r.get("correct") in {"1","True","true"} or gt==pred:
            correct[gt]+=1
        else:
            false_neg[gt]+=1
            confusions[(gt,pred)]+=1

    rec={}
    for cls,n in total.items():
        err=false_neg[cls]
        rate=err/n if n else 0.0
        # Collection recommendation: at least 10 extra examples for each observed false,
        # bounded to avoid absurdly large suggestions from small live tests.
        suggested=min(100,max(0,err*args.min_extra_per_error))
        rec[cls]={
            "live_trials":n,
            "correct":correct[cls],
            "false":err,
            "error_rate":rate,
            "suggested_new_train_sequences":suggested,
        }

    report={
        "total_trials":len(rows),
        "total_false":sum(false_neg.values()),
        "per_class":rec,
        "confusions":[
            {"ground_truth":a,"predicted":b,"count":n}
            for (a,b),n in confusions.most_common()
        ],
        "collection_rule":
            "Collect targeted TRAIN data for recurring false-negative classes and confusing class pairs; "
            "do not add locked TEST subjects to training."
    }

    out=ROOT/args.out
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    print("\nLIVE ERROR ANALYSIS")
    print("="*68)
    print(f"Trials : {len(rows)}")
    print(f"False  : {sum(false_neg.values())}")
    for cls,v in sorted(rec.items()):
        print(
            f"{cls:12} trials={v['live_trials']:3} false={v['false']:3} "
            f"error={v['error_rate']*100:5.1f}%  suggest+={v['suggested_new_train_sequences']}"
        )
    print("Top confusions:")
    for (gt,pred),n in confusions.most_common(10):
        print(f"  {gt} -> {pred}: {n}")
    print("Report :",out)
    print("="*68)

if __name__=="__main__":
    main()
