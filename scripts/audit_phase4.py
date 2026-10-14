from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
def rp(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p

def main():
    p=argparse.ArgumentParser(description="Audit Phase 4 feature outputs")
    p.add_argument("--config",default="configs/feature_phase4.yaml")
    a=p.parse_args()
    raw=yaml.safe_load(rp(a.config).read_text(encoding="utf-8")) or {}
    inp=raw.get("input",{}) or {}; sel=raw.get("selection",{}) or {}; out=raw.get("output",{}) or {}
    qpath=rp(inp.get("quality_manifest","dataset/processed/quality/quality_manifest.csv"))
    fpath=rp(out.get("manifest_csv","dataset/processed/features_phase4/feature_manifest.csv"))
    apath=rp(out.get("audit_json","dataset/processed/features_phase4/feature_audit.json"))

    def rows(path):
        with path.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
    q=rows(qpath); f=rows(fpath)
    include={str(x).upper() for x in sel.get("include_statuses",["ACCEPT","REVIEW"])}
    exclude={str(x).upper() for x in sel.get("exclude_statuses",["REJECT"])}
    selected=[r for r in q if (r.get("status") or "").upper() in include and (r.get("status") or "").upper() not in exclude]
    qmap={r["sample_id"]:r for r in selected}; fmap={r["sample_id"]:r for r in f}
    missing=sorted(set(qmap)-set(fmap)); extra=sorted(set(fmap)-set(qmap))
    failed=[]; split_mismatch=[]; feature_errors=[]

    for sid,row in fmap.items():
        if row.get("phase4_status")!="PASS": failed.append(sid)
        if sid in qmap and row.get("split","")!=qmap[sid].get("split",""): split_mismatch.append(sid)
        pth=row.get("features_path","")
        if not pth: continue
        pth=Path(pth)
        if not pth.is_absolute(): pth=ROOT/pth
        if not pth.is_file():
            feature_errors.append(f"{sid}:missing_features")
            continue
        try:
            with np.load(pth,allow_pickle=False) as d:
                x=np.asarray(d["normalized_xyz126"])
                pres=np.asarray(d["hand_presence2"])
                if x.ndim!=2 or x.shape[1]!=126: feature_errors.append(f"{sid}:shape={x.shape}")
                if pres.shape!=(x.shape[0],2): feature_errors.append(f"{sid}:presence={pres.shape}")
                if not np.isfinite(x).all(): feature_errors.append(f"{sid}:non_finite")
        except Exception as e:
            feature_errors.append(f"{sid}:{type(e).__name__}:{e}")

    passed=(not missing and not extra and not failed and not split_mismatch and not feature_errors)
    report={
        "phase":4,"selected_phase3_samples":len(selected),"phase4_rows":len(f),
        "pass_rows":sum(r.get("phase4_status")=="PASS" for r in f),
        "fail_rows":sum(r.get("phase4_status")!="PASS" for r in f),
        "missing":missing,"extra":extra,"split_mismatch":split_mismatch,
        "feature_errors":feature_errors,"phase4_pass":passed,
    }
    apath.parent.mkdir(parents=True,exist_ok=True)
    apath.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nPHASE 4 FEATURE AUDIT")
    print("="*60)
    print(f"Selected Phase-3 : {len(selected)}")
    print(f"Phase-4 rows     : {len(f)}")
    print(f"PASS rows        : {report['pass_rows']}")
    print(f"FAIL rows        : {report['fail_rows']}")
    print(f"Missing          : {len(missing)}")
    print(f"Extra            : {len(extra)}")
    print(f"Split mismatch   : {len(split_mismatch)}")
    print(f"Feature errors   : {len(feature_errors)}")
    print(f"PHASE 4          : {'PASS' if passed else 'FAIL'}")
    print(f"Report           : {apath}")
    print("="*60)
    raise SystemExit(0 if passed else 1)

if __name__=="__main__":
    main()
