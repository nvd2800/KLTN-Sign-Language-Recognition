from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path

def infer_from_path(path: Path, root: Path):
    rel=path.relative_to(root)
    parts=rel.parts
    category=parts[0] if parts else ""
    subject=""
    for x in parts:
        if re.fullmatch(r"S\d+", x, flags=re.I):
            subject=x.upper()
            break
    return category,subject

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features-root", default="dataset/processed/features_phase6_unknown")
    ap.add_argument("--out", default="dataset/unknown_phase8/unknown_manifest.csv")
    ap.add_argument("--val-subjects", default="")
    ap.add_argument("--test-subjects", default="")
    ap.add_argument("--default-split", choices=["val","test"], default="val")
    args=ap.parse_args()

    root=Path(args.features_root)
    if not root.is_absolute():
        root=Path.cwd()/root
    if not root.exists():
        raise FileNotFoundError(root)

    val_subjects={x.strip().upper() for x in args.val_subjects.split(",") if x.strip()}
    test_subjects={x.strip().upper() for x in args.test_subjects.split(",") if x.strip()}

    overlap=val_subjects & test_subjects
    if overlap:
        raise ValueError(f"Subject overlap between VAL/TEST: {sorted(overlap)}")

    rows=[]
    for feat in sorted(root.rglob("features_v3.npz")):
        sample_dir=feat.parent
        meta_path=sample_dir/"metadata.json"
        meta={}
        if meta_path.is_file():
            try:
                meta=json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        cat_path,sub_path=infer_from_path(feat,root)
        category=str(meta.get("category") or meta.get("label") or cat_path).strip()
        subject=str(meta.get("subject_id") or sub_path).strip().upper()
        sample_id=str(meta.get("sample_id") or sample_dir.name).strip()

        if subject in test_subjects:
            split="test"
        elif subject in val_subjects:
            split="val"
        else:
            split=str(meta.get("split") or args.default_split).strip().lower()

        if split not in {"val","test"}:
            raise ValueError(f"{feat}: invalid split={split}")

        try:
            feat_path=str(feat.relative_to(Path.cwd()))
        except Exception:
            feat_path=str(feat)

        rows.append({
            "sample_id":sample_id,
            "subject_id":subject,
            "category":category,
            "split":split,
            "feature_path":feat_path,
        })

    if not rows:
        raise RuntimeError(f"No features_v3.npz found under {root}")

    val={r["subject_id"] for r in rows if r["split"]=="val" and r["subject_id"]}
    test={r["subject_id"] for r in rows if r["split"]=="test" and r["subject_id"]}
    leakage=val & test
    if leakage:
        raise RuntimeError(f"UNKNOWN signer leakage VAL<->TEST: {sorted(leakage)}")

    out=Path(args.out)
    if not out.is_absolute():
        out=Path.cwd()/out
    out.parent.mkdir(parents=True,exist_ok=True)

    with out.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=[
            "sample_id","subject_id","category","split","feature_path"
        ])
        w.writeheader()
        w.writerows(rows)

    print("\nUNKNOWN MANIFEST CREATED")
    print("="*68)
    print("Rows :",len(rows))
    print("VAL  :",sum(r["split"]=="val" for r in rows))
    print("TEST :",sum(r["split"]=="test" for r in rows))
    print("VAL subjects :",sorted(val))
    print("TEST subjects:",sorted(test))
    print("Leakage      :",sorted(leakage))
    print("Output       :",out)

if __name__=="__main__":
    main()
