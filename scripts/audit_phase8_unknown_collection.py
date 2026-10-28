from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CATEGORIES = {
    "scratching_head","holding_phone","touching_face","random_hand",
    "waving","pointing","transition","no_hand","other_gesture"
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/unknown_phase8")
    ap.add_argument("--min-per-category", type=int, default=5)
    args=ap.parse_args()

    root=Path(args.root)
    if not root.is_absolute():
        root=Path.cwd()/root
    if not root.exists():
        raise FileNotFoundError(root)

    counts=Counter()
    subjects=defaultdict(set)
    errors=[]

    meta_files=sorted(root.rglob("metadata.json"))

    for meta_path in meta_files:
        try:
            m=json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{meta_path}: invalid json: {e}")
            continue

        category=str(m.get("category") or m.get("label") or "").strip()
        subject=str(m.get("subject_id") or "").strip()

        if not category:
            try:
                category=meta_path.relative_to(root).parts[0]
            except Exception:
                category=""

        if not category:
            errors.append(f"{meta_path}: missing category/label")
            continue

        counts[category]+=1
        if subject:
            subjects[category].add(subject)

    if not meta_files:
        for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            sample_dirs=[
                p for p in category_dir.rglob("*")
                if p.is_dir() and (
                    (p/"landmarks.npz").is_file()
                    or (p/"source_video.mp4").is_file()
                )
            ]
            counts[category_dir.name]+=len(sample_dirs)

    print("\nPHASE 08 UNKNOWN RAW AUDIT")
    print("="*72)
    print("Root:", root)
    print("Metadata files:", len(meta_files))
    print("Samples detected:", sum(counts.values()))
    print()

    all_categories=sorted(set(counts) | DEFAULT_CATEGORIES)
    for c in all_categories:
        print(f"{c:18} samples={counts[c]:4} subjects={len(subjects[c]):2}")

    missing=[c for c in DEFAULT_CATEGORIES if counts[c]==0]
    low=[c for c in DEFAULT_CATEGORIES if 0<counts[c]<args.min_per_category]

    print("\nMissing categories:", missing if missing else "None")
    print("Low-count categories:", low if low else "None")
    print("Metadata errors:", len(errors))

    summary={
        "root":str(root),
        "total_samples":sum(counts.values()),
        "counts":dict(counts),
        "subjects_per_category":{k:sorted(v) for k,v in subjects.items()},
        "missing_categories":missing,
        "low_count_categories":low,
        "errors":errors,
    }
    out=root/"unknown_collection_audit.json"
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Audit JSON:",out)

if __name__=="__main__":
    main()
