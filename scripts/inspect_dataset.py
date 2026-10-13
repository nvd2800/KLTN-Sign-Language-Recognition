from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json

from core.config import load_config, resolve_project_path
from core.collector import verify_saved_sample


def parse_args():
    p = argparse.ArgumentParser(description="Inspect Phase 1 raw sequence dataset")
    p.add_argument("--config", default=None)
    p.add_argument("--strict", action="store_true", help="exit nonzero if an invalid sample is found")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    root = resolve_project_path(cfg["dataset"]["raw_root"])

    sample_dirs = sorted(p.parent for p in root.rglob("metadata.json")) if root.exists() else []
    if not sample_dirs:
        print(f"No samples found under: {root}")
        return

    class_counts = Counter()
    subject_counts = Counter()
    session_counts = Counter()
    combo_counts = defaultdict(int)
    invalid = []

    print(f"Dataset root: {root}")
    print(f"Found samples: {len(sample_dirs)}\n")

    for sample_dir in sample_dirs:
        try:
            report = verify_saved_sample(sample_dir)
            with (sample_dir / "metadata.json").open("r", encoding="utf-8") as f:
                meta = json.load(f)
            sign = str(meta["sign"])
            subject = str(meta["subject_id"])
            session = str(meta["session_id"])
            class_counts[sign] += 1
            subject_counts[subject] += 1
            session_counts[(subject, session)] += 1
            combo_counts[(sign, subject)] += 1
        except Exception as e:
            invalid.append((sample_dir, str(e)))

    print("Samples by class:")
    for k, v in sorted(class_counts.items()):
        print(f"  {k:15s} {v}")
    print("\nSamples by subject:")
    for k, v in sorted(subject_counts.items()):
        print(f"  {k:8s} {v}")
    print("\nClass x subject:")
    for (sign, subject), v in sorted(combo_counts.items()):
        print(f"  {sign:15s} {subject:8s} {v}")

    print(f"\nValid samples  : {len(sample_dirs) - len(invalid)}")
    print(f"Invalid samples: {len(invalid)}")
    for path, err in invalid:
        print(f"  INVALID {path}: {err}")

    if invalid and args.strict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
