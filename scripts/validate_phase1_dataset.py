from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import Counter, defaultdict
import json

from core.config import load_config, resolve_project_path
from core.collector import verify_saved_sample


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 exit-gate validator")
    p.add_argument("--config", default=None)
    p.add_argument("--require-mini-complete", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    root = resolve_project_path(cfg["dataset"]["raw_root"])
    dataset_cfg = load_config("configs/dataset.yaml")
    required_classes = dataset_cfg["classes_phase1_mini"]
    guidance = dataset_cfg["collection_guidance"]
    target_subjects = int(guidance["target_subjects_mini"])
    target_per_combo = int(guidance["target_samples_per_class_per_subject"])
    min_frames = int(guidance["target_sequence_frames_min"])
    max_frames = int(guidance["target_sequence_frames_max"])

    sample_dirs = sorted(p.parent for p in root.rglob("metadata.json")) if root.exists() else []
    errors = []
    warnings = []
    if not sample_dirs:
        errors.append("dataset contains no samples")
    combo = Counter()
    subjects = set()
    classes = set()

    for d in sample_dirs:
        try:
            rep = verify_saved_sample(d)
            with (d / "metadata.json").open("r", encoding="utf-8") as f:
                m = json.load(f)
            sign, subj = m["sign"], m["subject_id"]
            classes.add(sign)
            subjects.add(subj)
            combo[(sign, subj)] += 1
            n = int(m["frame_count"])
            if not min_frames <= n <= max_frames:
                errors.append(f"{d}: frame_count={n}, expected {min_frames}-{max_frames}")
        except Exception as e:
            errors.append(f"{d}: {e}")

    for cls in required_classes:
        if cls not in classes:
            errors.append(f"missing required Phase-1 class: {cls}")

    if args.require_mini_complete:
        if len(subjects) < target_subjects:
            errors.append(f"need >= {target_subjects} subjects, found {len(subjects)}")
        for cls in required_classes:
            for subj in sorted(subjects):
                if combo[(cls, subj)] < target_per_combo:
                    errors.append(f"{cls}/{subj}: {combo[(cls, subj)]} samples, need >= {target_per_combo}")

    print("PHASE 1 DATASET VALIDATION")
    print(f"Root          : {root}")
    print(f"Samples       : {len(sample_dirs)}")
    print(f"Subjects      : {len(subjects)} {sorted(subjects)}")
    print(f"Classes       : {len(classes)} {sorted(classes)}")
    print(f"Errors        : {len(errors)}")
    print(f"Warnings      : {len(warnings)}")
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    if errors:
        print("\nPHASE 1 EXIT GATE: FAIL")
        raise SystemExit(2)
    print("\nPHASE 1 EXIT GATE: PASS")


if __name__ == "__main__":
    main()
