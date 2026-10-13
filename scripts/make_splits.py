from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config, resolve_project_path
from core.dataset_index import scan_raw_dataset
from core.split_utils import (
    audit_assignments,
    build_assignments,
    load_split_plan,
    make_summary,
    write_json,
    write_split_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 - Build signer-independent train/val/test split."
    )
    p.add_argument(
        "--split-config",
        default=str(ROOT / "configs" / "split_phase2.yaml"),
    )
    p.add_argument(
        "--project-config",
        default=None,
        help="Optional Phase-1/default project YAML.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_cfg = load_config(args.project_config)
    plan = load_split_plan(Path(args.split_config))

    raw_root = resolve_project_path(project_cfg["dataset"]["raw_root"])
    splits_root = resolve_project_path(
        project_cfg["dataset"].get("splits_root", "dataset/splits")
    )
    video_filename = project_cfg["dataset"].get(
        "video_filename", "source_video.mp4"
    )

    print("\nPHASE 2 - SIGNER-INDEPENDENT SPLIT")
    print("=" * 60)
    print(f"Raw dataset  : {raw_root}")
    print(f"Split config : {Path(args.split_config).resolve()}")
    print(f"Split version: {plan.split_version}")
    print(f"Strategy     : {plan.strategy}")
    print(f"TRAIN        : {' '.join(sorted(plan.train_subjects))}")
    print(f"VAL          : {' '.join(sorted(plan.val_subjects))}")
    print(f"TEST         : {' '.join(sorted(plan.test_subjects))}")
    print("=" * 60)

    scan = scan_raw_dataset(
        raw_root,
        video_filename=video_filename,
        strict_structure=True,
    )

    print(f"Samples found: {len(scan.records)}")
    print(f"Subjects     : {len(scan.subject_ids)} -> {' '.join(sorted(scan.subject_ids))}")
    print(f"Classes      : {len(scan.classes)} -> {' '.join(sorted(scan.classes))}")

    if scan.errors:
        print("\nDATASET SCAN FAILED")
        for err in scan.errors:
            print(f"  ERROR: {err}")
        raise SystemExit(2)

    if scan.warnings:
        print(f"\nScan warnings: {len(scan.warnings)}")
        for warning in scan.warnings[:20]:
            print(f"  WARNING: {warning}")
        if len(scan.warnings) > 20:
            print(f"  ... and {len(scan.warnings) - 20} more")

    assignments, unassigned = build_assignments(scan.records, plan)

    pre_audit = audit_assignments(
        scan.records,
        assignments,
        unassigned,
        plan,
        scan_errors=scan.errors,
        scan_warnings=scan.warnings,
    )

    if not pre_audit.passed:
        print("\nPHASE 2 PRE-AUDIT: FAIL")
        for err in pre_audit.errors:
            print(f"  ERROR: {err}")
        raise SystemExit(3)

    split_csv = splits_root / "split.csv"
    split_summary = splits_root / "split_summary.json"
    audit_report = splits_root / "audit_report.json"

    write_split_csv(split_csv, assignments, ROOT)
    write_json(
        split_summary,
        make_summary(scan.records, assignments, unassigned, plan),
    )
    write_json(audit_report, pre_audit.to_dict())

    summary = pre_audit.summary
    print("\nSPLIT CREATED")
    print("=" * 60)
    print(f"Train samples: {summary['train_samples']}")
    print(f"Val samples  : {summary['val_samples']}")
    print(f"Test samples : {summary['test_samples']}")
    print(f"Unassigned   : {summary['unassigned_samples']}")
    print("Signer leak  : 0")
    print(f"split.csv    : {split_csv}")
    print(f"summary      : {split_summary}")
    print(f"audit        : {audit_report}")
    print("=" * 60)
    print("PHASE 2 PRE-AUDIT: PASS")


if __name__ == "__main__":
    main()
