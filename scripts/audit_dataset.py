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
    load_split_plan,
    read_split_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 - Audit signer-independent dataset split."
    )
    p.add_argument(
        "--split-config",
        default=str(ROOT / "configs" / "split_phase2.yaml"),
    )
    p.add_argument("--project-config", default=None)
    p.add_argument(
        "--split",
        default=None,
        help="Default: dataset/splits/split.csv from project config.",
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
    split_csv = Path(args.split).resolve() if args.split else splits_root / "split.csv"
    report_path = splits_root / "audit_report.json"
    video_filename = project_cfg["dataset"].get(
        "video_filename", "source_video.mp4"
    )

    scan = scan_raw_dataset(
        raw_root,
        video_filename=video_filename,
        strict_structure=True,
    )

    rows = read_split_csv(split_csv)

    by_id = {r.sample_id: r for r in scan.records}
    assignments = []
    csv_ids: list[str] = []
    csv_errors: list[str] = []

    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        split = (row.get("split") or "").strip()

        if not sample_id:
            csv_errors.append("split.csv contains row without sample_id")
            continue

        csv_ids.append(sample_id)
        record = by_id.get(sample_id)

        if record is None:
            csv_errors.append(
                f"split.csv sample not found in raw dataset: {sample_id}"
            )
            continue

        for field, actual in (
            ("sign", record.sign),
            ("subject_id", record.subject_id),
            ("session_id", record.session_id),
        ):
            csv_value = (row.get(field) or "").strip()
            if csv_value != actual:
                csv_errors.append(
                    f"{sample_id}: split.csv {field}='{csv_value}' "
                    f"!= raw metadata '{actual}'"
                )

        assignments.append((record, split))

    duplicate_csv_ids = sorted(
        {x for x in csv_ids if csv_ids.count(x) > 1}
    )
    if duplicate_csv_ids:
        csv_errors.append(
            f"Duplicate sample_id rows in split.csv: {duplicate_csv_ids}"
        )

    raw_ids = set(by_id)
    split_ids = set(csv_ids)

    missing_from_split = sorted(raw_ids - split_ids)
    extra_in_split = sorted(split_ids - raw_ids)

    if missing_from_split:
        csv_errors.append(
            f"Raw samples missing from split.csv: {missing_from_split}"
        )
    if extra_in_split:
        csv_errors.append(
            f"split.csv contains unknown samples: {extra_in_split}"
        )

    assigned_ids = {r.sample_id for r, _ in assignments}
    unassigned = [r for r in scan.records if r.sample_id not in assigned_ids]

    result = audit_assignments(
        scan.records,
        assignments,
        unassigned,
        plan,
        scan_errors=scan.errors + csv_errors,
        scan_warnings=scan.warnings,
    )

    write_json(report_path, result.to_dict())

    print("\nPHASE 2 DATASET AUDIT")
    print("=" * 64)
    print(f"Raw samples          : {len(scan.records)}")
    print(f"split.csv rows       : {len(rows)}")
    print(f"Subjects             : {len(scan.subject_ids)}")
    print(f"Classes              : {len(scan.classes)}")
    print()
    print("TRAIN subjects       :", " ".join(result.summary["train_subjects"]))
    print("VAL subjects         :", " ".join(result.summary["val_subjects"]))
    print("TEST subjects        :", " ".join(result.summary["test_subjects"]))
    print()
    print("Train/Val leakage    :", len(result.summary["signer_leakage"]["train_val"]))
    print("Train/Test leakage   :", len(result.summary["signer_leakage"]["train_test"]))
    print("Val/Test leakage     :", len(result.summary["signer_leakage"]["val_test"]))
    print("Unassigned samples   :", result.summary["unassigned_samples"])
    print("Duplicate sample IDs :", len(result.summary["duplicate_sample_ids"]))
    print("Duplicate paths      :", len(result.summary["duplicate_paths"]))
    print("Missing classes      :", result.summary["missing_classes_by_split"])
    print()
    print("Class coverage:")
    coverage = result.summary["class_coverage"]
    expected = result.summary["expected_classes"]
    print(f"{'class':<18} {'train':>8} {'val':>8} {'test':>8}")
    for label in expected:
        print(
            f"{label:<18} "
            f"{coverage['train'].get(label, 0):>8} "
            f"{coverage['val'].get(label, 0):>8} "
            f"{coverage['test'].get(label, 0):>8}"
        )
    print()
    print(f"Audit report         : {report_path}")
    print("=" * 64)

    if result.passed:
        print("PHASE 2: PASS")
    else:
        print("PHASE 2: FAIL")
        for err in result.errors:
            print(f"  ERROR: {err}")
        for warning in result.warnings[:20]:
            print(f"  WARNING: {warning}")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
