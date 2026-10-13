from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import yaml

from .dataset_index import SampleRecord


VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitPlan:
    phase: int
    split_version: str
    strategy: str
    seed: int
    train_subjects: frozenset[str]
    val_subjects: frozenset[str]
    test_subjects: frozenset[str]
    fail_on_unassigned_subject: bool
    require_all_classes: bool
    expected_classes: tuple[str, ...]

    @property
    def all_subjects(self) -> frozenset[str]:
        return self.train_subjects | self.val_subjects | self.test_subjects


@dataclass
class AuditResult:
    status: str
    errors: list[str]
    warnings: list[str]
    summary: dict[str, object]

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            **self.summary,
        }


def load_split_plan(path: Path) -> SplitPlan:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split config not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def subject_set(key: str) -> frozenset[str]:
        values = data.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a YAML list")
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"{key} contains duplicate subject IDs")
        return frozenset(cleaned)

    train = subject_set("train_subjects")
    val = subject_set("val_subjects")
    test = subject_set("test_subjects")

    if not train or not val or not test:
        raise ValueError("train_subjects, val_subjects and test_subjects must be non-empty")

    overlap_tv = train & val
    overlap_tt = train & test
    overlap_vt = val & test
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            "Subject lists overlap: "
            f"train/val={sorted(overlap_tv)}, "
            f"train/test={sorted(overlap_tt)}, "
            f"val/test={sorted(overlap_vt)}"
        )

    expected_classes_raw = data.get("expected_classes", [])
    if not isinstance(expected_classes_raw, list):
        raise ValueError("expected_classes must be a YAML list")

    expected_classes = tuple(
        dict.fromkeys(str(x).strip() for x in expected_classes_raw if str(x).strip())
    )

    return SplitPlan(
        phase=int(data.get("phase", 2)),
        split_version=str(data.get("split_version", "v1")),
        strategy=str(data.get("strategy", "signer_independent")),
        seed=int(data.get("seed", 42)),
        train_subjects=train,
        val_subjects=val,
        test_subjects=test,
        fail_on_unassigned_subject=bool(data.get("fail_on_unassigned_subject", True)),
        require_all_classes=bool(data.get("require_all_classes", True)),
        expected_classes=expected_classes,
    )


def assign_subject(subject_id: str, plan: SplitPlan) -> str | None:
    if subject_id in plan.train_subjects:
        return "train"
    if subject_id in plan.val_subjects:
        return "val"
    if subject_id in plan.test_subjects:
        return "test"
    return None


def build_assignments(
    records: Iterable[SampleRecord],
    plan: SplitPlan,
) -> tuple[list[tuple[SampleRecord, str]], list[SampleRecord]]:
    assigned: list[tuple[SampleRecord, str]] = []
    unassigned: list[SampleRecord] = []

    for record in records:
        split = assign_subject(record.subject_id, plan)
        if split is None:
            unassigned.append(record)
        else:
            assigned.append((record, split))

    return assigned, unassigned


SPLIT_CSV_FIELDS = [
    "sample_id",
    "sign",
    "subject_id",
    "session_id",
    "split",
    "frame_count",
    "fps",
    "duration_s",
    "collection_mode",
    "dataset_version",
    "schema_version",
    "sample_path",
    "metadata_path",
    "landmarks_path",
    "video_path",
    "has_video",
]


def write_split_csv(
    path: Path,
    assignments: Iterable[tuple[SampleRecord, str]],
    project_root: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        record.to_row(project_root=project_root, split=split)
        for record, split in assignments
    ]
    rows.sort(
        key=lambda r: (
            r["split"],
            r["subject_id"],
            r["sign"],
            r["session_id"],
            r["sample_id"],
        )
    )

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_split_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"split.csv not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _coverage_table(
    assignments: Iterable[tuple[SampleRecord, str]],
    expected_classes: Iterable[str],
) -> dict[str, dict[str, int]]:
    classes = list(expected_classes)
    table = {
        split: {label: 0 for label in classes}
        for split in VALID_SPLITS
    }
    for record, split in assignments:
        if split not in table:
            continue
        if record.sign not in table[split]:
            table[split][record.sign] = 0
        table[split][record.sign] += 1
    return table


def make_summary(
    records: list[SampleRecord],
    assignments: list[tuple[SampleRecord, str]],
    unassigned: list[SampleRecord],
    plan: SplitPlan,
) -> dict[str, object]:
    split_counts = {s: 0 for s in VALID_SPLITS}
    split_subjects = {s: set() for s in VALID_SPLITS}

    for record, split in assignments:
        split_counts[split] += 1
        split_subjects[split].add(record.subject_id)

    expected = plan.expected_classes or tuple(sorted({r.sign for r in records}))
    coverage = _coverage_table(assignments, expected)

    return {
        "phase": plan.phase,
        "split_version": plan.split_version,
        "strategy": plan.strategy,
        "seed": plan.seed,
        "total_samples": len(records),
        "assigned_samples": len(assignments),
        "unassigned_samples": len(unassigned),
        "train_samples": split_counts["train"],
        "val_samples": split_counts["val"],
        "test_samples": split_counts["test"],
        "train_subjects": sorted(split_subjects["train"]),
        "val_subjects": sorted(split_subjects["val"]),
        "test_subjects": sorted(split_subjects["test"]),
        "expected_classes": list(expected),
        "class_coverage": coverage,
    }


def audit_assignments(
    records: list[SampleRecord],
    assignments: list[tuple[SampleRecord, str]],
    unassigned: list[SampleRecord],
    plan: SplitPlan,
    *,
    scan_errors: list[str] | None = None,
    scan_warnings: list[str] | None = None,
) -> AuditResult:
    errors = list(scan_errors or [])
    warnings = list(scan_warnings or [])

    train_val = sorted(plan.train_subjects & plan.val_subjects)
    train_test = sorted(plan.train_subjects & plan.test_subjects)
    val_test = sorted(plan.val_subjects & plan.test_subjects)

    if train_val:
        errors.append(f"Signer leakage train/val: {train_val}")
    if train_test:
        errors.append(f"Signer leakage train/test: {train_test}")
    if val_test:
        errors.append(f"Signer leakage val/test: {val_test}")

    subject_splits: dict[str, set[str]] = {}
    session_splits: dict[tuple[str, str], set[str]] = {}
    sample_id_counts: dict[str, int] = {}
    sample_path_counts: dict[str, int] = {}

    for record, split in assignments:
        if split not in VALID_SPLITS:
            errors.append(f"{record.sample_id}: invalid split '{split}'")
            continue

        subject_splits.setdefault(record.subject_id, set()).add(split)
        session_splits.setdefault((record.subject_id, record.session_id), set()).add(split)
        sample_id_counts[record.sample_id] = sample_id_counts.get(record.sample_id, 0) + 1
        path_key = str(record.sample_path.resolve())
        sample_path_counts[path_key] = sample_path_counts.get(path_key, 0) + 1

    for subject, splits in sorted(subject_splits.items()):
        if len(splits) != 1:
            errors.append(f"Subject {subject} appears in multiple splits: {sorted(splits)}")

    for (subject, session), splits in sorted(session_splits.items()):
        if len(splits) != 1:
            errors.append(f"Session leakage {subject}/{session}: {sorted(splits)}")

    duplicate_ids = sorted(k for k, v in sample_id_counts.items() if v > 1)
    duplicate_paths = sorted(k for k, v in sample_path_counts.items() if v > 1)

    if duplicate_ids:
        errors.append(f"Duplicate sample_id values: {duplicate_ids}")
    if duplicate_paths:
        errors.append(f"Duplicate sample paths: {duplicate_paths}")

    unassigned_subjects = sorted({r.subject_id for r in unassigned})
    if unassigned:
        message = (
            f"Unassigned samples={len(unassigned)}, "
            f"subjects={unassigned_subjects}"
        )
        if plan.fail_on_unassigned_subject:
            errors.append(message)
        else:
            warnings.append(message)

    present_subjects = {r.subject_id for r in records}
    missing_configured_subjects = sorted(plan.all_subjects - present_subjects)
    if missing_configured_subjects:
        errors.append(
            f"Configured subjects missing from dataset: {missing_configured_subjects}"
        )

    expected = plan.expected_classes or tuple(sorted({r.sign for r in records}))
    coverage = _coverage_table(assignments, expected)
    missing_classes: dict[str, list[str]] = {}

    for split in VALID_SPLITS:
        missing = [label for label in expected if coverage[split].get(label, 0) == 0]
        if missing:
            missing_classes[split] = missing

    if missing_classes:
        message = f"Missing classes by split: {missing_classes}"
        if plan.require_all_classes:
            errors.append(message)
        else:
            warnings.append(message)

    assigned_subjects = {r.subject_id for r, _ in assignments}
    unexpected_assigned = sorted(assigned_subjects - plan.all_subjects)
    if unexpected_assigned:
        errors.append(f"Unexpected assigned subjects: {unexpected_assigned}")

    summary = make_summary(records, assignments, unassigned, plan)
    summary.update(
        {
            "signer_leakage": {
                "train_val": train_val,
                "train_test": train_test,
                "val_test": val_test,
                "pass": not (train_val or train_test or val_test),
            },
            "duplicate_sample_ids": duplicate_ids,
            "duplicate_paths": duplicate_paths,
            "unassigned_subjects": unassigned_subjects,
            "missing_configured_subjects": missing_configured_subjects,
            "missing_classes_by_split": missing_classes,
        }
    )

    status = "PASS" if not errors else "FAIL"
    return AuditResult(status=status, errors=errors, warnings=warnings, summary=summary)


def write_json(path: Path, data: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
