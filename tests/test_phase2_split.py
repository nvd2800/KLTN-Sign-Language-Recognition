from __future__ import annotations

import json
from pathlib import Path

from core.dataset_index import scan_raw_dataset
from core.split_utils import (
    SplitPlan,
    assign_subject,
    audit_assignments,
    build_assignments,
)


def make_plan() -> SplitPlan:
    return SplitPlan(
        phase=2,
        split_version="v1",
        strategy="signer_independent",
        seed=42,
        train_subjects=frozenset({"S01", "S02"}),
        val_subjects=frozenset({"S03"}),
        test_subjects=frozenset({"S04", "S05"}),
        fail_on_unassigned_subject=True,
        require_all_classes=True,
        expected_classes=("xin_chao", "cam_on"),
    )


def add_sample(raw: Path, sign: str, subject: str, session: str, idx: int) -> None:
    sample_id = f"{sign}_{subject}_{session}_{idx:03d}"
    d = raw / sign / subject / session / sample_id
    d.mkdir(parents=True, exist_ok=True)

    (d / "landmarks.npz").write_bytes(b"npz-placeholder")
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "sign": sign,
                "subject_id": subject,
                "session_id": session,
                "frame_count": 45,
                "fps": 30.0,
                "duration_s": 1.466,
                "collection_mode": "fixed",
                "dataset_version": "v2.0-mini",
                "schema_version": "1.1",
            }
        ),
        encoding="utf-8",
    )


def build_complete_dataset(raw: Path) -> None:
    for subject in ("S01", "S02", "S03", "S04", "S05"):
        for sign in ("xin_chao", "cam_on"):
            add_sample(raw, sign, subject, "SES01", 1)


def test_subject_assignment_is_signer_independent():
    plan = make_plan()
    assert assign_subject("S01", plan) == "train"
    assert assign_subject("S03", plan) == "val"
    assert assign_subject("S04", plan) == "test"
    assert assign_subject("S99", plan) is None


def test_complete_split_passes(tmp_path: Path):
    raw = tmp_path / "dataset" / "raw"
    build_complete_dataset(raw)
    scan = scan_raw_dataset(raw)
    assert not scan.errors

    plan = make_plan()
    assignments, unassigned = build_assignments(scan.records, plan)
    result = audit_assignments(
        scan.records,
        assignments,
        unassigned,
        plan,
        scan_errors=scan.errors,
        scan_warnings=scan.warnings,
    )

    assert result.passed
    assert result.summary["signer_leakage"]["pass"] is True
    assert result.summary["unassigned_samples"] == 0


def test_unassigned_subject_fails(tmp_path: Path):
    raw = tmp_path / "dataset" / "raw"
    build_complete_dataset(raw)
    add_sample(raw, "xin_chao", "S99", "SES01", 1)

    scan = scan_raw_dataset(raw)
    plan = make_plan()
    assignments, unassigned = build_assignments(scan.records, plan)
    result = audit_assignments(scan.records, assignments, unassigned, plan)

    assert not result.passed
    assert "S99" in result.summary["unassigned_subjects"]


def test_missing_class_in_test_fails(tmp_path: Path):
    raw = tmp_path / "dataset" / "raw"

    for subject in ("S01", "S02", "S03"):
        for sign in ("xin_chao", "cam_on"):
            add_sample(raw, sign, subject, "SES01", 1)

    for subject in ("S04", "S05"):
        add_sample(raw, "xin_chao", subject, "SES01", 1)

    scan = scan_raw_dataset(raw)
    plan = make_plan()
    assignments, unassigned = build_assignments(scan.records, plan)
    result = audit_assignments(scan.records, assignments, unassigned, plan)

    assert not result.passed
    assert "test" in result.summary["missing_classes_by_split"]
    assert "cam_on" in result.summary["missing_classes_by_split"]["test"]


def test_all_sessions_of_subject_follow_same_split(tmp_path: Path):
    raw = tmp_path / "dataset" / "raw"

    for session in ("SES01", "SES02"):
        for sign in ("xin_chao", "cam_on"):
            add_sample(raw, sign, "S01", session, 1)

    for subject in ("S02", "S03", "S04", "S05"):
        for sign in ("xin_chao", "cam_on"):
            add_sample(raw, sign, subject, "SES01", 1)

    scan = scan_raw_dataset(raw)
    plan = make_plan()
    assignments, unassigned = build_assignments(scan.records, plan)

    s01_splits = {
        split
        for record, split in assignments
        if record.subject_id == "S01"
    }

    assert s01_splits == {"train"}
