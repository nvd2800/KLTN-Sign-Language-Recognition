from pathlib import Path

from scripts.manage_rejected_samples import (
    select_reject_rows,
    sample_destination_path,
)


def test_select_only_reject_and_not_train_eligible():
    rows = [
        {
            "sample_id": "A",
            "status": "ACCEPT",
            "train_eligible": "1",
        },
        {
            "sample_id": "B",
            "status": "REVIEW",
            "train_eligible": "0",
        },
        {
            "sample_id": "C",
            "status": "REJECT",
            "train_eligible": "0",
        },
    ]

    result = select_reject_rows(rows, set())

    assert [r["sample_id"] for r in result] == ["C"]


def test_filter_by_sample_id():
    rows = [
        {
            "sample_id": "C1",
            "status": "REJECT",
            "train_eligible": "0",
        },
        {
            "sample_id": "C2",
            "status": "REJECT",
            "train_eligible": "0",
        },
    ]

    result = select_reject_rows(
        rows,
        {"C2"},
    )

    assert [r["sample_id"] for r in result] == ["C2"]


def test_destination_preserves_semantic_structure(tmp_path: Path):
    row = {
        "sample_id": "xin_chao_S01_SES01_001",
        "sign": "xin_chao",
        "subject_id": "S01",
        "session_id": "SES01",
    }

    dst = sample_destination_path(
        row,
        tmp_path / "rejected",
    )

    assert dst.name == "xin_chao_S01_SES01_001"
    assert dst.parent.name == "SES01"
    assert dst.parent.parent.name == "S01"
    assert dst.parent.parent.parent.name == "xin_chao"


def test_reject_with_train_eligible_true_is_blocked():
    rows = [
        {
            "sample_id": "BAD",
            "status": "REJECT",
            "train_eligible": "1",
        }
    ]

    try:
        select_reject_rows(rows, set())
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
