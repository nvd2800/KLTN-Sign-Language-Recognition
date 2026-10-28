from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 - Audit QC artifacts and train eligibility."
    )
    p.add_argument("--project-config", default=None)
    p.add_argument(
        "--manifest",
        default=str(ROOT / "dataset" / "processed" / "quality" / "quality_manifest.csv"),
    )
    p.add_argument(
        "--require-no-review",
        action="store_true",
        help="Final Phase-3 exit gate: fail if any REVIEW remains.",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.project_config)

    split_csv = resolve_project_path(
        cfg["dataset"].get("splits_root", "dataset/splits")
    ) / "split.csv"

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest

    if not split_csv.is_file():
        raise SystemExit(f"Missing Phase-2 split.csv: {split_csv}")
    if not manifest.is_file():
        raise SystemExit(f"Missing Phase-3 manifest: {manifest}")

    split_rows = read_csv(split_csv)
    qc_rows = read_csv(manifest)

    errors: list[str] = []
    warnings: list[str] = []

    split_ids = [r.get("sample_id", "").strip() for r in split_rows]
    qc_ids = [r.get("sample_id", "").strip() for r in qc_rows]

    if len(split_ids) != len(set(split_ids)):
        errors.append("Duplicate sample IDs in Phase-2 split.csv")
    if len(qc_ids) != len(set(qc_ids)):
        errors.append("Duplicate sample IDs in quality_manifest.csv")

    missing_qc = sorted(set(split_ids) - set(qc_ids))
    extra_qc = sorted(set(qc_ids) - set(split_ids))

    if missing_qc:
        errors.append(f"Samples missing QC results: {missing_qc}")
    if extra_qc:
        errors.append(f"QC contains samples outside Phase-2 split: {extra_qc}")

    split_by_id = {
        r["sample_id"].strip(): r for r in split_rows
        if r.get("sample_id")
    }

    status_counts = Counter()
    split_status = defaultdict(Counter)
    class_status = defaultdict(Counter)
    subject_status = defaultdict(Counter)

    for row in qc_rows:
        sample_id = row.get("sample_id", "").strip()
        status = row.get("status", "").strip()
        eligible = as_bool(row.get("train_eligible", "0"))

        if status not in {"ACCEPT", "REVIEW", "REJECT"}:
            errors.append(f"{sample_id}: invalid status '{status}'")

        if status == "REJECT" and eligible:
            errors.append(f"{sample_id}: REJECT must not be train_eligible")
        if status == "REVIEW" and eligible:
            errors.append(
                f"{sample_id}: REVIEW is train_eligible before manual resolution"
            )

        original = split_by_id.get(sample_id)
        if original is not None:
            for field in ("sign", "subject_id", "session_id", "split"):
                if row.get(field, "").strip() != original.get(field, "").strip():
                    errors.append(
                        f"{sample_id}: QC {field} differs from Phase-2 split"
                    )

        qjson = row.get("quality_json", "").strip()
        qmask = row.get("quality_masks", "").strip()

        for label, pstr in (("quality_json", qjson), ("quality_masks", qmask)):
            if not pstr:
                errors.append(f"{sample_id}: missing {label} path")
                continue
            p = Path(pstr)
            if not p.is_absolute():
                p = ROOT / p
            if not p.is_file():
                errors.append(f"{sample_id}: {label} file not found: {p}")

        status_counts[status] += 1
        split_status[row.get("split", "")][status] += 1
        class_status[row.get("sign", "")][status] += 1
        subject_status[row.get("subject_id", "")][status] += 1

    if args.require_no_review and status_counts.get("REVIEW", 0) > 0:
        errors.append(
            f"Unresolved REVIEW samples: {status_counts['REVIEW']}"
        )

    report = {
        "phase": 3,
        "status": "PASS" if not errors else "FAIL",
        "total_phase2_samples": len(split_rows),
        "total_qc_samples": len(qc_rows),
        "status_counts": dict(status_counts),
        "missing_qc_samples": missing_qc,
        "extra_qc_samples": extra_qc,
        "by_split": {k: dict(v) for k, v in sorted(split_status.items())},
        "by_class": {k: dict(v) for k, v in sorted(class_status.items())},
        "by_subject": {k: dict(v) for k, v in sorted(subject_status.items())},
        "require_no_review": args.require_no_review,
        "errors": errors,
        "warnings": warnings,
    }

    report_path = (
        ROOT / "dataset" / "processed" / "quality" / "quality_audit.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 3 QUALITY AUDIT")
    print("=" * 68)
    print(f"Phase-2 samples : {len(split_rows)}")
    print(f"QC samples      : {len(qc_rows)}")
    print(f"ACCEPT          : {status_counts.get('ACCEPT', 0)}")
    print(f"REVIEW          : {status_counts.get('REVIEW', 0)}")
    print(f"REJECT          : {status_counts.get('REJECT', 0)}")
    print(f"Missing QC      : {len(missing_qc)}")
    print(f"Extra QC        : {len(extra_qc)}")
    print(f"Require review=0: {args.require_no_review}")
    print(f"Report          : {report_path}")
    print("=" * 68)

    if errors:
        print("PHASE 3: FAIL")
        for error in errors:
            print(f"  ERROR: {error}")
        raise SystemExit(3)

    print("PHASE 3 PIPELINE: PASS")
    if status_counts.get("REVIEW", 0):
        print(
            "NOTE: REVIEW samples are excluded from training. "
            "Resolve them before using --require-no-review."
        )
    else:
        print("FINAL EXIT GATE: no unresolved REVIEW samples.")


if __name__ == "__main__":
    main()
