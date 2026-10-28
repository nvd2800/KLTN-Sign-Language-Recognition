from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "dataset" / "processed" / "quality" / "quality_manifest.csv"
DEFAULT_REJECTED_ROOT = ROOT / "dataset" / "rejected"
DEFAULT_INDEX = DEFAULT_REJECTED_ROOT / "rejected_index.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Manage Phase-3 REJECT samples safely: list, quarantine(move), "
            "and restore."
        )
    )

    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List current REJECT samples.")
    action.add_argument(
        "--move",
        action="store_true",
        help=(
            "Move REJECT sample folders from dataset/raw to dataset/rejected. "
            "Requires --yes for actual changes; otherwise dry-run."
        ),
    )
    action.add_argument(
        "--restore",
        action="store_true",
        help="Restore quarantined sample folders to their original raw paths.",
    )

    p.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to Phase-3 quality_manifest.csv",
    )
    p.add_argument(
        "--rejected-root",
        default=str(DEFAULT_REJECTED_ROOT),
        help="Quarantine root directory.",
    )
    p.add_argument(
        "--index",
        default=str(DEFAULT_INDEX),
        help="JSON index used to track moved samples.",
    )
    p.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help=(
            "Operate only on a specific sample_id. Repeat this option for "
            "multiple samples. If omitted, operate on all matching samples."
        ),
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform move/restore. Without this flag, only dry-run.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "For restore only: replace an existing empty destination directory. "
            "Never overwrites a non-empty sample directory."
        ),
    )

    return p.parse_args()


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"quality_manifest.csv not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_index(path: Path) -> dict:
    if not path.is_file():
        return {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "entries": [],
        }

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError(f"Invalid rejected index: {path}")

    return data


def save_index(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def select_reject_rows(
    rows: list[dict[str, str]],
    sample_ids: set[str],
) -> list[dict[str, str]]:
    selected = []

    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        status = (row.get("status") or "").strip().upper()
        train_eligible = is_true(row.get("train_eligible", "0"))

        if status != "REJECT":
            continue

        # Safety: only quarantine samples already marked not train-eligible.
        if train_eligible:
            raise ValueError(
                f"{sample_id}: status=REJECT but train_eligible=true. "
                "Fix QC manifest before moving files."
            )

        if sample_ids and sample_id not in sample_ids:
            continue

        selected.append(row)

    return selected


def sample_source_path(row: dict[str, str]) -> Path:
    value = (row.get("sample_path") or "").strip()

    if not value:
        raise ValueError(
            f"{row.get('sample_id')}: sample_path missing from quality manifest."
        )

    return resolve_path(value).resolve()


def sample_destination_path(
    row: dict[str, str],
    rejected_root: Path,
) -> Path:
    sign = (row.get("sign") or "UNKNOWN_SIGN").strip()
    subject = (row.get("subject_id") or "UNKNOWN_SUBJECT").strip()
    session = (row.get("session_id") or "UNKNOWN_SESSION").strip()
    sample_id = (row.get("sample_id") or "UNKNOWN_SAMPLE").strip()

    return (
        rejected_root
        / sign
        / subject
        / session
        / sample_id
    ).resolve()


def print_rows(rows: list[dict[str, str]]) -> None:
    print("\nPHASE 3 - REJECT SAMPLES")
    print("=" * 120)

    if not rows:
        print("No REJECT samples found.")
        return

    print(
        f"{'#':>3}  {'sample_id':<36} {'sign':<14} "
        f"{'subject':<8} {'split':<7} {'hand':>7}  reason"
    )
    print("-" * 120)

    for i, row in enumerate(rows, 1):
        print(
            f"{i:>3}  "
            f"{(row.get('sample_id') or ''):<36} "
            f"{(row.get('sign') or ''):<14} "
            f"{(row.get('subject_id') or ''):<8} "
            f"{(row.get('split') or ''):<7} "
            f"{(row.get('any_hand_ratio') or ''):>7}  "
            f"{(row.get('hard_fail_reasons') or '')}"
        )

    print("-" * 120)
    print(f"Total REJECT: {len(rows)}")


def find_index_entry(entries: list[dict], sample_id: str) -> dict | None:
    for entry in entries:
        if entry.get("sample_id") == sample_id:
            return entry
    return None


def move_rejects(
    rows: list[dict[str, str]],
    rejected_root: Path,
    index_path: Path,
    actually_do_it: bool,
) -> None:
    index = load_index(index_path)
    entries = index["entries"]

    print("\nQUARANTINE REJECT SAMPLES")
    print("=" * 80)
    print("Mode:", "MOVE" if actually_do_it else "DRY-RUN")
    print(f"Rejected root: {rejected_root}")
    print()

    moved = 0
    skipped = 0
    errors = 0

    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()

        try:
            src = sample_source_path(row)
            dst = sample_destination_path(row, rejected_root)

            existing_entry = find_index_entry(entries, sample_id)

            if existing_entry and existing_entry.get("state") == "quarantined":
                print(f"SKIP  {sample_id}: already quarantined")
                skipped += 1
                continue

            if not src.is_dir():
                print(f"ERROR {sample_id}: source folder not found: {src}")
                errors += 1
                continue

            # Safety: must come from project dataset/raw.
            raw_root = (ROOT / "dataset" / "raw").resolve()
            try:
                src.relative_to(raw_root)
            except ValueError:
                print(
                    f"ERROR {sample_id}: source is outside dataset/raw: {src}"
                )
                errors += 1
                continue

            if dst.exists():
                print(f"ERROR {sample_id}: destination already exists: {dst}")
                errors += 1
                continue

            print(f"{'MOVE' if actually_do_it else 'WOULD MOVE'} {sample_id}")
            print(f"  FROM: {src}")
            print(f"  TO  : {dst}")

            if actually_do_it:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

                now = datetime.now(timezone.utc).isoformat()

                entry = existing_entry or {}
                entry.update(
                    {
                        "sample_id": sample_id,
                        "sign": (row.get("sign") or "").strip(),
                        "subject_id": (row.get("subject_id") or "").strip(),
                        "session_id": (row.get("session_id") or "").strip(),
                        "split": (row.get("split") or "").strip(),
                        "status": "REJECT",
                        "hard_fail_reasons": (
                            row.get("hard_fail_reasons") or ""
                        ).strip(),
                        "original_path": str(src),
                        "quarantine_path": str(dst),
                        "moved_at_utc": now,
                        "state": "quarantined",
                    }
                )

                if existing_entry is None:
                    entries.append(entry)

                moved += 1

        except Exception as exc:
            print(f"ERROR {sample_id}: {type(exc).__name__}: {exc}")
            errors += 1

    if actually_do_it:
        save_index(index_path, index)

    print()
    print("=" * 80)
    print(f"Moved   : {moved}")
    print(f"Skipped : {skipped}")
    print(f"Errors  : {errors}")

    if not actually_do_it:
        print("\nDRY-RUN ONLY. Add --yes to actually move files.")
    else:
        print(f"Index saved: {index_path}")
        print(
            "\nIMPORTANT: moving raw sample folders means Phase-2 raw-dataset "
            "audit will no longer see those samples. Keep rejected_index.json "
            "so they can be restored before reproducing the original Phase-2 audit."
        )


def restore_samples(
    index_path: Path,
    sample_ids: set[str],
    actually_do_it: bool,
    force: bool,
) -> None:
    index = load_index(index_path)
    entries = index["entries"]

    candidates = [
        e
        for e in entries
        if e.get("state") == "quarantined"
        and (
            not sample_ids
            or e.get("sample_id") in sample_ids
        )
    ]

    print("\nRESTORE REJECTED SAMPLES")
    print("=" * 80)
    print("Mode:", "RESTORE" if actually_do_it else "DRY-RUN")

    restored = 0
    skipped = 0
    errors = 0

    for entry in candidates:
        sample_id = entry.get("sample_id", "")
        src = Path(entry["quarantine_path"]).resolve()
        dst = Path(entry["original_path"]).resolve()

        if not src.is_dir():
            print(f"ERROR {sample_id}: quarantine folder missing: {src}")
            errors += 1
            continue

        if dst.exists():
            if force and dst.is_dir() and not any(dst.iterdir()):
                if actually_do_it:
                    dst.rmdir()
            else:
                print(
                    f"SKIP  {sample_id}: original path already exists: {dst}"
                )
                skipped += 1
                continue

        print(f"{'RESTORE' if actually_do_it else 'WOULD RESTORE'} {sample_id}")
        print(f"  FROM: {src}")
        print(f"  TO  : {dst}")

        if actually_do_it:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            entry["state"] = "restored"
            entry["restored_at_utc"] = datetime.now(timezone.utc).isoformat()
            restored += 1

    if actually_do_it:
        save_index(index_path, index)

    print()
    print("=" * 80)
    print(f"Restored: {restored}")
    print(f"Skipped : {skipped}")
    print(f"Errors  : {errors}")

    if not actually_do_it:
        print("\nDRY-RUN ONLY. Add --yes to actually restore files.")


def main() -> None:
    args = parse_args()

    manifest = resolve_path(args.manifest).resolve()
    rejected_root = resolve_path(args.rejected_root).resolve()
    index_path = resolve_path(args.index).resolve()
    sample_ids = {x.strip() for x in args.sample_id if x.strip()}

    if args.list:
        rows = load_manifest(manifest)
        selected = select_reject_rows(rows, sample_ids)
        print_rows(selected)
        return

    if args.move:
        rows = load_manifest(manifest)
        selected = select_reject_rows(rows, sample_ids)

        print_rows(selected)

        if not selected:
            return

        move_rejects(
            selected,
            rejected_root,
            index_path,
            actually_do_it=args.yes,
        )
        return

    if args.restore:
        restore_samples(
            index_path,
            sample_ids,
            actually_do_it=args.yes,
            force=args.force,
        )
        return


if __name__ == "__main__":
    main()
