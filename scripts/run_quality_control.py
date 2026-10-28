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
from core.quality_phase3 import (
    inspect_sample,
    load_quality_config,
    save_quality_sample,
)


MANIFEST_FIELDS = [
    "sample_id",
    "sign",
    "subject_id",
    "session_id",
    "split",
    "frame_count",
    "left_hand_ratio",
    "right_hand_ratio",
    "any_hand_ratio",
    "both_hand_ratio",
    "pose_ratio",
    "face_ratio",
    "crop_ratio",
    "jump_ratio",
    "motion_mean",
    "blur_checked",
    "blur_ratio",
    "blur_variance_mean",
    "status",
    "train_eligible",
    "hard_fail_reasons",
    "review_reasons",
    "warnings",
    "quality_json",
    "quality_masks",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 - Run Dataset Quality Control."
    )
    p.add_argument(
        "--quality-config",
        default=str(ROOT / "configs" / "quality_phase3.yaml"),
    )
    p.add_argument("--project-config", default=None)
    p.add_argument(
        "--split",
        default=None,
        help="Default: dataset/splits/split.csv",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug only: process first N samples.",
    )
    return p.parse_args()


def _resolve_from_root(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _join_reasons(values: list[str]) -> str:
    return " | ".join(values)


def main() -> None:
    args = parse_args()

    project_cfg = load_config(args.project_config)
    qcfg = load_quality_config(Path(args.quality_config))

    quality_yaml = Path(args.quality_config)
    import yaml
    qdata = yaml.safe_load(quality_yaml.read_text(encoding="utf-8")) or {}
    output_cfg = qdata.get("output", {}) or {}

    split_csv = (
        Path(args.split).resolve()
        if args.split
        else resolve_project_path(
            project_cfg["dataset"].get("splits_root", "dataset/splits")
        ) / "split.csv"
    )

    if not split_csv.is_file():
        raise SystemExit(
            f"Phase 2 split.csv not found: {split_csv}\n"
            "Run Phase 2 first."
        )

    output_root = _resolve_from_root(
        output_cfg.get("root", "dataset/processed/quality")
    )
    manifest_csv = _resolve_from_root(
        output_cfg.get(
            "manifest_csv",
            "dataset/processed/quality/quality_manifest.csv",
        )
    )
    summary_json = _resolve_from_root(
        output_cfg.get(
            "summary_json",
            "dataset/processed/quality/quality_summary.json",
        )
    )

    rows = _read_split_csv(split_csv)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    print("\nPHASE 3 - DATASET QUALITY CONTROL")
    print("=" * 72)
    print(f"split.csv      : {split_csv}")
    print(f"Samples        : {len(rows)}")
    print(f"Quality config : {Path(args.quality_config).resolve()}")
    print(f"Output         : {output_root}")
    print("=" * 72)

    manifest_rows: list[dict[str, object]] = []
    status_counter: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_subject: dict[str, Counter[str]] = defaultdict(Counter)

    for idx, row in enumerate(rows, start=1):
        sample_id = (row.get("sample_id") or "").strip()
        sign = (row.get("sign") or "").strip()
        subject_id = (row.get("subject_id") or "").strip()
        session_id = (row.get("session_id") or "").strip()
        split = (row.get("split") or "").strip()

        landmarks_path = _resolve_from_root(row["landmarks_path"])

        video_value = (row.get("video_path") or "").strip()
        if video_value:
            video_path = _resolve_from_root(video_value)
        else:
            sample_path = _resolve_from_root(row["sample_path"])
            video_filename = project_cfg["dataset"].get(
                "video_filename", "source_video.mp4"
            )
            video_path = sample_path / video_filename

        try:
            metadata_frame_count = int(row.get("frame_count") or 0)
        except ValueError:
            metadata_frame_count = None

        result, masks = inspect_sample(
            sample_id=sample_id,
            sign=sign,
            subject_id=subject_id,
            session_id=session_id,
            split=split,
            landmarks_path=landmarks_path,
            video_path=video_path,
            metadata_frame_count=metadata_frame_count,
            cfg=qcfg,
        )

        quality_json, quality_masks = save_quality_sample(
            output_root, result, masks
        )

        status_counter[result.status] += 1
        by_split[split][result.status] += 1
        by_class[sign][result.status] += 1
        by_subject[subject_id][result.status] += 1

        manifest_rows.append(
            {
                "sample_id": result.sample_id,
                "sign": result.sign,
                "subject_id": result.subject_id,
                "session_id": result.session_id,
                "split": result.split,
                "frame_count": result.frame_count,
                "left_hand_ratio": f"{result.left_hand_ratio:.6f}",
                "right_hand_ratio": f"{result.right_hand_ratio:.6f}",
                "any_hand_ratio": f"{result.any_hand_ratio:.6f}",
                "both_hand_ratio": f"{result.both_hand_ratio:.6f}",
                "pose_ratio": f"{result.pose_ratio:.6f}",
                "face_ratio": f"{result.face_ratio:.6f}",
                "crop_ratio": f"{result.crop_ratio:.6f}",
                "jump_ratio": f"{result.jump_ratio:.6f}",
                "motion_mean": f"{result.motion_mean:.8f}",
                "blur_checked": int(result.blur_checked),
                "blur_ratio": (
                    "" if result.blur_ratio is None
                    else f"{result.blur_ratio:.6f}"
                ),
                "blur_variance_mean": (
                    "" if result.blur_variance_mean is None
                    else f"{result.blur_variance_mean:.6f}"
                ),
                "status": result.status,
                "train_eligible": int(result.train_eligible),
                "hard_fail_reasons": _join_reasons(result.hard_fail_reasons),
                "review_reasons": _join_reasons(result.review_reasons),
                "warnings": _join_reasons(result.warnings),
                "quality_json": quality_json.relative_to(ROOT).as_posix(),
                "quality_masks": quality_masks.relative_to(ROOT).as_posix(),
            }
        )

        print(
            f"[{idx:>4}/{len(rows)}] "
            f"{sample_id:<35} "
            f"{result.status:<6} "
            f"hand={result.any_hand_ratio:.2f} "
            f"crop={result.crop_ratio:.2f} "
            f"jump={result.jump_ratio:.2f}"
        )

    _write_manifest(manifest_csv, manifest_rows)

    summary = {
        "phase": 3,
        "quality_version": qdata.get("quality_version", "v1"),
        "total_samples": len(manifest_rows),
        "status_counts": dict(status_counter),
        "train_eligible_count": sum(
            int(r["train_eligible"]) for r in manifest_rows
        ),
        "by_split": {
            key: dict(value) for key, value in sorted(by_split.items())
        },
        "by_class": {
            key: dict(value) for key, value in sorted(by_class.items())
        },
        "by_subject": {
            key: dict(value) for key, value in sorted(by_subject.items())
        },
        "policy": {
            "only_accept_is_train_eligible_by_default": True,
            "review_requires_manual_resolution": True,
            "raw_dataset_modified": False,
            "phase2_split_modified": False,
        },
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nQUALITY CONTROL FINISHED")
    print("=" * 72)
    print(f"ACCEPT : {status_counter.get('ACCEPT', 0)}")
    print(f"REVIEW : {status_counter.get('REVIEW', 0)}")
    print(f"REJECT : {status_counter.get('REJECT', 0)}")
    print(
        "TRAIN ELIGIBLE : "
        f"{sum(int(r['train_eligible']) for r in manifest_rows)}"
    )
    print(f"Manifest : {manifest_csv}")
    print(f"Summary  : {summary_json}")
    print("=" * 72)
    print(
        "Next: python scripts/audit_quality.py\n"
        "Use --require-no-review only after manually resolving REVIEW samples."
    )


if __name__ == "__main__":
    main()
