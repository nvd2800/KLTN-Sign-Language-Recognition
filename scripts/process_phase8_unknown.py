"""Phase 08: process category-structured UNKNOWN raw samples with frozen Phase 04-06 code.

Does not modify raw UNKNOWN, Phase 02 splits, known features, or TCN weights.
Run from project root: python scripts/process_phase8_unknown.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_phase4 import (
    build_phase4_features, inspect_feature_arrays, load_phase4_config,
)
from core.feature_phase5 import (
    build_phase5_features, inspect_phase5_arrays, load_phase5_config,
)
from core.feature_phase6 import (
    build_phase6_features, inspect_phase6_arrays, load_phase6_config,
)


def at_root(raw_path: str, project_root: Path = ROOT) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else project_root / p


def csv_subjects(value: str) -> set[str]:
    return {x.strip().upper() for x in value.split(",") if x.strip()}


def canonical_sample_id(category: str, subject: str, session: str, raw_name: str) -> str:
    """Unique across categories even when old RAW filenames repeat *_001."""
    for label, value in (("category", category), ("subject", subject), ("session", session)):
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", value):
            raise ValueError(f"Invalid {label}: {value!r}")
    match = re.search(r"_(\d+)$", raw_name)
    suffix = match.group(1) if match else raw_name
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", suffix):
        raise ValueError(f"Invalid raw sample name: {raw_name!r}")
    return f"UNKNOWN_{category}_{subject}_{session}_{suffix}"


def optional_points(data, keys, T, minimum_points: int, field: str):
    for key in keys:
        if key in data.files:
            arr = np.asarray(data[key], dtype=np.float32)
            if arr.ndim != 3 or arr.shape[0] != T or arr.shape[1] < minimum_points or arr.shape[2] < 3:
                raise ValueError(f"{field} must have shape (T,N>= {minimum_points},XYZ), got {arr.shape}")
            return np.asarray(arr[:, :, :3], dtype=np.float32)
    return None


def mask(data, names, T: int, required: bool) -> np.ndarray:
    for name in names:
        if name in data.files:
            a = np.asarray(data[name]).reshape(-1)
            if a.size != T:
                raise ValueError(f"{name} mask length {a.size} differs from T={T}")
            return a.astype(bool)
    if required:
        raise KeyError("Missing required mask: " + "/".join(names))
    return np.zeros(T, dtype=bool)


def extract_one(raw_path: Path, cfg4, cfg5, cfg6):
    with np.load(raw_path, allow_pickle=False) as data:
        needed = ("timestamps", "left_hand", "right_hand")
        if any(n not in data.files for n in needed):
            raise KeyError("Missing required raw arrays: " + ",".join(n for n in needed if n not in data.files))
        timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
        T = timestamps.size
        if T < 2:
            raise ValueError(f"Need >= 2 frames, got T={T}")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
            raise ValueError("Invalid or decreasing timestamps")
        left = np.asarray(data["left_hand"], dtype=np.float32)
        right = np.asarray(data["right_hand"], dtype=np.float32)
        if left.shape != (T, 21, 3) or right.shape != (T, 21, 3):
            raise ValueError(f"Wrong hand shapes: {left.shape}, {right.shape}; expected (T,21,3)")
        lp = mask(data, ("left_present", "left_hand_present"), T, required=True)
        rp = mask(data, ("right_present", "right_hand_present"), T, required=True)
        pose = optional_points(data, ("pose", "pose_landmarks", "pose_xyz"), T, 17, "pose")
        face = optional_points(data, ("face", "face_landmarks", "face_xyz"), T, 387, "face")
        pp = mask(data, ("pose_present",), T, required=False)
        fp = mask(data, ("face_present",), T, required=False)

    # Phase 04, with the same normalization/mask policy as known sign processing.
    a4 = build_phase4_features(left, right, lp, rp, cfg4)
    a4["timestamps"] = timestamps
    a4["pose_present"] = pp
    a4["face_present"] = fp
    qc4 = inspect_feature_arrays(a4, cfg4)
    if not qc4["pass"]:
        raise ValueError("Phase04 audit failed: " + ";".join(qc4["issues"]))

    # Phase 05 uses raw wrist/pose to preserve body-relative motion.
    a5 = build_phase5_features(
        phase4_arrays=a4, raw_left=left, raw_right=right,
        pose_xyz=pose, pose_present=pp if pose is not None else None,
        cfg=cfg5,
    )
    qc5 = inspect_phase5_arrays(a5)
    if not qc5["pass"]:
        raise ValueError("Phase05 audit failed: " + ";".join(qc5["issues"]))

    # Phase 06; missing face/pose is zero+presence mask (not invented landmarks).
    a6 = build_phase6_features(
        phase5_full417=a5["full_features417"],
        pose_xyz=pose, pose_present=pp if pose is not None else None,
        face_xyz=face, face_present=fp if face is not None else None,
        cfg=cfg6,
    )
    qc6 = inspect_phase6_arrays(a6)
    if not qc6["pass"]:
        raise ValueError("Phase06 audit failed: " + ";".join(qc6["issues"]))
    features = a6["ablation_hands_pose_face453"]
    if features.shape != (T, 453) or not np.isfinite(features).all():
        raise ValueError(f"Invalid Phase06 feature array {features.shape}")

    coverage = {
        "any_hand_ratio": float(np.mean(lp | rp)),
        "pose_present_ratio": float(np.mean(pp)),
        "face_present_ratio": float(np.mean(fp)),
    }
    return a4, a5, a6, (qc4, qc5, qc6), coverage


def save_npz(path: Path, arrays: dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace):
    root = ROOT
    raw_root = at_root(args.raw_root, root)
    processed_root = at_root(args.processed_root, root)
    manifest_path = at_root(args.manifest, root)
    report_path = at_root(args.report, root)
    cfg4, _ = load_phase4_config(at_root(args.phase4_config, root))
    cfg5, _ = load_phase5_config(at_root(args.phase5_config, root))
    cfg6, _ = load_phase6_config(at_root(args.phase6_config, root))
    if not raw_root.is_dir():
        raise FileNotFoundError(f"UNKNOWN raw root not found: {raw_root}")

    val_subjects, test_subjects = csv_subjects(args.val_subjects), csv_subjects(args.test_subjects)
    if val_subjects.intersection(test_subjects):
        raise ValueError(f"VAL/TEST subject overlap: {sorted(val_subjects & test_subjects)}")
    # Phase08 accepts ANY SUBSET of UNKNOWN categories; no fixed nine-category requirement.
    # Quick-test mode is exploratory: all observed UNKNOWN become VAL, no independent TEST.
    quick_test = bool(args.quick_test)
    if quick_test and (val_subjects or test_subjects):
        raise ValueError("--quick-test cannot be combined with --val-subjects/--test-subjects")
    split_requested = bool(val_subjects or test_subjects)
    if split_requested and (not val_subjects or not test_subjects):
        raise ValueError("Both --val-subjects and --test-subjects must be supplied")

    audit_rows = []
    prepared_rows = []
    used_ids: set[str] = set()
    status_counts = Counter()
    detected_files = sorted(raw_root.rglob("landmarks.npz"))
    if not detected_files:
        raise RuntimeError(f"No landmarks.npz found beneath {raw_root}")

    for num, path in enumerate(detected_files, 1):
        rel = path.relative_to(raw_root)
        # Required: <category>/<subject>/<session>/<sample>/landmarks.npz
        if len(rel.parts) != 5:
            if len(rel.parts) == 4 and re.fullmatch(r"S\d+", rel.parts[0].upper()):
                # Legacy UNKNOWN/S01/SES01/sample has no category. Do not guess it.
                audit_rows.append({"source": str(path), "status": "SKIP_LEGACY", "reason": "legacy_raw_without_unknown_category", "sample_id": ""})
                status_counts["SKIP_LEGACY"] += 1
                print(f"[{num}/{len(detected_files)}] SKIP legacy category-less sample: {rel}")
            else:
                audit_rows.append({"source": str(path), "status": "FAIL", "reason": "bad_structure_expected_category_subject_session_sample", "sample_id": ""})
                status_counts["FAIL"] += 1
                print(f"[{num}/{len(detected_files)}] FAIL unexpected path: {rel}")
            continue

        category, subject, session, raw_id, _ = rel.parts
        try:
            sid = canonical_sample_id(category, subject, session, raw_id)
            if sid in used_ids:
                raise ValueError(f"Duplicate canonical ID: {sid}")
            used_ids.add(sid)
            if category.upper().startswith("S") and re.fullmatch(r"S\d+", category.upper()):
                raise ValueError("Legacy UNKNOWN/S01 layout: move to UNKNOWN/<category>/S01/...")
            meta_path = path.parent / "metadata.json"
            raw_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            if str(raw_meta.get("sign", raw_meta.get("label", "UNKNOWN"))).upper() != "UNKNOWN":
                raise ValueError("Raw metadata sign is not UNKNOWN")
            for key, expected in (("subject_id", subject), ("session_id", session)):
                if key in raw_meta and str(raw_meta[key]).upper() != expected.upper():
                    raise ValueError(f"Raw metadata {key} disagrees with folder name")
            meta_category = raw_meta.get("category") or raw_meta.get("unknown_category")
            if meta_category and str(meta_category) != category:
                raise ValueError(f"Category mismatch: folder={category}, metadata={meta_category}")

            a4, a5, a6, checks, cover = extract_one(path, cfg4, cfg5, cfg6)
            rel_out = Path(category) / subject / session / sid
            p4 = processed_root / "features_phase4_unknown" / rel_out / "features.npz"
            p5 = processed_root / "features_phase5_unknown" / rel_out / "features_v2.npz"
            p6 = processed_root / "features_phase6_unknown" / rel_out / "features_v3.npz"
            save_npz(p4, a4)
            save_npz(p5, a5)
            save_npz(p6, a6)
            split = ("val" if quick_test else
                     "val" if subject.upper() in val_subjects else
                     "test" if subject.upper() in test_subjects else "unassigned")
            detail = {
                "phase": 8, "sample_id": sid, "raw_sample_id": raw_meta.get("sample_id", raw_id),
                "sign": "UNKNOWN", "category": category, "subject_id": subject,
                "session_id": session, "split": split, "source": str(path),
                "feature_dim": 453, "frame_count": int(a6["ablation_hands_pose_face453"].shape[0]),
                "phase4_audit": checks[0], "phase5_audit": checks[1], "phase6_audit": checks[2],
                "coverage": cover,
                "qc_note": "Feature/structure checks only; manual semantic QC not replaced",
            }
            save_json(p6.parent / "feature_metadata.json", detail)
            prepared_rows.append({
                "sample_id": sid, "subject_id": subject, "category": category,
                "split": split, "feature_path": str(p6),
            })
            audit_rows.append({"source": str(path), "status": "PASS", "reason": "", "sample_id": sid})
            status_counts["PASS"] += 1
            print(f"[{num}/{len(detected_files)}] PASS {sid} ({detail['frame_count']}x453, split={split})")
        except Exception as exc:
            audit_rows.append({"source": str(path), "status": "FAIL", "reason": f"{type(exc).__name__}: {exc}", "sample_id": ""})
            status_counts["FAIL"] += 1
            print(f"[{num}/{len(detected_files)}] FAIL {rel}: {type(exc).__name__}: {exc}")

    inventory = processed_root / "features_phase6_unknown" / "unknown_inventory.csv"
    write_csv(inventory, prepared_rows, ["sample_id", "subject_id", "category", "split", "feature_path"])
    save_json(report_path, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root), "total": len(detected_files),
        "pass": status_counts["PASS"], "fail": status_counts["FAIL"],
        "split_requested": split_requested, "quick_test": quick_test,
        "subjects_val": sorted(val_subjects),
        "subjects_test": sorted(test_subjects), "rows": audit_rows,
    })

    print("\nPHASE08 UNKNOWN FEATURES")
    print(f"TOTAL={len(detected_files)} PASS={status_counts['PASS']} FAIL={status_counts['FAIL']} SKIP_LEGACY={status_counts['SKIP_LEGACY']}")
    print(f"Inventory: {inventory}")
    print(f"Audit:     {report_path}")
    if not prepared_rows:
        print("ERROR: No category-labeled UNKNOWN samples were processed. Check raw folder structure.")
        return 2
    if split_requested or quick_test:
        selected = [r for r in prepared_rows if r["split"] in {"val", "test"}]
        val_n = sum(r["split"] == "val" for r in selected)
        test_n = sum(r["split"] == "test" for r in selected)
        if not val_n or (not quick_test and not test_n):
            print("ERROR: Official evaluation needs UNKNOWN in both VAL and TEST")
            return 2
        if status_counts["FAIL"]:
            print("ERROR: Invalid raw sample(s) detected; inspect audit before generating manifest")
            return 2
        if manifest_path.exists() and not args.overwrite_manifest:
            print(f"ERROR: Existing manifest will NOT be overwritten: {manifest_path}")
            print("Use --overwrite-manifest only after checking dataset split.")
            return 2
        write_csv(manifest_path, selected, ["sample_id", "subject_id", "category", "split", "feature_path"])
        # Marker is deliberately separate from model/artifacts to distinguish exploratory
        # confidence analysis from signer-independent official evaluation.
        mode_path = manifest_path.parent / "unknown_manifest_mode.json"
        mode = "exploratory_val_only" if quick_test else "signer_independent_val_test"
        save_json(mode_path, {
            "mode": mode, "val_samples": val_n, "test_samples": test_n,
            "categories": sorted({r["category"] for r in selected}),
            "subjects": sorted({r["subject_id"] for r in selected}),
            "warning": ("DEVELOPMENT ONLY: same collected UNKNOWN used for threshold exploration; "
                        "no independent UNKNOWN TEST exists." if quick_test else ""),
        })
        print(f"UNKNOWN manifest: {manifest_path} (VAL={val_n}, TEST={test_n})")
        print(f"Mode: {mode}; categories={sorted({r['category'] for r in selected})}")
        if quick_test:
            print("WARNING: QUICK TEST ONLY. No independent TEST, no Phase08 final PASS.")
            print("NEXT: python scripts/prepare_phase8_unknown.py")
            print("THEN: python scripts/calibrate_phase8_dev.py")
        else:
            print("NEXT: python scripts/prepare_phase8_unknown.py")
    else:
        print("No VAL/TEST split selected: manifest NOT created.")
        print("To generate: --val-subjects S11 --test-subjects S12")
    return 0 if status_counts["FAIL"] == 0 else 2


def main():
    ap = argparse.ArgumentParser(description="Phase08 UNKNOWN raw -> Phase04/05/06 -> manifest")
    ap.add_argument("--raw-root", default="dataset/raw/UNKNOWN")
    ap.add_argument("--processed-root", default="dataset/processed")
    ap.add_argument("--manifest", default="dataset/unknown_phase8/unknown_manifest.csv")
    ap.add_argument("--report", default="dataset/processed/features_phase6_unknown/unknown_processing_audit.json")
    ap.add_argument("--phase4-config", default="configs/feature_phase4.yaml")
    ap.add_argument("--phase5-config", default="configs/feature_phase5.yaml")
    ap.add_argument("--phase6-config", default="configs/feature_phase6.yaml")
    ap.add_argument("--val-subjects", default="", help="Comma separated signer IDs, e.g. S11,S12")
    ap.add_argument("--test-subjects", default="", help="Different signer IDs, e.g. S13,S14")
    ap.add_argument("--quick-test", action="store_true",
                    help="EXPLORATORY ONLY: use all collected UNKNOWN as VAL; no independent TEST")
    ap.add_argument("--overwrite-manifest", action="store_true", help="Allow overwriting existing UNKNOWN manifest")
    raise SystemExit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
