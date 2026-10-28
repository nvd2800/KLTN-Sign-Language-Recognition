"""Phase08 live tester: pure preprocessing/decision/log helpers (no TF/MediaPipe)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameSample:
    left: np.ndarray
    right: np.ndarray
    pose: np.ndarray
    face: np.ndarray
    left_present: bool
    right_present: bool
    pose_present: bool
    face_present: bool
    timestamp: float


@dataclass(frozen=True)
class Decision:
    top1: str
    outcome: str
    confidence: float
    margin: float
    probabilities: dict[str, float]
    threshold: float


def phase08_decision(probs: np.ndarray, labels: list[str], threshold: float,
                     min_margin: float = 0.0) -> Decision:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size != len(labels) or p.size < 2:
        raise ValueError(f"Expected {len(labels)} probabilities, got {p.shape}")
    if len(set(labels)) != len(labels) or any(not str(x) for x in labels):
        raise ValueError("Labels must be unique and nonempty")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities contain NaN/Inf")
    if np.any(p < -1e-5) or np.any(p > 1+1e-5):
        raise ValueError("Probabilities must be in [0,1]")
    if not 0 <= float(threshold) <= 1 or not 0 <= float(min_margin) <= 1:
        raise ValueError("Threshold/margin must be in [0,1]")
    order = np.argsort(p)[::-1]
    i, j = int(order[0]), int(order[1])
    conf, margin = float(p[i]), float(p[i] - p[j])
    outcome = labels[i] if conf >= threshold and margin >= min_margin else "UNKNOWN"
    return Decision(
        top1=labels[i], outcome=outcome, confidence=conf, margin=margin,
        probabilities={s: float(v) for s,v in zip(labels,p)}, threshold=float(threshold),
    )


def judge_trial(gt: str | None, result: Decision) -> str:
    if not gt:
        return "UNLABELED"
    if gt == result.outcome:
        return "CORRECT"
    if gt == "UNKNOWN":
        return "FALSE_ACCEPT_UNKNOWN"
    if result.outcome == "UNKNOWN":
        return "FALSE_REJECT_KNOWN"
    return "WRONG_KNOWN_SIGN"


def build_model_input(frames: list[FrameSample], cfg4: Any, cfg5: Any, cfg6: Any,
                      mean: np.ndarray, std: np.ndarray, target_frames: int = 32,
                      feature_mode: str = "phase6_453"):
    """Reuse Phase04→05→06 funcs and Phase07 temporal preprocessing, exactly.

    Returns standardized [32,453], valid temporal mask, raw [T,453].
    Does NOT refit the standardizer or reuse Phase08 test data for fitting.
    """
    from core.feature_phase4 import build_phase4_features
    from core.feature_phase5 import build_phase5_features
    from core.feature_phase6 import build_phase6_features
    from core.preprocess_phase7 import uniform_resample_or_pad

    if not frames:
        raise ValueError("Empty camera sequence")
    T = len(frames)
    left = np.stack([f.left for f in frames]).astype(np.float32)
    right = np.stack([f.right for f in frames]).astype(np.float32)
    pose = np.stack([f.pose for f in frames]).astype(np.float32)
    face = np.stack([f.face for f in frames]).astype(np.float32)
    lp = np.array([f.left_present for f in frames], dtype=bool)
    rp = np.array([f.right_present for f in frames], dtype=bool)
    pp = np.array([f.pose_present for f in frames], dtype=bool)
    fp = np.array([f.face_present for f in frames], dtype=bool)
    timestamps = np.asarray([f.timestamp for f in frames], dtype=np.float64)

    if left.shape != (T,21,3) or right.shape != (T,21,3):
        raise ValueError(f"Bad hand shapes {left.shape}, {right.shape}")
    if pose.shape != (T,33,3) or face.shape != (T,468,3):
        raise ValueError(f"Bad pose/face shapes {pose.shape}, {face.shape}")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError("Invalid or non-monotonic timestamps")

    a4 = build_phase4_features(left, right, lp, rp, cfg4)
    a4["timestamps"] = timestamps
    a5 = build_phase5_features(
        phase4_arrays=a4, raw_left=left, raw_right=right,
        pose_xyz=pose, pose_present=pp, cfg=cfg5,
    )
    a6 = build_phase6_features(
        phase5_full417=a5["full_features417"],
        pose_xyz=pose, pose_present=pp, face_xyz=face, face_present=fp, cfg=cfg6,
    )
    raw453 = np.asarray(a6["ablation_hands_pose_face453"], np.float32)
    if raw453.shape != (T,453) or not np.isfinite(raw453).all():
        raise ValueError(f"Invalid Phase06 features: {raw453.shape}")

    if feature_mode == "spatial513":
        from core.feature_spatial import spatial60
        spatial = spatial60(left, right, pose, lp, rp, pp, timestamps)
        raw = np.concatenate((raw453, spatial), axis=1).astype(np.float32)
    elif feature_mode == "phase6_453":
        raw = raw453
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    feature_dim = raw.shape[1]
    fixed, mask, _ = uniform_resample_or_pad(raw, target_frames)
    mean = np.asarray(mean,np.float32).reshape(-1)
    std = np.asarray(std,np.float32).reshape(-1)
    if mean.shape != (feature_dim,) or std.shape != (feature_dim,):
        raise ValueError(f"Scaler/feature mismatch: expected ({feature_dim},), got {mean.shape}/{std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError("Invalid frozen scaler")
    scaled=np.zeros_like(fixed,np.float32)
    scaled[mask] = (fixed[mask] - mean) / std
    scaled[~mask] = 0
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite scaled features")
    return scaled, mask, raw


def append_csv_row(path: Path, row: dict[str, Any], fields: list[str]):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        with path.open("r", encoding="utf-8-sig",newline="") as f:
            existing=next(csv.reader(f),[])
        if existing != fields:
            raise ValueError(f"Existing log uses different columns: {path}")
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def load_threshold(path: Path | None, override: float | None):
    if override is not None:
        if not 0<=override<=1:
            raise ValueError("--threshold must be between 0 and 1")
        return float(override), "manual override", None
    if not path or not Path(path).is_file():
        raise FileNotFoundError(
            f"Calibrated Phase08 threshold not found: {path}. "
            "Run Phase08 calibration, or use --threshold 0.90 for an EXPERIMENT only."
        )
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    threshold=float(obj["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid calibrated threshold")
    validation=obj.get("validation",{})
    return threshold, "Phase08 VAL calibration", validation
