from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class QualityConfig:
    min_frames: int = 30
    max_frames: int = 60
    require_monotonic_timestamps: bool = True
    require_finite_landmarks: bool = True

    rest_signs: tuple[str, ...] = ("REST",)
    hand_presence_exempt_signs: tuple[str, ...] = ("REST", "UNKNOWN")
    two_hand_signs: tuple[str, ...] = ()

    min_any_hand_ratio_accept: float = 0.70
    min_any_hand_ratio_reject: float = 0.55
    min_both_hand_ratio_accept: float = 0.70
    min_both_hand_ratio_reject: float = 0.55

    rest_max_any_hand_ratio_accept: float = 0.05
    rest_max_motion_mean_accept: float = 0.010

    min_pose_ratio_warning: float = 0.50
    min_face_ratio_warning: float = 0.50

    crop_margin: float = 0.03
    crop_ratio_review: float = 0.25
    jump_distance_threshold: float = 0.25
    jump_ratio_review: float = 0.15
    min_motion_mean_review: float = 0.002
    low_motion_exempt_signs: tuple[str, ...] = ("REST",)

    blur_enabled: bool = True
    blur_variance_threshold: float = 50.0
    blur_ratio_review: float = 0.40
    blur_sample_every_n_frames: int = 3
    video_missing_is_review: bool = False

    review_train_eligible: bool = False
    reject_train_eligible: bool = False


@dataclass
class QualityResult:
    sample_id: str
    sign: str
    subject_id: str
    session_id: str
    split: str
    frame_count: int
    left_hand_ratio: float
    right_hand_ratio: float
    any_hand_ratio: float
    both_hand_ratio: float
    pose_ratio: float
    face_ratio: float
    crop_ratio: float
    jump_ratio: float
    motion_mean: float
    blur_checked: bool
    blur_ratio: float | None
    blur_variance_mean: float | None
    status: str
    train_eligible: bool
    hard_fail_reasons: list[str]
    review_reasons: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_quality_config(path: Path) -> QualityConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Quality config not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = data.get("rules", {}) or {}
    policy = data.get("policy", {}) or {}

    cfg = QualityConfig(
        min_frames=int(rules.get("min_frames", 30)),
        max_frames=int(rules.get("max_frames", 60)),
        require_monotonic_timestamps=bool(rules.get("require_monotonic_timestamps", True)),
        require_finite_landmarks=bool(rules.get("require_finite_landmarks", True)),

        rest_signs=tuple(str(x) for x in rules.get("rest_signs", ["REST"])),
        hand_presence_exempt_signs=tuple(
            str(x) for x in rules.get("hand_presence_exempt_signs", ["REST", "UNKNOWN"])
        ),
        two_hand_signs=tuple(str(x) for x in rules.get("two_hand_signs", [])),

        min_any_hand_ratio_accept=float(rules.get("min_any_hand_ratio_accept", 0.70)),
        min_any_hand_ratio_reject=float(rules.get("min_any_hand_ratio_reject", 0.55)),
        min_both_hand_ratio_accept=float(rules.get("min_both_hand_ratio_accept", 0.70)),
        min_both_hand_ratio_reject=float(rules.get("min_both_hand_ratio_reject", 0.55)),

        rest_max_any_hand_ratio_accept=float(
            rules.get("rest_max_any_hand_ratio_accept", 0.05)
        ),
        rest_max_motion_mean_accept=float(
            rules.get("rest_max_motion_mean_accept", 0.010)
        ),

        min_pose_ratio_warning=float(rules.get("min_pose_ratio_warning", 0.50)),
        min_face_ratio_warning=float(rules.get("min_face_ratio_warning", 0.50)),

        crop_margin=float(rules.get("crop_margin", 0.03)),
        crop_ratio_review=float(rules.get("crop_ratio_review", 0.25)),
        jump_distance_threshold=float(rules.get("jump_distance_threshold", 0.25)),
        jump_ratio_review=float(rules.get("jump_ratio_review", 0.15)),
        min_motion_mean_review=float(rules.get("min_motion_mean_review", 0.002)),
        low_motion_exempt_signs=tuple(
            str(x) for x in rules.get("low_motion_exempt_signs", ["REST"])
        ),

        blur_enabled=bool(rules.get("blur_enabled", True)),
        blur_variance_threshold=float(rules.get("blur_variance_threshold", 50.0)),
        blur_ratio_review=float(rules.get("blur_ratio_review", 0.40)),
        blur_sample_every_n_frames=max(
            1, int(rules.get("blur_sample_every_n_frames", 3))
        ),
        video_missing_is_review=bool(rules.get("video_missing_is_review", False)),

        review_train_eligible=bool(policy.get("review_train_eligible", False)),
        reject_train_eligible=bool(policy.get("reject_train_eligible", False)),
    )

    if cfg.min_frames < 1 or cfg.max_frames < cfg.min_frames:
        raise ValueError("Invalid min_frames/max_frames")

    if cfg.min_any_hand_ratio_reject >= cfg.min_any_hand_ratio_accept:
        raise ValueError(
            "min_any_hand_ratio_reject must be lower than min_any_hand_ratio_accept"
        )

    if cfg.min_both_hand_ratio_reject >= cfg.min_both_hand_ratio_accept:
        raise ValueError(
            "min_both_hand_ratio_reject must be lower than min_both_hand_ratio_accept"
        )

    return cfg


def _pick_array(data: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    return None


def _mask(
    data: Any,
    aliases: tuple[str, ...],
    frame_count: int,
    *,
    required: bool = True,
) -> np.ndarray:
    arr = _pick_array(data, aliases)

    if arr is None:
        if required:
            raise ValueError(
                "Missing presence mask. Expected one of: " + ", ".join(aliases)
            )
        return np.zeros(frame_count, dtype=bool)

    arr = np.asarray(arr).reshape(-1)

    if arr.shape[0] != frame_count:
        raise ValueError(
            f"Mask {aliases[0]} length={arr.shape[0]} != frame_count={frame_count}"
        )

    return arr.astype(bool)


def _ratio(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def _validate_hand(arr: np.ndarray, frame_count: int, name: str) -> None:
    if arr.shape != (frame_count, 21, 3):
        raise ValueError(
            f"{name} must have shape ({frame_count},21,3), got {arr.shape}"
        )


def _finite_present(arr: np.ndarray, present: np.ndarray) -> bool:
    return True if not present.any() else bool(np.isfinite(arr[present]).all())


def _crop_flags(
    hand: np.ndarray,
    present: np.ndarray,
    margin: float,
) -> np.ndarray:
    flags = np.zeros(len(present), dtype=bool)

    if not present.any():
        return flags

    xy = hand[:, :, :2]

    edge = (
        (xy[:, :, 0] <= margin)
        | (xy[:, :, 0] >= 1.0 - margin)
        | (xy[:, :, 1] <= margin)
        | (xy[:, :, 1] >= 1.0 - margin)
    )

    flags[present] = edge[present].any(axis=1)

    return flags


def _pair_motion(
    hand: np.ndarray,
    present: np.ndarray,
) -> np.ndarray:
    values: list[float] = []

    for i in range(1, len(present)):
        if not (present[i - 1] and present[i]):
            continue

        delta = hand[i, :, :2] - hand[i - 1, :, :2]
        values.append(
            float(
                np.linalg.norm(delta, axis=1).mean()
            )
        )

    return np.asarray(values, dtype=np.float32)


def compute_blur_metrics(
    video_path: Path,
    *,
    variance_threshold: float,
    every_n_frames: int,
) -> tuple[bool, float | None, float | None, str | None]:
    video_path = Path(video_path)

    if not video_path.is_file():
        return False, None, None, "source_video_missing"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return False, None, None, "source_video_unreadable"

    values: list[float] = []
    index = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if index % every_n_frames == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                values.append(
                    float(
                        cv2.Laplacian(gray, cv2.CV_64F).var()
                    )
                )

            index += 1

    finally:
        cap.release()

    if not values:
        return False, None, None, "source_video_no_frames"

    arr = np.asarray(values, dtype=np.float64)

    return (
        True,
        float(np.mean(arr < variance_threshold)),
        float(arr.mean()),
        None,
    )


def inspect_sample(
    *,
    sample_id: str,
    sign: str,
    subject_id: str,
    session_id: str,
    split: str,
    landmarks_path: Path,
    video_path: Path,
    metadata_frame_count: int | None,
    cfg: QualityConfig,
) -> tuple[QualityResult, dict[str, np.ndarray]]:
    hard: list[str] = []
    review: list[str] = []
    warnings: list[str] = []

    frame_count = 0
    left_ratio = right_ratio = any_ratio = both_ratio = 0.0
    pose_ratio = face_ratio = 0.0
    crop_ratio = jump_ratio = motion_mean = 0.0
    masks: dict[str, np.ndarray] = {}

    try:
        if not Path(landmarks_path).is_file():
            raise FileNotFoundError("missing_landmarks_npz")

        with np.load(landmarks_path, allow_pickle=False) as data:
            timestamps = np.asarray(data["timestamps"], dtype=np.float64)
            frame_count = int(len(timestamps))

            left = np.asarray(data["left_hand"], dtype=np.float32)
            right = np.asarray(data["right_hand"], dtype=np.float32)

            _validate_hand(left, frame_count, "left_hand")
            _validate_hand(right, frame_count, "right_hand")

            left_present = _mask(
                data, ("left_present", "left_hand_present"), frame_count
            )
            right_present = _mask(
                data, ("right_present", "right_hand_present"), frame_count
            )
            pose_present = _mask(
                data, ("pose_present",), frame_count, required=False
            )
            face_present = _mask(
                data, ("face_present",), frame_count, required=False
            )

            # -------- Hard structural checks --------
            if metadata_frame_count is not None and metadata_frame_count != frame_count:
                hard.append(
                    f"metadata_frame_count_mismatch:{metadata_frame_count}!={frame_count}"
                )

            if not cfg.min_frames <= frame_count <= cfg.max_frames:
                hard.append(f"frame_count_outside_range:{frame_count}")

            if (
                cfg.require_monotonic_timestamps
                and frame_count > 1
                and not np.all(np.diff(timestamps) > 0)
            ):
                hard.append("timestamps_not_strictly_increasing")

            if cfg.require_finite_landmarks:
                if not _finite_present(left, left_present):
                    hard.append("left_hand_non_finite")

                if not _finite_present(right, right_present):
                    hard.append("right_hand_non_finite")

            # -------- Presence --------
            any_present = left_present | right_present
            both_present = left_present & right_present

            left_ratio = _ratio(left_present)
            right_ratio = _ratio(right_present)
            any_ratio = _ratio(any_present)
            both_ratio = _ratio(both_present)
            pose_ratio = _ratio(pose_present)
            face_ratio = _ratio(face_present)

            # -------- Crop --------
            crop_flags = (
                _crop_flags(left, left_present, cfg.crop_margin)
                | _crop_flags(right, right_present, cfg.crop_margin)
            )

            crop_ratio = _ratio(crop_flags)

            # -------- Motion / jump --------
            pieces = [
                arr
                for arr in (
                    _pair_motion(left, left_present),
                    _pair_motion(right, right_present),
                )
                if arr.size
            ]

            if pieces:
                displacement = np.concatenate(pieces)

                motion_mean = float(displacement.mean())

                jump_ratio = float(
                    np.mean(
                        displacement
                        > cfg.jump_distance_threshold
                    )
                )

            # -------- Semantic rules --------
            is_rest = sign in cfg.rest_signs
            is_two_hand = sign in cfg.two_hand_signs

            if is_rest:
                # REST:
                # no-hand + idle => ACCEPT
                # hand detected / motion => REVIEW
                # structural data error => REJECT

                if any_ratio > cfg.rest_max_any_hand_ratio_accept:
                    review.append(
                        "rest_hand_detected:"
                        f"{any_ratio:.4f}>"
                        f"{cfg.rest_max_any_hand_ratio_accept:.4f}"
                    )

                if motion_mean > cfg.rest_max_motion_mean_accept:
                    review.append(
                        "rest_motion_too_high:"
                        f"{motion_mean:.6f}>"
                        f"{cfg.rest_max_motion_mean_accept:.6f}"
                    )

            elif is_two_hand:
                # >= .70 pass
                # .55-.70 review
                # < .55 reject
                if both_ratio < cfg.min_both_hand_ratio_reject:
                    hard.append(
                        "both_hand_ratio_too_low:"
                        f"{both_ratio:.4f}<"
                        f"{cfg.min_both_hand_ratio_reject:.4f}"
                    )

                elif both_ratio < cfg.min_both_hand_ratio_accept:
                    review.append(
                        "both_hand_ratio_review:"
                        f"{both_ratio:.4f}<"
                        f"{cfg.min_both_hand_ratio_accept:.4f}"
                    )

            elif sign not in cfg.hand_presence_exempt_signs:
                # Normal known sign / one-hand sign:
                # >= .70 pass
                # .55-.70 review
                # < .55 reject
                if any_ratio < cfg.min_any_hand_ratio_reject:
                    hard.append(
                        "any_hand_ratio_too_low:"
                        f"{any_ratio:.4f}<"
                        f"{cfg.min_any_hand_ratio_reject:.4f}"
                    )

                elif any_ratio < cfg.min_any_hand_ratio_accept:
                    review.append(
                        "any_hand_ratio_review:"
                        f"{any_ratio:.4f}<"
                        f"{cfg.min_any_hand_ratio_accept:.4f}"
                    )

            # -------- Warnings --------
            if pose_ratio < cfg.min_pose_ratio_warning:
                warnings.append(f"low_pose_ratio:{pose_ratio:.4f}")

            if face_ratio < cfg.min_face_ratio_warning:
                warnings.append(f"low_face_ratio:{face_ratio:.4f}")

            # -------- Generic review heuristics --------
            if not is_rest:
                if crop_ratio > cfg.crop_ratio_review:
                    review.append(
                        f"crop_ratio_high:{crop_ratio:.4f}>{cfg.crop_ratio_review:.4f}"
                    )

                if jump_ratio > cfg.jump_ratio_review:
                    review.append(
                        "landmark_jump_ratio_high:"
                        f"{jump_ratio:.4f}>{cfg.jump_ratio_review:.4f}"
                    )

                if (
                    sign not in cfg.low_motion_exempt_signs
                    and motion_mean < cfg.min_motion_mean_review
                ):
                    review.append(
                        "motion_too_low:"
                        f"{motion_mean:.6f}<"
                        f"{cfg.min_motion_mean_review:.6f}"
                    )

            masks = {
                "left_hand_present": left_present.astype(np.bool_),
                "right_hand_present": right_present.astype(np.bool_),
                "any_hand_present": any_present.astype(np.bool_),
                "both_hands_present": both_present.astype(np.bool_),
                "pose_present": pose_present.astype(np.bool_),
                "face_present": face_present.astype(np.bool_),
                "crop_flag": crop_flags.astype(np.bool_),
            }

    except FileNotFoundError as exc:
        hard.append(str(exc))

    except Exception as exc:
        hard.append(
            f"npz_schema_error:{type(exc).__name__}:{exc}"
        )

    # -------- Blur --------
    blur_checked = False
    blur_ratio: float | None = None
    blur_mean: float | None = None

    if cfg.blur_enabled:
        blur_checked, blur_ratio, blur_mean, video_error = compute_blur_metrics(
            video_path,
            variance_threshold=cfg.blur_variance_threshold,
            every_n_frames=cfg.blur_sample_every_n_frames,
        )

        if video_error:
            warnings.append(video_error)

            if cfg.video_missing_is_review:
                review.append(video_error)

        if (
            blur_checked
            and blur_ratio is not None
            and blur_ratio > cfg.blur_ratio_review
        ):
            review.append(
                f"blur_ratio_high:{blur_ratio:.4f}>{cfg.blur_ratio_review:.4f}"
            )

    # -------- Final status --------
    if hard:
        status = "REJECT"
        train_eligible = cfg.reject_train_eligible

    elif review:
        status = "REVIEW"
        train_eligible = cfg.review_train_eligible

    else:
        status = "ACCEPT"
        train_eligible = True

    result = QualityResult(
        sample_id=sample_id,
        sign=sign,
        subject_id=subject_id,
        session_id=session_id,
        split=split,
        frame_count=frame_count,

        left_hand_ratio=left_ratio,
        right_hand_ratio=right_ratio,
        any_hand_ratio=any_ratio,
        both_hand_ratio=both_ratio,
        pose_ratio=pose_ratio,
        face_ratio=face_ratio,

        crop_ratio=crop_ratio,
        jump_ratio=jump_ratio,
        motion_mean=motion_mean,

        blur_checked=blur_checked,
        blur_ratio=blur_ratio,
        blur_variance_mean=blur_mean,

        status=status,
        train_eligible=train_eligible,

        hard_fail_reasons=hard,
        review_reasons=review,
        warnings=warnings,
    )

    return result, masks


def save_quality_sample(
    output_root: Path,
    result: QualityResult,
    masks: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    sample_dir = Path(output_root) / result.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    json_path = sample_dir / "quality.json"

    json_path.write_text(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    npz_path = sample_dir / "quality_masks.npz"

    if masks:
        np.savez_compressed(
            npz_path,
            **masks,
        )
    else:
        np.savez_compressed(
            npz_path
        )

    return json_path, npz_path
