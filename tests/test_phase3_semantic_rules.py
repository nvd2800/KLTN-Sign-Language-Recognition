from __future__ import annotations

from pathlib import Path
import numpy as np

from core.quality_phase3 import QualityConfig, inspect_sample


def make_npz(
    path: Path,
    *,
    frames: int = 45,
    left_present_ratio: float = 1.0,
    right_present_ratio: float = 0.0,
    motion_step: float = 0.0,
) -> None:
    timestamps = np.arange(frames, dtype=np.float64) / 30.0
    left = np.zeros((frames, 21, 3), dtype=np.float32)
    right = np.zeros((frames, 21, 3), dtype=np.float32)

    left[:, :, 0] = 0.5
    left[:, :, 1] = 0.5
    right[:, :, 0] = 0.5
    right[:, :, 1] = 0.5

    for t in range(frames):
        left[t, :, 0] += t * motion_step
        right[t, :, 0] += t * motion_step

    lp = np.zeros(frames, dtype=bool)
    rp = np.zeros(frames, dtype=bool)

    lp[: int(round(frames * left_present_ratio))] = True
    rp[: int(round(frames * right_present_ratio))] = True

    np.savez_compressed(
        path,
        timestamps=timestamps,
        left_hand=left,
        right_hand=right,
        left_present=lp,
        right_present=rp,
        pose_present=np.ones(frames, dtype=bool),
        face_present=np.ones(frames, dtype=bool),
    )


def cfg() -> QualityConfig:
    return QualityConfig(
        min_frames=30,
        max_frames=60,
        rest_signs=("REST",),
        hand_presence_exempt_signs=("REST", "UNKNOWN"),
        two_hand_signs=(),
        min_any_hand_ratio=0.80,
        min_both_hand_ratio=0.80,
        rest_max_any_hand_ratio_accept=0.05,
        rest_max_motion_mean_accept=0.010,
        crop_ratio_review=0.25,
        jump_distance_threshold=0.25,
        jump_ratio_review=0.15,
        min_motion_mean_review=0.0,
        blur_enabled=False,
    )


def run(path: Path, sign: str):
    return inspect_sample(
        sample_id=f"{sign}_S01_SES01_001",
        sign=sign,
        subject_id="S01",
        session_id="SES01",
        split="train",
        landmarks_path=path,
        video_path=path.parent / "missing.mp4",
        metadata_frame_count=45,
        cfg=cfg(),
    )


def test_one_hand_sign_accepts_one_visible_hand(tmp_path: Path):
    p = tmp_path / "landmarks.npz"
    make_npz(p, left_present_ratio=1.0, right_present_ratio=0.0)

    result, _ = run(p, "xin_chao")

    assert result.status == "ACCEPT"
    assert result.any_hand_ratio == 1.0
    assert result.both_hand_ratio == 0.0


def test_one_hand_sign_rejects_when_hand_missing_too_much(tmp_path: Path):
    p = tmp_path / "landmarks.npz"
    make_npz(p, left_present_ratio=0.50)

    result, _ = run(p, "xin_chao")

    assert result.status == "REJECT"
    assert any("any_hand_ratio_below_min" in x for x in result.hard_fail_reasons)


def test_rest_accepts_no_hand(tmp_path: Path):
    p = tmp_path / "landmarks.npz"
    make_npz(p, left_present_ratio=0.0, right_present_ratio=0.0)

    result, _ = run(p, "REST")

    assert result.status == "ACCEPT"
    assert result.any_hand_ratio == 0.0


def test_rest_reviews_when_hand_detected(tmp_path: Path):
    p = tmp_path / "landmarks.npz"
    make_npz(p, left_present_ratio=0.30)

    result, _ = run(p, "REST")

    assert result.status == "REVIEW"
    assert any("rest_hand_detected" in x for x in result.review_reasons)


def test_rest_structural_error_still_rejects(tmp_path: Path):
    p = tmp_path / "landmarks.npz"
    make_npz(p, frames=20, left_present_ratio=0.0)

    result, _ = inspect_sample(
        sample_id="REST_S01_SES01_001",
        sign="REST",
        subject_id="S01",
        session_id="SES01",
        split="train",
        landmarks_path=p,
        video_path=tmp_path / "missing.mp4",
        metadata_frame_count=20,
        cfg=cfg(),
    )

    assert result.status == "REJECT"
    assert any("frame_count_outside_range" in x for x in result.hard_fail_reasons)
