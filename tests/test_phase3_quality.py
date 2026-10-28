from __future__ import annotations

from pathlib import Path

import numpy as np

from core.quality_phase3 import QualityConfig, inspect_sample


def save_npz(
    path: Path,
    *,
    frames: int = 45,
    left_present: bool = True,
    right_present: bool = False,
    crop: bool = False,
    jump: bool = False,
) -> None:
    timestamps = np.arange(frames, dtype=np.float64) / 30.0

    left = np.zeros((frames, 21, 3), dtype=np.float32)
    right = np.zeros((frames, 21, 3), dtype=np.float32)

    # Stable synthetic hand inside the image.
    for t in range(frames):
        left[t, :, 0] = 0.5
        left[t, :, 1] = 0.5
        left[t, :, 2] = 0.0

    if crop:
        left[:, 0, 0] = 0.005

    if jump:
        for t in range(1, frames, 2):
            left[t, :, 0] += 0.35

    lp = np.full(frames, left_present, dtype=bool)
    rp = np.full(frames, right_present, dtype=bool)

    np.savez_compressed(
        path,
        timestamps=timestamps,
        left_hand=left,
        right_hand=right,
        pose=np.zeros((frames, 0, 3), dtype=np.float32),
        face=np.zeros((frames, 0, 3), dtype=np.float32),
        left_present=lp,
        right_present=rp,
        pose_present=np.ones(frames, dtype=bool),
        face_present=np.ones(frames, dtype=bool),
    )


def cfg() -> QualityConfig:
    return QualityConfig(
        min_frames=30,
        max_frames=60,
        min_any_hand_ratio=0.90,
        hand_presence_exempt_signs=("REST", "UNKNOWN"),
        crop_ratio_review=0.20,
        jump_distance_threshold=0.20,
        jump_ratio_review=0.10,
        min_motion_mean_review=0.0,
        blur_enabled=False,
    )


def inspect(path: Path, sign: str = "xin_chao"):
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


def test_good_sample_accept(tmp_path: Path):
    p = tmp_path / "landmarks.npz"
    save_npz(p)
    result, masks = inspect(p)

    assert result.status == "ACCEPT"
    assert result.train_eligible is True
    assert result.any_hand_ratio == 1.0
    assert masks["left_hand_present"].shape == (45,)


def test_missing_hand_rejects_normal_sign(tmp_path: Path):
    p = tmp_path / "landmarks.npz"
    save_npz(p, left_present=False, right_present=False)

    result, _ = inspect(p, "xin_chao")

    assert result.status == "REJECT"
    assert result.train_eligible is False
    assert any("any_hand_ratio_below_min" in x for x in result.hard_fail_reasons)


def test_rest_can_have_no_hand(tmp_path: Path):
    p = tmp_path / "landmarks.npz"
    save_npz(p, left_present=False, right_present=False)

    result, _ = inspect(p, "REST")

    assert result.status == "ACCEPT"
    assert result.train_eligible is True


def test_crop_becomes_review_not_reject(tmp_path: Path):
    p = tmp_path / "landmarks.npz"
    save_npz(p, crop=True)

    result, _ = inspect(p)

    assert result.status == "REVIEW"
    assert result.train_eligible is False
    assert any("crop_ratio_high" in x for x in result.review_reasons)


def test_wrong_frame_count_rejects(tmp_path: Path):
    p = tmp_path / "landmarks.npz"
    save_npz(p, frames=20)

    result, _ = inspect_sample(
        sample_id="xin_chao_S01_SES01_001",
        sign="xin_chao",
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
