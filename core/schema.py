from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Any
import json
import shutil

import numpy as np

from .types import FrameLandmarks


@dataclass
class SampleMetadata:
    dataset_version: str
    schema_version: str
    sample_id: str
    sign: str
    subject_id: str
    session_id: str
    camera_id: int
    fps: float
    frame_count: int
    resolution: list[int]
    lighting: str = "unspecified"
    distance: str = "unspecified"
    left_hand_ratio: float = 0.0
    right_hand_ratio: float = 0.0
    pose_ratio: float = 0.0
    face_ratio: float = 0.0
    collection_mode: str = "fixed"
    target_fps: float | None = None
    duration_s: float | None = None
    created_at_utc: str = ""

    def validate(self) -> None:
        if not self.dataset_version:
            raise ValueError("dataset_version must not be empty")
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.sign:
            raise ValueError("sign must not be empty")
        if not self.subject_id.startswith("S"):
            raise ValueError("subject_id must use anonymous Sxx format")
        if not self.session_id.startswith("SES"):
            raise ValueError("session_id must use SESxx format")
        if self.frame_count < 1:
            raise ValueError("frame_count must be >= 1")
        if self.fps <= 0 or not np.isfinite(self.fps):
            raise ValueError("fps must be finite and > 0")
        effective_target = self.fps if self.target_fps is None else self.target_fps
        if effective_target <= 0 or not np.isfinite(effective_target):
            raise ValueError("target_fps must be finite and > 0")
        if self.duration_s is not None and (self.duration_s < 0 or not np.isfinite(self.duration_s)):
            raise ValueError("duration_s must be finite and >= 0")
        if len(self.resolution) != 2 or int(self.resolution[0]) <= 0 or int(self.resolution[1]) <= 0:
            raise ValueError("resolution must be [width, height] with positive values")
        if self.collection_mode not in {"fixed", "manual", "video_import"}:
            raise ValueError("collection_mode must be fixed/manual/video_import")
        for name in ("left_hand_ratio", "right_hand_ratio", "pose_ratio", "face_ratio"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        if data["target_fps"] is None:
            data["target_fps"] = float(self.fps)
        if data["duration_s"] is None:
            data["duration_s"] = max((self.frame_count - 1) / float(self.fps), 0.0)
        return data


def _stack_optional(frames: list[FrameLandmarks], attr: str) -> tuple[np.ndarray, np.ndarray]:
    values = [getattr(f, attr) for f in frames]
    present = np.asarray([v is not None for v in values], dtype=np.uint8)
    available = [v for v in values if v is not None]
    if not available:
        return np.zeros((len(frames), 0, 3), dtype=np.float32), present

    shape = available[0].shape
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(f"{attr} must have shape (N, 3), got {shape}")

    out = np.zeros((len(frames), *shape), dtype=np.float32)
    for i, value in enumerate(values):
        if value is None:
            continue
        if value.shape != shape:
            raise ValueError(f"Inconsistent {attr} shape at frame {i}: {value.shape} vs {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{attr} contains NaN/Inf at frame {i}")
        out[i] = value.astype(np.float32, copy=False)
    return out, present


def _validate_timestamps(timestamps: np.ndarray) -> None:
    if timestamps.ndim != 1:
        raise ValueError("timestamps must have shape (T,)")
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps contain NaN/Inf")
    if timestamps.size > 1 and np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be monotonic non-decreasing")


def frames_to_arrays(frames: Iterable[FrameLandmarks]) -> dict[str, np.ndarray]:
    fs = list(frames)
    if not fs:
        raise ValueError("Cannot serialize empty sequence")

    for i, frame in enumerate(fs):
        try:
            frame.validate()
        except Exception as exc:
            raise ValueError(f"Invalid frame at index {i}: {exc}") from exc

    timestamps = np.asarray([f.timestamp for f in fs], dtype=np.float64)
    _validate_timestamps(timestamps)
    left_hand = np.stack([f.left_hand for f in fs]).astype(np.float32)
    right_hand = np.stack([f.right_hand for f in fs]).astype(np.float32)
    expected = (len(fs), 21, 3)
    if left_hand.shape != expected:
        raise ValueError(f"left_hand must have shape {expected}, got {left_hand.shape}")
    if right_hand.shape != expected:
        raise ValueError(f"right_hand must have shape {expected}, got {right_hand.shape}")

    pose, pose_present = _stack_optional(fs, "pose")
    face, face_present = _stack_optional(fs, "face")

    return {
        "timestamps": timestamps,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "pose": pose,
        "face": face,
        "left_present": np.asarray([f.presence.left_hand for f in fs], dtype=np.uint8),
        "right_present": np.asarray([f.presence.right_hand for f in fs], dtype=np.uint8),
        "pose_present": pose_present,
        "face_present": face_present,
    }


def save_sequence(sample_dir: Path, frames: Iterable[FrameLandmarks], metadata: SampleMetadata) -> None:
    sample_dir = Path(sample_dir)
    if sample_dir.exists():
        raise FileExistsError(f"Sample directory already exists: {sample_dir}")
    fs = list(frames)
    arrays = frames_to_arrays(fs)
    if metadata.frame_count != len(fs):
        raise ValueError("metadata.frame_count does not match actual sequence length")
    metadata.validate()
    sample_dir.mkdir(parents=True, exist_ok=False)
    try:
        np.savez_compressed(sample_dir / "landmarks.npz", **arrays)
        with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
    except Exception:
        shutil.rmtree(sample_dir, ignore_errors=True)
        raise


def load_sequence(sample_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    sample_dir = Path(sample_dir)
    npz_path = sample_dir / "landmarks.npz"
    meta_path = sample_dir / "metadata.json"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing landmarks.npz: {npz_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.json: {meta_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return arrays, meta


def validate_loaded_sequence(arrays: dict[str, np.ndarray], metadata: dict) -> None:
    required_arrays = {
        "timestamps", "left_hand", "right_hand", "pose", "face",
        "left_present", "right_present", "pose_present", "face_present",
    }
    missing = required_arrays.difference(arrays)
    if missing:
        raise ValueError(f"Missing arrays: {sorted(missing)}")

    n = int(arrays["timestamps"].shape[0])
    if n < 1:
        raise ValueError("Sequence contains zero frames")
    _validate_timestamps(arrays["timestamps"])
    if arrays["left_hand"].shape != (n, 21, 3):
        raise ValueError(f"Invalid left_hand shape: {arrays['left_hand'].shape}")
    if arrays["right_hand"].shape != (n, 21, 3):
        raise ValueError(f"Invalid right_hand shape: {arrays['right_hand'].shape}")
    for key in ("left_present", "right_present", "pose_present", "face_present"):
        if arrays[key].shape != (n,):
            raise ValueError(f"Invalid {key} shape: {arrays[key].shape}")
    for key in ("pose", "face"):
        arr = arrays[key]
        if arr.ndim != 3 or arr.shape[0] != n or (arr.shape[1] > 0 and arr.shape[2] != 3):
            raise ValueError(f"Invalid {key} shape: {arr.shape}")

    required_meta = {
        "dataset_version", "schema_version", "sample_id", "sign", "subject_id",
        "session_id", "camera_id", "fps", "frame_count", "resolution", "collection_mode",
    }
    missing_meta = required_meta.difference(metadata)
    if missing_meta:
        raise ValueError(f"Missing metadata fields: {sorted(missing_meta)}")
    if int(metadata["frame_count"]) != n:
        raise ValueError("metadata.frame_count does not match landmarks.npz")

    SampleMetadata(
        dataset_version=str(metadata["dataset_version"]),
        schema_version=str(metadata["schema_version"]),
        sample_id=str(metadata["sample_id"]),
        sign=str(metadata["sign"]),
        subject_id=str(metadata["subject_id"]),
        session_id=str(metadata["session_id"]),
        camera_id=int(metadata["camera_id"]),
        fps=float(metadata["fps"]),
        frame_count=int(metadata["frame_count"]),
        resolution=list(metadata["resolution"]),
        lighting=str(metadata.get("lighting", "unspecified")),
        distance=str(metadata.get("distance", "unspecified")),
        left_hand_ratio=float(metadata.get("left_hand_ratio", 0.0)),
        right_hand_ratio=float(metadata.get("right_hand_ratio", 0.0)),
        pose_ratio=float(metadata.get("pose_ratio", 0.0)),
        face_ratio=float(metadata.get("face_ratio", 0.0)),
        collection_mode=str(metadata.get("collection_mode", "fixed")),
        target_fps=float(metadata.get("target_fps", metadata["fps"])),
        duration_s=float(metadata.get("duration_s", max((n - 1) / float(metadata["fps"]), 0.0))),
        created_at_utc=str(metadata.get("created_at_utc", "")),
    ).validate()
