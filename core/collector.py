from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import json
import os
import shutil

import numpy as np

from .dataset_contract import (
    validate_session_id,
    validate_sign_name,
    validate_subject_id,
)

from .schema import (
    SampleMetadata,
    frames_to_arrays,
)

from .types import FrameLandmarks


# ============================================================
# COLLECTION IDENTITY
# ============================================================

@dataclass(frozen=True)
class CollectionIdentity:
    sign: str
    subject_id: str
    session_id: str

    def validate(self) -> None:
        validate_sign_name(
            self.sign
        )

        validate_subject_id(
            self.subject_id
        )

        validate_session_id(
            self.session_id
        )


# ============================================================
# COLLECTION SETTINGS
# ============================================================

@dataclass(frozen=True)
class CollectionSettings:
    dataset_version: str
    schema_version: str

    camera_id: int

    resolution: tuple[int, int]

    target_fps: float

    mode: str

    lighting: str = "unspecified"
    distance: str = "unspecified"

    save_video: bool = False

    def validate(self) -> None:

        allowed_modes = {
            "fixed",
            "manual",
            "video_import",
        }

        if self.mode not in allowed_modes:

            raise ValueError(
                "mode must be one of: "
                "fixed, manual, video_import"
            )

        if self.target_fps <= 0:

            raise ValueError(
                "target_fps must be > 0"
            )

        if len(
            self.resolution
        ) != 2:

            raise ValueError(
                "resolution must be "
                "(width, height)"
            )

        width, height = (
            self.resolution
        )

        if width <= 0 or height <= 0:

            raise ValueError(
                "resolution width/height "
                "must be > 0"
            )


# ============================================================
# SAMPLE INDEX
# ============================================================

def next_sample_index(
    session_dir: Path,
    sign: str,
    subject_id: str,
    session_id: str,
) -> int:
    """
    Return next available 1-based sample index.

    Example:
        _001
        _002
        _003

    returns:
        4
    """

    session_dir = Path(
        session_dir
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"{sign}_"
        f"{subject_id}_"
        f"{session_id}_"
    )

    max_index = 0

    for child in (
        session_dir.iterdir()
    ):

        if not child.is_dir():
            continue

        if not child.name.startswith(
            prefix
        ):
            continue

        suffix = (
            child.name[
                len(prefix):
            ]
        )

        if suffix.isdigit():

            max_index = max(
                max_index,
                int(suffix),
            )

    return max_index + 1


# ============================================================
# SAMPLE ID
# ============================================================

def make_sample_id(
    identity: CollectionIdentity,
    index: int,
) -> str:

    identity.validate()

    if index < 1:

        raise ValueError(
            "sample index must be >= 1"
        )

    return (
        f"{identity.sign}_"
        f"{identity.subject_id}_"
        f"{identity.session_id}_"
        f"{index:03d}"
    )


# ============================================================
# SAMPLE DIRECTORY
# ============================================================

def make_sample_dir(
    raw_root: Path,
    identity: CollectionIdentity,
    index: int,
) -> Path:

    identity.validate()

    sample_id = (
        make_sample_id(
            identity,
            index,
        )
    )

    return (
        Path(raw_root)
        / identity.sign
        / identity.subject_id
        / identity.session_id
        / sample_id
    )


# ============================================================
# PRESENCE RATIO
# ============================================================

def _ratio(
    values: Iterable[bool],
) -> float:

    values_list = list(
        values
    )

    if not values_list:
        return 0.0

    return float(
        np.mean(
            np.asarray(
                values_list,
                dtype=np.float32,
            )
        )
    )


# ============================================================
# METADATA BUILDER
# ============================================================

def build_metadata(
    *,
    frames: list[FrameLandmarks],
    sample_id: str,
    identity: CollectionIdentity,
    settings: CollectionSettings,
    actual_fps: float,
) -> SampleMetadata:

    if not frames:

        raise ValueError(
            "Cannot build metadata "
            "for an empty sequence"
        )

    identity.validate()
    settings.validate()

    for frame in frames:
        frame.validate()

    first_timestamp = float(
        frames[0].timestamp
    )

    last_timestamp = float(
        frames[-1].timestamp
    )

    duration_s = max(
        last_timestamp
        - first_timestamp,
        0.0,
    )

    return SampleMetadata(
        dataset_version=(
            settings.dataset_version
        ),

        schema_version=(
            settings.schema_version
        ),

        sample_id=sample_id,

        sign=identity.sign,

        subject_id=(
            identity.subject_id
        ),

        session_id=(
            identity.session_id
        ),

        camera_id=(
            settings.camera_id
        ),

        fps=float(
            actual_fps
        ),

        target_fps=float(
            settings.target_fps
        ),

        frame_count=len(
            frames
        ),

        duration_s=(
            duration_s
        ),

        resolution=[
            int(
                settings.resolution[0]
            ),
            int(
                settings.resolution[1]
            ),
        ],

        collection_mode=(
            settings.mode
        ),

        lighting=(
            settings.lighting
        ),

        distance=(
            settings.distance
        ),

        left_hand_ratio=(
            _ratio(
                frame.presence.left_hand
                for frame in frames
            )
        ),

        right_hand_ratio=(
            _ratio(
                frame.presence.right_hand
                for frame in frames
            )
        ),

        pose_ratio=(
            _ratio(
                frame.presence.pose
                for frame in frames
            )
        ),

        face_ratio=(
            _ratio(
                frame.presence.face
                for frame in frames
            )
        ),

        created_at_utc=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    )


# ============================================================
# ATOMIC SAVE
# ============================================================

def atomic_save_sequence(
    sample_dir: Path,
    frames: list[FrameLandmarks],
    metadata: SampleMetadata,
    video_temp_path: Path | None = None,
    video_filename: str = "source_video.mp4",
) -> None:
    """
    Atomically save one sample.

    Output:

        sample/
            landmarks.npz
            metadata.json
            source_video.mp4   optional

    Uses sibling *.tmp directory.

    Final sample directory is created only when
    all files have been written successfully.
    """

    sample_dir = Path(
        sample_dir
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if sample_dir.exists():

        raise FileExistsError(
            "Refusing to overwrite "
            f"existing sample: "
            f"{sample_dir}"
        )

    if not frames:

        raise ValueError(
            "Cannot save "
            "an empty sequence"
        )

    arrays = (
        frames_to_arrays(
            frames
        )
    )

    metadata.validate()

    # --------------------------------------------------------
    # Temporary sample directory
    # --------------------------------------------------------

    temp_dir = (
        sample_dir.with_name(
            sample_dir.name
            + ".tmp"
        )
    )

    if temp_dir.exists():

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

    temp_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:

        # ====================================================
        # LANDMARKS
        # ====================================================

        np.savez_compressed(
            temp_dir
            / "landmarks.npz",
            **arrays,
        )

        # ====================================================
        # METADATA
        # ====================================================

        metadata_path = (
            temp_dir
            / "metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.flush()

            os.fsync(
                file.fileno()
            )

        # ====================================================
        # VIDEO
        # ====================================================

        if (
            video_temp_path
            is not None
        ):

            video_temp_path = Path(
                video_temp_path
            )

            if not (
                video_temp_path.is_file()
            ):

                raise FileNotFoundError(
                    "Temporary video "
                    "does not exist: "
                    f"{video_temp_path}"
                )

            # Empty video = failed recording.
            video_size = (
                video_temp_path.stat()
                .st_size
            )

            if video_size <= 0:

                raise RuntimeError(
                    "Temporary video exists "
                    "but is empty: "
                    f"{video_temp_path}"
                )

            destination_video = (
                temp_dir
                / video_filename
            )

            shutil.move(
                str(
                    video_temp_path
                ),
                str(
                    destination_video
                ),
            )

        # ====================================================
        # FINAL COMMIT
        # ====================================================

        temp_dir.replace(
            sample_dir
        )

    except Exception:

        # Remove incomplete sample directory.
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

        raise


# ============================================================
# VERIFY SAVED SAMPLE
# ============================================================

def verify_saved_sample(
    sample_dir: Path,
    require_video: bool = False,
    video_filename: str = "source_video.mp4",
) -> dict[str, object]:

    sample_dir = Path(
        sample_dir
    )

    npz_path = (
        sample_dir
        / "landmarks.npz"
    )

    metadata_path = (
        sample_dir
        / "metadata.json"
    )

    video_path = (
        sample_dir
        / video_filename
    )

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    if not npz_path.is_file():

        raise FileNotFoundError(
            "Missing landmarks.npz: "
            f"{npz_path}"
        )

    if not metadata_path.is_file():

        raise FileNotFoundError(
            "Missing metadata.json: "
            f"{metadata_path}"
        )

    if (
        require_video
        and not video_path.is_file()
    ):

        raise FileNotFoundError(
            "Missing source video: "
            f"{video_path}"
        )

    if (
        require_video
        and video_path.stat().st_size
        <= 0
    ):

        raise RuntimeError(
            "Source video exists "
            "but is empty: "
            f"{video_path}"
        )

    # --------------------------------------------------------
    # Load landmarks
    # --------------------------------------------------------

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:

        arrays = {
            key: data[key]
            for key in data.files
        }

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = (
            json.load(
                file
            )
        )

    # --------------------------------------------------------
    # Required arrays
    # --------------------------------------------------------

    required_arrays = {
        "timestamps",
        "left_hand",
        "right_hand",
        "pose",
        "face",
        "left_present",
        "right_present",
        "pose_present",
        "face_present",
    }

    missing = (
        required_arrays
        .difference(
            arrays
        )
    )

    if missing:

        raise ValueError(
            "Missing arrays: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Sequence length
    # --------------------------------------------------------

    frame_count = int(
        arrays[
            "timestamps"
        ].shape[0]
    )

    if frame_count < 1:

        raise ValueError(
            "Saved sequence "
            "has zero frames"
        )

    # --------------------------------------------------------
    # Validate frame count
    # --------------------------------------------------------

    for key in (
        "left_hand",
        "right_hand",
        "left_present",
        "right_present",
        "pose_present",
        "face_present",
    ):

        if (
            arrays[key].shape[0]
            != frame_count
        ):

            raise ValueError(
                "Frame count mismatch "
                f"in {key}"
            )

    # --------------------------------------------------------
    # Hand shape
    # --------------------------------------------------------

    if (
        arrays[
            "left_hand"
        ].shape[1:]
        != (21, 3)
    ):

        raise ValueError(
            "left_hand must have "
            "shape (T, 21, 3)"
        )

    if (
        arrays[
            "right_hand"
        ].shape[1:]
        != (21, 3)
    ):

        raise ValueError(
            "right_hand must have "
            "shape (T, 21, 3)"
        )

    # --------------------------------------------------------
    # Metadata consistency
    # --------------------------------------------------------

    metadata_frame_count = int(
        metadata.get(
            "frame_count",
            -1,
        )
    )

    if (
        metadata_frame_count
        != frame_count
    ):

        raise ValueError(
            "metadata.frame_count "
            "does not match "
            "landmarks.npz"
        )

    # --------------------------------------------------------
    # Return verification report
    # --------------------------------------------------------

    return {
        "sample_id": (
            metadata.get(
                "sample_id"
            )
        ),

        "frame_count": (
            frame_count
        ),

        "sign": (
            metadata.get(
                "sign"
            )
        ),

        "subject_id": (
            metadata.get(
                "subject_id"
            )
        ),

        "session_id": (
            metadata.get(
                "session_id"
            )
        ),

        "video_saved": (
            video_path.is_file()
            and video_path.stat().st_size > 0
        ),

        "video_size_bytes": (
            video_path.stat().st_size
            if video_path.is_file()
            else 0
        ),
    }