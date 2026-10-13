from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from core.collector import (
    CollectionIdentity,
    CollectionSettings,
    atomic_save_sequence,
    build_metadata,
    make_sample_dir,
    next_sample_index,
    verify_saved_sample,
)

from core.config import (
    load_config,
    resolve_project_path,
)

from core.dataset_contract import (
    validate_session_id,
    validate_sign_name,
    validate_subject_id,
)

from core.detector import MediaPipeDetector


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import one isolated sign video "
            "into Phase-1 Dataset V2."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path.",
    )

    parser.add_argument(
        "--sign",
        required=True,
        help=(
            "xin_chao, cam_on, khong, "
            "REST, UNKNOWN, ..."
        ),
    )

    parser.add_argument(
        "--subject",
        required=True,
        help="Example: S01",
    )

    parser.add_argument(
        "--session",
        required=True,
        help="Example: SES01",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Trim start time in seconds.",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Trim end time in seconds.",
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "FPS of trimmed video and "
            "landmark sequence."
        ),
    )

    parser.add_argument(
        "--lighting",
        default=None,
    )

    parser.add_argument(
        "--distance",
        default=None,
    )

    parser.add_argument(
        "--preview",
        action="store_true",
    )

    parser.add_argument(
        "--allow-outside-range",
        action="store_true",
    )

    parser.add_argument(
        "--no-save-video",
        action="store_true",
    )

    parser.add_argument(
        "--config",
        default=None,
    )

    return parser.parse_args()


# ============================================================
# SHA256
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            hasher.update(
                chunk
            )

    return hasher.hexdigest()


# ============================================================
# CHECK FFMPEG
# ============================================================

def check_ffmpeg() -> None:

    ffmpeg = shutil.which(
        "ffmpeg"
    )

    if ffmpeg is None:

        raise RuntimeError(
            "FFmpeg was not found.\n"
            "Run: ffmpeg -version\n"
            "and install/add FFmpeg "
            "to PATH first."
        )


# ============================================================
# DELETE TEMP
# ============================================================

def remove_file(
    path: Path | None,
) -> None:

    if path is None:
        return

    path = Path(path)

    if path.exists():

        try:
            path.unlink()

        except OSError as exc:

            print(
                "WARNING: cannot remove "
                f"{path}: {exc}"
            )


# ============================================================
# READ SOURCE INFORMATION
# ============================================================

def read_video_info(
    video_path: Path,
) -> dict:

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "OpenCV cannot open video: "
            f"{video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 0.0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        or 0
    )

    cap.release()

    if fps <= 0:

        raise RuntimeError(
            "Could not determine "
            "source video FPS."
        )

    duration = (
        frame_count / fps
        if frame_count > 0
        else 0.0
    )

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration": duration,
    }


# ============================================================
# FFMPEG TRIM + FPS NORMALIZATION
# ============================================================

def create_trimmed_video(
    *,
    source_path: Path,
    output_path: Path,
    start_s: float,
    end_s: float,
    target_fps: float,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    remove_file(
        output_path
    )

    duration = (
        end_s
        - start_s
    )

    if duration <= 0:

        raise ValueError(
            "Trim duration must be > 0"
        )

    # Important:
    # -ss is placed after -i for more
    # accurate decoding/trim.
    #
    # fps filter ensures the video used
    # for landmarks has exactly target FPS.

    command = [
        "ffmpeg",
        "-y",

        "-i",
        str(source_path),

        "-ss",
        f"{start_s:.6f}",

        "-t",
        f"{duration:.6f}",

        "-vf",
        f"fps={target_fps}",

        "-an",

        "-c:v",
        "libx264",

        "-preset",
        "fast",

        "-crf",
        "18",

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        str(output_path),
    ]

    print()
    print(
        "Creating trimmed video..."
    )

    print(
        f"Start      : "
        f"{start_s:.3f}s"
    )

    print(
        f"End        : "
        f"{end_s:.3f}s"
    )

    print(
        f"Duration   : "
        f"{duration:.3f}s"
    )

    print(
        f"Target FPS : "
        f"{target_fps:.3f}"
    )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        remove_file(
            output_path
        )

        raise RuntimeError(
            "FFmpeg trimming failed:\n"
            + result.stderr
        )

    if not output_path.is_file():

        raise RuntimeError(
            "FFmpeg finished but "
            "output video is missing: "
            f"{output_path}"
        )

    if (
        output_path.stat().st_size
        <= 0
    ):

        remove_file(
            output_path
        )

        raise RuntimeError(
            "Trimmed video is empty."
        )


# ============================================================
# PREVIEW
# ============================================================

def show_preview(
    image,
    *,
    result,
    frame_count: int,
    identity: CollectionIdentity,
    mirror: bool,
) -> bool:

    display = (
        image.copy()
    )

    if mirror:

        display = cv2.flip(
            display,
            1,
        )

    presence = (
        result.presence
    )

    lines = [
        (
            f"VIDEO IMPORT "
            f"{frame_count}"
        ),

        (
            f"{identity.sign} | "
            f"{identity.subject_id} | "
            f"{identity.session_id}"
        ),

        (
            f"Left:"
            f"{'YES' if presence.left_hand else 'NO'} "
            f"Right:"
            f"{'YES' if presence.right_hand else 'NO'}"
        ),

        (
            f"Pose:"
            f"{'YES' if presence.pose else 'NO'} "
            f"Face:"
            f"{'YES' if presence.face else 'NO'}"
        ),

        "Q / ESC = cancel",
    ]

    y = 32

    for text in lines:

        cv2.putText(
            display,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        y += 30

    cv2.imshow(
        "Phase 1 Video Import",
        display,
    )

    key = (
        cv2.waitKey(1)
        & 0xFF
    )

    return key not in (
        ord("q"),
        ord("Q"),
        27,
    )


# ============================================================
# EXTRACT LANDMARKS FROM TRIMMED VIDEO
# ============================================================

def extract_landmarks(
    *,
    video_path: Path,
    cfg: dict,
    identity: CollectionIdentity,
    preview: bool,
) -> tuple[list, float, int, int]:

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Cannot open trimmed video: "
            f"{video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:

        cap.release()

        raise RuntimeError(
            "Trimmed video FPS "
            "could not be read."
        )

    frames = []

    mirror_preview = bool(
        cfg["camera"].get(
            "mirror_preview",
            True,
        )
    )

    cancelled = False

    try:

        with MediaPipeDetector(
            cfg["mediapipe"]
        ) as detector:

            frame_index = 0

            while True:

                ok, image = (
                    cap.read()
                )

                if not ok:
                    break

                # Use deterministic timestamps
                # based on the normalized video FPS.
                timestamp = (
                    frame_index
                    / fps
                )

                result = (
                    detector.process(
                        image_bgr=image,
                        frame_index=frame_index,
                        timestamp=timestamp,
                    )
                )

                frames.append(
                    result
                )

                if preview:

                    keep_running = (
                        show_preview(
                            image,
                            result=result,
                            frame_count=(
                                len(frames)
                            ),
                            identity=identity,
                            mirror=(
                                mirror_preview
                            ),
                        )
                    )

                    if not keep_running:

                        cancelled = True
                        break

                frame_index += 1

    finally:

        cap.release()

        cv2.destroyAllWindows()

    if cancelled:

        raise KeyboardInterrupt

    return (
        frames,
        fps,
        width,
        height,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    cfg = load_config(
        args.config
    )

    check_ffmpeg()

    # ========================================================
    # SOURCE VIDEO
    # ========================================================

    source_video = (
        Path(args.video)
        .expanduser()
        .resolve()
    )

    if not source_video.is_file():

        raise SystemExit(
            "Video not found: "
            f"{source_video}"
        )

    # ========================================================
    # VALIDATE IDS
    # ========================================================

    validate_sign_name(
        args.sign
    )

    validate_subject_id(
        args.subject
    )

    validate_session_id(
        args.session
    )

    if args.start < 0:

        raise SystemExit(
            "--start must be >= 0"
        )

    identity = CollectionIdentity(
        sign=args.sign,
        subject_id=args.subject,
        session_id=args.session,
    )

    identity.validate()

    # ========================================================
    # SOURCE INFORMATION
    # ========================================================

    source_info = (
        read_video_info(
            source_video
        )
    )

    native_fps = (
        source_info["fps"]
    )

    source_duration = (
        source_info["duration"]
    )

    if (
        source_duration > 0
        and args.start >= source_duration
    ):

        raise SystemExit(
            f"--start={args.start:.3f}s "
            "is outside source video "
            f"duration={source_duration:.3f}s"
        )

    if args.end is None:

        end_s = (
            source_duration
        )

    else:

        end_s = float(
            args.end
        )

    if source_duration > 0:

        end_s = min(
            end_s,
            source_duration,
        )

    if end_s <= args.start:

        raise SystemExit(
            "--end must be greater "
            "than --start"
        )

    # ========================================================
    # TARGET FPS
    # ========================================================

    target_fps = float(
        args.target_fps
        or cfg["camera"]["fps"]
    )

    if target_fps <= 0:

        raise SystemExit(
            "--target-fps must be > 0"
        )

    # No artificial upsampling.
    target_fps = min(
        target_fps,
        native_fps,
    )

    # ========================================================
    # OUTPUT PATH
    # ========================================================

    raw_root = (
        resolve_project_path(
            cfg["dataset"][
                "raw_root"
            ]
        )
    )

    session_dir = (
        raw_root
        / identity.sign
        / identity.subject_id
        / identity.session_id
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_index = (
        next_sample_index(
            session_dir,
            identity.sign,
            identity.subject_id,
            identity.session_id,
        )
    )

    sample_dir = (
        make_sample_dir(
            raw_root,
            identity,
            sample_index,
        )
    )

    sample_id = (
        sample_dir.name
    )

    # ========================================================
    # VIDEO SAVE CONFIG
    # ========================================================

    save_video = bool(
        cfg["dataset"].get(
            "save_video",
            True,
        )
    )

    if args.no_save_video:

        save_video = False

    video_filename = (
        cfg["dataset"].get(
            "video_filename",
            "source_video.mp4",
        )
    )

    # Temporary trimmed video.
    temp_video = (
        session_dir
        / (
            f".{sample_id}"
            "_trimmed.mp4"
        )
    )

    remove_file(
        temp_video
    )

    # ========================================================
    # CREATE EXACT TRIMMED VIDEO
    # ========================================================

    try:

        create_trimmed_video(
            source_path=(
                source_video
            ),
            output_path=(
                temp_video
            ),
            start_s=(
                args.start
            ),
            end_s=(
                end_s
            ),
            target_fps=(
                target_fps
            ),
        )

        # ====================================================
        # READ BACK TRIMMED VIDEO
        # ====================================================

        trimmed_info = (
            read_video_info(
                temp_video
            )
        )

        print()
        print(
            "TRIMMED VIDEO"
        )

        print(
            "======================================"
        )

        print(
            f"File       : "
            f"{temp_video}"
        )

        print(
            f"FPS        : "
            f"{trimmed_info['fps']:.3f}"
        )

        print(
            f"Frames     : "
            f"{trimmed_info['frame_count']}"
        )

        print(
            f"Resolution : "
            f"{trimmed_info['width']}x"
            f"{trimmed_info['height']}"
        )

        print(
            f"Duration   : "
            f"{trimmed_info['duration']:.3f}s"
        )

        print(
            "======================================"
        )

        # ====================================================
        # LANDMARK EXTRACTION
        # ====================================================

        (
            frames,
            actual_video_fps,
            width,
            height,
        ) = extract_landmarks(
            video_path=(
                temp_video
            ),
            cfg=cfg,
            identity=identity,
            preview=args.preview,
        )

    except KeyboardInterrupt:

        remove_file(
            temp_video
        )

        print(
            "Import cancelled. "
            "Nothing saved."
        )

        return

    except Exception:

        remove_file(
            temp_video
        )

        raise

    # ========================================================
    # VALIDATE FRAME COUNT
    # ========================================================

    if not frames:

        remove_file(
            temp_video
        )

        raise SystemExit(
            "No landmark frames extracted."
        )

    collection_cfg = (
        cfg["collection"]
    )

    min_frames = int(
        collection_cfg.get(
            "min_frames",
            30,
        )
    )

    max_frames = int(
        collection_cfg.get(
            "max_frames",
            60,
        )
    )

    if (
        not args.allow_outside_range
        and not (
            min_frames
            <= len(frames)
            <= max_frames
        )
    ):

        count = len(
            frames
        )

        remove_file(
            temp_video
        )

        raise SystemExit(
            f"Extracted {count} frames, "
            "but Phase 1 expects "
            f"{min_frames}-{max_frames}.\n"
            "Adjust --start/--end or "
            "--target-fps."
        )

    # ========================================================
    # ACTUAL FPS
    # ========================================================

    if (
        len(frames) > 1
        and (
            frames[-1].timestamp
            >
            frames[0].timestamp
        )
    ):

        actual_fps = (
            len(frames) - 1
        ) / (
            frames[-1].timestamp
            -
            frames[0].timestamp
        )

    else:

        actual_fps = (
            actual_video_fps
        )

    # ========================================================
    # SETTINGS
    # ========================================================

    settings = CollectionSettings(
        dataset_version=(
            cfg["project"][
                "dataset_version"
            ]
        ),

        schema_version=(
            cfg["project"][
                "schema_version"
            ]
        ),

        camera_id=-1,

        resolution=(
            width,
            height,
        ),

        target_fps=(
            actual_video_fps
        ),

        mode=(
            "video_import"
        ),

        lighting=(
            args.lighting
            or collection_cfg.get(
                "lighting_default",
                "normal",
            )
        ),

        distance=(
            args.distance
            or collection_cfg.get(
                "distance_default",
                "medium",
            )
        ),

        save_video=(
            save_video
        ),
    )

    settings.validate()

    # ========================================================
    # METADATA
    # ========================================================

    metadata = (
        build_metadata(
            frames=frames,
            sample_id=sample_id,
            identity=identity,
            settings=settings,
            actual_fps=actual_fps,
        )
    )

    # ========================================================
    # SAVE SAMPLE
    # ========================================================

    try:

        atomic_save_sequence(
            sample_dir=sample_dir,
            frames=frames,
            metadata=metadata,

            video_temp_path=(
                temp_video
                if save_video
                else None
            ),

            video_filename=(
                video_filename
            ),
        )

    except Exception:

        remove_file(
            temp_video
        )

        raise

    # If video saving is disabled,
    # atomic_save_sequence does not move it.
    if not save_video:

        remove_file(
            temp_video
        )

    # ========================================================
    # PROVENANCE
    # ========================================================

    provenance = {
        "source_type":
            "video_import",

        "source_filename":
            source_video.name,

        "source_sha256":
            sha256_file(
                source_video
            ),

        "source_native_fps":
            native_fps,

        "source_native_frame_count":
            source_info[
                "frame_count"
            ],

        "source_resolution": [
            source_info[
                "width"
            ],
            source_info[
                "height"
            ],
        ],

        "source_duration_s":
            source_duration,

        "trim_start_s":
            args.start,

        "trim_end_s":
            end_s,

        "trim_duration_s":
            end_s
            - args.start,

        "target_fps":
            target_fps,

        "saved_video_fps":
            actual_video_fps,

        "extracted_frame_count":
            len(frames),

        "saved_video":
            save_video,

        "saved_video_filename":
            (
                video_filename
                if save_video
                else None
            ),
    }

    provenance_path = (
        sample_dir
        / "source_video.json"
    )

    with provenance_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            provenance,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # VERIFY
    # ========================================================

    report = (
        verify_saved_sample(
            sample_dir,
            require_video=(
                save_video
            ),
            video_filename=(
                video_filename
            ),
        )
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print(
        "======================================"
    )

    print(
        "IMPORT SUCCESS"
    )

    print(
        "======================================"
    )

    print(
        f"Sample      : "
        f"{sample_id}"
    )

    print(
        f"Frames      : "
        f"{len(frames)}"
    )

    print(
        f"FPS         : "
        f"{actual_video_fps:.3f}"
    )

    print(
        f"Trim        : "
        f"{args.start:.3f}s "
        f"-> {end_s:.3f}s"
    )

    print(
        f"Duration    : "
        f"{end_s - args.start:.3f}s"
    )

    print(
        f"Video saved : "
        f"{save_video}"
    )

    if save_video:

        final_video = (
            sample_dir
            / video_filename
        )

        print(
            f"Video       : "
            f"{final_video}"
        )

        print(
            f"Video size  : "
            f"{final_video.stat().st_size} "
            "bytes"
        )

    print(
        f"Output      : "
        f"{sample_dir}"
    )

    print()
    print(
        report
    )

    print(
        "======================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()