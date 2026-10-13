from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


# ============================================================
# PROJECT ROOT BOOTSTRAP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from core.camera import CameraManager

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
            "Phase 1 - Collect sign-language "
            "sequence dataset"
        )
    )

    parser.add_argument(
        "--sign",
        required=True,
        help=(
            "Sign label, e.g. "
            "xin_chao, cam_on, khong, REST, UNKNOWN"
        ),
    )

    parser.add_argument(
        "--subject",
        required=True,
        help="Anonymous subject id, e.g. S01",
    )

    parser.add_argument(
        "--session",
        required=True,
        help="Session id, e.g. SES01",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of sequences to collect",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "fixed",
            "manual",
        ],
        default=None,
        help="Collection mode",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=(
            "Target frame count in fixed mode"
        ),
    )

    parser.add_argument(
        "--lighting",
        default=None,
        help=(
            "Lighting metadata: "
            "normal/bright/dim/backlight/..."
        ),
    )

    parser.add_argument(
        "--distance",
        default=None,
        help=(
            "Distance metadata: "
            "near/medium/far"
        ),
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Optional config YAML path",
    )

    return parser.parse_args()


# ============================================================
# UI HELPERS
# ============================================================

def draw_status(
    display,
    lines,
    *,
    recording: bool = False,
) -> None:
    """
    Draw status information on PREVIEW only.

    Dataset video itself remains raw:
        - no mirror
        - no text overlay
    """

    y = 32

    color = (
        (0, 0, 255)
        if recording
        else (0, 255, 0)
    )

    for text in lines:

        cv2.putText(
            display,
            str(text),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )

        y += 30


# ============================================================
# COUNTDOWN
# ============================================================

def countdown(
    cam: CameraManager,
    seconds: int,
    mirror: bool,
) -> bool:
    """
    Countdown before one sequence starts.

    Returns:
        True  -> continue recording
        False -> user requested quit
    """

    if seconds <= 0:
        return True

    end_time = (
        time.perf_counter()
        + seconds
    )

    while (
        time.perf_counter()
        < end_time
    ):

        item = cam.read()

        if item is None:
            continue

        remaining = max(
            1,
            int(
                end_time
                - time.perf_counter()
            )
            + 1,
        )

        display = (
            item.image_bgr.copy()
        )

        # Mirror only preview
        if mirror:

            display = cv2.flip(
                display,
                1,
            )

        h, w = (
            display.shape[:2]
        )

        cv2.putText(
            display,
            str(remaining),
            (
                w // 2 - 45,
                h // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (0, 0, 255),
            6,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "Prepare sign - Q/ESC to quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Phase 1 Sequence Collector",
            display,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key in (
            ord("q"),
            ord("Q"),
            27,
        ):
            return False

    return True


# ============================================================
# VIDEO WRITER
# ============================================================

def create_video_writer(
    path: Path,
    width: int,
    height: int,
    fps: float,
    codec: str = "mp4v",
):
    """
    Create OpenCV VideoWriter.

    IMPORTANT:
        Video dataset stores RAW frames:
            - not mirrored
            - no UI text
            - no landmark overlay
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if width <= 0:
        raise ValueError(
            "Video width must be > 0"
        )

    if height <= 0:
        raise ValueError(
            "Video height must be > 0"
        )

    if fps <= 0:
        raise ValueError(
            "Video FPS must be > 0"
        )

    if len(codec) != 4:
        raise ValueError(
            "video_codec must contain "
            "exactly 4 characters"
        )

    fourcc = (
        cv2.VideoWriter_fourcc(
            *codec
        )
    )

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        float(fps),
        (
            int(width),
            int(height),
        ),
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create "
            f"VideoWriter: {path}"
        )

    return writer


# ============================================================
# TEMP VIDEO CLEANUP
# ============================================================

def remove_temp_video(
    video_temp_path: Path | None,
) -> None:
    """
    Safely remove unfinished temporary video.
    """

    if video_temp_path is None:
        return

    video_temp_path = Path(
        video_temp_path
    )

    if video_temp_path.exists():

        try:
            video_temp_path.unlink()

        except OSError as exc:

            print(
                "WARNING: "
                "Could not remove temporary "
                f"video: {exc}"
            )


# ============================================================
# FIXED MODE
# ============================================================

def collect_fixed(
    cam,
    det,
    cfg,
    identity,
    settings,
    target_frames: int,
    video_temp_path: Path | None = None,
):
    """
    Fixed collection mode.

    Records exactly target_frames.

    Returns:
        (frames, quit_requested)
    """

    if target_frames < 1:

        raise ValueError(
            "target_frames must be >= 1"
        )

    frames = []

    mirror_preview = bool(
        cfg["camera"].get(
            "mirror_preview",
            True,
        )
    )

    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    video_writer = None

    if video_temp_path is not None:

        width, height = (
            settings.resolution
        )

        codec = (
            cfg["dataset"].get(
                "video_codec",
                "mp4v",
            )
        )

        video_writer = (
            create_video_writer(
                path=video_temp_path,
                width=width,
                height=height,
                fps=settings.target_fps,
                codec=codec,
            )
        )

    try:

        while (
            len(frames)
            < target_frames
        ):

            # ==================================================
            # READ CAMERA
            # ==================================================

            item = cam.read()

            if item is None:
                continue

            raw_frame = (
                item.image_bgr
            )

            # ==================================================
            # MEDIAPIPE DETECTION
            # ==================================================

            result = det.process(
                raw_frame,
                item.index,
                item.timestamp,
            )

            frames.append(
                result
            )

            # ==================================================
            # SAVE RAW VIDEO
            # ==================================================

            if (
                video_writer
                is not None
            ):

                video_writer.write(
                    raw_frame
                )

            # ==================================================
            # PREVIEW
            # ==================================================

            display = (
                raw_frame.copy()
            )

            if mirror_preview:

                display = cv2.flip(
                    display,
                    1,
                )

            presence = (
                result.presence
            )

            status_lines = [
                (
                    "RECORDING FIXED"
                ),

                (
                    f"Frames: "
                    f"{len(frames)}/"
                    f"{target_frames}"
                ),

                (
                    f"Sign: "
                    f"{identity.sign}"
                ),

                (
                    f"Subject: "
                    f"{identity.subject_id}    "
                    f"Session: "
                    f"{identity.session_id}"
                ),

                (
                    f"Left: "
                    f"{'YES' if presence.left_hand else 'NO'}    "
                    f"Right: "
                    f"{'YES' if presence.right_hand else 'NO'}"
                ),

                (
                    f"Pose: "
                    f"{'YES' if presence.pose else 'NO'}    "
                    f"Face: "
                    f"{'YES' if presence.face else 'NO'}"
                ),

                (
                    "Q / ESC = abort"
                ),
            ]

            draw_status(
                display,
                status_lines,
                recording=True,
            )

            cv2.imshow(
                "Phase 1 Sequence Collector",
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ==================================================
            # ABORT
            # ==================================================

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):

                print(
                    "\nFixed sample aborted."
                )

                return (
                    None,
                    True,
                )

    finally:

        # Important:
        # release before temp video is moved
        if video_writer is not None:
            video_writer.release()

    return (
        frames,
        False,
    )


# ============================================================
# MANUAL MODE
# ============================================================

def collect_manual(
    cam,
    det,
    cfg,
    identity,
    settings,
    video_temp_path: Path | None = None,
):
    """
    Manual collection mode.

    Controls:
        SPACE:
            finish sample after min_frames

        Q / ESC:
            abort current sample and stop collection

    Automatically finishes at max_frames.
    """

    frames = []

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

    if min_frames < 1:

        raise ValueError(
            "min_frames must be >= 1"
        )

    if max_frames < min_frames:

        raise ValueError(
            "max_frames must be >= min_frames"
        )

    mirror_preview = bool(
        cfg["camera"].get(
            "mirror_preview",
            True,
        )
    )

    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    video_writer = None

    if video_temp_path is not None:

        width, height = (
            settings.resolution
        )

        codec = (
            cfg["dataset"].get(
                "video_codec",
                "mp4v",
            )
        )

        video_writer = (
            create_video_writer(
                path=video_temp_path,
                width=width,
                height=height,
                fps=settings.target_fps,
                codec=codec,
            )
        )

    print()
    print(
        "MANUAL COLLECTION"
    )
    print(
        "------------------------------"
    )
    print(
        f"Sign       : "
        f"{identity.sign}"
    )
    print(
        f"Subject    : "
        f"{identity.subject_id}"
    )
    print(
        f"Session    : "
        f"{identity.session_id}"
    )
    print(
        f"Min frames : "
        f"{min_frames}"
    )
    print(
        f"Max frames : "
        f"{max_frames}"
    )
    print()
    print(
        "SPACE = finish sample"
    )
    print(
        "Q/ESC = abort and quit"
    )
    print()

    try:

        while True:

            # ==================================================
            # READ CAMERA
            # ==================================================

            item = cam.read()

            if item is None:
                continue

            raw_frame = (
                item.image_bgr
            )

            # ==================================================
            # MEDIAPIPE
            # ==================================================

            result = det.process(
                raw_frame,
                item.index,
                item.timestamp,
            )

            # ==================================================
            # LANDMARK SEQUENCE
            # ==================================================

            frames.append(
                result
            )

            # ==================================================
            # RAW VIDEO
            # ==================================================

            if (
                video_writer
                is not None
            ):

                video_writer.write(
                    raw_frame
                )

            # ==================================================
            # PREVIEW
            # ==================================================

            display = (
                raw_frame.copy()
            )

            if mirror_preview:

                display = cv2.flip(
                    display,
                    1,
                )

            presence = (
                result.presence
            )

            frame_count = len(
                frames
            )

            if (
                frame_count
                < min_frames
            ):

                finish_status = (
                    "SPACE locked: "
                    f"{frame_count}/"
                    f"{min_frames}"
                )

            else:

                finish_status = (
                    "SPACE = finish sample"
                )

            status_lines = [
                (
                    "RECORDING MANUAL"
                ),

                (
                    f"Frames: "
                    f"{frame_count}/"
                    f"{max_frames}"
                ),

                (
                    f"Sign: "
                    f"{identity.sign}"
                ),

                (
                    f"Subject: "
                    f"{identity.subject_id}    "
                    f"Session: "
                    f"{identity.session_id}"
                ),

                (
                    f"Left: "
                    f"{'YES' if presence.left_hand else 'NO'}    "
                    f"Right: "
                    f"{'YES' if presence.right_hand else 'NO'}"
                ),

                (
                    f"Pose: "
                    f"{'YES' if presence.pose else 'NO'}    "
                    f"Face: "
                    f"{'YES' if presence.face else 'NO'}"
                ),

                finish_status,

                (
                    "Q / ESC = abort"
                ),
            ]

            draw_status(
                display,
                status_lines,
                recording=True,
            )

            cv2.imshow(
                "Phase 1 Sequence Collector",
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ==================================================
            # ABORT
            # ==================================================

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):

                print(
                    "\nManual sample aborted."
                )

                return (
                    None,
                    True,
                )

            # ==================================================
            # FINISH WITH SPACE
            # ==================================================

            if key == ord(" "):

                if (
                    frame_count
                    < min_frames
                ):

                    print(
                        "Cannot finish yet: "
                        f"{frame_count}/"
                        f"{min_frames} frames."
                    )

                    continue

                print(
                    "\nManual sample "
                    f"finished with "
                    f"{frame_count} frames."
                )

                return (
                    frames,
                    False,
                )

            # ==================================================
            # AUTO FINISH
            # ==================================================

            if (
                frame_count
                >= max_frames
            ):

                print(
                    "\nReached maximum "
                    f"{max_frames} frames."
                )

                print(
                    "Manual sample "
                    "auto-finished."
                )

                return (
                    frames,
                    False,
                )

    finally:

        # Must be released before moving video.
        if video_writer is not None:
            video_writer.release()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    cfg = load_config(
        args.config
    )
    print("\nCONFIG CHECK")
    print("save_video     :", cfg["dataset"].get("save_video"))
    print("video_filename :", cfg["dataset"].get("video_filename"))
    print("video_codec    :", cfg["dataset"].get("video_codec"))
    print()
    # ========================================================
    # VALIDATE CLI INPUT
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

    if args.samples < 1:

        raise SystemExit(
            "--samples must be >= 1"
        )

    # ========================================================
    # IDENTITY
    # ========================================================

    identity = CollectionIdentity(
        sign=args.sign,
        subject_id=args.subject,
        session_id=args.session,
    )

    identity.validate()

    # ========================================================
    # COLLECTION CONFIG
    # ========================================================

    collection_cfg = (
        cfg["collection"]
    )

    mode = (
        args.mode
        or collection_cfg.get(
            "default_mode",
            "fixed",
        )
    )

    if mode not in {
        "fixed",
        "manual",
    }:

        raise SystemExit(
            "Collection mode must be "
            "'fixed' or 'manual'"
        )

    fixed_frames = int(
        args.frames
        or collection_cfg.get(
            "fixed_frames",
            45,
        )
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

    if min_frames < 1:

        raise SystemExit(
            "collection.min_frames "
            "must be >= 1"
        )

    if max_frames < min_frames:

        raise SystemExit(
            "collection.max_frames "
            "must be >= min_frames"
        )

    if (
        mode == "fixed"
        and not (
            min_frames
            <= fixed_frames
            <= max_frames
        )
    ):

        raise SystemExit(
            "fixed frames must be "
            f"inside configured range "
            f"{min_frames}-{max_frames}"
        )

    # ========================================================
    # DATASET PATH
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

    # ========================================================
    # CAMERA CONFIG
    # ========================================================

    camera_cfg = (
        cfg["camera"]
    )

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

        camera_id=int(
            camera_cfg["id"]
        ),

        resolution=(
            int(
                camera_cfg["width"]
            ),
            int(
                camera_cfg["height"]
            ),
        ),

        target_fps=float(
            camera_cfg["fps"]
        ),

        mode=mode,

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

        save_video=bool(
            cfg["dataset"].get(
                "save_video",
                True,
            )
        ),
    )

    settings.validate()

    # ========================================================
    # SAMPLE INDEX
    # ========================================================

    start_index = next_sample_index(
        session_dir,
        identity.sign,
        identity.subject_id,
        identity.session_id,
    )

    # ========================================================
    # START INFO
    # ========================================================

    print()
    print(
        "PHASE 1 SEQUENCE COLLECTION"
    )
    print(
        "======================================"
    )
    print(
        f"Sign        : "
        f"{identity.sign}"
    )
    print(
        f"Subject     : "
        f"{identity.subject_id}"
    )
    print(
        f"Session     : "
        f"{identity.session_id}"
    )
    print(
        f"Mode        : "
        f"{mode}"
    )
    print(
        f"Samples     : "
        f"{args.samples}"
    )

    if mode == "fixed":

        print(
            f"Frames      : "
            f"{fixed_frames}/sample"
        )

    else:

        print(
            f"Frame range : "
            f"{min_frames}-"
            f"{max_frames}"
        )

    print(
        f"Start index : "
        f"{start_index:03d}"
    )

    print(
        f"Save video  : "
        f"{settings.save_video}"
    )

    print(
        f"Output      : "
        f"{session_dir}"
    )

    print(
        "======================================"
    )
    print()

    saved = 0

    quit_requested = False

    # ========================================================
    # CAMERA + DETECTOR
    # ========================================================

    try:

        with (
            CameraManager(
                int(
                    camera_cfg["id"]
                ),
                int(
                    camera_cfg["width"]
                ),
                int(
                    camera_cfg["height"]
                ),
                int(
                    camera_cfg["fps"]
                ),
            )
            as cam,
            MediaPipeDetector(
                cfg["mediapipe"]
            )
            as detector,
        ):

            # Actual camera resolution
            actual_resolution = (
                cam.resolution
            )

            settings = CollectionSettings(
                **{
                    **settings.__dict__,
                    "resolution":
                        actual_resolution,
                }
            )

            # ==================================================
            # SAMPLE LOOP
            # ==================================================

            for offset in range(
                args.samples
            ):

                sample_index = (
                    start_index
                    + offset
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

                print(
                    f"[{offset + 1}/"
                    f"{args.samples}] "
                    f"Ready: "
                    f"{sample_id}"
                )

                # ==============================================
                # VIDEO CONFIG
                # ==============================================

                save_video = bool(
                    cfg["dataset"].get(
                        "save_video",
                        True,
                    )
                )

                video_filename = (
                    cfg["dataset"].get(
                        "video_filename",
                        "source_video.mp4",
                    )
                )

                video_temp_path = None

                if save_video:

                    # Temporary file outside sample_dir.
                    #
                    # Example:
                    # .xin_chao_S01_SES01_001_recording.mp4
                    #
                    # Only moved into the sample directory
                    # after landmarks + metadata save succeeds.

                    video_temp_path = (
                        session_dir
                        / (
                            f".{sample_id}"
                            "_recording.mp4"
                        )
                    )

                    # Remove stale temp file
                    # from a previous crash.
                    remove_temp_video(
                        video_temp_path
                    )

                # ==============================================
                # COUNTDOWN
                # ==============================================

                countdown_ok = countdown(
                    cam=cam,

                    seconds=int(
                        collection_cfg.get(
                            "countdown_seconds",
                            3,
                        )
                    ),

                    mirror=bool(
                        camera_cfg.get(
                            "mirror_preview",
                            True,
                        )
                    ),
                )

                if not countdown_ok:

                    quit_requested = True

                    remove_temp_video(
                        video_temp_path
                    )

                    break

                # ==============================================
                # RECORD SAMPLE
                # ==============================================

                if mode == "fixed":

                    (
                        frames,
                        quit_requested,
                    ) = collect_fixed(
                        cam=cam,
                        det=detector,
                        cfg=cfg,
                        identity=identity,
                        settings=settings,
                        target_frames=(
                            fixed_frames
                        ),
                        video_temp_path=(
                            video_temp_path
                        ),
                    )

                else:

                    (
                        frames,
                        quit_requested,
                    ) = collect_manual(
                        cam=cam,
                        det=detector,
                        cfg=cfg,
                        identity=identity,
                        settings=settings,
                        video_temp_path=(
                            video_temp_path
                        ),
                    )

                # ==============================================
                # ABORT
                # ==============================================

                if (
                    quit_requested
                    or frames is None
                ):

                    remove_temp_video(
                        video_temp_path
                    )

                    print(
                        "Sample aborted; "
                        "nothing was saved."
                    )

                    break

                # ==============================================
                # ACTUAL FPS
                # ==============================================

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

                    actual_fps = float(
                        cam.stats.actual_fps
                    )

                # Guard against invalid FPS.
                if (
                    actual_fps <= 0
                ):

                    actual_fps = float(
                        settings.target_fps
                    )

                # ==============================================
                # METADATA
                # ==============================================

                metadata = build_metadata(
                    frames=frames,
                    sample_id=sample_id,
                    identity=identity,
                    settings=settings,
                    actual_fps=actual_fps,
                )

                # ==============================================
                # ATOMIC SAVE
                # ==============================================

                try:

                    atomic_save_sequence(
                        sample_dir=(
                            sample_dir
                        ),

                        frames=frames,

                        metadata=metadata,

                        video_temp_path=(
                            video_temp_path
                            if save_video
                            else None
                        ),

                        video_filename=(
                            video_filename
                        ),
                    )

                except Exception:

                    # If atomic save failed before
                    # moving the temporary video,
                    # remove it.

                    remove_temp_video(
                        video_temp_path
                    )

                    raise

                # ==============================================
                # VERIFY
                # ==============================================

                if bool(
                    collection_cfg.get(
                        "auto_verify_after_save",
                        True,
                    )
                ):

                    report = (
                        verify_saved_sample(
                            sample_dir,
                            require_video=save_video,
                            video_filename=video_filename,
                        )
                    )

                    print(
                        "SAVED + VERIFIED:"
                    )

                    print(
                        report
                    )

                else:

                    print(
                        f"SAVED: "
                        f"{sample_dir}"
                    )

                saved += 1

                # ==============================================
                # COOLDOWN
                # ==============================================

                cooldown = float(
                    collection_cfg.get(
                        "cooldown_seconds",
                        1.0,
                    )
                )

                if cooldown > 0:

                    time.sleep(
                        cooldown
                    )

    finally:

        cv2.destroyAllWindows()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "COLLECTION FINISHED"
    )
    print(
        f"Saved: "
        f"{saved}/"
        f"{args.samples}"
    )

    if quit_requested:

        print(
            "Collection stopped "
            "by user."
        )

    print(
        "======================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()