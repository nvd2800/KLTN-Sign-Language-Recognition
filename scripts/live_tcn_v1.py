from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_phase5 import load_phase5_config, build_phase5_features
from core.feature_phase6 import load_phase6_config, build_phase6_features


# ============================================================
# COLORS (OpenCV BGR)
# ============================================================

COLOR_BG = (12, 18, 28)
COLOR_PANEL = (18, 25, 38)
COLOR_PANEL_BORDER = (75, 95, 120)
COLOR_TEXT = (245, 245, 245)

COLOR_LEFT = (255, 80, 20)        # blue
COLOR_LEFT_CONN = (255, 165, 80)

COLOR_RIGHT = (0, 120, 255)       # orange
COLOR_RIGHT_CONN = (0, 190, 255)

COLOR_POSE = (0, 220, 90)         # green
COLOR_POSE_CONN = (80, 255, 150)

COLOR_FACE = (190, 70, 255)       # purple
COLOR_FACE_CONN = (225, 130, 255)

COLOR_OK = (60, 230, 100)
COLOR_WARN = (0, 220, 255)
COLOR_BAD = (70, 70, 255)
COLOR_ACCENT = (255, 220, 80)


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_hand(hand, eps=1e-6):
    if hand is None or np.asarray(hand).shape != (21, 3):
        return np.zeros((21, 3), np.float32), False, 0.0

    hand = np.asarray(hand, np.float32)

    if not np.isfinite(hand).all():
        return np.zeros((21, 3), np.float32), False, 0.0

    scale = float(np.linalg.norm(hand[9] - hand[0]))

    if not np.isfinite(scale) or scale <= eps:
        return np.zeros((21, 3), np.float32), False, 0.0

    out = (hand - hand[0]) / scale

    if not np.isfinite(out).all():
        return np.zeros((21, 3), np.float32), False, scale

    return out.astype(np.float32), True, scale


def lm_np(lst, count):
    if lst is None:
        return np.zeros((count, 3), np.float32), False

    arr = np.asarray(
        [[x.x, x.y, x.z] for x in lst.landmark],
        dtype=np.float32,
    )

    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((count, 3), np.float32), False

    return arr[:, :3], True


def append_csv(path, row, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    exists = path.is_file()

    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def put_text(
    frame,
    text,
    xy,
    color=COLOR_TEXT,
    scale=0.65,
    thickness=2,
):
    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def alpha_rect(frame, p1, p2, color, alpha=0.78, border=None):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        p1,
        p2,
        color,
        -1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
        frame,
    )

    if border is not None:
        cv2.rectangle(
            frame,
            p1,
            p2,
            border,
            1,
        )


def fit_text_scale(width):
    if width >= 1600:
        return 0.82
    if width >= 1280:
        return 0.72
    if width >= 960:
        return 0.62
    return 0.52


# ============================================================
# DRAWING SPECS
# ============================================================

def create_specs(mp_drawing):
    return {
        "left_lm": mp_drawing.DrawingSpec(
            color=COLOR_LEFT,
            thickness=3,
            circle_radius=4,
        ),
        "left_conn": mp_drawing.DrawingSpec(
            color=COLOR_LEFT_CONN,
            thickness=3,
            circle_radius=1,
        ),
        "right_lm": mp_drawing.DrawingSpec(
            color=COLOR_RIGHT,
            thickness=3,
            circle_radius=4,
        ),
        "right_conn": mp_drawing.DrawingSpec(
            color=COLOR_RIGHT_CONN,
            thickness=3,
            circle_radius=1,
        ),
        "pose_lm": mp_drawing.DrawingSpec(
            color=COLOR_POSE,
            thickness=3,
            circle_radius=4,
        ),
        "pose_conn": mp_drawing.DrawingSpec(
            color=COLOR_POSE_CONN,
            thickness=3,
            circle_radius=1,
        ),
        "face_conn": mp_drawing.DrawingSpec(
            color=COLOR_FACE_CONN,
            thickness=2,
            circle_radius=1,
        ),
    }


# ============================================================
# RESPONSIVE UI
# ============================================================

def draw_status_badge(frame, x, y, label, detected, color, scale):
    badge_w = int(185 * scale / 0.65)
    badge_h = int(34 * scale / 0.65)

    fill = color if detected else COLOR_BAD

    alpha_rect(
        frame,
        (x, y - badge_h + 4),
        (x + badge_w, y + 6),
        fill,
        alpha=0.70,
        border=(230, 230, 230),
    )

    put_text(
        frame,
        f"{label}: {'ON' if detected else 'OFF'}",
        (x + 10, y - 4),
        color=(255, 255, 255),
        scale=scale,
        thickness=2,
    )


def draw_probability_bars(frame, labels, probs, x, y, width, scale):
    if probs is None:
        return

    bar_h = max(18, int(24 * scale / 0.65))
    gap = 8

    for i, name in enumerate(labels):
        p = float(probs[i])
        py = y + i * (bar_h + gap)

        put_text(
            frame,
            name,
            (x, py + bar_h - 4),
            color=COLOR_TEXT,
            scale=max(0.42, scale * 0.72),
            thickness=1,
        )

        bar_x = x + int(110 * scale / 0.65)
        bar_w = max(120, width - (bar_x - x) - 70)

        cv2.rectangle(
            frame,
            (bar_x, py),
            (bar_x + bar_w, py + bar_h),
            (55, 65, 80),
            -1,
        )

        fill_w = int(bar_w * max(0.0, min(1.0, p)))

        cv2.rectangle(
            frame,
            (bar_x, py),
            (bar_x + fill_w, py + bar_h),
            COLOR_ACCENT,
            -1,
        )

        put_text(
            frame,
            f"{p * 100:5.1f}%",
            (bar_x + bar_w + 8, py + bar_h - 4),
            color=COLOR_TEXT,
            scale=max(0.42, scale * 0.72),
            thickness=1,
        )


def draw_ui(
    frame,
    *,
    labels,
    probs,
    pred,
    confidence,
    inference_ms,
    buffer_size,
    target_frames,
    fps,
    left_present,
    right_present,
    pose_present,
    face_present,
    fullscreen,
):
    h, w = frame.shape[:2]
    scale = fit_text_scale(w)

    top_h = max(170, int(h * 0.24))
    side_w = max(280, int(w * 0.27))

    # Top translucent panel
    alpha_rect(
        frame,
        (0, 0),
        (w, top_h),
        COLOR_BG,
        alpha=0.82,
        border=COLOR_PANEL_BORDER,
    )

    put_text(
        frame,
        f"TCN V1 LIVE | BUFFER {buffer_size}/{target_frames}",
        (22, 36),
        color=COLOR_TEXT,
        scale=scale,
        thickness=2,
    )

    put_text(
        frame,
        f"FPS {fps:.1f} | Inference {inference_ms:.1f} ms",
        (22, 72),
        color=(210, 220, 235),
        scale=max(0.48, scale * 0.80),
        thickness=1,
    )

    if pred is None:
        put_text(
            frame,
            "Prediction: warming up...",
            (22, 122),
            color=COLOR_WARN,
            scale=max(0.70, scale * 1.10),
            thickness=2,
        )
    else:
        pred_color = COLOR_OK if confidence >= 0.70 else COLOR_WARN

        put_text(
            frame,
            f"Prediction: {pred}",
            (22, 122),
            color=pred_color,
            scale=max(0.82, scale * 1.25),
            thickness=3,
        )

        put_text(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (22, 158),
            color=pred_color,
            scale=max(0.58, scale * 0.90),
            thickness=2,
        )

    # Landmark status badges
    badge_x = int(w * 0.52)
    badge_y = 42
    badge_gap_x = max(190, int(210 * scale / 0.65))
    badge_gap_y = 46

    draw_status_badge(
        frame,
        badge_x,
        badge_y,
        "L-HAND",
        left_present,
        COLOR_LEFT,
        max(0.50, scale * 0.78),
    )
    draw_status_badge(
        frame,
        badge_x + badge_gap_x,
        badge_y,
        "R-HAND",
        right_present,
        COLOR_RIGHT,
        max(0.50, scale * 0.78),
    )
    draw_status_badge(
        frame,
        badge_x,
        badge_y + badge_gap_y,
        "POSE",
        pose_present,
        COLOR_POSE,
        max(0.50, scale * 0.78),
    )
    draw_status_badge(
        frame,
        badge_x + badge_gap_x,
        badge_y + badge_gap_y,
        "FACE",
        face_present,
        COLOR_FACE,
        max(0.50, scale * 0.78),
    )

    # Side panel with probabilities
    if probs is not None and w >= 900:
        panel_x = w - side_w
        panel_y = top_h + 12
        panel_bottom = h - 54

        alpha_rect(
            frame,
            (panel_x, panel_y),
            (w - 8, panel_bottom),
            COLOR_PANEL,
            alpha=0.78,
            border=COLOR_PANEL_BORDER,
        )

        put_text(
            frame,
            "CLASS PROBABILITIES",
            (panel_x + 16, panel_y + 30),
            color=COLOR_TEXT,
            scale=max(0.48, scale * 0.78),
            thickness=2,
        )

        draw_probability_bars(
            frame,
            labels,
            probs,
            panel_x + 16,
            panel_y + 55,
            side_w - 24,
            max(0.46, scale * 0.72),
        )

    # Footer
    footer_h = 42
    alpha_rect(
        frame,
        (0, h - footer_h),
        (w, h),
        COLOR_BG,
        alpha=0.86,
        border=COLOR_PANEL_BORDER,
    )

    footer = (
        "1..9 = log ground truth | R = reset buffer | "
        "F = fullscreen | +/- = resize | Q/ESC = quit"
    )

    put_text(
        frame,
        footer,
        (14, h - 14),
        color=COLOR_TEXT,
        scale=max(0.42, scale * 0.68),
        thickness=1,
    )

    if fullscreen:
        put_text(
            frame,
            "FULLSCREEN",
            (w - 145, h - 14),
            color=COLOR_ACCENT,
            scale=max(0.42, scale * 0.68),
            thickness=1,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--experiment",
        default="tcn_v1_160samples",
    )

    ap.add_argument(
        "--log",
        default="logs/phase7_live_tcn_v1.csv",
    )

    ap.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    ap.add_argument(
        "--height",
        type=int,
        default=720,
    )

    ap.add_argument(
        "--window-width",
        type=int,
        default=1280,
    )

    ap.add_argument(
        "--window-height",
        type=int,
        default=800,
    )

    ap.add_argument(
        "--fullscreen",
        action="store_true",
    )

    ap.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = ap.parse_args()

    # --------------------------------------------------------
    # Frozen TCN V1
    # --------------------------------------------------------

    exp = (
        ROOT
        / "artifacts"
        / "experiments"
        / args.experiment
    )

    model_path = exp / "model.keras"
    stats_path = exp / "preprocess_stats.npz"
    labels_path = exp / "labels.json"

    for p in (
        model_path,
        stats_path,
        labels_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing frozen experiment file: {p}\n"
                "Run: python scripts/freeze_tcn_v1.py"
            )

    import tensorflow as tf

    model = tf.keras.models.load_model(
        model_path
    )

    label_info = json.loads(
        labels_path.read_text(
            encoding="utf-8"
        )
    )

    labels = label_info["labels"]

    with np.load(
        stats_path,
        allow_pickle=False,
    ) as d:
        mean = np.asarray(
            d["mean"],
            np.float32,
        )

        std = np.asarray(
            d["std"],
            np.float32,
        )

        target_frames = int(
            np.asarray(
                d["target_frames"]
            ).reshape(-1)[0]
        )

        feature_dim = int(
            np.asarray(
                d["feature_dim"]
            ).reshape(-1)[0]
        )

    if feature_dim != 453:
        raise RuntimeError(
            f"Frozen model expects feature_dim={feature_dim}. "
            "This tester expects Phase06 Hands+Pose+Face 453."
        )

    # --------------------------------------------------------
    # Feature configuration
    # --------------------------------------------------------

    p5cfg, _ = load_phase5_config(
        ROOT
        / "configs"
        / "feature_phase5.yaml"
    )

    p6cfg, _ = load_phase6_config(
        ROOT
        / "configs"
        / "feature_phase6.yaml"
    )

    # --------------------------------------------------------
    # Rolling buffer
    # --------------------------------------------------------

    buf = deque(
        maxlen=target_frames
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    mp_h = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    specs = create_specs(mp_drawing)

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        args.camera,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        cap = cv2.VideoCapture(
            args.camera
        )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {args.camera}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        args.width,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        args.height,
    )

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    window_name = (
        "Phase07 TCN V1 Live Inference"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        args.window_width,
        args.window_height,
    )

    fullscreen = bool(args.fullscreen)

    if fullscreen:
        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

    # --------------------------------------------------------
    # Runtime state
    # --------------------------------------------------------

    log_path = ROOT / args.log

    last_pred = None
    last_conf = 0.0
    last_probs = None
    last_ms = 0.0
    fps = 0.0
    last_frame_time = time.perf_counter()

    key_help = " | ".join(
        f"{i + 1}:{name}"
        for i, name
        in enumerate(labels[:9])
    )

    # --------------------------------------------------------
    # Holistic
    # --------------------------------------------------------

    with mp_h.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print(
                    "[WARNING] Cannot read webcam frame."
                )
                break

            # -----------------------------------------------
            # FPS
            # -----------------------------------------------

            now = time.perf_counter()
            dt = max(
                now - last_frame_time,
                1e-9,
            )

            inst_fps = 1.0 / dt

            if fps <= 0:
                fps = inst_fps
            else:
                fps = (
                    0.90 * fps
                    + 0.10 * inst_fps
                )

            last_frame_time = now

            # -----------------------------------------------
            # MediaPipe inference
            # -----------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            results = holistic.process(
                rgb
            )

            lraw, lp = lm_np(
                results.left_hand_landmarks,
                21,
            )

            rraw, rp = lm_np(
                results.right_hand_landmarks,
                21,
            )

            praw, pp = lm_np(
                results.pose_landmarks,
                33,
            )

            fraw, fp = lm_np(
                results.face_landmarks,
                468,
            )

            # Face refined model may return 478 points.
            if fraw.shape[0] < 468:
                tmp = np.zeros(
                    (468, 3),
                    dtype=np.float32,
                )
                tmp[:fraw.shape[0]] = fraw
                fraw = tmp

            buf.append(
                (
                    lraw,
                    rraw,
                    praw,
                    fraw,
                    lp,
                    rp,
                    pp,
                    fp,
                    time.perf_counter(),
                )
            )

            # -----------------------------------------------
            # Build the exact Phase04->05->06 input
            # -----------------------------------------------

            if len(buf) == target_frames:
                L = np.zeros(
                    (target_frames, 21, 3),
                    np.float32,
                )

                R = np.zeros_like(L)
                LN = np.zeros_like(L)
                RN = np.zeros_like(L)

                HP = np.zeros(
                    (target_frames, 2),
                    dtype=bool,
                )

                S = np.zeros(
                    (target_frames, 2),
                    dtype=np.float32,
                )

                P = np.zeros(
                    (target_frames, 33, 3),
                    dtype=np.float32,
                )

                PP = np.zeros(
                    target_frames,
                    dtype=bool,
                )

                F = np.zeros(
                    (target_frames, 468, 3),
                    dtype=np.float32,
                )

                FP = np.zeros(
                    target_frames,
                    dtype=bool,
                )

                TS = np.zeros(
                    target_frames,
                    dtype=np.float64,
                )

                for i, item in enumerate(buf):
                    (
                        li,
                        ri,
                        pi,
                        fi,
                        lpi,
                        rpi,
                        ppi,
                        fpi,
                        tsi,
                    ) = item

                    L[i] = li[:21]
                    R[i] = ri[:21]

                    P[i] = pi[:33]
                    F[i] = fi[:468]

                    PP[i] = ppi
                    FP[i] = fpi
                    TS[i] = tsi

                    ln, lv, ls = normalize_hand(
                        li[:21]
                        if lpi
                        else None
                    )

                    rn, rv, rs = normalize_hand(
                        ri[:21]
                        if rpi
                        else None
                    )

                    LN[i] = ln
                    RN[i] = rn
                    HP[i] = [lv, rv]
                    S[i] = [ls, rs]

                p4 = {
                    "normalized_xyz126":
                        np.concatenate(
                            [
                                LN.reshape(
                                    target_frames,
                                    63,
                                ),
                                RN.reshape(
                                    target_frames,
                                    63,
                                ),
                            ],
                            axis=1,
                        ),

                    "left_normalized_xyz":
                        LN,

                    "right_normalized_xyz":
                        RN,

                    "hand_presence2":
                        HP,

                    "scale2":
                        S,

                    "timestamps":
                        TS,
                }

                p5 = build_phase5_features(
                    phase4_arrays=p4,
                    raw_left=L,
                    raw_right=R,
                    pose_xyz=P,
                    pose_present=PP,
                    cfg=p5cfg,
                )

                p6 = build_phase6_features(
                    phase5_full417=
                        p5["full_features417"],

                    pose_xyz=P,
                    pose_present=PP,

                    face_xyz=F,
                    face_present=FP,

                    cfg=p6cfg,
                )

                x = np.asarray(
                    p6[
                        "ablation_hands_pose_face453"
                    ],
                    dtype=np.float32,
                )

                x = (
                    x - mean
                ) / std

                if not np.isfinite(x).all():
                    raise RuntimeError(
                        "Live preprocessing produced NaN/Inf"
                    )

                t0 = time.perf_counter()

                probs = model.predict(
                    x[None, ...],
                    verbose=0,
                )[0]

                last_ms = (
                    time.perf_counter()
                    - t0
                ) * 1000.0

                idx = int(
                    np.argmax(probs)
                )

                last_pred = labels[idx]
                last_conf = float(
                    probs[idx]
                )
                last_probs = probs.astype(
                    float
                )

            # -----------------------------------------------
            # Draw ALL landmarks on original frame first
            # -----------------------------------------------

            display = frame.copy()

            if results.left_hand_landmarks is not None:
                mp_drawing.draw_landmarks(
                    display,
                    results.left_hand_landmarks,
                    mp_h.HAND_CONNECTIONS,
                    specs["left_lm"],
                    specs["left_conn"],
                )

            if results.right_hand_landmarks is not None:
                mp_drawing.draw_landmarks(
                    display,
                    results.right_hand_landmarks,
                    mp_h.HAND_CONNECTIONS,
                    specs["right_lm"],
                    specs["right_conn"],
                )

            if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    display,
                    results.pose_landmarks,
                    mp_h.POSE_CONNECTIONS,
                    landmark_drawing_spec=
                        specs["pose_lm"],
                    connection_drawing_spec=
                        specs["pose_conn"],
                )

            if results.face_landmarks is not None:
                mp_drawing.draw_landmarks(
                    display,
                    results.face_landmarks,
                    mp_h.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=
                        specs["face_conn"],
                )

            # Preview mirror only AFTER inference/drawing
            if args.mirror:
                display = cv2.flip(
                    display,
                    1,
                )

            # -----------------------------------------------
            # Responsive UI
            # -----------------------------------------------

            draw_ui(
                display,
                labels=labels,
                probs=last_probs,
                pred=last_pred,
                confidence=last_conf,
                inference_ms=last_ms,
                buffer_size=len(buf),
                target_frames=target_frames,
                fps=fps,
                left_present=bool(lp),
                right_present=bool(rp),
                pose_present=bool(pp),
                face_present=bool(fp),
                fullscreen=fullscreen,
            )

            # Ground-truth key legend
            h, w = display.shape[:2]

            put_text(
                display,
                key_help,
                (
                    18,
                    max(
                        200,
                        int(h * 0.24) - 10,
                    ),
                ),
                color=(230, 230, 230),
                scale=max(
                    0.42,
                    fit_text_scale(w) * 0.65,
                ),
                thickness=1,
            )

            cv2.imshow(
                window_name,
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # -----------------------------------------------
            # Controls
            # -----------------------------------------------

            if key in (
                ord("q"),
                27,
            ):
                break

            if key in (
                ord("r"),
                ord("R"),
            ):
                buf.clear()
                last_pred = None
                last_probs = None
                last_conf = 0.0
                last_ms = 0.0
                print(
                    "[RESET] Buffer cleared."
                )
                continue

            if key in (
                ord("f"),
                ord("F"),
            ):
                fullscreen = not fullscreen

                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    (
                        cv2.WINDOW_FULLSCREEN
                        if fullscreen
                        else cv2.WINDOW_NORMAL
                    ),
                )

                if not fullscreen:
                    cv2.resizeWindow(
                        window_name,
                        args.window_width,
                        args.window_height,
                    )

                continue

            # Increase window size
            if key in (
                ord("+"),
                ord("="),
            ):
                if not fullscreen:
                    args.window_width = min(
                        1920,
                        args.window_width + 160,
                    )
                    args.window_height = min(
                        1080,
                        args.window_height + 90,
                    )

                    cv2.resizeWindow(
                        window_name,
                        args.window_width,
                        args.window_height,
                    )
                continue

            # Decrease window size
            if key in (
                ord("-"),
                ord("_"),
            ):
                if not fullscreen:
                    args.window_width = max(
                        800,
                        args.window_width - 160,
                    )
                    args.window_height = max(
                        540,
                        args.window_height - 90,
                    )

                    cv2.resizeWindow(
                        window_name,
                        args.window_width,
                        args.window_height,
                    )
                continue

            # Press 1..9 to log the CURRENT window ground truth
            if (
                ord("1")
                <= key
                <= ord("9")
                and last_pred
                is not None
            ):
                gt_i = (
                    key - ord("1")
                )

                if gt_i < len(labels):
                    gt = labels[gt_i]
                    correct = (
                        gt == last_pred
                    )

                    row = {
                        "timestamp":
                            time.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "experiment":
                            args.experiment,

                        "ground_truth":
                            gt,

                        "prediction":
                            last_pred,

                        "correct":
                            int(correct),

                        "confidence":
                            f"{last_conf:.6f}",

                        "inference_ms":
                            f"{last_ms:.3f}",
                    }

                    for i, name in enumerate(labels):
                        row[
                            f"p_{name}"
                        ] = (
                            f"{last_probs[i]:.6f}"
                        )

                    append_csv(
                        log_path,
                        row,
                        list(row.keys()),
                    )

                    print(
                        f"[LOG] "
                        f"GT={gt:<12} "
                        f"PRED={last_pred:<12} "
                        f"conf={last_conf:.3f} "
                        f"{'OK' if correct else 'FALSE'}"
                    )

    cap.release()
    cv2.destroyAllWindows()

    print(
        "Live log:",
        log_path,
    )


if __name__ == "__main__":
    main()
