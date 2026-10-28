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
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_phase5 import load_phase5_config, build_phase5_features
from core.feature_phase6 import load_phase6_config, build_phase6_features
from core.preprocess_phase7 import uniform_resample_or_pad
from core.realtime_decision_phase7 import (
    DecisionConfig,
    RealtimeDecisionEngine,
    SegmentState,
    gate_prediction,
    motion_score,
)


def rp(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_realtime_config(path):
    raw = yaml.safe_load(rp(path).read_text(encoding="utf-8")) or {}
    dec = raw.get("decision", {}) or {}
    seg = raw.get("segmentation", {}) or {}
    unk = raw.get("unknown", {}) or {}

    cfg = DecisionConfig(
        min_confidence=float(dec.get("min_confidence", 0.85)),
        min_top1_margin=float(dec.get("min_top1_margin", 0.20)),
        stable_window_count=int(dec.get("stable_window_count", 5)),
        stable_required=int(dec.get("stable_required", 4)),
        min_predict_frames=int(dec.get("min_predict_frames", 12)),
        preview_stride_frames=int(dec.get("preview_stride_frames", 4)),
        start_motion_threshold=float(seg.get("start_motion_threshold", 0.012)),
        end_motion_threshold=float(seg.get("end_motion_threshold", 0.004)),
        start_required_frames=int(seg.get("start_required_frames", 3)),
        end_still_frames=int(seg.get("end_still_frames", 6)),
        min_active_frames=int(seg.get("min_active_frames", 12)),
        max_active_frames=int(seg.get("max_active_frames", 60)),
        min_hand_presence_ratio=float(seg.get("min_hand_presence_ratio", 0.45)),
        cooldown_seconds=float(seg.get("cooldown_seconds", 0.90)),
        require_rest_before_repeat=bool(seg.get("require_rest_before_repeat", True)),
        unknown_label=str(unk.get("label", "UNKNOWN")),
    )

    pre_roll = int(seg.get("pre_roll_frames", 4))
    return cfg, pre_roll, raw


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


def build_feature453(segment, p5cfg, p6cfg):
    T = len(segment)

    L = np.zeros((T, 21, 3), np.float32)
    R = np.zeros_like(L)
    LN = np.zeros_like(L)
    RN = np.zeros_like(L)

    HP = np.zeros((T, 2), bool)
    S = np.zeros((T, 2), np.float32)

    P = np.zeros((T, 33, 3), np.float32)
    PP = np.zeros(T, bool)

    F = np.zeros((T, 468, 3), np.float32)
    FP = np.zeros(T, bool)

    TS = np.zeros(T, np.float64)

    for i, item in enumerate(segment):
        li, ri, pi, fi, lp, rp_, pp, fp, ts = item

        L[i] = li[:21]
        R[i] = ri[:21]
        P[i] = pi[:33]
        F[i] = fi[:468]

        PP[i] = pp
        FP[i] = fp
        TS[i] = ts

        ln, lv, ls = normalize_hand(li[:21] if lp else None)
        rn, rv, rs = normalize_hand(ri[:21] if rp_ else None)

        LN[i] = ln
        RN[i] = rn
        HP[i] = [lv, rv]
        S[i] = [ls, rs]

    p4 = {
        "normalized_xyz126": np.concatenate(
            [
                LN.reshape(T, 63),
                RN.reshape(T, 63),
            ],
            axis=1,
        ),
        "left_normalized_xyz": LN,
        "right_normalized_xyz": RN,
        "hand_presence2": HP,
        "scale2": S,
        "timestamps": TS,
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
        phase5_full417=p5["full_features417"],
        pose_xyz=P,
        pose_present=PP,
        face_xyz=F,
        face_present=FP,
        cfg=p6cfg,
    )

    return np.asarray(
        p6["ablation_hands_pose_face453"],
        dtype=np.float32,
    )


def prepare_model_input(feature453, target_frames, mean, std):
    fixed, mask, _ = uniform_resample_or_pad(
        feature453,
        target_frames,
    )

    out = np.zeros_like(fixed, dtype=np.float32)
    out[mask] = (fixed[mask] - mean) / std
    out[~mask] = 0.0

    if not np.isfinite(out).all():
        raise RuntimeError("Realtime preprocessing produced NaN/Inf")

    return out


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()

    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            + "\n"
        )


def put(frame, text, xy, color=(255, 255, 255), scale=0.58, thickness=1):
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


def alpha_panel(frame, p1, p2, color=(10, 18, 28), alpha=0.80):
    overlay = frame.copy()
    cv2.rectangle(overlay, p1, p2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/realtime_phase7.yaml",
    )

    ap.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--fullscreen",
        action="store_true",
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

    args = ap.parse_args()

    cfg, pre_roll_frames, raw = load_realtime_config(args.config)

    experiment = raw.get("model", {}).get(
        "experiment",
        "tcn_v1_160samples",
    )

    exp = (
        ROOT
        / "artifacts"
        / "experiments"
        / experiment
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
                f"Missing frozen TCN V1 artifact: {p}"
            )

    import tensorflow as tf

    model = tf.keras.models.load_model(
        model_path
    )

    labels = json.loads(
        labels_path.read_text(
            encoding="utf-8"
        )
    )["labels"]

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
            f"TCN V1 expects feature_dim={feature_dim}, "
            "but this realtime engine expects Phase06 453."
        )

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

    engine = RealtimeDecisionEngine(cfg)

    mp_h = mp.solutions.holistic
    draw = mp.solutions.drawing_utils

    left_spec = draw.DrawingSpec(
        color=(255, 70, 0),
        thickness=3,
        circle_radius=4,
    )

    left_conn = draw.DrawingSpec(
        color=(255, 160, 70),
        thickness=3,
        circle_radius=1,
    )

    right_spec = draw.DrawingSpec(
        color=(0, 110, 255),
        thickness=3,
        circle_radius=4,
    )

    right_conn = draw.DrawingSpec(
        color=(0, 185, 255),
        thickness=3,
        circle_radius=1,
    )

    pose_spec = draw.DrawingSpec(
        color=(0, 215, 85),
        thickness=3,
        circle_radius=4,
    )

    pose_conn = draw.DrawingSpec(
        color=(80, 255, 145),
        thickness=3,
        circle_radius=1,
    )

    face_conn = draw.DrawingSpec(
        color=(200, 90, 255),
        thickness=2,
        circle_radius=1,
    )

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
            "Cannot open webcam."
        )

    window_name = (
        "Phase07 Realtime Decision Engine V2"
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

    pre_roll = deque(
        maxlen=max(1, pre_roll_frames)
    )

    segment = []

    prev_left = None
    prev_right = None
    prev_pose = None

    current_motion = 0.0

    preview_result = None
    final_label = cfg.unknown_label
    final_details = {}
    last_probs = None
    last_inference_ms = 0.0

    live_csv = rp(
        raw.get("logging", {}).get(
            "live_csv",
            "logs/phase7_decision_v2.csv",
        )
    )

    event_jsonl = rp(
        raw.get("logging", {}).get(
            "event_jsonl",
            "logs/phase7_decision_events.jsonl",
        )
    )

    last_frame_t = time.perf_counter()
    fps = 0.0

    def classify_segment(items):
        nonlocal last_inference_ms

        feature453 = build_feature453(
            items,
            p5cfg,
            p6cfg,
        )

        X = prepare_model_input(
            feature453,
            target_frames,
            mean,
            std,
        )

        t0 = time.perf_counter()

        probs = model.predict(
            X[None, ...],
            verbose=0,
        )[0]

        last_inference_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        result = gate_prediction(
            labels,
            probs,
            cfg,
        )

        return result, probs

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
                break

            now = time.perf_counter()

            dt = max(
                now - last_frame_t,
                1e-9,
            )

            instant_fps = 1.0 / dt

            if fps <= 0:
                fps = instant_fps
            else:
                fps = (
                    0.90 * fps
                    + 0.10 * instant_fps
                )

            last_frame_t = now

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            results = holistic.process(
                rgb
            )

            left, lp = lm_np(
                results.left_hand_landmarks,
                21,
            )

            right, rp_ = lm_np(
                results.right_hand_landmarks,
                21,
            )

            pose, pp = lm_np(
                results.pose_landmarks,
                33,
            )

            face, fp = lm_np(
                results.face_landmarks,
                468,
            )

            if face.shape[0] < 468:
                tmp = np.zeros(
                    (468, 3),
                    np.float32,
                )
                tmp[:face.shape[0]] = face
                face = tmp

            current_motion = motion_score(
                left_now=left if lp else None,
                left_prev=prev_left,
                right_now=right if rp_ else None,
                right_prev=prev_right,
                pose_now=pose if pp else None,
                pose_prev=prev_pose,
            )

            prev_left = (
                left.copy()
                if lp
                else None
            )

            prev_right = (
                right.copy()
                if rp_
                else None
            )

            prev_pose = (
                pose.copy()
                if pp
                else None
            )

            item = (
                left,
                right,
                pose,
                face,
                lp,
                rp_,
                pp,
                fp,
                now,
            )

            pre_roll.append(item)

            state_before = engine.state

            event = engine.update(
                motion=current_motion,
                any_hand_present=bool(
                    lp or rp_
                ),
                now_s=now,
            )

            if event == "START":
                segment = list(pre_roll)

                append_jsonl(
                    event_jsonl,
                    {
                        "time": time.time(),
                        "event": "START",
                        "motion": current_motion,
                    },
                )

            elif engine.state == SegmentState.ACTIVE:
                if (
                    not segment
                    or segment[-1][-1] != item[-1]
                ):
                    segment.append(item)

                # Preview predictions for stabilizer.
                if (
                    len(segment)
                    >= cfg.min_predict_frames
                    and (
                        len(segment)
                        - cfg.min_predict_frames
                    )
                    % cfg.preview_stride_frames
                    == 0
                ):
                    try:
                        preview_result, probs = classify_segment(
                            segment
                        )

                        engine.add_preview_label(
                            preview_result.gated_label
                        )

                        last_probs = probs

                    except Exception as e:
                        print(
                            "[PREVIEW ERROR]",
                            type(e).__name__,
                            e,
                        )

            if event in (
                "END",
                "MAX_END",
            ):
                final_details = {}

                if not segment:
                    final_label = (
                        cfg.unknown_label
                    )
                else:
                    try:
                        final_result, probs = classify_segment(
                            segment
                        )

                        last_probs = probs

                        final_label, final_details = (
                            engine.confirm_prediction(
                                final_result,
                                now,
                            )
                        )

                        append_jsonl(
                            event_jsonl,
                            {
                                "time": time.time(),
                                "event": event,
                                "output": final_label,
                                **final_details,
                            },
                        )

                    except Exception as e:
                        final_label = (
                            cfg.unknown_label
                        )

                        final_details = {
                            "error":
                                f"{type(e).__name__}:{e}"
                        }

                print(
                    f"[{event}] "
                    f"output={final_label} "
                    f"frames={len(segment)} "
                    f"motion={current_motion:.4f} "
                    f"details={final_details}"
                )

            if event == "READY":
                segment = []

            display = frame.copy()

            if results.left_hand_landmarks:
                draw.draw_landmarks(
                    display,
                    results.left_hand_landmarks,
                    mp_h.HAND_CONNECTIONS,
                    left_spec,
                    left_conn,
                )

            if results.right_hand_landmarks:
                draw.draw_landmarks(
                    display,
                    results.right_hand_landmarks,
                    mp_h.HAND_CONNECTIONS,
                    right_spec,
                    right_conn,
                )

            if results.pose_landmarks:
                draw.draw_landmarks(
                    display,
                    results.pose_landmarks,
                    mp_h.POSE_CONNECTIONS,
                    landmark_drawing_spec=
                        pose_spec,
                    connection_drawing_spec=
                        pose_conn,
                )

            if results.face_landmarks:
                draw.draw_landmarks(
                    display,
                    results.face_landmarks,
                    mp_h.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=
                        face_conn,
                )

            mirror = bool(
                raw.get("ui", {}).get(
                    "mirror_preview",
                    True,
                )
            )

            if mirror:
                display = cv2.flip(
                    display,
                    1,
                )

            h, w = display.shape[:2]

            alpha_panel(
                display,
                (0, 0),
                (w, 190),
            )

            state_color = {
                SegmentState.REST:
                    (220, 220, 220),

                SegmentState.ACTIVE:
                    (0, 220, 255),

                SegmentState.COOLDOWN:
                    (255, 180, 70),

            }[engine.state]

            put(
                display,
                f"STATE: {engine.state.value}",
                (20, 34),
                color=state_color,
                scale=0.78,
                thickness=2,
            )

            put(
                display,
                (
                    f"Motion={current_motion:.4f} | "
                    f"segment={len(segment)} | "
                    f"FPS={fps:.1f}"
                ),
                (20, 66),
                color=(230, 230, 230),
                scale=0.52,
            )

            put(
                display,
                (
                    f"START >= {cfg.start_motion_threshold:.3f} | "
                    f"END <= {cfg.end_motion_threshold:.3f}"
                ),
                (20, 94),
                color=(190, 200, 220),
                scale=0.48,
            )

            preview_text = (
                cfg.unknown_label
                if preview_result is None
                else (
                    f"{preview_result.gated_label} "
                    f"conf={preview_result.top1_prob*100:.1f}% "
                    f"margin={preview_result.margin:.2f}"
                )
            )

            put(
                display,
                f"CANDIDATE: {preview_text}",
                (20, 128),
                color=(0, 220, 255),
                scale=0.62,
                thickness=2,
            )

            output_color = (
                (0, 255, 110)
                if final_label
                != cfg.unknown_label
                else (70, 70, 255)
            )

            put(
                display,
                f"CONFIRMED: {final_label}",
                (20, 166),
                color=output_color,
                scale=0.76,
                thickness=2,
            )

            # Detection status
            x0 = max(
                580,
                int(w * 0.56),
            )

            put(
                display,
                f"L-HAND {'ON' if lp else 'OFF'}",
                (x0, 34),
                color=(
                    (255, 160, 70)
                    if lp
                    else (70, 70, 255)
                ),
                scale=0.52,
                thickness=2,
            )

            put(
                display,
                f"R-HAND {'ON' if rp_ else 'OFF'}",
                (x0, 66),
                color=(
                    (0, 180, 255)
                    if rp_
                    else (70, 70, 255)
                ),
                scale=0.52,
                thickness=2,
            )

            put(
                display,
                f"POSE {'ON' if pp else 'OFF'}",
                (x0, 98),
                color=(
                    (0, 220, 90)
                    if pp
                    else (70, 70, 255)
                ),
                scale=0.52,
                thickness=2,
            )

            put(
                display,
                f"FACE {'ON' if fp else 'OFF'}",
                (x0, 130),
                color=(
                    (200, 100, 255)
                    if fp
                    else (70, 70, 255)
                ),
                scale=0.52,
                thickness=2,
            )

            put(
                display,
                (
                    f"Gate: conf>={cfg.min_confidence:.2f} | "
                    f"margin>={cfg.min_top1_margin:.2f} | "
                    f"stable={cfg.stable_required}/{cfg.stable_window_count}"
                ),
                (x0, 162),
                color=(215, 215, 215),
                scale=0.45,
            )

            footer = (
                "1..9 log GT | R reset | F fullscreen | Q quit"
            )

            alpha_panel(
                display,
                (0, h - 42),
                (w, h),
            )

            put(
                display,
                footer,
                (15, h - 14),
                scale=0.48,
            )

            cv2.imshow(
                window_name,
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                break

            if key in (
                ord("r"),
                ord("R"),
            ):
                engine = RealtimeDecisionEngine(
                    cfg
                )
                segment = []
                pre_roll.clear()
                preview_result = None
                final_label = cfg.unknown_label
                final_details = {}
                last_probs = None
                print("[RESET]")
                continue

            if key in (
                ord("f"),
                ord("F"),
            ):
                fullscreen = (
                    not fullscreen
                )

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

            # Log current confirmed output using numeric ground truth.
            if (
                ord("1")
                <= key
                <= ord("9")
                and last_probs
                is not None
            ):
                gt_index = (
                    key
                    - ord("1")
                )

                if gt_index < len(labels):
                    gt = labels[
                        gt_index
                    ]

                    row = {
                        "timestamp":
                            time.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "ground_truth":
                            gt,

                        "confirmed_output":
                            final_label,

                        "correct":
                            int(
                                gt
                                == final_label
                            ),

                        "state":
                            engine.state.value,

                        "segment_frames":
                            len(segment),

                        "motion":
                            f"{current_motion:.6f}",

                        "inference_ms":
                            f"{last_inference_ms:.3f}",
                    }

                    for i, name in enumerate(
                        labels
                    ):
                        row[
                            f"p_{name}"
                        ] = (
                            f"{float(last_probs[i]):.6f}"
                        )

                    append_csv(
                        live_csv,
                        row,
                    )

                    print(
                        "[GROUND TRUTH]",
                        gt,
                        "->",
                        final_label,
                    )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
