from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


HAND_ANGLE_TRIPLES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

POSE_IDX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}

FACE_IDX = {
    "mouth_upper": 13,
    "mouth_lower": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "left_eye_upper": 159,
    "left_eye_lower": 145,
    "right_eye_upper": 386,
    "right_eye_lower": 374,
    "left_brow": 105,
    "right_brow": 334,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "nose_tip": 1,
}


def lm_list_to_np(landmark_list):
    if landmark_list is None:
        return None
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark],
        dtype=np.float32,
    )


def euclid(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float64)
            - np.asarray(b, dtype=np.float64)
        )
    )


def normalize_hand(hand, eps=1e-6):
    if hand is None or hand.shape != (21, 3):
        return None, False, 0.0

    if not np.isfinite(hand).all():
        return None, False, 0.0

    wrist = hand[0]
    scale = euclid(hand[9], wrist)

    if not np.isfinite(scale) or scale <= eps:
        return None, False, float(scale if np.isfinite(scale) else 0.0)

    normalized = (hand - wrist[None, :]) / scale

    if not np.isfinite(normalized).all():
        return None, False, scale

    return normalized.astype(np.float32), True, scale


def angle_abc(a, b, c, eps=1e-8):
    u = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)

    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)

    if nu <= eps or nv <= eps or not np.isfinite([nu, nv]).all():
        return None

    cosine = float(np.dot(u, v) / (nu * nv))
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return float(np.arccos(cosine) / np.pi)


def hand_angle_mean(normalized_hand):
    if normalized_hand is None:
        return None

    values = []
    for a, b, c in HAND_ANGLE_TRIPLES:
        value = angle_abc(
            normalized_hand[a],
            normalized_hand[b],
            normalized_hand[c],
        )
        if value is not None:
            values.append(value)

    return float(np.mean(values)) if values else None


def thumb_index_distance(normalized_hand):
    if normalized_hand is None:
        return None

    value = euclid(normalized_hand[4], normalized_hand[8])
    return value if np.isfinite(value) else None


def mean_abs_motion(current, previous):
    if current is None or previous is None:
        return None

    if current.shape != previous.shape:
        return None

    difference = current - previous

    if not np.isfinite(difference).all():
        return None

    return float(np.mean(np.abs(difference)))


def pose_features(pose, eps=1e-6):
    if pose is None or pose.ndim != 2 or pose.shape[0] <= 16:
        return None

    points = {
        name: pose[index, :3]
        for name, index in POSE_IDX.items()
    }

    if not all(np.isfinite(value).all() for value in points.values()):
        return None

    nose = points["nose"]
    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    left_wrist = points["left_wrist"]
    right_wrist = points["right_wrist"]

    chest = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = euclid(left_shoulder, right_shoulder)

    if not np.isfinite(shoulder_width) or shoulder_width <= eps:
        return None

    return {
        "shoulder_width": shoulder_width,
        "left_wrist_to_nose": euclid(left_wrist, nose) / shoulder_width,
        "right_wrist_to_nose": euclid(right_wrist, nose) / shoulder_width,
        "left_wrist_to_shoulder": euclid(left_wrist, left_shoulder) / shoulder_width,
        "right_wrist_to_shoulder": euclid(right_wrist, right_shoulder) / shoulder_width,
        "left_wrist_to_chest": euclid(left_wrist, chest) / shoulder_width,
        "right_wrist_to_chest": euclid(right_wrist, chest) / shoulder_width,
    }


def face_features(face, eps=1e-6):
    if face is None or face.ndim != 2:
        return None

    if face.shape[0] <= max(FACE_IDX.values()):
        return None

    def point(name):
        return face[FACE_IDX[name], :3]

    required = [point(name) for name in FACE_IDX]

    if not all(np.isfinite(value).all() for value in required):
        return None

    left_eye_outer = point("left_eye_outer")
    right_eye_outer = point("right_eye_outer")

    face_scale = euclid(left_eye_outer, right_eye_outer)

    if not np.isfinite(face_scale) or face_scale <= eps:
        return None

    left_eye_center = (
        point("left_eye_upper") + point("left_eye_lower")
    ) / 2.0

    right_eye_center = (
        point("right_eye_upper") + point("right_eye_lower")
    ) / 2.0

    eye_midpoint = (
        left_eye_outer + right_eye_outer
    ) / 2.0

    nose = point("nose_tip")
    eye_vector = right_eye_outer[:2] - left_eye_outer[:2]

    values = {
        "mouth_open": euclid(
            point("mouth_upper"),
            point("mouth_lower"),
        ) / face_scale,

        "mouth_width": euclid(
            point("mouth_left"),
            point("mouth_right"),
        ) / face_scale,

        "left_eye_open": euclid(
            point("left_eye_upper"),
            point("left_eye_lower"),
        ) / face_scale,

        "right_eye_open": euclid(
            point("right_eye_upper"),
            point("right_eye_lower"),
        ) / face_scale,

        "left_brow_height": euclid(
            point("left_brow"),
            left_eye_center,
        ) / face_scale,

        "right_brow_height": euclid(
            point("right_brow"),
            right_eye_center,
        ) / face_scale,

        "yaw_proxy": float(
            (nose[0] - eye_midpoint[0]) / face_scale
        ),

        "pitch_proxy": float(
            (nose[1] - eye_midpoint[1]) / face_scale
        ),

        "roll_proxy": float(
            np.arctan2(
                eye_vector[1],
                eye_vector[0],
            ) / np.pi
        ),
    }

    if not np.isfinite(list(values.values())).all():
        return None

    return values


@dataclass
class State:
    mode: int = 3
    show_values: bool = True
    mirror_preview: bool = True
    previous_left: np.ndarray | None = None
    previous_right: np.ndarray | None = None
    previous_left_velocity: float | None = None
    previous_right_velocity: float | None = None
    fps: float = 0.0


def put_text(
    frame,
    text,
    xy,
    color=(255, 255, 255),
    scale=0.55,
    thickness=1,
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


def format_value(value, decimals=4):
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def draw_information_panel(
    frame,
    lines,
    start=(15, 30),
    line_height=23,
):
    x, y = start

    panel_width = min(frame.shape[1] - 20, 550)
    panel_height = line_height * len(lines) + 18

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (5, 5),
        (5 + panel_width, 5 + panel_height),
        (10, 18, 28),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.80,
        frame,
        0.20,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (5, 5),
        (5 + panel_width, 5 + panel_height),
        (70, 90, 110),
        1,
    )

    for index, (text, color) in enumerate(lines):
        put_text(
            frame,
            text,
            (x, y + index * line_height),
            color=color,
            scale=0.50,
            thickness=1,
        )


def create_thesis_drawing_specs(mp_drawing):
    """
    OpenCV BGR:
    LEFT HAND  = strong blue
    RIGHT HAND = strong orange
    POSE       = strong green
    FACE       = strong purple
    """

    return {
        "left_hand_landmarks": mp_drawing.DrawingSpec(
            color=(255, 70, 0),
            thickness=3,
            circle_radius=4,
        ),

        "left_hand_connections": mp_drawing.DrawingSpec(
            color=(255, 150, 40),
            thickness=3,
            circle_radius=1,
        ),

        "right_hand_landmarks": mp_drawing.DrawingSpec(
            color=(0, 105, 255),
            thickness=3,
            circle_radius=4,
        ),

        "right_hand_connections": mp_drawing.DrawingSpec(
            color=(0, 175, 255),
            thickness=3,
            circle_radius=1,
        ),

        "pose_landmarks": mp_drawing.DrawingSpec(
            color=(0, 200, 60),
            thickness=3,
            circle_radius=4,
        ),

        "pose_connections": mp_drawing.DrawingSpec(
            color=(70, 255, 120),
            thickness=3,
            circle_radius=1,
        ),

        "face_connections": mp_drawing.DrawingSpec(
            color=(185, 35, 255),
            thickness=2,
            circle_radius=1,
        ),
    }


def save_snapshot(frame, metrics, output_directory):
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    image_path = (
        output_directory
        / f"debug_{timestamp}.jpg"
    )

    json_path = (
        output_directory
        / f"debug_{timestamp}.json"
    )

    cv2.imwrite(
        str(image_path),
        frame,
    )

    json_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[SNAPSHOT] {image_path}")
    print(f"[METRICS ] {json_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--snapshot-dir",
        default="debug_snapshots",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Legacy MediaPipe Solutions.
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    thesis_specs = create_thesis_drawing_specs(
        mp_drawing
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
            f"Cannot open camera index {args.camera}. "
            "Try --camera 1."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        args.width,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        args.height,
    )

    state = State()

    print()
    print("KLTN LIVE LANDMARK DEBUG")
    print("=" * 68)
    print("1 Hands | 2 Hands+Pose | 3 Hands+Pose+Face")
    print("4 Values | M Mirror | S Snapshot | R Reset | Q Quit")
    print("=" * 68)

    previous_time = time.perf_counter()

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as holistic:

        while True:
            success, frame = cap.read()

            if not success or frame is None:
                print("[WARNING] Cannot read webcam frame.")
                break

            # Detect on original frame.
            # Mirror only the preview after inference.
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            left_raw = lm_list_to_np(
                results.left_hand_landmarks
            )

            right_raw = lm_list_to_np(
                results.right_hand_landmarks
            )

            pose_raw = lm_list_to_np(
                results.pose_landmarks
            )

            face_raw = lm_list_to_np(
                results.face_landmarks
            )

            (
                left_normalized,
                left_valid,
                left_scale,
            ) = normalize_hand(left_raw)

            (
                right_normalized,
                right_valid,
                right_scale,
            ) = normalize_hand(right_raw)

            left_angle = hand_angle_mean(
                left_normalized
            )

            right_angle = hand_angle_mean(
                right_normalized
            )

            left_thumb_index = thumb_index_distance(
                left_normalized
            )

            right_thumb_index = thumb_index_distance(
                right_normalized
            )

            left_velocity = mean_abs_motion(
                left_normalized,
                state.previous_left,
            )

            right_velocity = mean_abs_motion(
                right_normalized,
                state.previous_right,
            )

            left_acceleration = None
            if (
                left_velocity is not None
                and state.previous_left_velocity is not None
            ):
                left_acceleration = (
                    left_velocity
                    - state.previous_left_velocity
                )

            right_acceleration = None
            if (
                right_velocity is not None
                and state.previous_right_velocity is not None
            ):
                right_acceleration = (
                    right_velocity
                    - state.previous_right_velocity
                )

            pose_values = (
                pose_features(pose_raw)
                if state.mode >= 2
                else None
            )

            face_values = (
                face_features(face_raw)
                if state.mode >= 3
                else None
            )

            state.previous_left = (
                left_normalized.copy()
                if left_valid
                else None
            )

            state.previous_right = (
                right_normalized.copy()
                if right_valid
                else None
            )

            state.previous_left_velocity = (
                left_velocity
            )

            state.previous_right_velocity = (
                right_velocity
            )

            current_time = time.perf_counter()

            delta_time = max(
                current_time - previous_time,
                1e-9,
            )

            instant_fps = 1.0 / delta_time

            if state.fps <= 0:
                state.fps = instant_fps
            else:
                state.fps = (
                    0.9 * state.fps
                    + 0.1 * instant_fps
                )

            previous_time = current_time

            display = frame.copy()

            # LEFT HAND
            if (
                state.mode >= 1
                and results.left_hand_landmarks is not None
            ):
                mp_drawing.draw_landmarks(
                    display,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    thesis_specs["left_hand_landmarks"],
                    thesis_specs["left_hand_connections"],
                )

            # RIGHT HAND
            if (
                state.mode >= 1
                and results.right_hand_landmarks is not None
            ):
                mp_drawing.draw_landmarks(
                    display,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    thesis_specs["right_hand_landmarks"],
                    thesis_specs["right_hand_connections"],
                )

            # POSE
            if (
                state.mode >= 2
                and results.pose_landmarks is not None
            ):
                mp_drawing.draw_landmarks(
                    display,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=thesis_specs["pose_landmarks"],
                    connection_drawing_spec=thesis_specs["pose_connections"],
                )

            # FACE
            if (
                state.mode >= 3
                and results.face_landmarks is not None
            ):
                mp_drawing.draw_landmarks(
                    display,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=thesis_specs["face_connections"],
                )

            if state.mirror_preview:
                display = cv2.flip(
                    display,
                    1,
                )

            pose_valid = pose_values is not None
            face_valid = face_values is not None

            lines = [
                (
                    f"KLTN SIGN LANGUAGE | MODE {state.mode} | FPS {state.fps:.1f}",
                    (255, 255, 255),
                ),
                (
                    (
                        "LEFT HAND  : "
                        f"{'DETECTED' if left_valid else 'NOT DETECTED'} "
                        f"| scale={left_scale:.4f}"
                    ),
                    (
                        (255, 140, 40)
                        if left_valid
                        else (70, 70, 255)
                    ),
                ),
                (
                    (
                        "RIGHT HAND : "
                        f"{'DETECTED' if right_valid else 'NOT DETECTED'} "
                        f"| scale={right_scale:.4f}"
                    ),
                    (
                        (0, 170, 255)
                        if right_valid
                        else (70, 70, 255)
                    ),
                ),
            ]

            if state.mode >= 2:
                lines.append(
                    (
                        "POSE       : "
                        f"{'DETECTED' if pose_valid else 'NOT DETECTED'}",
                        (
                            (0, 220, 90)
                            if pose_valid
                            else (70, 70, 255)
                        ),
                    )
                )

            if state.mode >= 3:
                lines.append(
                    (
                        "FACE       : "
                        f"{'DETECTED' if face_valid else 'NOT DETECTED'}",
                        (
                            (190, 80, 255)
                            if face_valid
                            else (70, 70, 255)
                        ),
                    )
                )

            if state.show_values:
                lines.extend(
                    [
                        (
                            (
                                "L angle="
                                f"{format_value(left_angle)} "
                                "| thumb-index="
                                f"{format_value(left_thumb_index)}"
                            ),
                            (225, 225, 225),
                        ),
                        (
                            (
                                "R angle="
                                f"{format_value(right_angle)} "
                                "| thumb-index="
                                f"{format_value(right_thumb_index)}"
                            ),
                            (225, 225, 225),
                        ),
                        (
                            (
                                "L velocity="
                                f"{format_value(left_velocity)} "
                                "| accel="
                                f"{format_value(left_acceleration)}"
                            ),
                            (225, 225, 225),
                        ),
                        (
                            (
                                "R velocity="
                                f"{format_value(right_velocity)} "
                                "| accel="
                                f"{format_value(right_acceleration)}"
                            ),
                            (225, 225, 225),
                        ),
                    ]
                )

                if pose_values is not None:
                    lines.extend(
                        [
                            (
                                (
                                    "POSE wrist->nose "
                                    f"L={pose_values['left_wrist_to_nose']:.3f} "
                                    f"R={pose_values['right_wrist_to_nose']:.3f}"
                                ),
                                (80, 255, 130),
                            ),
                            (
                                (
                                    "POSE wrist->chest "
                                    f"L={pose_values['left_wrist_to_chest']:.3f} "
                                    f"R={pose_values['right_wrist_to_chest']:.3f}"
                                ),
                                (80, 255, 130),
                            ),
                        ]
                    )

                if face_values is not None:
                    lines.extend(
                        [
                            (
                                (
                                    "FACE mouth "
                                    f"open={face_values['mouth_open']:.3f} "
                                    f"width={face_values['mouth_width']:.3f}"
                                ),
                                (210, 120, 255),
                            ),
                            (
                                (
                                    "FACE eyes "
                                    f"L={face_values['left_eye_open']:.3f} "
                                    f"R={face_values['right_eye_open']:.3f}"
                                ),
                                (210, 120, 255),
                            ),
                            (
                                (
                                    "FACE brow "
                                    f"L={face_values['left_brow_height']:.3f} "
                                    f"R={face_values['right_brow_height']:.3f}"
                                ),
                                (210, 120, 255),
                            ),
                            (
                                (
                                    "HEAD "
                                    f"yaw={face_values['yaw_proxy']:.3f} "
                                    f"pitch={face_values['pitch_proxy']:.3f} "
                                    f"roll={face_values['roll_proxy']:.3f}"
                                ),
                                (210, 120, 255),
                            ),
                        ]
                    )

            draw_information_panel(
                display,
                lines,
            )

            height = display.shape[0]
            width = display.shape[1]

            legend_x = max(
                10,
                width - 295,
            )
            legend_y = 32

            legend_overlay = display.copy()

            cv2.rectangle(
                legend_overlay,
                (legend_x - 12, legend_y - 24),
                (width - 10, legend_y + 112),
                (10, 18, 28),
                -1,
            )

            cv2.addWeighted(
                legend_overlay,
                0.78,
                display,
                0.22,
                0,
                display,
            )

            put_text(
                display,
                "LANDMARK LEGEND",
                (legend_x, legend_y),
                color=(255, 255, 255),
                scale=0.50,
            )

            put_text(
                display,
                "LEFT HAND",
                (legend_x, legend_y + 26),
                color=(255, 140, 40),
                scale=0.47,
                thickness=2,
            )

            put_text(
                display,
                "RIGHT HAND",
                (legend_x, legend_y + 50),
                color=(0, 170, 255),
                scale=0.47,
                thickness=2,
            )

            put_text(
                display,
                "POSE",
                (legend_x, legend_y + 74),
                color=(0, 220, 90),
                scale=0.47,
                thickness=2,
            )

            put_text(
                display,
                "FACE",
                (legend_x, legend_y + 98),
                color=(190, 80, 255),
                scale=0.47,
                thickness=2,
            )

            footer_overlay = display.copy()

            cv2.rectangle(
                footer_overlay,
                (0, height - 42),
                (width, height),
                (10, 18, 28),
                -1,
            )

            cv2.addWeighted(
                footer_overlay,
                0.80,
                display,
                0.20,
                0,
                display,
            )

            put_text(
                display,
                (
                    "1 Hands | 2 +Pose | 3 +Face | "
                    "4 Values | M Mirror | S Save | "
                    "R Reset | Q Quit"
                ),
                (12, height - 16),
                color=(245, 245, 245),
                scale=0.44,
            )

            metrics = {
                "timestamp": time.time(),
                "fps": state.fps,
                "mode": state.mode,
                "mirror_preview": state.mirror_preview,

                "left_hand_present": bool(left_valid),
                "right_hand_present": bool(right_valid),
                "pose_present": bool(pose_valid),
                "face_present": bool(face_valid),

                "left_scale": float(left_scale),
                "right_scale": float(right_scale),

                "left_angle_mean": left_angle,
                "right_angle_mean": right_angle,

                "left_thumb_index": left_thumb_index,
                "right_thumb_index": right_thumb_index,

                "left_velocity": left_velocity,
                "right_velocity": right_velocity,

                "left_acceleration": left_acceleration,
                "right_acceleration": right_acceleration,

                "pose": pose_values,
                "face": face_values,
            }

            cv2.imshow(
                "KLTN Sign Language - Live Landmark Debug",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("1"):
                state.mode = 1

            elif key == ord("2"):
                state.mode = 2

            elif key == ord("3"):
                state.mode = 3

            elif key == ord("4"):
                state.show_values = not state.show_values

            elif key in (ord("m"), ord("M")):
                state.mirror_preview = not state.mirror_preview

            elif key in (ord("r"), ord("R")):
                state.previous_left = None
                state.previous_right = None
                state.previous_left_velocity = None
                state.previous_right_velocity = None
                print("[RESET] Motion history cleared.")

            elif key in (ord("s"), ord("S")):
                save_snapshot(
                    display,
                    metrics,
                    Path(args.snapshot_dir),
                )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
