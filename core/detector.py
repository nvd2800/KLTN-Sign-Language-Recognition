from __future__ import annotations
import importlib
from typing import Any
import cv2
import numpy as np
from .types import FrameLandmarks, PresenceMask, empty_hand


class MediaPipeDetector:
    """Landmark detector for Phase 0.

    Canonical identity is the signer's physical left/right hand.
    Mirroring is presentation-only and is never applied inside this class.
    """

    def __init__(self, config: dict[str, Any]):
        try:
            mp = importlib.import_module("mediapipe")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "mediapipe is not installed. Run: pip install -r requirements.txt"
            ) from e
        self.mp = mp
        self.cfg = config
        hcfg = config["hands"]
        pcfg = config["pose"]
        fcfg = config["face"]

        self.hands = None
        self.pose = None
        self.face = None
        if hcfg.get("enabled", True):
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=int(hcfg.get("max_num_hands", 2)),
                min_detection_confidence=float(hcfg.get("min_detection_confidence", 0.6)),
                min_tracking_confidence=float(hcfg.get("min_tracking_confidence", 0.6)),
            )
        if pcfg.get("enabled", True):
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                min_detection_confidence=float(pcfg.get("min_detection_confidence", 0.6)),
                min_tracking_confidence=float(pcfg.get("min_tracking_confidence", 0.6)),
            )
        if fcfg.get("enabled", True):
            self.face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=int(fcfg.get("max_num_faces", 1)),
                refine_landmarks=True,
                min_detection_confidence=float(fcfg.get("min_detection_confidence", 0.6)),
                min_tracking_confidence=float(fcfg.get("min_tracking_confidence", 0.6)),
            )

    @staticmethod
    def _xyz(landmark_list) -> np.ndarray:
        return np.asarray([[lm.x, lm.y, lm.z] for lm in landmark_list.landmark], dtype=np.float32)

    def process(self, image_bgr: np.ndarray, frame_index: int, timestamp: float) -> FrameLandmarks:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        hand_result = self.hands.process(rgb) if self.hands else None
        pose_result = self.pose.process(rgb) if self.pose else None
        face_result = self.face.process(rgb) if self.face else None

        left = empty_hand()
        right = empty_hand()
        left_present = False
        right_present = False
        handedness_meta: list[dict[str, Any]] = []

        if hand_result and hand_result.multi_hand_landmarks and hand_result.multi_handedness:
            for lm_list, handed in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                label = handed.classification[0].label
                score = float(handed.classification[0].score)
                points = self._xyz(lm_list)
                handedness_meta.append({"label": label, "score": score})
                # MediaPipe Hands assumes a mirrored/selfie input for its handedness labels.
                # We feed the raw camera frame. Therefore swap the reported label so storage
                # remains the signer's PHYSICAL left/right. Verify once with check_landmarks.py.
                if label == "Right":
                    left = points
                    left_present = True
                elif label == "Left":
                    right = points
                    right_present = True

        pose = None
        pose_present = False
        if pose_result and pose_result.pose_landmarks:
            pose = self._xyz(pose_result.pose_landmarks)
            pose_present = True

        face = None
        face_present = False
        if face_result and face_result.multi_face_landmarks:
            face = self._xyz(face_result.multi_face_landmarks[0])
            face_present = True

        item = FrameLandmarks(
            frame_index=frame_index,
            timestamp=timestamp,
            left_hand=left,
            right_hand=right,
            pose=pose,
            face=face,
            presence=PresenceMask(
                left_hand=left_present,
                right_hand=right_present,
                pose=pose_present,
                face=face_present,
            ),
            detector_meta={"handedness_raw": handedness_meta},
        )
        item.validate()
        return item

    def close(self) -> None:
        for obj in (self.hands, self.pose, self.face):
            if obj is not None and hasattr(obj, "close"):
                obj.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
