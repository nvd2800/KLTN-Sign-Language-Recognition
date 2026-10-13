from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

HAND_LANDMARKS = 21
COORD_DIMS = 3


def empty_hand() -> np.ndarray:
    return np.zeros((HAND_LANDMARKS, COORD_DIMS), dtype=np.float32)


@dataclass
class PresenceMask:
    left_hand: bool = False
    right_hand: bool = False
    pose: bool = False
    face: bool = False

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.left_hand, self.right_hand, self.pose, self.face],
            dtype=np.uint8,
        )


@dataclass
class FrameLandmarks:
    frame_index: int
    timestamp: float
    left_hand: np.ndarray = field(default_factory=empty_hand)
    right_hand: np.ndarray = field(default_factory=empty_hand)
    pose: Optional[np.ndarray] = None
    face: Optional[np.ndarray] = None
    presence: PresenceMask = field(default_factory=PresenceMask)
    detector_meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name, hand in (("left_hand", self.left_hand), ("right_hand", self.right_hand)):
            if hand.shape != (HAND_LANDMARKS, COORD_DIMS):
                raise ValueError(f"{name} must have shape (21, 3), got {hand.shape}")
            if hand.dtype.kind not in "fc":
                raise TypeError(f"{name} must be floating point")
        if not np.isfinite(self.left_hand).all() or not np.isfinite(self.right_hand).all():
            raise ValueError("Hand landmarks contain NaN/Inf")
        if self.timestamp < 0:
            raise ValueError("timestamp must be >= 0")


@dataclass
class CameraFrame:
    index: int
    timestamp: float
    image_bgr: np.ndarray


@dataclass
class QualityInfo:
    valid: bool
    left_hand_present: bool
    right_hand_present: bool
    pose_present: bool
    face_present: bool
    notes: list[str] = field(default_factory=list)
