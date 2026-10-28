"""Pure, testable window decision and release helpers for Phase 09 webcam.

No TensorFlow, MediaPipe, or OpenCV dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np


@dataclass(frozen=True)
class LiveGateConfig:
    min_confidence: float = 0.80  # provisional live threshold; NOT a calibrated OOV threshold
    min_margin: float = 0.0
    min_hand_presence_ratio: float = 0.30
    min_motion: float = 0.0   # default disabled: measure/tune before enabling
    release_no_hand_frames: int = 12
    auto_release_no_hand: bool = True

    def validate(self):
        for k in ("min_confidence", "min_margin", "min_hand_presence_ratio"):
            v = getattr(self, k)
            if not math.isfinite(v) or not (0 <= v <= 1):
                raise ValueError(f"{k} must be in [0,1]")
        if not math.isfinite(self.min_motion) or self.min_motion < 0:
            raise ValueError("min_motion must be >= 0")
        if self.release_no_hand_frames < 1:
            raise ValueError("release_no_hand_frames must be >= 1")


@dataclass(frozen=True)
class WindowDecision:
    raw_label: str
    label: str
    confidence: float
    margin: float
    presence_ratio: float
    motion: float
    reason: str
    probabilities: dict[str, float]


def hand_presence_ratio(frames: Sequence) -> float:
    if not frames:
        return 0.0
    return sum(bool(f.left_present or f.right_present) for f in frames) / len(frames)


def window_motion_score(frames: Sequence) -> float:
    """Mean successive landmark displacement (raw MediaPipe image XY).

    Includes fingertips: also captures gestures with fixed wrist but finger motion.
    This is a debug/quality score, not a calibrated gesture segmentation method.
    """
    if len(frames) < 2:
        return 0.0
    displacements = []
    indices = [0, 4, 8, 12, 16, 20]
    for prev, curr in zip(frames[:-1], frames[1:]):
        for side, present in (("left", "left_present"), ("right", "right_present")):
            if not (getattr(prev, present) and getattr(curr, present)):
                continue
            a = np.asarray(getattr(prev, side), dtype=np.float32)
            b = np.asarray(getattr(curr, side), dtype=np.float32)
            if a.shape != (21, 3) or b.shape != (21, 3):
                continue
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            d = np.linalg.norm(b[indices, :2] - a[indices, :2], axis=1)
            displacements.append(float(d.mean()))
    return float(np.mean(displacements)) if displacements else 0.0


def decide_window(
    probabilities: np.ndarray,
    labels: Sequence[str],
    frames: Sequence,
    gate: LiveGateConfig,
) -> WindowDecision:
    gate.validate()
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(labels) != len(p) or len(p) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Invalid label/probability dimensions")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1 + 1e-4):
        raise ValueError("Bad model probabilities")
    idx = np.argsort(p)[::-1]
    top = str(labels[int(idx[0])])
    conf = float(p[idx[0]])
    margin = conf - float(p[idx[1]])
    presence = hand_presence_ratio(frames)
    motion = window_motion_score(frames)
    outcome, reason = top, "ACCEPT"
    if top not in ("REST", "UNKNOWN"):
        if presence < gate.min_hand_presence_ratio:
            outcome, reason = "UNKNOWN", "LOW_HAND_PRESENCE"
        elif gate.min_motion > 0 and motion < gate.min_motion:
            outcome, reason = "UNKNOWN", "LOW_MOTION"
    if conf < gate.min_confidence or margin < gate.min_margin:
        outcome, reason = "UNKNOWN", "LOW_CONFIDENCE_OR_MARGIN"
    return WindowDecision(
        raw_label=top, label=outcome, confidence=conf, margin=margin,
        presence_ratio=presence, motion=motion, reason=reason,
        probabilities={str(l): float(v) for l,v in zip(labels, p)},
    )


def frame_rest_hint(no_hand_count: int, gate: LiveGateConfig) -> bool:
    """Heuristic release signal, not a classification of sign-language REST.

    Avoids permanently locking WAIT_RELEASE if the hands have fully left frame.
    False tracking loss is possible, so provide a toggle and log its source.
    """
    return bool(gate.auto_release_no_hand and no_hand_count >= gate.release_no_hand_frames)


def readable_tokens(tokens: Sequence[str], max_visible: int = 8) -> str:
    out = list(tokens)[-max_visible:]
    return "  |  ".join(out) if out else "(waiting for confirmed sign)"
