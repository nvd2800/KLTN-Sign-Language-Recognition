from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Iterable

import numpy as np


class SegmentState(str, Enum):
    REST = "REST"
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class DecisionConfig:
    min_confidence: float = 0.85
    min_top1_margin: float = 0.20
    stable_window_count: int = 5
    stable_required: int = 4
    min_predict_frames: int = 24
    preview_stride_frames: int = 4

    start_motion_threshold: float = 0.012
    end_motion_threshold: float = 0.004
    start_required_frames: int = 3
    end_still_frames: int = 6
    min_active_frames: int = 12
    max_active_frames: int = 60
    min_hand_presence_ratio: float = 0.45
    cooldown_seconds: float = 0.90
    require_rest_before_repeat: bool = True

    unknown_label: str = "UNKNOWN"


@dataclass
class PredictionResult:
    top1_label: str
    top1_prob: float
    top2_prob: float
    margin: float
    gated_label: str
    gate_pass: bool


def gate_prediction(
    labels: list[str],
    probs: np.ndarray,
    cfg: DecisionConfig,
) -> PredictionResult:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)

    if p.size != len(labels):
        raise ValueError("probability/label size mismatch")

    if not np.isfinite(p).all():
        raise ValueError("non-finite probabilities")

    order = np.argsort(p)[::-1]
    i1 = int(order[0])
    i2 = int(order[1]) if len(order) > 1 else i1

    top1 = float(p[i1])
    top2 = float(p[i2]) if len(order) > 1 else 0.0
    margin = float(top1 - top2)

    passed = (
        top1 >= cfg.min_confidence
        and margin >= cfg.min_top1_margin
    )

    return PredictionResult(
        top1_label=labels[i1],
        top1_prob=top1,
        top2_prob=top2,
        margin=margin,
        gated_label=labels[i1] if passed else cfg.unknown_label,
        gate_pass=passed,
    )


def stable_label(
    recent_labels: Iterable[str],
    final_label: str,
    cfg: DecisionConfig,
) -> tuple[bool, int, int]:
    vals = list(recent_labels)[-cfg.stable_window_count:]

    if final_label == cfg.unknown_label:
        return False, 0, len(vals)

    count = sum(x == final_label for x in vals)

    return (
        count >= cfg.stable_required,
        count,
        len(vals),
    )


def _valid_xy(arr: np.ndarray | None):
    if arr is None:
        return None

    x = np.asarray(arr, dtype=np.float32)

    if x.ndim != 2 or x.shape[1] < 2:
        return None

    if not np.isfinite(x[:, :2]).all():
        return None

    return x[:, :2]


def motion_score(
    *,
    left_now: np.ndarray | None,
    left_prev: np.ndarray | None,
    right_now: np.ndarray | None,
    right_prev: np.ndarray | None,
    pose_now: np.ndarray | None = None,
    pose_prev: np.ndarray | None = None,
) -> float:
    """
    Realtime segmentation motion in raw normalized-image XY coordinates.

    This intentionally does NOT use Phase04 hand-centered coordinates because
    those remove hand translation, which is exactly what START/END detection
    needs to observe.
    """
    values: list[float] = []

    def add_pair(now, prev):
        a = _valid_xy(now)
        b = _valid_xy(prev)

        if a is None or b is None or a.shape != b.shape:
            return

        values.append(
            float(np.mean(np.linalg.norm(a - b, axis=1)))
        )

    add_pair(left_now, left_prev)
    add_pair(right_now, right_prev)

    # Pose only uses wrists/elbows/shoulders if a standard 33-landmark pose is available.
    pn = _valid_xy(pose_now)
    pp = _valid_xy(pose_prev)

    if pn is not None and pp is not None and pn.shape == pp.shape and pn.shape[0] > 16:
        idx = [11, 12, 13, 14, 15, 16]
        values.append(
            float(np.mean(np.linalg.norm(pn[idx] - pp[idx], axis=1)))
        )

    return max(values) if values else 0.0


class RealtimeDecisionEngine:
    def __init__(self, cfg: DecisionConfig):
        self.cfg = cfg
        self.state = SegmentState.REST

        self.start_counter = 0
        self.still_counter = 0
        self.active_frames = 0
        self.hand_present_frames = 0

        self.preview_labels: Deque[str] = deque(
            maxlen=cfg.stable_window_count
        )

        self.last_emitted_label: str | None = None
        self.last_emit_time: float = -1e9

        self.rest_seen_since_emit = True

    def reset_segment(self):
        self.start_counter = 0
        self.still_counter = 0
        self.active_frames = 0
        self.hand_present_frames = 0
        self.preview_labels.clear()

    def add_preview_label(self, label: str):
        self.preview_labels.append(label)

    def update(
        self,
        *,
        motion: float,
        any_hand_present: bool,
        now_s: float,
    ) -> str | None:
        """
        Returns event:
          "START"
          "END"
          "MAX_END"
          "READY"
          None
        """

        if self.state == SegmentState.REST:
            if motion >= self.cfg.start_motion_threshold and any_hand_present:
                self.start_counter += 1
            else:
                self.start_counter = 0

            if self.start_counter >= self.cfg.start_required_frames:
                self.state = SegmentState.ACTIVE
                self.active_frames = 0
                self.hand_present_frames = 0
                self.still_counter = 0
                self.preview_labels.clear()
                return "START"

            return None

        if self.state == SegmentState.ACTIVE:
            self.active_frames += 1

            if any_hand_present:
                self.hand_present_frames += 1

            if (
                self.active_frames >= self.cfg.min_active_frames
                and motion <= self.cfg.end_motion_threshold
            ):
                self.still_counter += 1
            else:
                self.still_counter = 0

            if self.active_frames >= self.cfg.max_active_frames:
                self.state = SegmentState.COOLDOWN
                return "MAX_END"

            if self.still_counter >= self.cfg.end_still_frames:
                self.state = SegmentState.COOLDOWN
                return "END"

            return None

        # COOLDOWN
        cooldown_ok = (
            now_s - self.last_emit_time >= self.cfg.cooldown_seconds
        )

        rest_now = (
            motion <= self.cfg.end_motion_threshold
        )

        if rest_now:
            self.rest_seen_since_emit = True

        repeat_ok = (
            (not self.cfg.require_rest_before_repeat)
            or self.rest_seen_since_emit
        )

        if cooldown_ok and repeat_ok:
            self.state = SegmentState.REST
            self.reset_segment()
            return "READY"

        return None

    def hand_presence_ratio(self) -> float:
        if self.active_frames <= 0:
            return 0.0
        return self.hand_present_frames / self.active_frames

    def can_accept_segment(self) -> bool:
        return (
            self.active_frames >= self.cfg.min_active_frames
            and self.hand_presence_ratio()
                >= self.cfg.min_hand_presence_ratio
        )

    def confirm_prediction(
        self,
        final: PredictionResult,
        now_s: float,
    ) -> tuple[str, dict]:
        stable_ok, stable_count, seen = stable_label(
            self.preview_labels,
            final.gated_label,
            self.cfg,
        )

        accepted = (
            final.gate_pass
            and self.can_accept_segment()
            and stable_ok
        )

        label = (
            final.top1_label
            if accepted
            else self.cfg.unknown_label
        )

        if accepted:
            self.last_emitted_label = label
            self.last_emit_time = now_s
            self.rest_seen_since_emit = False

        details = {
            "accepted": accepted,
            "final_top1": final.top1_label,
            "final_confidence": final.top1_prob,
            "final_margin": final.margin,
            "stable_count": stable_count,
            "stable_seen": seen,
            "hand_presence_ratio": self.hand_presence_ratio(),
            "segment_frames": self.active_frames,
        }

        return label, details
