from __future__ import annotations
from .types import FrameLandmarks, QualityInfo


class BasicQualityChecker:
    """Phase-0 checks only. Full blur/crop/jump QC belongs to Phase 3."""

    def inspect(self, frame: FrameLandmarks) -> QualityInfo:
        notes: list[str] = []
        p = frame.presence
        if not p.left_hand and not p.right_hand:
            notes.append("no_hand_detected")
        if not p.pose:
            notes.append("pose_missing")
        if not p.face:
            notes.append("face_missing")
        return QualityInfo(
            valid=True,  # Phase 0 never rejects merely because a modality is missing.
            left_hand_present=p.left_hand,
            right_hand_present=p.right_hand,
            pose_present=p.pose,
            face_present=p.face,
            notes=notes,
        )
