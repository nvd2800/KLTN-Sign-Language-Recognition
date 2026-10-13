from core.types import FrameLandmarks, PresenceMask
from core.quality import BasicQualityChecker


def test_missing_modalities_do_not_make_phase0_frame_invalid():
    q=BasicQualityChecker().inspect(FrameLandmarks(0,0.0,presence=PresenceMask()))
    assert q.valid is True
    assert "no_hand_detected" in q.notes
