import numpy as np
import pytest
from core.types import FrameLandmarks


def test_hand_shape_is_21x3():
    f=FrameLandmarks(frame_index=0,timestamp=0.0)
    f.validate()
    assert f.left_hand.shape == (21,3)
    assert f.right_hand.shape == (21,3)


def test_invalid_hand_shape_rejected():
    f=FrameLandmarks(frame_index=0,timestamp=0.0,left_hand=np.zeros((21,2),dtype=np.float32))
    with pytest.raises(ValueError):
        f.validate()
