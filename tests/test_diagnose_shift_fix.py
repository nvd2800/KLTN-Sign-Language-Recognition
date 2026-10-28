"""Regression checks for broadcasting bug, missing hands, and Pose masking."""
import numpy as np
import pytest
from scripts.diagnose_position_sensitivity import shift_hands_horizontally_by_shoulder_width


def landmarks(T=5):
    l = np.zeros((T, 21, 3), np.float32)
    r = np.zeros((T, 21, 3), np.float32)
    p = np.zeros((T, 33, 3), np.float32)
    l[:, :, 0] = 0.2
    r[:, :, 0] = 0.6
    p[:, 11, 0] = 0.3
    p[:, 12, 0] = 0.7  # shoulder width 0.4
    return l, r, p


def test_only_one_hand_visible_no_broadcast():
    l, r, p = landmarks()
    lp = np.ones(5, bool)
    rp = np.zeros(5, bool)
    pp = np.ones(5, bool)
    sl, sr = shift_hands_horizontally_by_shoulder_width(l,r,p,lp,rp,pp,1.0)
    np.testing.assert_allclose(sl[:, :, 0], 0.6, rtol=1e-6)
    np.testing.assert_array_equal(sr, r)
    np.testing.assert_allclose(l[:, :, 0], 0.2)


def test_both_hands_and_pose_gaps():
    l, r, p = landmarks(5)
    lp = np.array([1, 0, 1, 0, 1], bool)
    rp = np.array([1, 1, 0, 0, 1], bool)
    pp = np.array([1, 1, 1, 0, 1], bool)
    sl, sr = shift_hands_horizontally_by_shoulder_width(l,r,p,lp,rp,pp,1.0)
    np.testing.assert_allclose(sl[0, :, 0], 0.6)
    np.testing.assert_allclose(sl[1, :, 0], 0.2)
    np.testing.assert_allclose(sl[4, :, 0], 0.6)
    np.testing.assert_allclose(sr[0, :, 0], 1.0)
    np.testing.assert_allclose(sr[1, :, 0], 1.0)
    np.testing.assert_allclose(sr[3, :, 0], 0.6)


def test_zero_present_hand_is_not_a_broadcasting_error():
    l, r, p = landmarks()
    with pytest.raises(ValueError, match='No hand/Pose co-visible'):
        shift_hands_horizontally_by_shoulder_width(l,r,p,np.zeros(5,bool),np.zeros(5,bool),np.ones(5,bool),1.0)


def test_invalid_pose_is_reported():
    l,r,p = landmarks()
    with pytest.raises(ValueError, match='No valid Pose'):
        shift_hands_horizontally_by_shoulder_width(l,r,p,np.ones(5,bool),np.ones(5,bool),np.zeros(5,bool),1.0)


def test_shift_respects_shoulder_scale_and_only_x():
    l, r, p = landmarks()
    p[3, 12, 0] = 0.9  # shoulder width 0.6 for frame 3
    sl, sr = shift_hands_horizontally_by_shoulder_width(l,r,p,np.ones(5,bool),np.zeros(5,bool),np.ones(5,bool),2)
    np.testing.assert_allclose(sl[0, :, 0], 1.0)
    np.testing.assert_allclose(sl[3, :, 0], 1.4)
    np.testing.assert_array_equal(sl[:, :, 1:], l[:, :, 1:])
    np.testing.assert_array_equal(sr, r)


def test_full_candidate_pipeline_513d_shift(tmp_path):
    """Real Phase04/05/06 recompute + Spatial60 + resampling for single-hand sample."""
    from scripts.diagnose_position_sensitivity import candidate
    T = 32
    left = np.zeros((T, 21, 3), np.float32)
    right = np.zeros_like(left)
    pose = np.zeros((T, 33, 3), np.float32)
    face = np.zeros((T, 468, 3), np.float32)
    for t in range(T):
        for j in range(21):
            left[t, j] = [0.3 + 0.002*t + 0.008*(j%5), 0.4 + 0.009*(j//5), 0.01*(j//5)]
    pose[:, 11, :2] = [0.4, 0.4]
    pose[:, 12, :2] = [0.6, 0.4]
    pose[:, 0, :2] = [0.5, 0.2]
    pose[:, 9, :2] = [0.48, 0.25]
    pose[:, 10, :2] = [0.52, 0.25]
    np.savez_compressed(tmp_path/'raw_landmarks.npz', left_hand=left,right_hand=right,
                        pose=pose,face=face,left_present=np.ones(T,bool),
                        right_present=np.zeros(T,bool),pose_present=np.ones(T,bool),
                        face_present=np.zeros(T,bool),timestamps=np.arange(T)/30)
    np.savez_compressed(tmp_path/'features.npz',features453=np.zeros((T,453),np.float32))
    mean=np.zeros(513,np.float32)
    std=np.ones(513,np.float32)
    x,sp,mask=candidate(tmp_path,mean,std)
    xs,sps,masks=candidate(tmp_path,mean,std,shift_shoulders=1.0)
    assert x.shape==xs.shape==(32,513)
    assert np.isfinite(xs).all()
    np.testing.assert_allclose((sps[:,0]-sp[:,0])[mask[:,0]],1.0,atol=1e-5)
