import numpy as np

from core.feature_phase6 import (
    Phase6Config,
    compute_pose_features,
    compute_face_features,
    build_phase6_features,
    inspect_phase6_arrays,
)

CFG=Phase6Config()


def make_pose(T=5):
    pose=np.zeros((T,33,3),dtype=np.float32)

    for t in range(T):
        # shoulder line
        pose[t,11]=[-0.2,0.0,0.0]
        pose[t,12]=[ 0.2,0.0,0.0]

        # nose
        pose[t,0]=[0.0,-0.5,0.0]

        # elbows
        pose[t,13]=[-0.35,0.25,0.0]
        pose[t,14]=[ 0.35,0.25,0.0]

        # wrists
        pose[t,15]=[-0.45,0.45,0.0]
        pose[t,16]=[ 0.45,0.45,0.0]

    return pose


def make_face(T=5):
    face=np.zeros((T,468,3),dtype=np.float32)

    for t in range(T):
        # Eye outer corners define scale = 1.0
        face[t,33]=[-0.5,0.0,0.0]
        face[t,263]=[0.5,0.0,0.0]

        # Nose
        face[t,1]=[0.0,0.25,0.0]

        # Mouth
        face[t,13]=[0.0,0.50,0.0]
        face[t,14]=[0.0,0.60,0.0]
        face[t,61]=[-0.20,0.55,0.0]
        face[t,291]=[0.20,0.55,0.0]

        # Eyes
        face[t,159]=[-0.25,-0.02,0.0]
        face[t,145]=[-0.25,0.03,0.0]
        face[t,386]=[0.25,-0.02,0.0]
        face[t,374]=[0.25,0.03,0.0]

        # Brows
        face[t,105]=[-0.25,-0.12,0.0]
        face[t,334]=[0.25,-0.12,0.0]

    return face


def test_pose_shape():
    pose=make_pose()
    f,m,names=compute_pose_features(
        pose,
        np.ones(5,dtype=bool),
        CFG,
    )
    assert f.shape==(5,27)
    assert m.shape==(5,)
    assert len(names)==27
    assert m.all()


def test_pose_translation_and_scale_invariance():
    pose=make_pose()

    f1,m1,_=compute_pose_features(
        pose,
        np.ones(5,bool),
        CFG,
    )

    pose2=pose*2.5 + np.array([0.7,-0.4,0.2],dtype=np.float32)

    f2,m2,_=compute_pose_features(
        pose2,
        np.ones(5,bool),
        CFG,
    )

    assert m1.all() and m2.all()
    assert np.allclose(f1,f2,atol=1e-5)


def test_pose_missing_zero():
    pose=make_pose()
    present=np.array([1,0,1,0,1],dtype=bool)

    f,m,_=compute_pose_features(
        pose,present,CFG
    )

    assert np.all(f[~present]==0)
    assert not np.any(m[~present])


def test_face_shape_and_expected_mouth():
    face=make_face()

    f,m,names=compute_face_features(
        face,
        np.ones(5,bool),
        CFG,
    )

    assert f.shape==(5,9)
    assert m.all()
    assert len(names)==9

    # face scale = 1, mouth gap = 0.1
    assert np.allclose(f[:,0],0.1,atol=1e-6)

    # mouth width = 0.4
    assert np.allclose(f[:,1],0.4,atol=1e-6)


def test_face_missing_zero():
    face=make_face()
    present=np.array([1,0,1,0,1],dtype=bool)

    f,m,_=compute_face_features(
        face,present,CFG
    )

    assert np.all(f[~present]==0)
    assert not np.any(m[~present])


def test_build_dimensions():
    T=5
    hands=np.zeros((T,417),dtype=np.float32)

    arr=build_phase6_features(
        phase5_full417=hands,
        pose_xyz=make_pose(T),
        pose_present=np.ones(T,bool),
        face_xyz=make_face(T),
        face_present=np.ones(T,bool),
        cfg=CFG,
    )

    assert arr["hands_full417"].shape==(T,417)
    assert arr["pose27"].shape==(T,27)
    assert arr["face9"].shape==(T,9)
    assert arr["ablation_hands_pose444"].shape==(T,444)
    assert arr["ablation_hands_pose_face453"].shape==(T,453)


def test_missing_modalities_are_masked():
    T=5
    hands=np.zeros((T,417),dtype=np.float32)

    arr=build_phase6_features(
        phase5_full417=hands,
        pose_xyz=None,
        pose_present=None,
        face_xyz=None,
        face_present=None,
        cfg=CFG,
    )

    assert np.all(arr["pose27"]==0)
    assert np.all(arr["face9"]==0)
    assert not np.any(arr["pose_valid"])
    assert not np.any(arr["face_valid"])


def test_no_nan_inf_and_audit_pass():
    T=5
    hands=np.ones((T,417),dtype=np.float32)

    arr=build_phase6_features(
        phase5_full417=hands,
        pose_xyz=make_pose(T),
        pose_present=np.ones(T,bool),
        face_xyz=make_face(T),
        face_present=np.ones(T,bool),
        cfg=CFG,
    )

    audit=inspect_phase6_arrays(arr)

    assert audit["pass"]
    assert np.isfinite(
        arr["ablation_hands_pose_face453"]
    ).all()
