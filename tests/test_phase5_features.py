import numpy as np
from core.feature_phase5 import (
    Phase5Config, angle_abc, compute_hand_angles,
    compute_velocity_acceleration, build_phase5_features,
    inspect_phase5_arrays
)

CFG=Phase5Config()

def make_hand(T=5,scale=1.0,offset=(0.2,0.3,0.1)):
    h=np.zeros((T,21,3),dtype=np.float32)
    for t in range(T):
        for j in range(21):
            h[t,j]=np.array(offset,dtype=np.float32)+scale*np.array(
                [0.01*j,0.004*j,0.002*j],dtype=np.float32
            )
        # Make all finger chains non-collinear enough for stable angles.
        h[t,4,1]+=0.03
        h[t,8,1]+=0.04
        h[t,12,1]+=0.05
        h[t,16,1]+=0.04
        h[t,20,1]+=0.03
    return h

def make_phase4(T=5):
    L=make_hand(T)
    R=make_hand(T,scale=1.1,offset=(0.6,0.3,0.1))
    # Approximate Phase-4 normalized hands by wrist+scale normalization.
    def norm(h):
        out=np.zeros_like(h)
        for t in range(T):
            c=h[t]-h[t,0]
            s=np.linalg.norm(h[t,9]-h[t,0])
            out[t]=c/s
        return out
    ln=norm(L); rn=norm(R)
    pos=np.concatenate([ln.reshape(T,63),rn.reshape(T,63)],axis=1)
    hp=np.ones((T,2),dtype=bool)
    return {
        "normalized_xyz126":pos.astype(np.float32),
        "left_normalized_xyz":ln.astype(np.float32),
        "right_normalized_xyz":rn.astype(np.float32),
        "hand_presence2":hp,
        "scale2":np.ones((T,2),dtype=np.float32)*0.1,
        "timestamps":np.arange(T,dtype=np.float64)/30.0,
    },L,R

def test_angle_right_angle():
    a=np.array([1.,0.,0.])
    b=np.array([0.,0.,0.])
    c=np.array([0.,1.,0.])
    theta=angle_abc(a,b,c,1e-8)
    assert np.isclose(theta,np.pi/2,atol=1e-6)

def test_angles_shape_and_range():
    h=make_hand()
    a,m=compute_hand_angles(h,np.ones(5,bool),eps=1e-8,normalize_by_pi=True)
    assert a.shape==(5,15)
    assert m.shape==(5,15)
    assert np.all(a[m]>=0.0)
    assert np.all(a[m]<=1.0)

def test_missing_hand_angles_zero_mask_false():
    h=make_hand()
    p=np.array([1,0,1,0,1],dtype=bool)
    a,m=compute_hand_angles(h,p,eps=1e-8,normalize_by_pi=True)
    assert np.all(a[~p]==0)
    assert not np.any(m[~p])

def test_velocity_constant_motion_and_acceleration_zero():
    T=5
    X=np.zeros((T,126),dtype=np.float32)
    for t in range(T):
        X[t,:63]=t*0.1
        X[t,63:]=t*0.2
    hp=np.ones((T,2),dtype=bool)
    ts=np.arange(T,dtype=np.float64)
    v,a,vm,am=compute_velocity_acceleration(X,hp,ts,CFG)
    assert np.allclose(v[1:,:63],0.1,atol=1e-6)
    assert np.allclose(v[1:,63:],0.2,atol=1e-6)
    assert np.allclose(a[2:],0.0,atol=1e-6)
    assert not vm[0].any()
    assert not am[:2].any()

def test_motion_does_not_bridge_missing_frames():
    T=5
    X=np.zeros((T,126),dtype=np.float32)
    for t in range(T):
        X[t]=t
    hp=np.ones((T,2),dtype=bool)
    hp[2,0]=False
    ts=np.arange(T,dtype=np.float64)
    v,a,vm,am=compute_velocity_acceleration(X,hp,ts,CFG)
    assert not vm[2,0]
    assert not vm[3,0]
    assert np.all(v[2,:63]==0)
    assert np.all(v[3,:63]==0)

def test_build_dimensions():
    p4,L,R=make_phase4()
    arr=build_phase5_features(
        phase4_arrays=p4,
        raw_left=L,raw_right=R,
        pose_xyz=None,pose_present=None,cfg=CFG
    )
    assert arr["position126"].shape==(5,126)
    assert arr["angles30"].shape==(5,30)
    assert arr["distances9"].shape==(5,9)
    assert arr["velocity126"].shape==(5,126)
    assert arr["acceleration126"].shape==(5,126)
    assert arr["ablation_position_velocity252"].shape==(5,252)
    assert arr["ablation_position_velocity_acceleration378"].shape==(5,378)
    assert arr["full_features417"].shape==(5,417)

def test_optional_body_distances_missing_are_masked():
    p4,L,R=make_phase4()
    arr=build_phase5_features(
        phase4_arrays=p4,
        raw_left=L,raw_right=R,
        pose_xyz=None,pose_present=None,cfg=CFG
    )
    # body distances indices 3..8
    assert np.all(arr["distances9"][:,3:]==0)
    assert not np.any(arr["distance_mask9"][:,3:])

def test_no_nan_inf():
    p4,L,R=make_phase4()
    arr=build_phase5_features(
        phase4_arrays=p4,
        raw_left=L,raw_right=R,
        pose_xyz=None,pose_present=None,cfg=CFG
    )
    audit=inspect_phase5_arrays(arr)
    assert audit["pass"]
    assert np.isfinite(arr["full_features417"]).all()
