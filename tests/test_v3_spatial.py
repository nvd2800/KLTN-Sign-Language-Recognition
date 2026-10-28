import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.feature_spatial import spatial60,SPATIAL_DIM,load_raw_npz
from core.release_v3 import NoHandReleaseScheduler
from core.preprocess_phase7 import uniform_resample_or_pad,apply_standardizer,FEATURE_ARRAYS


def sample(T=32):
    l=np.zeros((T,21,3),np.float32);r=np.zeros_like(l)
    p=np.zeros((T,33,3),np.float32)
    p[:,11,:2]=[.35,.45];p[:,12,:2]=[.55,.45]
    p[:,0,:2]=[.45,.2];p[:,9,:2]=[.44,.28];p[:,10,:2]=[.46,.28]
    for t in range(T):
        for i in range(21):l[t,i]=[.37+(i%5)*.01,.34+(i//5)*.02,(i%4)*.005]
        # ensure wrist-middleMCP, normals nonzero
    return l,r,p,np.ones(T,bool),np.zeros(T,bool),np.ones(T,bool)


def test_shape_mask_and_missing_hand():
    inp=sample()
    feat,mask=spatial60(*inp,return_mask=True)
    assert feat.shape==(32,60) and mask.shape==(32,60)
    assert np.all(feat[:,30:]==0)
    assert mask[:,0].all() and mask[:,8].all()
    assert np.isfinite(feat).all()


def test_pose_missing_handshape_survives():
    l,r,p,lp,rp,pp=sample()
    pp[:]=False
    feat,mask=spatial60(l,r,p,lp,rp,pp,return_mask=True)
    assert not mask[:,0:8].any()
    assert mask[:,8:28].any()
    assert np.allclose(feat[:,0:8],0)


def test_shift_by_one_shoulder_changes_position_not_curl():
    l,r,p,lp,rp,pp=sample()
    a=spatial60(l,r,p,lp,rp,pp)
    b=l.copy();b[:,:,0]+=.2 # shoulder width exactly .2
    z=spatial60(b,r,p,lp,rp,pp)
    assert np.allclose(z[:,0]-a[:,0],1,atol=1e-5)
    assert np.allclose(z[:,8:13],a[:,8:13],atol=1e-5)


def test_bend_changes_curl():
    l,r,p,lp,rp,pp=sample()
    a=spatial60(l,r,p,lp,rp,pp)
    b=l.copy();b[:,7,:]+=[.03,.02,0];b[:,8,:]+=[.03,-.03,0]
    z=spatial60(b,r,p,lp,rp,pp)
    assert np.max(np.abs(z[:,9]-a[:,9]))>.01


def test_no_hand_release_scheduler():
    schedule=NoHandReleaseScheduler(4,2)
    flags=[schedule.update(False) for _ in range(10)]
    assert [i for i,v in enumerate(flags,1) if v]==[4,6,8,10]
    assert not schedule.update(True)
    assert not schedule.update(False)


def test_513_dim_feature_config():
    assert FEATURE_ARRAYS['hands_pose_face453'][1]==453
    assert FEATURE_ARRAYS['hands_pose_face_spatial513'][1]==513


def test_padding_standardizer_513():
    seq=np.ones((25,513),np.float32)
    x,mask,src=uniform_resample_or_pad(seq,32)
    z=apply_standardizer(x[None],mask[None],np.zeros(513),np.ones(513))
    assert z.shape==(1,32,513) and np.all(z[0,25:]==0)


def test_nan_rejected():
    inp=list(sample())
    inp[0][0,0,0]=np.nan
    with pytest.raises(ValueError):spatial60(*inp)


def test_mismatched_presence_rejected():
    inp=list(sample());inp[3]=np.zeros(2,bool)
    with pytest.raises(ValueError):spatial60(*inp)


def test_raw_npz_read(tmp_path):
    l,r,p,lp,rp,pp=sample()
    path=tmp_path/'landmarks.npz'
    np.savez_compressed(path,left_hand=l,right_hand=r,pose=p,
                        left_present=lp,right_present=rp,pose_present=pp)
    z=load_raw_npz(path)
    assert z[0].shape==(32,21,3)
    assert z[6] is None
