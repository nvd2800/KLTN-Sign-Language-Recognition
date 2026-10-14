from pathlib import Path
import numpy as np
from core.feature_phase4 import Phase4Config, normalize_hand_xyz, build_phase4_features, inspect_feature_arrays

CFG=Phase4Config()

def hand_seq(T=4,offset=(0.2,0.3,0.1),scale=1.0):
    base=np.zeros((T,21,3),dtype=np.float32)
    # construct a non-degenerate simple hand
    for j in range(21):
        base[:,j,0]=offset[0]+scale*(j*0.01)
        base[:,j,1]=offset[1]+scale*(j*0.005)
        base[:,j,2]=offset[2]+scale*(j*0.002)
    # make middle MCP reference explicit
    base[:,9,:]=np.array(offset,dtype=np.float32)+scale*np.array([0.09,0.04,0.02],dtype=np.float32)
    return base

def test_wrist_becomes_zero():
    h=hand_seq()
    r=normalize_hand_xyz(h,np.ones(4,bool),CFG)
    assert np.allclose(r["normalized"][:,0,:],0.0,atol=1e-6)

def test_translation_invariance():
    h1=hand_seq(offset=(0.2,0.3,0.1))
    h2=h1+np.array([0.4,-0.2,0.3],dtype=np.float32)
    a=normalize_hand_xyz(h1,np.ones(4,bool),CFG)["normalized"]
    b=normalize_hand_xyz(h2,np.ones(4,bool),CFG)["normalized"]
    assert np.allclose(a,b,atol=1e-5)

def test_scale_invariance():
    h1=hand_seq(scale=1.0)
    h2=hand_seq(scale=2.5)
    a=normalize_hand_xyz(h1,np.ones(4,bool),CFG)["normalized"]
    b=normalize_hand_xyz(h2,np.ones(4,bool),CFG)["normalized"]
    assert np.allclose(a,b,atol=1e-5)

def test_output_dimension_126_and_left_right_order():
    L=hand_seq(offset=(0.1,0.2,0.0))
    R=hand_seq(offset=(0.6,0.2,0.0),scale=1.3)
    a=build_phase4_features(L,R,np.ones(4,bool),np.ones(4,bool),CFG)
    assert a["normalized_xyz126"].shape==(4,126)
    assert np.allclose(a["normalized_xyz126"][:,:63],a["left_normalized_xyz"].reshape(4,63))
    assert np.allclose(a["normalized_xyz126"][:,63:],a["right_normalized_xyz"].reshape(4,63))

def test_missing_hand_is_zero_with_mask():
    h=hand_seq()
    p=np.array([1,0,1,0],dtype=bool)
    r=normalize_hand_xyz(h,p,CFG)
    assert np.all(r["normalized"][~p]==0)
    assert np.array_equal(r["output_present"],p)

def test_zero_scale_invalidated():
    h=hand_seq()
    h[1,9,:]=h[1,0,:]
    r=normalize_hand_xyz(h,np.ones(4,bool),CFG)
    assert not r["output_present"][1]
    assert np.all(r["normalized"][1]==0)

def test_features_are_finite():
    L=hand_seq(); R=hand_seq(scale=1.2)
    a=build_phase4_features(L,R,np.ones(4,bool),np.ones(4,bool),CFG)
    assert np.isfinite(a["normalized_xyz126"]).all()

def test_audit_pass():
    L=hand_seq(); R=hand_seq(scale=1.2)
    a=build_phase4_features(L,R,np.ones(4,bool),np.ones(4,bool),CFG)
    audit=inspect_feature_arrays(a,CFG)
    assert audit["pass"]
    assert audit["feature_dim"]==126
