from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from core.phase8_live import FrameSample, build_model_input
from core.feature_phase4 import Phase4Config
from core.feature_phase5 import Phase5Config
from core.feature_phase6 import Phase6Config
from core.spatial_handshape_diagnostic import (
    v4_candidate32, extract_diagnostic_features,compare_files,
    pairwise_report,save_trial,GROUP453,GROUP32,
)


def hand(center=(.3,.43), folded=False):
    a=np.zeros((21,3),np.float32)
    a[0,:2]=center
    anchor={1:(.02,-.015),5:(-.075,-.04),9:(-.025,-.06),13:(.03,-.06),17:(.08,-.038)}
    for i,(dx,dy) in anchor.items():
        a[i,:2]=np.array(center)+[dx,dy]
    chains=((1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20))
    for c in chains:
        base=a[c[0],:2].copy()
        for j,k in enumerate(c[1:],1):
            a[k,:2]=base+[.003*j, -.018*j]
            if folded and j>=2:
                a[k,:2]=base+[.016*(4-j),.015*j]
    return a


def frames(kind='correct',T=45,pose=True,face=True,has_hand=True):
    out=[]
    for t in range(T):
        drift=.005*np.sin(t/9)
        shift=(.15,.1) if kind=='wrong_position' else (0,0)
        center=np.array([.30+shift[0]+drift,.43+shift[1]],np.float32)
        l=hand(center,folded=kind=='wrong_handshape') if has_hand else np.zeros((21,3),np.float32)
        r=np.zeros((21,3),np.float32)
        p=np.zeros((33,3),np.float32)
        p[11,:2]=[.39,.42];p[12,:2]=[.65,.42]
        p[0,:2]=[.52,.18]
        p[15,:2]=[.30,.47];p[16,:2]=[.7,.47]
        f=np.zeros((468,3),np.float32)
        f[33,:2]=[.47,.19];f[263,:2]=[.58,.19]
        f[13,:2]=[.51,.26];f[14,:2]=[.52,.275]
        out.append(FrameSample(l,r,p,f,has_hand,False,pose,face,t/30+0.0001))
    return out


CFG=(Phase4Config(),Phase5Config(),Phase6Config())


def test_dimensions():
    raw,derived,qc=extract_diagnostic_features(frames(),*CFG)
    assert derived['features453'].shape==(45,453)
    assert derived['candidate_v4_32'].shape==(45,32)
    assert derived['candidate_v4_mask32'].shape==(45,32)
    assert qc['left_ratio']==1.0


def test_hand_centering_loses_position_but_body_features_recover():
    _,a,_=extract_diagnostic_features(frames('correct'),*CFG)
    _,b,_=extract_diagnostic_features(frames('wrong_position'),*CFG)
    assert np.allclose(a['phase4_normalized_xyz126'],b['phase4_normalized_xyz126'],atol=3e-5)
    assert np.linalg.norm(a['candidate_v4_32'][:,:12]-b['candidate_v4_32'][:,:12]) > 0.5


def test_handshape_bend_detectable():
    _,a,_=extract_diagnostic_features(frames('correct'),*CFG)
    _,b,_=extract_diagnostic_features(frames('wrong_handshape'),*CFG)
    assert np.linalg.norm(a['phase5_angles30']-b['phase5_angles30'])>0.1
    assert np.linalg.norm(a['candidate_v4_32'][:,12:22]-b['candidate_v4_32'][:,12:22])>0.1


def test_candidate_invalid_without_hand():
    _,features,qc=extract_diagnostic_features(frames(has_hand=False),*CFG)
    assert qc['any_hand_ratio']==0
    assert not features['candidate_v4_mask32'].any()
    assert not np.isnan(features['features453']).any()


def test_no_pose_invalidates_spatial_but_not_curl():
    x,m=v4_candidate32(frames(pose=False))
    assert not m[:,:12].any()
    assert m[:,12:17].all()


def test_no_face_invalidates_mouth_only():
    _,mask=v4_candidate32(frames(face=False))
    assert not mask[:,3].any()
    assert mask[:,0].all()
    assert mask[:,4].all()


def test_existing_453_matches_phase08_live_model_preprocessing():
    f=frames()
    _,derived,_=extract_diagnostic_features(f,*CFG)
    X,temporal_mask,raw=build_model_input(f,*CFG,np.zeros(453),np.ones(453),32)
    assert np.array_equal(raw,derived['features453'])
    assert X.shape==(32,453)
    assert temporal_mask.sum()==32


def test_feature_groups_sum_to_dim():
    assert list(GROUP453.values())[0][0]==0
    assert list(GROUP453.values())[-1][-1]==453
    assert list(GROUP32.values())[0][0]==0
    assert list(GROUP32.values())[-1][-1]==32


def test_save_and_compare_reports(tmp_path):
    root=tmp_path/'clips'
    for kind in ('correct','wrong_position','wrong_handshape','correct_repeat'):
        raw,features,qc=extract_diagnostic_features(frames(kind if kind in ('wrong_position','wrong_handshape') else 'correct'),*CFG)
        save_trial(root/'xin_chao'/'S01'/kind/'20261018T100000Z',raw,features,
                   {'condition':kind,'label':'xin_chao','subject':'S01','captured_at':kind,'qc':qc})
    rows,path=pairwise_report(root,'xin_chao','S01',tmp_path/'reports')
    assert path.exists() and path.with_suffix('.json').exists()
    assert len(rows)==3*(len(GROUP453)+len(GROUP32))
    ours={(r['condition'],r['group']):r for r in rows}
    assert ours[('wrong_position','hand_body_position12')]['rms']>.01
    assert ours[('wrong_handshape','finger_curl10')]['rms']>.01


def test_no_correct_reference_fails(tmp_path):
    root=tmp_path/'clips'
    raw,feature,qc=extract_diagnostic_features(frames(),*CFG)
    save_trial(root/'xin_chao'/'S01'/'wrong_position'/'1',raw,feature,
               {'condition':'wrong_position','label':'xin_chao','subject':'S01'})
    with pytest.raises(RuntimeError,match='correct'):
        pairwise_report(root,'xin_chao','S01',tmp_path/'reports')


def test_duplicate_save_no_overwrite(tmp_path):
    raw,feature,_=extract_diagnostic_features(frames(),*CFG)
    target=tmp_path/'same'
    save_trial(target,raw,feature,{'condition':'correct'})
    with pytest.raises(FileExistsError):
        save_trial(target,raw,feature,{'condition':'correct'})


def test_compare_with_missing_modality_masks(tmp_path):
    a,fa,_=extract_diagnostic_features(frames(pose=False),*CFG)
    b,fb,_=extract_diagnostic_features(frames(pose=False),*CFG)
    d1=tmp_path/'a';d2=tmp_path/'b'
    save_trial(d1,a,fa,{'condition':'correct'})
    save_trial(d2,b,fb,{'condition':'correct_repeat'})
    comparison=compare_files(d1/'features.npz',d2/'features.npz')
    assert comparison['hand_body_position12']['compared_ratio']==0.0
    assert comparison['hand_body_position12']['rms'] is None


def test_no_compatibility_drift_for_timestamps():
    f=frames()
    f[1]=FrameSample(f[1].left,f[1].right,f[1].pose,f[1].face,
                    f[1].left_present,f[1].right_present,True,True,f[0].timestamp)
    with pytest.raises(ValueError,match='Timestamps'):
        extract_diagnostic_features(f,*CFG)
