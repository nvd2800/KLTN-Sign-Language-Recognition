import numpy as np
import pytest
from scripts.prepare_v3_blind_test import fresh_signers,make_dataset


def test_blind_rejects_seen_signer():
    with pytest.raises(ValueError,match='leakage'):
        fresh_signers('S11,S09',{'S01':'train','S09':'test'})
    assert fresh_signers('S12,s11',{'S01':'train'})==['S11','S12']


def test_blind_needs_positives_and_negatives(tmp_path):
    labels=['REST','cam_on','xin_chao','xin_loi','UNKNOWN']
    with pytest.raises(RuntimeError,match='needs NEW known positives'):
        make_dataset([],tmp_path,['S11'],labels,np.zeros(513),np.ones(513))


def test_blind_prepare_reuses_frozen_scaler(tmp_path,monkeypatch):
    import json
    from scripts import prepare_v3_blind_test as mod
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    T=32
    rawdir=tmp_path/'dataset/raw/xin_chao/S11/SES01/001'
    rawdir.mkdir(parents=True)
    left=np.zeros((T,21,3),np.float32)
    right=np.zeros((T,21,3),np.float32)
    pose=np.zeros((T,33,3),np.float32)
    for i in range(21):left[:,i,:]=(.45+i*.003,.40+i*.002,i*.0001)
    pose[:,11,:2]=[.3,.4];pose[:,12,:2]=[.5,.4]
    np.savez_compressed(rawdir/'landmarks.npz',left_hand=left,right_hand=right,pose=pose,
                        left_present=np.ones(T,bool),right_present=np.zeros(T,bool),pose_present=np.ones(T,bool))
    feature_dir=tmp_path/'processed_features';feature_dir.mkdir()
    np.savez_compressed(feature_dir/'item.npz',ablation_hands_pose_face453=np.ones((T,453),np.float32))
    rows=[{'sign':'xin_chao','subject_id':'S11','session_id':'SES01','sample_id':'001','sample_path':str(rawdir),
           'features_path':str(feature_dir/'item.npz'),'phase6_status':'PASS'}]
    trial=tmp_path/'logs/feature_diagnostic/xin_chao/S11/wrong_position/t1';trial.mkdir(parents=True)
    np.savez_compressed(trial/'raw_landmarks.npz',left_hand=left,right_hand=right,pose=pose,
                        left_present=np.ones(T,bool),right_present=np.zeros(T,bool),pose_present=np.ones(T,bool))
    np.savez_compressed(trial/'features.npz',features453=np.zeros((T,453),np.float32))
    (trial/'metadata.json').write_text(json.dumps({'subject':'S11','condition':'wrong_position','label':'xin_chao'}))
    labels=['REST','cam_on','xin_chao','xin_loi','UNKNOWN']
    X,M,y,ids,neg,kn,nnum=mod.make_dataset(rows,tmp_path/'logs/feature_diagnostic',['S11'],labels,
                                             np.ones(513),np.ones(513))
    assert X.shape==(2,32,513) and y.tolist()==[2,4]
    assert kn==1 and nnum==1 and M.shape==(2,32)
    # Frozen scaler mean=1, so original known Phase06 values of 1 -> zero (not refitted).
    assert np.allclose(X[0,:,:453],0.)
