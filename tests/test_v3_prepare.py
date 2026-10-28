import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_v3_spatial import main
from core.feature_spatial import spatial60


def test_merge_real_annotated_negative_dev(tmp_path,monkeypatch):
    known=tmp_path/'known';known.mkdir()
    trials=tmp_path/'trials'/'xin_chao'/'S01'/'wrong_position'/'trial_1';trials.mkdir(parents=True)
    out=tmp_path/'out'
    labels=['REST','cam_on','xin_chao','xin_loi']
    (known/'labels.json').write_text(json.dumps({'labels':labels}))
    np.savez_compressed(known/'preprocess_stats.npz',mean=np.zeros(513),std=np.ones(513),target_frames=[32],feature_dim=[513])
    for sp in ('train','val','test'):
        np.savez_compressed(known/f'{sp}.npz',X=np.zeros((4 if sp=='train' else 1,32,513),np.float32),
                            mask=np.ones((4 if sp=='train' else 1,32),bool),
                            y=np.arange(4) if sp=='train' else np.array([0]),
                            sample_ids=[f'{sp}_{i}' for i in range(4 if sp=='train' else 1)])
    summary=tmp_path/'knownsummary.json'
    summary.write_text(json.dumps({'subjects':{'train':['S01'],'val':['S03'],'test':['S04']}}))
    T=32;l=np.zeros((T,21,3),np.float32);r=np.zeros_like(l)
    for j in range(21):l[:,j]=[.3+j*.001,.4+j*.002,j*.001]
    p=np.zeros((T,33,3),np.float32);p[:,11,:2]=[.3,.4];p[:,12,:2]=[.5,.4]
    np.savez_compressed(trials/'raw_landmarks.npz',left_hand=l,right_hand=r,pose=p,
                        left_present=np.ones(T,bool),right_present=np.zeros(T,bool),pose_present=np.ones(T,bool),
                        timestamps=np.arange(T)/30)
    np.savez_compressed(trials/'features.npz',features453=np.zeros((T,453),np.float32))
    (trials/'metadata.json').write_text(json.dumps({'condition':'wrong_position','label':'xin_chao','subject':'S01'}))
    monkeypatch.setattr(sys,'argv',['script','--known',str(known),'--trials',str(tmp_path/'trials'),
                                  '--known-summary',str(summary),'--output',str(out),'--mode','dev'])
    main()
    with np.load(out/'train.npz') as f:
        assert f['X'].shape==(5,32,513)
        assert f['y'][-1]==4
    with np.load(out/'val.npz') as f:
        assert f['y'].shape==(1,)
    assert json.loads((out/'dataset_summary.json').read_text())['mode']=='dev'
