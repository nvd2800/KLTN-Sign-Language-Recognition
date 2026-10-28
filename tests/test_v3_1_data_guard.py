import json
from pathlib import Path
import numpy as np
import pytest
from scripts.guard_v3_1_dataset import check_dataset
from scripts.collect_v3_train_negative import validate_capture
from scripts.audit_v3_train_hardnegatives import audit


def _dataset(p,train,val,test):
    p.mkdir(parents=True)
    for name,ids in [('train',train),('val',val),('test',test)]:
        np.savez_compressed(p/f'{name}.npz',sample_ids=np.asarray(ids))
    (p/'dataset_summary.json').write_text(json.dumps({'mode':'strict','signer_assignment':{'S01':'train','S08':'val','S09':'test'},'hard_negatives':[]}))


def test_test_baseline_not_in_training(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new'
    _dataset(old,['train01'],['val08'],['test09','test10'])
    _dataset(new,['train01','new_train'],['val08'],['test09','test10'])
    result=check_dataset(old,new)
    assert result['pass'] and result['baseline_test_reused_for_fitting']==0


def test_test_leak_guard_fails(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new'
    _dataset(old,['a'],['b'],['test09'])
    _dataset(new,['a','test09'],['b'],['test09'])
    result=check_dataset(old,new)
    assert not result['pass'] and result['baseline_test_reused_for_fitting']>0


def test_collect_train_only():
    assignment={'S01':'train','S08':'val','S09':'test'}
    labs=['REST','cam_on','xin_chao','xin_loi','UNKNOWN']
    assert validate_capture('xin_chao','s01',assignment,labs)==('xin_chao','S01')
    with pytest.raises(ValueError,match='not assigned to TRAIN'):
        validate_capture('xin_chao','S09',assignment,labs)
    with pytest.raises(ValueError,match='must be a known sign'):
        validate_capture('UNKNOWN','S01',assignment,labs)


def test_audit_only_train_counts(tmp_path):
    for sub in ('S01','S09'):
        d=tmp_path/'xin_chao'/sub/'wrong_position'/'trial01';d.mkdir(parents=True)
        (d/'metadata.json').write_text(json.dumps({'subject':sub,'label':'xin_chao','condition':'wrong_position'}))
        np.savez_compressed(d/'raw_landmarks.npz',left_hand=np.zeros((2,21,3)),right_hand=np.zeros((2,21,3)),pose=np.zeros((2,33,3)))
        np.savez_compressed(d/'features.npz',features453=np.zeros((2,453)))
    result=audit(tmp_path,{'S01':'train','S09':'test'},minimum=10)
    assert result['train_trials']==1 and result['test_trials']==1
    assert result['counts_train']['xin_chao/wrong_position']==1
    assert result['gaps_train'][0]['suggested_additional']>=0
