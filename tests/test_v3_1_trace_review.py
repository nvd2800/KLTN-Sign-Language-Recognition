import csv,json
from pathlib import Path
import numpy as np
import pytest
from core.v3_eval_trace import (build_provenance_index,make_prediction_rows,compute_landmark_qc,
                                error_type,condition_summary,write_csv)
from scripts.review_v3_test_errors import summarize,review

LABELS=['REST','cam_on','xin_chao','xin_loi','UNKNOWN']


def dataset_fixture(tmp_path):
    data=tmp_path/'temporal_v3';data.mkdir()
    meta=[]
    for i in range(3):
        meta.append({'id':f'DIAG_{i}','split':'test','subject':'S09',
                     'condition':['wrong_position','wrong_handshape','partial'][i],
                     'reference_sign':'xin_chao','raw':str(tmp_path/'trials'/str(i))})
    (data/'dataset_summary.json').write_text(json.dumps({'mode':'strict','hard_negatives':meta}))
    return data


def test_three_unknown_test_mistakes_resolved_by_id(tmp_path):
    data=dataset_fixture(tmp_path)
    index,summary=build_provenance_index(data,tmp_path)
    y=np.asarray([4,4,4]);probs=np.array([[.80,.05,.05,.05,.05],
                   [.04,.88,.03,.03,.02],[.03,.03,.02,.91,.01]],np.float32)
    r=make_prediction_rows(LABELS,y,probs,['DIAG_0','DIAG_1','DIAG_2'],'test',index,tmp_path)
    assert [x['predicted_label'] for x in r]==['REST','cam_on','xin_loi']
    assert [x['condition'] for x in r]==['wrong_position','wrong_handshape','partial']
    assert r[0]['error_type']=='UNKNOWN_AS_REST_NO_TOKEN'
    assert r[1]['error_type']=='UNKNOWN_AS_KNOWN_TOKEN_RISK'
    s=summarize(r)
    assert s['n_errors']==3 and s['unknown_to_rest']==1 and s['unknown_to_token_risk']==2
    assert len(review(r))==3
    out=tmp_path/'predict.csv';write_csv(out,r)
    with out.open(encoding='utf-8-sig',newline='') as f:
        assert len(list(csv.DictReader(f)))==3


def test_raw_landmark_qc_flags_incomplete(tmp_path):
    T=10;left=np.zeros((T,21,3),np.float32);right=left.copy()
    pose=np.zeros((T,33,3),np.float32)
    p=tmp_path/'raw_landmarks.npz'
    np.savez_compressed(p,left_hand=left,right_hand=right,pose=pose,
                        left_present=np.array([1,1,1,0,0,0,0,0,0,0],bool),
                        right_present=np.zeros(T,bool),pose_present=np.ones(T,bool))
    q=compute_landmark_qc(p,'xin_chao')
    assert q['main_hand_valid_ratio']==.3
    assert q['any_hand_valid_ratio']==.3
    assert q['qc_status']=='REVIEW_LOW_MAIN_HAND'
    assert q['frames_raw']==10


def test_split_mismatch_fails(tmp_path):
    data=dataset_fixture(tmp_path);index,_=build_provenance_index(data,tmp_path)
    with pytest.raises(ValueError,match='split conflict'):
        make_prediction_rows(LABELS,np.array([4]),np.eye(5)[[4]],['DIAG_0'],'train',index,tmp_path)


def test_unresolved_negative_cannot_silently_pass(tmp_path):
    with pytest.raises(ValueError,match='Missing hard-negative metadata'):
        make_prediction_rows(LABELS,np.array([4]),np.eye(5)[[4]],['DIAG_NOPE'],'test',{},tmp_path)


def test_provenance_conflict_rejected(tmp_path):
    d=dataset_fixture(tmp_path)
    (d/'negative_manifest.csv').write_text('id,split,subject,condition,reference_sign,raw\nDIAG_0,train,S09,wrong_position,xin_chao,x\n')
    with pytest.raises(ValueError,match='Conflicting negative provenance'):
        build_provenance_index(d,tmp_path)


def test_expected_error_classifications():
    assert error_type('UNKNOWN','REST')=='UNKNOWN_AS_REST_NO_TOKEN'
    assert error_type('UNKNOWN','cam_on')=='UNKNOWN_AS_KNOWN_TOKEN_RISK'
    assert error_type('cam_on','UNKNOWN')=='KNOWN_REJECTED'
    assert error_type('cam_on','xin_loi')=='KNOWN_CLASS_CONFUSION'
    assert error_type('cam_on','cam_on')=='CORRECT'


def test_group_summary():
    rs=[{'condition':'wrong_position','true_label':'UNKNOWN','predicted_label':'REST','prediction_correct':False},
        {'condition':'wrong_position','true_label':'UNKNOWN','predicted_label':'cam_on','prediction_correct':False}]
    assert condition_summary(rs)['wrong_position']['unknown_to_token_risk_count']==1
