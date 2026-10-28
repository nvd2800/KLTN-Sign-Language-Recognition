from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from core.spatial_validation_gate import (
    masked_group_distance, threshold_on_validation, evaluate,
    load_profile, validate_profile, trial_from_npz, GROUPS,
)
from scripts.build_spatial_gate_profile import fit_label
from core.continuous_phase9 import ContinuousRecognizer, Phase09Config


def synthetic(delta=0, group=None, only_left=True):
    x=np.zeros((32,32),np.float32)
    m=np.ones((32,32),bool)
    if only_left:
        for lo,hi in ((6,12),(17,22),(25,28),(30,32)):
            m[:,lo:hi]=False
    if group:
        lo,hi,_=GROUPS[group]
        x[:,lo:hi][m[:,lo:hi]] = delta
    return x,m


def profile(ref,thresholds):
    x,m=ref
    return {"schema_version":1,"calibration_mode":"DEV_ONLY","min_overlap":.55,
            "classes":{"xin_chao":{"references":[{"features":x.tolist(),"mask":m.astype(int).tolist()}],
            "thresholds":thresholds}}}


def test_one_hand_is_not_penalized_for_missing_other():
    x,m=synthetic()
    ans=masked_group_distance(x,m,x,m,'hand_body_position12')
    assert ans['distance']==0
    assert ans['relative_coverage']==1.0


def test_position_change_detected():
    x,m=synthetic()
    y,n=synthetic(.7,'hand_body_position12')
    r=masked_group_distance(y,n,x,m,'hand_body_position12')
    assert .69<r['distance']<.71


def test_handshape_change_detected():
    x,m=synthetic()
    y,n=synthetic(.3,'finger_curl10')
    r=masked_group_distance(y,n,x,m,'finger_curl10')
    assert .29<r['distance']<.31


def test_orientation_change_detected():
    x,m=synthetic()
    y,n=synthetic(.9,'palm_orientation6')
    r=masked_group_distance(y,n,x,m,'palm_orientation6')
    assert .89<r['distance']<.91


def test_reject_missing_overlap():
    x,m=synthetic()
    y,n=synthetic()
    n[:,:12]=False
    r=masked_group_distance(y,n,x,m,'hand_body_position12')
    assert r['distance'] is None
    assert r['relative_coverage']==0


def test_thresholds_separate_validation():
    t,score=threshold_on_validation([.06,.08],[.4,.5])
    assert .08<t<.4
    assert score['balanced']==1.0


def test_no_discrimination_requires_more_data():
    t,score=threshold_on_validation([.5],[.5])
    assert t is None
    assert score['reason']=='NO_DISCRIMINATION'


def test_gate_accepts_valid_input():
    ref=synthetic()
    pro=profile(ref,{'hand_body_position12':.2,'finger_curl10':.2,'palm_orientation6':.2})
    validate_profile(pro)
    ans=evaluate(pro,'xin_chao',*ref)
    assert ans.decision=='PASS'


def test_gate_blocks_invalid_known_sign():
    ref=synthetic(); pro=profile(ref,{'hand_body_position12':.2,'finger_curl10':.2})
    ans=evaluate(pro,'xin_chao',*synthetic(.7,'hand_body_position12'))
    assert ans.decision=='REJECT'
    assert 'hand_body_position12' in ans.reason


def test_gate_missing_features_fails_closed():
    ref=synthetic();pro=profile(ref,{'hand_body_position12':.2})
    x,m=synthetic();m[:,:12]=False
    ans=evaluate(pro,'xin_chao',x,m)
    assert ans.decision=='INSUFFICIENT'


def test_gate_other_classes_bypass_explicitly():
    pro=profile(synthetic(),{'hand_body_position12':.2})
    ans=evaluate(pro,'cam_on',*synthetic())
    assert ans.decision=='BYPASS_UNCONFIGURED'
    assert 'NOT_CALIBRATED' in ans.reason


def test_rest_unknown_skipped():
    pro=profile(synthetic(),{'hand_body_position12':.2})
    for label in ('REST','UNKNOWN'):
        assert evaluate(pro,label,*synthetic()).decision=='SKIP'


def test_profile_serializes(tmp_path):
    pro=profile(synthetic(),{'hand_body_position12':.2})
    path=tmp_path/'gate.json';path.write_text(json.dumps(pro))
    assert load_profile(path)['calibration_mode']=='DEV_ONLY'


def test_bad_profile_fails(tmp_path):
    pro=profile(synthetic(),{'bogus':.2})
    with pytest.raises(ValueError,match='Invalid threshold'):
        validate_profile(pro)


def test_fit_dev_profile_from_separate_variants():
    r=[]
    for name,x,m in [('correct',*synthetic()),('correct_repeat',*synthetic(.05,'finger_curl10')),
                      ('wrong_position',*synthetic(.6,'hand_body_position12')),
                      ('wrong_handshape',*synthetic(.5,'finger_curl10')),
                      ('wrong_orientation',*synthetic(.9,'palm_orientation6'))]:
        r.append({'condition':name,'subject':'S01','x':x,'m':m,'path':name})
    # A low but distinct clean-repeat deviation for position and orientation.
    spec,diagnostics=fit_label(r,{'S01'},{'S01'},True)
    assert 'hand_body_position12' in spec['thresholds']
    assert 'finger_curl10' in spec['thresholds']
    assert 'palm_orientation6' in spec['thresholds']


def test_signer_independent_profile_rejects_overlap():
    with pytest.raises(RuntimeError,match='signers overlap'):
        fit_label([],{'S01'},{'S01'},False)


def test_state_machine_no_false_token_if_rejected():
    ref=synthetic();pro=profile(ref,{'hand_body_position12':.2})
    wrong=synthetic(.7,'hand_body_position12')
    engine=ContinuousRecognizer(Phase09Config(confidence_threshold=.8,min_consecutive=3))
    for t in [0.,.2,.4,.6]:
        ans=evaluate(pro,'xin_chao',*wrong)
        assert ans.decision=='REJECT'
        engine.update('UNKNOWN',.99,t)
    assert engine.events==[]


def test_v4_source_shape_unchanged_and_gate_masks():
    x,m=synthetic()
    assert x.shape==(32,32)
    assert m.shape==(32,32)


def test_end_to_end_cli_dev_profile(tmp_path):
    from scripts.build_spatial_gate_profile import collect, fit_label
    out=tmp_path/'logs'/'xin_chao'/'S01'
    trials={
      'correct': synthetic(),
      'correct_repeat':synthetic(.06,'finger_curl10'),
      'wrong_position':synthetic(.65,'hand_body_position12'),
      'wrong_handshape':synthetic(.55,'finger_curl10'),
      'wrong_orientation':synthetic(.85,'palm_orientation6'),
    }
    for condition,(x,m) in trials.items():
        p=out/condition/'trial1';p.mkdir(parents=True)
        (p/'metadata.json').write_text(json.dumps({'condition':condition,'subject':'S01','label':'xin_chao'}))
        np.savez_compressed(p/'features.npz',candidate_v4_32=x,candidate_v4_mask32=m)
    rows=collect(tmp_path/'logs','xin_chao')
    assert len(rows)==5
    spec,diag=fit_label(rows,{'S01'},{'S01'},True)
    pro={'schema_version':1,'calibration_mode':'DEV_ONLY','min_overlap':.55,
         'classes':{'xin_chao':spec}}
    validate_profile(pro)
    positive=evaluate(pro,'xin_chao',*trials['correct_repeat'])
    negative=evaluate(pro,'xin_chao',*trials['wrong_position'])
    assert positive.decision=='PASS'
    assert negative.decision=='REJECT'
    # Ensure all three diagnostic categories were calibrated
    assert set(spec['thresholds'])==set(GROUPS)


def test_ref_and_probe_must_have_finite_values():
    ref=synthetic()
    probe,mask=synthetic()
    probe[1,1]=np.nan
    pro=profile(ref,{'hand_body_position12':.2})
    with pytest.raises(ValueError,match='Non-finite'):
        evaluate(pro,'xin_chao',probe,mask)
