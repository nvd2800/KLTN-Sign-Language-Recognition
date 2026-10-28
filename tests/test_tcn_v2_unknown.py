import json
from pathlib import Path
import numpy as np
import pytest
from core.tcn_v2_unknown import (
    assign_unknown_subject, get_extras, compute_class_weights, resample_and_scale,
    summarize_multiclass, subject_map_from_summary,
)

def test_mapping_known_train_cannot_be_unknown_test():
    with pytest.raises(ValueError,match='Signer leakage'):
        assign_unknown_subject('S02',{'S02':'train'},{'S02':'test'})

def test_known_mapping_used_even_if_old_unknown_manifest_says_test():
    assert assign_unknown_subject('S02',{'S02':'train'})=='train'

def test_unknown_new_subject_requires_explicit_assignment():
    with pytest.raises(ValueError,match='not in Phase07'):
        assign_unknown_subject('S11',{'S01':'train'})
    assert assign_unknown_subject('S11',{'S01':'train'}, {'S11':'val'})=='val'

def test_extras_leakage():
    with pytest.raises(ValueError):get_extras(train='S11',test='S11')

def test_class_weight_all_5_classes_required():
    with pytest.raises(ValueError,match='Every class'):
        compute_class_weights(np.array([0,1,2,3]),5)
    weights=compute_class_weights(np.array([0,0,0,0,1,2,3,4]),5)
    assert len(weights)==5
    assert weights[4]>weights[0]

def test_unknown_raw_to_v1_scaled_32x453():
    x=np.ones((45,453),np.float32)*4
    out,mask=resample_and_scale(x,np.ones(453,np.float32)*2,
                                np.ones(453,np.float32)*2)
    assert out.shape==(32,453)
    assert mask.all()
    assert np.allclose(out,1)

def test_padding_stays_zero():
    x=np.ones((10,453),np.float32)*5
    out,mask=resample_and_scale(x,np.zeros(453,np.float32),np.ones(453,np.float32))
    assert mask.sum()==10
    assert np.array_equal(out[10:],np.zeros((22,453),np.float32))

def test_metrics_with_no_unknown_test_does_not_fake_score():
    report=summarize_multiclass(np.array([0,1,1]),np.array([0,1,0]),
        ['REST','cam_on','xin_chao','xin_loi','UNKNOWN'])
    assert report['n']==3
    assert report['classes'][4]['recall'] is None

def test_summary_known_subjects_leakage(tmp_path):
    p=tmp_path/'s.json';p.write_text(json.dumps({'subjects':{'train':['S01'],
        'val':['S08'],'test':['S01']}}))
    with pytest.raises(ValueError,match='leakage'):subject_map_from_summary(p)
