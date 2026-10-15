import numpy as np

from core.preprocess_phase7 import (
    uniform_resample_or_pad,
    fit_standardizer_train_only,
    apply_standardizer,
    validate_split_subject_leakage,
    label_mapping_from_rows,
)


def test_long_sequence_resamples_to_32():
    x=np.arange(40*3,dtype=np.float32).reshape(40,3)
    out,mask,src=uniform_resample_or_pad(x,32)
    assert out.shape==(32,3)
    assert mask.all()
    assert src[0]==0
    assert src[-1]==39


def test_short_sequence_pads_and_masks():
    x=np.ones((25,4),dtype=np.float32)
    out,mask,src=uniform_resample_or_pad(x,32)
    assert out.shape==(32,4)
    assert mask.sum()==25
    assert np.all(out[25:]==0)
    assert np.all(src[25:]==-1)


def test_exact_32_preserves_sequence():
    x=np.random.default_rng(1).normal(size=(32,5)).astype(np.float32)
    out,mask,src=uniform_resample_or_pad(x,32)
    assert np.allclose(out,x)
    assert mask.all()
    assert np.array_equal(src,np.arange(32))


def test_standardizer_train_only_values():
    X=np.zeros((2,3,2),dtype=np.float32)
    M=np.ones((2,3),dtype=bool)
    X[0]=[[1,10],[2,20],[3,30]]
    X[1]=[[4,40],[5,50],[6,60]]
    mean,std=fit_standardizer_train_only(X,M,1e-6)
    assert np.allclose(mean,[3.5,35.0])
    assert np.all(std>0)


def test_padding_remains_zero_after_standardization():
    X=np.zeros((1,4,2),dtype=np.float32)
    M=np.array([[1,1,0,0]],dtype=bool)
    X[0,0]=[1,2]
    X[0,1]=[3,4]
    mean,std=fit_standardizer_train_only(X,M,1e-6)
    out=apply_standardizer(X,M,mean,std)
    assert np.all(out[0,2:]==0)


def test_no_signer_leakage():
    split_map={
        "train":[{"subject_id":"S01"},{"subject_id":"S02"}],
        "val":[{"subject_id":"S08"}],
        "test":[{"subject_id":"S09"},{"subject_id":"S10"}],
    }
    report=validate_split_subject_leakage(split_map)
    assert report["pass"]
    assert report["leakage_count"]==0


def test_detect_signer_leakage():
    split_map={
        "train":[{"subject_id":"S01"}],
        "val":[{"subject_id":"S01"}],
        "test":[{"subject_id":"S09"}],
    }
    report=validate_split_subject_leakage(split_map)
    assert not report["pass"]
    assert report["leakage"]["train_val"]==["S01"]


def test_label_mapping_is_deterministic():
    rows=[
        {"sign":"xin_loi"},
        {"sign":"REST"},
        {"sign":"cam_on"},
        {"sign":"xin_chao"},
    ]
    labels,mapping=label_mapping_from_rows(rows)
    assert labels==sorted(labels)
    assert set(mapping)==set(labels)
