import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from core.continuous_phase9 import ContinuousRecognizer, Phase09Config, State


def engine(**kwargs):
    return ContinuousRecognizer(Phase09Config(**kwargs))


def feed(e, series, start=0.0, step=0.1):
    emissions=[]
    for i, (lab, conf) in enumerate(series):
        v=e.update(lab, conf, start+i*step)
        if v: emissions.append(v.label)
    return emissions


def test_one_sign_only_one_token_while_held():
    e=engine()
    assert feed(e,[('xin_chao',.96)]*15)==['xin_chao']
    assert e.state==State.WAIT_RELEASE


def test_release_enables_same_sign_twice():
    e=engine()
    arr=[('xin_chao',.96)]*8+[('REST',.99)]*3+[('xin_chao',.97)]*6
    assert feed(e,arr)==['xin_chao','xin_chao']


def test_unknown_does_not_release():
    e=engine()
    arr=[('cam_on',.96)]*5+[('UNKNOWN',.90)]*9+[('cam_on',.96)]*5
    assert feed(e,arr)==['cam_on']


def test_partial_noise_not_confirmed():
    e=engine()
    assert feed(e,[('cam_on',.96),('UNKNOWN',.7),('cam_on',.96),('REST',.99)])==[]


def test_low_confidence_is_unknown():
    e=engine()
    assert feed(e,[('xin_loi',.5)]*12)==[]


def test_rest_does_not_emit_token():
    e=engine()
    assert feed(e,[('REST',1)]*20)==[]


def test_change_class_before_confirmation_resets_candidate():
    e=engine()
    assert feed(e,[('cam_on',.99)]*2+[('xin_loi',.99)]*3)==['xin_loi']


def test_no_second_token_without_full_release():
    e=engine()
    s=[('cam_on',.99)]*5+[('REST',.99)]*2+[('cam_on',.99)]*8
    assert feed(e,s)==['cam_on']


def test_reset():
    e=engine()
    assert feed(e,[('cam_on',.99)]*4)==['cam_on']
    e.reset()
    assert feed(e,[('cam_on',.99)]*4,start=4)==['cam_on']


def test_timestamp_regression_rejected():
    e=engine()
    e.update('REST',1.0,5.0)
    with pytest.raises(ValueError):
        e.update('REST',1.0,4.0)


def test_gap_breaks_candidate():
    e=engine()
    feed(e,[('cam_on',.99)]*2)
    assert e.update('cam_on',.99,5.0) is None
    assert e.update('cam_on',.99,5.1) is None
    assert e.update('cam_on',.99,5.2).label=='cam_on'


def test_can_release_on_unknown_if_explicitly_enabled():
    e=engine(unknown_counts_as_release=True)
    s=[('cam_on',.99)]*4+[('UNKNOWN',.4)]*3+[('cam_on',.99)]*5
    assert feed(e,s)==['cam_on','cam_on']
