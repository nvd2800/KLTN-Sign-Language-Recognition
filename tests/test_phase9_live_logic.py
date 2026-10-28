import sys
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.phase9_live_logic import (
    LiveGateConfig, decide_window, hand_presence_ratio,
    window_motion_score, frame_rest_hint, readable_tokens,
)
from core.continuous_phase9 import ContinuousRecognizer, Phase09Config, State

LABELS=['REST','UNKNOWN','cam_on','xin_chao','xin_loi']


def frame(hand=True, x=0.0):
    left=np.zeros((21,3),np.float32)
    right=np.zeros((21,3),np.float32)
    left[:,0]=x
    return SimpleNamespace(left=left,right=right,left_present=hand,right_present=False)


def test_decision_known_high_confidence():
    p=np.array([.02,.01,.03,.92,.02])
    res=decide_window(p,LABELS,[frame()]*5,LiveGateConfig())
    assert res.raw_label=='xin_chao' and res.label=='xin_chao'
    assert res.reason=='ACCEPT'


def test_explicit_unknown_even_when_high_confidence():
    res=decide_window([.01,.98,.003,.004,.003],LABELS,[frame()]*5,LiveGateConfig())
    assert res.label=='UNKNOWN' and res.reason=='ACCEPT'


def test_low_confidence_rejected():
    res=decide_window([.01,.06,.12,.78,.03],LABELS,[frame()]*5,LiveGateConfig())
    assert res.label=='UNKNOWN' and res.reason=='LOW_CONFIDENCE_OR_MARGIN'


def test_low_hand_presence_rejected():
    seq=[frame(False)]*4+[frame(True)]
    res=decide_window([.01,.01,.96,.01,.01],LABELS,seq,LiveGateConfig())
    assert res.label=='UNKNOWN' and res.reason=='LOW_HAND_PRESENCE'
    assert res.presence_ratio==.2


def test_rest_not_rejected_by_hand_presence():
    res=decide_window([.99,.003,.003,.002,.002],LABELS,[frame(False)]*6,LiveGateConfig())
    assert res.label=='REST'


def test_motion_score_positive_for_moving_hand():
    score=window_motion_score([frame(x=0.),frame(x=.03),frame(x=.05)])
    assert score>0.02


def test_motion_zero_for_missing():
    assert window_motion_score([frame(False),frame(False)])==0.


def test_motion_gate_optional():
    seq=[frame()]*8
    res=decide_window([.01,.01,.95,.02,.01],LABELS,seq,LiveGateConfig(min_motion=.001))
    assert res.reason=='LOW_MOTION'


def test_release_heuristic_opt_out():
    c=LiveGateConfig(release_no_hand_frames=12)
    assert not frame_rest_hint(11,c) and frame_rest_hint(12,c)
    assert not frame_rest_hint(100,replace(c,auto_release_no_hand=False))


def test_fsm_from_window_labels_twice_after_rest():
    rec=ContinuousRecognizer(Phase09Config(confidence_threshold=.8,min_consecutive=3,release_consecutive=3))
    feed=[('xin_chao',.95)]*6+[('REST',.99)]*3+[('xin_chao',.96)]*4
    emitted=[]
    for i,(label,conf) in enumerate(feed):
        ev=rec.update(label,conf,i*.15)
        if ev:emitted.append(ev.label)
    assert emitted==['xin_chao','xin_chao']


def test_no_repeat_without_rest():
    rec=ContinuousRecognizer(Phase09Config(confidence_threshold=.8,min_consecutive=3,release_consecutive=3))
    feed=[('xin_chao',.96)]*12+[('UNKNOWN',.98)]*4+[('xin_chao',.99)]*6
    emits=[e.label for i,(lab,conf) in enumerate(feed) if (e:=rec.update(lab,conf,i*.15))]
    assert emits==['xin_chao']
    assert rec.state==State.WAIT_RELEASE


def test_window_validation():
    with pytest.raises(ValueError):
        LiveGateConfig(min_confidence=1.1).validate()
    with pytest.raises(ValueError):
        decide_window(np.full(5,np.nan),LABELS,[frame()],LiveGateConfig())


def test_readable_token_list():
    assert 'xin_chao' in readable_tokens(['xin_chao'])
    assert readable_tokens([]).startswith('(waiting')
