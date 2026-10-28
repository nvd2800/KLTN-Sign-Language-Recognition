import numpy as np

from core.realtime_decision_phase7 import (
    DecisionConfig,
    RealtimeDecisionEngine,
    SegmentState,
    gate_prediction,
    motion_score,
    stable_label,
)


CFG = DecisionConfig(
    min_confidence=0.85,
    min_top1_margin=0.20,
    stable_window_count=5,
    stable_required=4,
    min_predict_frames=12,
    preview_stride_frames=4,
    start_motion_threshold=0.012,
    end_motion_threshold=0.004,
    start_required_frames=3,
    end_still_frames=3,
    min_active_frames=4,
    max_active_frames=20,
    min_hand_presence_ratio=0.40,
    cooldown_seconds=0.0,
    require_rest_before_repeat=True,
)


def test_gate_accepts_high_confidence_and_margin():
    labels=["REST","cam_on","xin_chao","xin_loi"]
    p=np.array([0.03,0.04,0.90,0.03],dtype=float)
    r=gate_prediction(labels,p,CFG)
    assert r.gate_pass
    assert r.gated_label=="xin_chao"


def test_gate_rejects_low_confidence():
    labels=["REST","cam_on","xin_chao","xin_loi"]
    p=np.array([0.20,0.25,0.50,0.05],dtype=float)
    r=gate_prediction(labels,p,CFG)
    assert not r.gate_pass
    assert r.gated_label=="UNKNOWN"


def test_gate_rejects_small_margin():
    labels=["REST","cam_on","xin_chao","xin_loi"]
    p=np.array([0.01,0.44,0.46,0.09],dtype=float)
    r=gate_prediction(labels,p,CFG)
    assert not r.gate_pass


def test_stability_4_of_5():
    ok,count,seen=stable_label(
        ["xin_chao","xin_chao","cam_on","xin_chao","xin_chao"],
        "xin_chao",
        CFG,
    )
    assert ok
    assert count==4
    assert seen==5


def test_motion_score_detects_translation():
    prev=np.zeros((21,3),dtype=np.float32)
    now=prev.copy()
    now[:,0]+=0.03
    s=motion_score(
        left_now=now,left_prev=prev,
        right_now=None,right_prev=None,
    )
    assert s > 0.02


def test_state_rest_to_active():
    e=RealtimeDecisionEngine(CFG)
    assert e.state==SegmentState.REST
    assert e.update(motion=0.02,any_hand_present=True,now_s=0.0) is None
    assert e.update(motion=0.02,any_hand_present=True,now_s=0.1) is None
    ev=e.update(motion=0.02,any_hand_present=True,now_s=0.2)
    assert ev=="START"
    assert e.state==SegmentState.ACTIVE


def test_state_active_to_end():
    e=RealtimeDecisionEngine(CFG)
    for t in [0.0,0.1]:
        e.update(motion=0.02,any_hand_present=True,now_s=t)
    e.update(motion=0.02,any_hand_present=True,now_s=0.2)
    # 4 active frames first
    for i in range(4):
        e.update(motion=0.02,any_hand_present=True,now_s=0.3+i*0.1)
    # then 3 still frames
    ev=None
    for i in range(3):
        ev=e.update(motion=0.0,any_hand_present=True,now_s=1+i*0.1)
    assert ev=="END"
    assert e.state==SegmentState.COOLDOWN


def test_confirm_requires_gate_presence_and_stability():
    labels=["REST","cam_on","xin_chao","xin_loi"]
    e=RealtimeDecisionEngine(CFG)
    e.state=SegmentState.ACTIVE
    e.active_frames=10
    e.hand_present_frames=8
    for x in ["xin_chao","xin_chao","cam_on","xin_chao","xin_chao"]:
        e.add_preview_label(x)
    final=gate_prediction(
        labels,
        np.array([0.01,0.02,0.94,0.03]),
        CFG,
    )
    label,details=e.confirm_prediction(final,now_s=10.0)
    assert label=="xin_chao"
    assert details["accepted"]


def test_confirm_returns_unknown_when_unstable():
    labels=["REST","cam_on","xin_chao","xin_loi"]
    e=RealtimeDecisionEngine(CFG)
    e.state=SegmentState.ACTIVE
    e.active_frames=10
    e.hand_present_frames=8
    for x in ["cam_on","xin_loi","cam_on","xin_chao","REST"]:
        e.add_preview_label(x)
    final=gate_prediction(
        labels,
        np.array([0.01,0.02,0.94,0.03]),
        CFG,
    )
    label,details=e.confirm_prediction(final,now_s=10.0)
    assert label=="UNKNOWN"
    assert not details["accepted"]
