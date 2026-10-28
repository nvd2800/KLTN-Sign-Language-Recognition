import numpy as np
from core.unknown_phase8 import (
    threshold_metrics, choose_threshold, apply_unknown_threshold,
    known_classification_with_rejection, unknown_rejection_metrics,
)

def test_threshold_metrics():
    k=np.array([.95,.90,.80,.70])
    u=np.array([.20,.40,.60,.88])
    r=threshold_metrics(k,u,.85)
    assert abs(r.known_accept_rate-.5)<1e-9
    assert abs(r.unknown_reject_rate-.75)<1e-9
    assert abs(r.false_accept_rate-.25)<1e-9

def test_choose_threshold_balanced():
    rows=[
        threshold_metrics([.9,.8],[.7,.6],.65),
        threshold_metrics([.9,.8],[.7,.6],.75),
    ]
    best=choose_threshold(rows)
    assert best.threshold in {0.65,0.75}

def test_apply_threshold():
    labels=["A","B"]
    p=np.array([[.9,.1],[.55,.45]],float)
    pred,conf=apply_unknown_threshold(p,labels,.8)
    assert pred.tolist()==["A","UNKNOWN"]
    assert np.allclose(conf,[.9,.55])

def test_known_metrics():
    y=np.array([0,1,1])
    p=np.array([[.9,.1],[.1,.9],[.6,.4]])
    m=known_classification_with_rejection(y,p,.8)
    assert m["accept_rate"]==2/3
    assert m["accuracy_on_accepted"]==1.0

def test_unknown_metrics():
    p=np.array([[.7,.3],[.9,.1]])
    m=unknown_rejection_metrics(p,.8)
    assert m["unknown_reject_rate"]==0.5
    assert m["false_accept_rate"]==0.5
