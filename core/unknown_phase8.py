from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    known_accept_rate: float
    false_reject_rate: float
    unknown_reject_rate: float
    false_accept_rate: float
    balanced_accept_reject: float

def threshold_metrics(known_conf, unknown_conf, threshold: float) -> ThresholdMetrics:
    known_conf=np.asarray(known_conf,dtype=float)
    unknown_conf=np.asarray(unknown_conf,dtype=float)
    known_accept=float(np.mean(known_conf >= threshold)) if known_conf.size else 0.0
    unknown_reject=float(np.mean(unknown_conf < threshold)) if unknown_conf.size else 0.0
    return ThresholdMetrics(
        threshold=float(threshold),
        known_accept_rate=known_accept,
        false_reject_rate=1.0-known_accept,
        unknown_reject_rate=unknown_reject,
        false_accept_rate=1.0-unknown_reject,
        balanced_accept_reject=0.5*(known_accept+unknown_reject),
    )

def choose_threshold(rows):
    if not rows:
        raise ValueError("No threshold rows")
    # Highest balanced score; tie -> lower FAR; tie -> lower FRR; tie -> higher threshold.
    return sorted(
        rows,
        key=lambda r:(
            -r.balanced_accept_reject,
            r.false_accept_rate,
            r.false_reject_rate,
            -r.threshold,
        )
    )[0]

def apply_unknown_threshold(probs, labels, threshold: float, unknown_label="UNKNOWN"):
    p=np.asarray(probs,dtype=float)
    if p.ndim==1:
        p=p[None,:]
    idx=np.argmax(p,axis=1)
    conf=np.max(p,axis=1)
    out=[]
    for i,c in zip(idx,conf):
        out.append(labels[int(i)] if float(c)>=threshold else unknown_label)
    return np.asarray(out), conf

def known_classification_with_rejection(y_true, probs, threshold: float):
    probs=np.asarray(probs,dtype=float)
    y_true=np.asarray(y_true,dtype=int)
    pred=np.argmax(probs,axis=1)
    conf=np.max(probs,axis=1)
    accepted=conf>=threshold
    correct=np.logical_and(accepted,pred==y_true)
    return {
        "n": int(y_true.size),
        "accept_rate": float(np.mean(accepted)) if y_true.size else 0.0,
        "false_reject_rate": float(np.mean(~accepted)) if y_true.size else 0.0,
        "correct_and_accepted_rate": float(np.mean(correct)) if y_true.size else 0.0,
        "accuracy_on_accepted": float(np.mean(pred[accepted]==y_true[accepted])) if np.any(accepted) else 0.0,
    }

def unknown_rejection_metrics(unknown_probs, threshold: float):
    probs=np.asarray(unknown_probs,dtype=float)
    conf=np.max(probs,axis=1) if probs.size else np.asarray([],dtype=float)
    rejected=conf<threshold
    return {
        "n": int(conf.size),
        "unknown_reject_rate": float(np.mean(rejected)) if conf.size else 0.0,
        "false_accept_rate": float(np.mean(~rejected)) if conf.size else 0.0,
        "mean_max_confidence": float(np.mean(conf)) if conf.size else 0.0,
        "median_max_confidence": float(np.median(conf)) if conf.size else 0.0,
    }
