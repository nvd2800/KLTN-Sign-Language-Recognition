"""Pure dataset/preprocessing helpers for TCN-V2 UNKNOWN training.
Reuses Phase07 frozen scaler; never trains on test or refits the standardizer.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np


def read_csv(p):
    with Path(p).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_phase7_split(path):
    with np.load(path, allow_pickle=False) as d:
        X=np.asarray(d['X'],np.float32)
        y=np.asarray(d['y'],np.int64)
        mask=np.asarray(d['mask'],bool) if 'mask' in d.files else np.ones(X.shape[:2],bool)
        ids=[str(s) for s in d['sample_ids']] if 'sample_ids' in d.files else [f'known_{i}' for i in range(len(y))]
    if X.ndim!=3 or X.shape[1:]!=(32,453) or len(y)!=len(X) or len(ids)!=len(X):
        raise ValueError(f'Invalid Phase07 split shapes in {path}: {X.shape}/{y.shape}')
    if not np.isfinite(X).all() or y.size and y.min()<0:
        raise ValueError('Invalid known sequence/label')
    return {'X': X, 'y':y, 'mask':mask, 'ids':ids}


def load_v1_scaler(path):
    with np.load(path,allow_pickle=False) as d:
        mean=np.asarray(d['mean'],np.float32)
        std=np.asarray(d['std'],np.float32)
        T=int(np.asarray(d['target_frames']).reshape(-1)[0])
        F=int(np.asarray(d['feature_dim']).reshape(-1)[0])
    if (T,F)!=(32,453) or mean.shape!=(453,) or std.shape!=(453,):
        raise ValueError(f'Scaler must have 32x453: received {T}x{F}')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('Invalid frozen V1 scaler')
    return mean,std


def assign_unknown_subject(subject,known_subject_splits,extra_subject_splits=None):
    """Use original Phase07 signer assignment as single source of truth."""
    subject=str(subject).upper().strip()
    if not subject:
        raise ValueError('UNKNOWN subject_id is empty')
    extra_subject_splits=extra_subject_splits or {}
    known=known_subject_splits.get(subject)
    extra=extra_subject_splits.get(subject)
    if known and extra and known!=extra:
        raise ValueError(f'Signer leakage: {subject} is KNOWN {known} but UNKNOWN {extra}')
    result=known or extra
    if not result:
        raise ValueError(f'Subject {subject} not in Phase07 subject map; supply --{subject} allocation via --*-subjects')
    return result


def subject_map_from_summary(path):
    summary=json.loads(Path(path).read_text(encoding='utf-8'))
    split_subjects=summary.get('subjects',{})
    mapping={}
    for split in ('train','val','test'):
        for sid in split_subjects.get(split,[]):
            sid=str(sid).upper().strip()
            if sid in mapping and mapping[sid]!=split:
                raise ValueError(f'Known signer leakage: {sid}')
            mapping[sid]=split
    if not mapping:
        raise ValueError('Phase07 dataset_summary subjects missing; regenerate Phase07 summary')
    return mapping


def parse_subject_list(s):
    return {x.strip().upper() for x in str(s).split(',') if x.strip()}


def get_extras(train='',val='',test=''):
    result={}
    for split,subject_list in (('train',train),('val',val),('test',test)):
        for subject in parse_subject_list(subject_list):
            if subject in result and result[subject]!=split:
                raise ValueError(f'New signer {subject} assigned to two splits')
            result[subject]=split
    return result


def resample_and_scale(phase6,mean,std):
    from core.preprocess_phase7 import uniform_resample_or_pad, apply_standardizer
    z=np.asarray(phase6,np.float32)
    if z.ndim!=2 or z.shape[1]!=453 or not len(z) or not np.isfinite(z).all():
        raise ValueError(f'UNKNOWN Phase06 expected finite [T,453], got {z.shape}')
    fixed,mask,_=uniform_resample_or_pad(z,32)
    out=apply_standardizer(fixed[np.newaxis],mask[np.newaxis],mean,std)[0]
    return out,mask


def compute_class_weights(y, num_classes, cap=10.0):
    y=np.asarray(y,np.int64)
    counts=np.bincount(y,minlength=num_classes)
    if np.any(counts==0):
        raise ValueError(f'Every class must occur in TRAIN, counts={counts.tolist()}')
    raw=len(y)/(num_classes*counts)
    return {i:float(min(cap,max(0.5,w))) for i,w in enumerate(raw)}


def summarize_multiclass(y,pred,labels):
    y=np.asarray(y,np.int64);pred=np.asarray(pred,np.int64)
    if y.shape!=pred.shape: raise ValueError('y/pred size mismatch')
    C=len(labels)
    cm=np.zeros((C,C),np.int64)
    for a,b in zip(y,pred):
        if not (0<=a<C and 0<=b<C):raise ValueError('Label out of range')
        cm[a,b]+=1
    result=[]
    for c,name in enumerate(labels):
        tp=int(cm[c,c]);support=int(cm[c,:].sum())
        fp=int(cm[:,c].sum()-tp)
        prec=tp/(tp+fp) if tp+fp else 0.0
        rec=tp/support if support else None
        f1=(2*prec*rec/(prec+rec) if prec+rec else 0.0) if rec is not None else None
        result.append({'label':name,'support':support,'precision':prec,'recall':rec,'f1':f1})
    supported=[r['f1'] for r in result if r['f1'] is not None]
    return {'n':len(y),'accuracy':float(np.mean(y==pred)) if len(y) else None,
            'macro_f1_supported':float(np.mean(supported)) if supported else None,
            'classes':result,'confusion_matrix':cm.tolist()}
