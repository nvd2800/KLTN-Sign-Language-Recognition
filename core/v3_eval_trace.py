"""V3.1 trace helpers: explicit provenance, no fabricated metadata.

This module performs no model inference. It is shared by evaluator, reviewer,
and unit tests so the 513D checkpoint/scaler are never changed.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Mapping
import numpy as np

CONDITIONS = {'wrong_position', 'wrong_handshape', 'wrong_orientation', 'partial'}
PRED_COLUMNS = [
    'sample_id','split','subject','session','condition','reference_sign',
    'true_label','predicted_label','confidence','true_label_probability',
    'unknown_probability','prediction_correct','error_type',
    'left_hand_valid_ratio','right_hand_valid_ratio','any_hand_valid_ratio',
    'pose_valid_ratio','main_hand_valid_ratio','frames_raw','qc_status',
    'raw_path','video_path','source_type','probs_json',
]


def read_csv_rows(path:Path) -> list[dict]:
    if not path.is_file(): return []
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def _make_path(p, root):
    if not p: return None
    path=Path(str(p))
    return path if path.is_absolute() else Path(root)/path


def build_provenance_index(data:Path, project_root:Path, phase6_manifest:Path|None=None):
    """Map sample_id to metadata and paths, reject conflicting IDs.

    UNKNOWN provenance comes from the dataset actually used in the model,
    not from an arbitrary rglob of current diagnostic files.
    """
    dataset=json.loads((data/'dataset_summary.json').read_text(encoding='utf-8'))
    entries={}
    for row in dataset.get('hard_negatives',[]):
        key=str(row.get('id',''))
        if not key: continue
        split=str(row.get('split','')).lower()
        if key in entries: raise ValueError(f'Duplicate hard-negative ID: {key}')
        entries[key]={
            'sample_id':key,'split':split,
            'subject':row.get('subject',''),'condition':row.get('condition',''),
            'reference_sign':row.get('reference_sign',''),'raw_path':row.get('raw',''),
            'source_type':'REAL_HARD_NEGATIVE',
        }
    # Also supports manifest when summary comes from another preparation version.
    for row in read_csv_rows(data/'negative_manifest.csv'):
        k=str(row.get('id',''))
        if not k:continue
        new={'sample_id':k,'split':str(row.get('split','')).lower(),
             'subject':row.get('subject',''),'condition':row.get('condition',''),
             'reference_sign':row.get('reference_sign',''),'raw_path':row.get('raw',''),
             'source_type':'REAL_HARD_NEGATIVE'}
        if k in entries and any(entries[k].get(f,'')!=new.get(f,'') for f in ('split','subject','condition')):
            raise ValueError(f'Conflicting negative provenance {k}')
        entries[k]=new
    if phase6_manifest is not None:
        for row in read_csv_rows(phase6_manifest):
            k=str(row.get('sample_id','')).strip()
            if not k:continue
            if k in entries:
                if entries[k]['source_type']=='REAL_HARD_NEGATIVE':
                    raise ValueError(f'Known/negative sample ID collision: {k}')
                continue
            entries[k]={
                'sample_id':k,'split':str(row.get('split','')).strip().lower(),
                'subject':row.get('subject_id',''),'session':row.get('session_id',''),
                'condition':'correct' if str(row.get('sign','')).upper()!='REST' else 'REST',
                'reference_sign':row.get('sign',''),
                'raw_path':row.get('sample_path',''),
                'source_type':'KNOWN_PHASE06',
            }
    return entries,dataset


def resolve_raw_and_video(meta:Mapping,project_root:Path):
    """Only return *existing* paths; never imply a missing source was reviewed."""
    root=Path(project_root)
    raw_value=meta.get('raw_path','')
    source=_make_path(raw_value,root)
    dirs=[]
    if source is not None:
        dirs.append(source if source.is_dir() else source.parent)
    if meta.get('source_type')=='KNOWN_PHASE06':
        sign=meta.get('reference_sign','');subject=meta.get('subject','');session=meta.get('session','');sid=meta.get('sample_id','')
        if all((sign,subject,session,sid)):
            dirs.append(root/'dataset'/'raw'/str(sign)/str(subject)/str(session)/str(sid))
    raw=None;video=None
    for d in dirs:
        if not d.is_dir():continue
        for name in ('raw_landmarks.npz','landmarks.npz'):
            candidate=d/name
            if candidate.is_file() and raw is None:raw=candidate
        for name in ('source_video.avi','source_video.mp4','video.mp4','video.avi','video.mov'):
            candidate=d/name
            if candidate.is_file() and video is None:video=candidate
    return raw,video


def compute_landmark_qc(raw_path:Path|None,reference_sign:str):
    empty=dict(left_hand_valid_ratio=None,right_hand_valid_ratio=None,
        any_hand_valid_ratio=None,pose_valid_ratio=None,
        main_hand_valid_ratio=None,frames_raw=None,qc_status='RAW_UNAVAILABLE')
    if raw_path is None or not raw_path.is_file():return empty
    try:
        from core.feature_spatial import load_raw_npz
        l,r,p,lp,rp,pp,_=load_raw_npz(raw_path)
        n=len(lp)
        if n==0:raise ValueError('empty')
        lp=np.asarray(lp,bool);rp=np.asarray(rp,bool);pp=np.asarray(pp,bool)
        # A true dominant hand cannot be inferred from this metadata alone:
        # use max(left,right) explicitly as a tracking proxy, not ground truth.
        q=dict(left_hand_valid_ratio=float(lp.mean()),right_hand_valid_ratio=float(rp.mean()),
               any_hand_valid_ratio=float((lp|rp).mean()),pose_valid_ratio=float(pp.mean()),
               main_hand_valid_ratio=float(max(lp.mean(),rp.mean())),frames_raw=int(n),qc_status='AVAILABLE')
        # Low coverage is a REVIEW flag, not silent deletion of hard examples.
        if str(reference_sign).upper()!='REST' and q['main_hand_valid_ratio']<.80:
            q['qc_status']='REVIEW_LOW_MAIN_HAND'
        if q['pose_valid_ratio']<.90:q['qc_status']+='|REVIEW_POSE' if q['qc_status']!='AVAILABLE' else 'REVIEW_POSE'
        return q
    except (ValueError,KeyError,OSError,IndexError) as exc:
        return {**empty,'qc_status':f'RAW_INVALID:{type(exc).__name__}'}


def error_type(gt:str,pred:str):
    if gt==pred:return 'CORRECT'
    if gt=='UNKNOWN' and pred=='REST':return 'UNKNOWN_AS_REST_NO_TOKEN'
    if gt=='UNKNOWN' and pred!='UNKNOWN':return 'UNKNOWN_AS_KNOWN_TOKEN_RISK'
    if gt!='UNKNOWN' and pred=='UNKNOWN':return 'KNOWN_REJECTED'
    return 'KNOWN_CLASS_CONFUSION'


def make_prediction_rows(labels,y,probs,sample_ids,split,provenance,project_root,with_qc=True):
    y=np.asarray(y);pr=np.asarray(probs)
    if pr.ndim!=2 or pr.shape!=(len(y),len(labels)):raise ValueError('Probability matrix mismatch')
    if len(sample_ids)!=len(y):raise ValueError('Sample ids length mismatch')
    if len(set(map(str,sample_ids)))!=len(sample_ids):raise ValueError('Duplicate IDs in split')
    out=[]
    for i,(yi,prob,sid) in enumerate(zip(y,pr,sample_ids)):
        sid=str(sid);gt=labels[int(yi)];idx=int(np.argmax(prob));pred=labels[idx]
        m=provenance.get(sid,{}).copy()
        if m.get('split') and m['split']!=split:
            raise ValueError(f'Provenance split conflict {sid}: {m["split"]}!={split}')
        if sid.startswith('DIAG_') and not m:
            raise ValueError(f'Missing hard-negative metadata for {sid}: cannot audit leakage')
        raw,vid=resolve_raw_and_video({**m,'sample_id':sid},project_root)
        qc=compute_landmark_qc(raw,m.get('reference_sign',gt)) if with_qc else dict(
            left_hand_valid_ratio=None,right_hand_valid_ratio=None,any_hand_valid_ratio=None,
            pose_valid_ratio=None,main_hand_valid_ratio=None,frames_raw=None,qc_status='NOT_REQUESTED')
        row={
            'sample_id':sid,'split':split,'subject':m.get('subject',''),
            'session':m.get('session',''),'condition':m.get('condition','UNRESOLVED'),
            'reference_sign':m.get('reference_sign',''),
            'true_label':gt,'predicted_label':pred,
            'confidence':float(prob[idx]),'true_label_probability':float(prob[int(yi)]),
            'unknown_probability':float(prob[labels.index('UNKNOWN')]) if 'UNKNOWN' in labels else None,
            'prediction_correct':bool(gt==pred),'error_type':error_type(gt,pred),
            **qc,'raw_path':str(raw) if raw else '',
            'video_path':str(vid) if vid else '',
            'source_type':m.get('source_type','UNRESOLVED'),
            'probs_json':json.dumps({label:float(v) for label,v in zip(labels,prob)},ensure_ascii=False),
        }
        out.append(row)
    return out


def write_csv(path,rows,columns=PRED_COLUMNS):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)


def condition_summary(rows):
    out={}
    for cond in sorted({r['condition'] for r in rows}):
        subset=[r for r in rows if r['condition']==cond]
        n=len(subset);correct=sum(r['prediction_correct'] for r in subset)
        accepted=sum(r['true_label']=='UNKNOWN' and r['predicted_label'] not in ('UNKNOWN','REST') for r in subset)
        out[cond]={'n':n,'accuracy':correct/n if n else None,
                   'unknown_to_token_risk_count':accepted,
                   'predicted_labels':{q:sum(r['predicted_label']==q for r in subset) for q in sorted({r['predicted_label'] for r in subset})}}
    return out
