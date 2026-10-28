"""Diagnostic-only spatial + handshape features and paired-trial comparison.

No training, no mutation of existing 453-dimensional model/scaler.
Inputs are FrameSample from existing core.phase8_live (Phase09 pipeline).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import numpy as np

from core.feature_phase4 import build_phase4_features, inspect_feature_arrays
from core.feature_phase5 import build_phase5_features, inspect_phase5_arrays, angle_abc
from core.feature_phase6 import build_phase6_features, inspect_phase6_arrays

GROUP453 = {
    "hand_position126": (0,126),
    "finger_angles30": (126,156),
    "distance9_legacy": (156,165),
    "local_velocity126": (165,291),
    "local_acceleration126": (291,417),
    "pose27": (417,444),
    "face9": (444,453),
}
GROUP32 = {
    "hand_body_position12": (0,12),
    "finger_curl10": (12,22),
    "palm_orientation6": (22,28),
    "wrist_relative_velocity4": (28,32),
}
CONDITIONS = ("correct", "wrong_position", "wrong_handshape", "wrong_orientation", "partial", "correct_repeat")


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)))


def _curl(hand: np.ndarray) -> np.ndarray:
    # Shape proxy: average flexion at two distal joints of each finger.
    # 0 ~ straight, 1 ~ tightly bent; not calibrated hand-pose probability.
    triplets = [((1,2,3),(2,3,4)), ((5,6,7),(6,7,8)),
                ((9,10,11),(10,11,12)), ((13,14,15),(14,15,16)),
                ((17,18,19),(18,19,20))]
    return np.array([
        float(np.mean([np.clip(1.0-angle_abc(hand[a],hand[b],hand[c],1e-8)/np.pi,0,1)
                       for (a,b,c) in pair])) for pair in triplets
    ], dtype=np.float32)


def _palm_normal(hand: np.ndarray) -> tuple[np.ndarray, bool]:
    # Raw MediaPipe single-view depth is noisy; this is orientation proxy only.
    a=hand[5]-hand[0]; b=hand[17]-hand[0]
    cross=np.cross(a,b); length=float(np.linalg.norm(cross))
    if not np.isfinite(length) or length < 1e-7:
        return np.zeros(3,np.float32), False
    return np.asarray(cross/length,dtype=np.float32), True


def v4_candidate32(frames: list[Any]) -> tuple[np.ndarray,np.ndarray]:
    """Output candidate-only spatial/handshape 32 features and explicit 32 masks.

    [hand-body(6+6), finger-curl(5+5), palm-normal(3+3), wrist-vxy(2+2)]
    All hand-body distances are XY image coordinate distances scaled by shoulder
    width; never combine Hand and Pose landmark Z values.
    """
    T=len(frames)
    feat=np.zeros((T,32),np.float32)
    mask=np.zeros((T,32),bool)
    for t,f in enumerate(frames):
        pose_ok=bool(f.pose_present) and f.pose.shape==(33,3) and np.isfinite(f.pose).all()
        face_ok=bool(f.face_present) and f.face.shape[0]>=14 and np.isfinite(f.face[[13,14],:2]).all()
        valid_body=False
        if pose_ok:
            ps=f.pose
            center=(ps[11,:2]+ps[12,:2])*0.5
            shoulder_width=_dist(ps[11,:2],ps[12,:2])
            valid_body=np.isfinite(shoulder_width) and shoulder_width > 1e-6
            if face_ok:
                mouth=(f.face[13,:2]+f.face[14,:2])*0.5
        for h,(hand,present,wrist_i,shoulder_i) in enumerate([
            (f.left,f.left_present,15,11),(f.right,f.right_present,16,12)
        ]):
            if not present or hand.shape!=(21,3) or not np.isfinite(hand).all():
                continue
            ix=h*6
            if valid_body:
                wrist=hand[0,:2]
                feat[t,ix:ix+2]=(wrist-center)/shoulder_width
                feat[t,ix+2]=_dist(wrist,ps[0,:2])/shoulder_width
                feat[t,ix+4]=_dist(wrist,ps[shoulder_i,:2])/shoulder_width
                feat[t,ix+5]=_dist(wrist,center)/shoulder_width
                mask[t,[ix,ix+1,ix+2,ix+4,ix+5]]=True
                if face_ok:
                    feat[t,ix+3]=_dist(wrist,mouth)/shoulder_width
                    mask[t,ix+3]=True
            cix=12+h*5
            feat[t,cix:cix+5]=_curl(hand)
            mask[t,cix:cix+5]=True
            oix=22+h*3
            normal,ok=_palm_normal(hand)
            if ok:
                feat[t,oix:oix+3]=normal
                mask[t,oix:oix+3]=True
    # Relative wrist motion (per frame); valid only if consecutive poses/hands valid.
    for t in range(1,T):
        for h in range(2):
            ix=h*6; out=28+h*2
            if mask[t,ix:ix+2].all() and mask[t-1,ix:ix+2].all():
                feat[t,out:out+2]=feat[t,ix:ix+2]-feat[t-1,ix:ix+2]
                mask[t,out:out+2]=True
    feat[~mask]=0.0
    return feat,mask


def extract_diagnostic_features(frames: list[Any], cfg4: Any, cfg5: Any, cfg6: Any):
    if not frames:
        raise ValueError("No frames")
    T=len(frames)
    left=np.stack([f.left for f in frames]).astype(np.float32)
    right=np.stack([f.right for f in frames]).astype(np.float32)
    pose=np.stack([f.pose for f in frames]).astype(np.float32)
    face=np.stack([f.face for f in frames]).astype(np.float32)
    lp=np.array([f.left_present for f in frames],bool)
    rp=np.array([f.right_present for f in frames],bool)
    pp=np.array([f.pose_present for f in frames],bool)
    fp=np.array([f.face_present for f in frames],bool)
    timestamps=np.array([f.timestamp for f in frames],np.float64)
    if left.shape!=(T,21,3) or right.shape!=(T,21,3) or pose.shape!=(T,33,3) or face.shape!=(T,468,3):
        raise ValueError(f"Bad landmark shapes: {left.shape},{right.shape},{pose.shape},{face.shape}")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps)<=0):
        raise ValueError("Timestamps must increase strictly")
    p4=build_phase4_features(left,right,lp,rp,cfg4)
    p4['timestamps']=timestamps
    p5=build_phase5_features(phase4_arrays=p4,raw_left=left,raw_right=right,
                            pose_xyz=pose,pose_present=pp,cfg=cfg5)
    p6=build_phase6_features(phase5_full417=p5['full_features417'],
                            pose_xyz=pose,pose_present=pp,face_xyz=face,face_present=fp,cfg=cfg6)
    old=np.asarray(p6['ablation_hands_pose_face453'],np.float32)
    candidate,mask=v4_candidate32(frames)
    qc4=inspect_feature_arrays(p4,cfg4)
    qc5=inspect_phase5_arrays(p5)
    qc6=inspect_phase6_arrays(p6)
    if not all((qc4['pass'],qc5['pass'],qc6['pass'])):
        raise ValueError(f"Feature QC failed: phase4={qc4['issues']} phase5={qc5['issues']} phase6={qc6['issues']}")
    if old.shape!=(T,453) or candidate.shape!=(T,32) or not np.isfinite(old).all():
        raise ValueError("Feature shape/finite check failed")
    raw={
        'left_hand':left,'right_hand':right,'pose':pose,'face':face,
        'left_present':lp,'right_present':rp,'pose_present':pp,'face_present':fp,
        'timestamps':timestamps,
    }
    derived={
        'features453':old,
        'phase4_normalized_xyz126':p4['normalized_xyz126'],
        'phase5_angles30':p5['angles30'],
        'phase5_distance9':p5['distances9'],
        'candidate_v4_32':candidate,
        'candidate_v4_mask32':mask,
        'hand_presence2':p4['hand_presence2'],
        'pose_valid':p6['pose_valid'], 'face_valid':p6['face_valid'],
    }
    cover={
        'frames':T,
        'left_ratio':float(lp.mean()),'right_ratio':float(rp.mean()),
        'any_hand_ratio':float((lp|rp).mean()),
        'pose_ratio':float(pp.mean()),'face_ratio':float(fp.mean()),
        'valid_pose_ratio':float(p6['pose_valid'].mean()),
        'valid_face_ratio':float(p6['face_valid'].mean()),
        'v4_candidate_valid_ratio':float(mask.mean()),
        'duration_sec':float(timestamps[-1]-timestamps[0]),
    }
    return raw,derived,cover


def _resample(x: np.ndarray,n:int=32)->np.ndarray:
    x=np.asarray(x)
    if len(x)==0:
        raise ValueError("Empty sequence")
    idx=np.rint(np.linspace(0,len(x)-1,n)).astype(int)
    return x[idx]


def _masked_rms(a:np.ndarray,b:np.ndarray,ma:np.ndarray | None=None, mb:np.ndarray | None=None)->dict:
    a=_resample(a).astype(np.float64);b=_resample(b).astype(np.float64)
    if a.shape!=b.shape: raise ValueError("Incompatible feature shapes")
    valid=np.isfinite(a)&np.isfinite(b)
    if ma is not None and mb is not None:
        valid &= _resample(ma).astype(bool) & _resample(mb).astype(bool)
    if not valid.any():return {'rms':None,'mae':None,'compared_ratio':0.0}
    diff=a[valid]-b[valid]
    return {'rms':float(np.sqrt(np.mean(diff**2))),
            'mae':float(np.mean(np.abs(diff))),
            'compared_ratio':float(valid.mean())}


def compare_files(baseline:Path, variant:Path)->dict:
    with np.load(baseline,allow_pickle=False) as da, np.load(variant,allow_pickle=False) as db:
        f0=da['features453'];f1=db['features453']
        c0=da['candidate_v4_32'];c1=db['candidate_v4_32']
        m0=da['candidate_v4_mask32'];m1=db['candidate_v4_mask32']
        result={}
        for name,(lo,hi) in GROUP453.items():
            result[name]=_masked_rms(f0[:,lo:hi],f1[:,lo:hi])
        for name,(lo,hi) in GROUP32.items():
            result[name]=_masked_rms(c0[:,lo:hi],c1[:,lo:hi],m0[:,lo:hi],m1[:,lo:hi])
        return result


def save_trial(trial_dir:Path, raw:dict,features:dict, metadata:dict)->None:
    trial_dir.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(trial_dir/'raw_landmarks.npz',**raw)
    np.savez_compressed(trial_dir/'features.npz',**features)
    (trial_dir/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')


def pairwise_report(data_root:Path,label:str,subject:str,out_dir:Path)->tuple[list[dict],Path]:
    trial_root=Path(data_root)/label/subject
    if not trial_root.is_dir():
        raise FileNotFoundError(f"No recorded trials for {label} / {subject}: {trial_root}")
    samples=[]
    for mp in trial_root.rglob('metadata.json'):
        try:
            meta=json.loads(mp.read_text(encoding='utf-8'))
            fp=mp.parent/'features.npz'
            if fp.is_file() and meta.get('condition') in CONDITIONS:
                samples.append((meta,fp))
        except (ValueError,OSError,TypeError):
            continue
    refs=[(m,f) for m,f in samples if m['condition']=='correct']
    if not refs:raise RuntimeError('Need at least one condition=correct reference')
    # Nearest stored reference in time where available; preserves paired signer+label.
    refs.sort(key=lambda pair:str(pair[0].get('captured_at','')))
    refm,reff=refs[-1]
    records=[]
    for meta,fp in sorted(samples,key=lambda pair:str(pair[0].get('captured_at',''))):
        if fp==reff:continue
        group=compare_files(reff,fp)
        for g,v in group.items():
            records.append({
                'label':label,'subject':subject,
                'baseline':str(reff),'trial':str(fp),
                'condition':meta['condition'],'group':g,
                'rms':v['rms'],'mae':v['mae'],'compared_ratio':v['compared_ratio'],
                'baseline_prediction':refm.get('model_prediction',''),
                'trial_prediction':meta.get('model_prediction',''),
                'baseline_confidence':refm.get('model_confidence',''),
                'trial_confidence':meta.get('model_confidence',''),
            })
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    csv_path=out_dir/f'comparison_{label}_{subject}.csv'
    fields=list(records[0]) if records else ['label','subject','condition','group','rms','mae','compared_ratio']
    with csv_path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)
    summary={'label':label,'subject':subject,'reference':str(reff),
             'trial_count':len(samples),'comparison_count':len(samples)-1,
             'feature_groups':{'existing_453':list(GROUP453),'candidate_32':list(GROUP32)},
             'results':records,
             'interpretation_note':('RMS is a diagnostic distance, not model importance or accuracy. '
                 'Different recording speeds and tracking quality can alter RMS. '
                 'Use correct_repeat as an empirical noise reference; do not tune on locked TEST.')}
    js=out_dir/f'comparison_{label}_{subject}.json'
    js.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return records,csv_path
