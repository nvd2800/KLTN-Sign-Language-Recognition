"""TCN V3 positional sensitivity: annotated REAL negatives + synthetic 1-shoulder probe.
This is a diagnostic tool, not evidence of signer-independent generalization.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.feature_spatial import load_raw_npz,spatial60
from core.preprocess_phase7 import uniform_resample_or_pad,apply_standardizer



def shift_hands_horizontally_by_shoulder_width(left, right, pose, left_present, right_present,
                                                pose_present, shift_shoulders):
    """Synthetic perturbation in unmirrored image XY; don't move missing hands.

    Hand X is shifted by shift_shoulders * same-frame Pose shoulder width.
    This is a counterfactual robustness probe, NOT a real recorded negative.
    """
    l = np.array(left, dtype=np.float32, copy=True)
    r = np.array(right, dtype=np.float32, copy=True)
    p = np.asarray(pose, dtype=np.float32)
    lp = np.asarray(left_present, dtype=bool).reshape(-1)
    rp = np.asarray(right_present, dtype=bool).reshape(-1)
    pp = np.asarray(pose_present, dtype=bool).reshape(-1)
    if l.ndim != 3 or l.shape[1:] != (21, 3) or r.shape != l.shape or p.shape != (len(l), 33, 3):
        raise ValueError('Invalid landmark shapes for horizontal hand shift')
    if any(len(mask) != len(l) for mask in (lp, rp, pp)):
        raise ValueError('Presence mask length mismatch')
    widths = np.linalg.norm(p[:, 11, :2] - p[:, 12, :2], axis=1)
    valid_pose = pp & np.isfinite(widths) & (widths > 1e-4)
    if not valid_pose.any():
        raise ValueError('No valid Pose shoulder widths: cannot synthesize shift')
    moved = 0
    for arr, present in ((l, lp), (r, rp)):
        valid_hand = present & valid_pose
        count = int(np.count_nonzero(valid_hand))
        if count:
            # arr[valid_hand, :, 0] has shape [N_valid, 21].
            # The per-frame shifts MUST have shape [N_valid, 1], not [N_valid].
            arr[valid_hand, :, 0] += (float(shift_shoulders) * widths[valid_hand])[:, None]
            moved += count
    if moved == 0:
        raise ValueError('No hand/Pose co-visible frames to synthesize shift')
    return l, r


def candidate(folder,mean,std,shift_shoulders=0.0):
    from core.feature_spatial import load_raw_npz
    folder=Path(folder)
    with np.load(folder/'features.npz',allow_pickle=False) as d:
        old=np.asarray(d['features453'],np.float32)
    l,r,p,lp,rp,pp,ts=load_raw_npz(folder/'raw_landmarks.npz')
    if shift_shoulders:
        # Synthetic counterfactual; the old453 MUST also be recomputed, otherwise
        # it would have unshifted legacy pose-hand distances. Need source configs.
        from core.feature_phase4 import load_phase4_config,build_phase4_features
        from core.feature_phase5 import load_phase5_config,build_phase5_features
        from core.feature_phase6 import load_phase6_config,build_phase6_features
        with np.load(folder/'raw_landmarks.npz',allow_pickle=False) as d:
            face=np.asarray(d['face'],np.float32)
            fp=np.asarray(d['face_present'],bool)
        l,r=shift_hands_horizontally_by_shoulder_width(l,r,p,lp,rp,pp,shift_shoulders)
        c4,_=load_phase4_config(ROOT/'configs/feature_phase4.yaml')
        c5,_=load_phase5_config(ROOT/'configs/feature_phase5.yaml')
        c6,_=load_phase6_config(ROOT/'configs/feature_phase6.yaml')
        z4=build_phase4_features(l,r,lp,rp,c4);z4['timestamps']=ts if ts is not None else np.arange(len(l),dtype=np.float64)/30
        z5=build_phase5_features(phase4_arrays=z4,raw_left=l,raw_right=r,pose_xyz=p,pose_present=pp,cfg=c5)
        z6=build_phase6_features(phase5_full417=z5['full_features417'],pose_xyz=p,pose_present=pp,
                                 face_xyz=face,face_present=fp,cfg=c6)
        old=np.asarray(z6['ablation_hands_pose_face453'],np.float32)
    spatial,mask60=spatial60(l,r,p,lp,rp,pp,ts,return_mask=True)
    if old.shape!=(len(l),453):raise ValueError(f'Wrong features453 at {folder}')
    fixed,mask,_=uniform_resample_or_pad(np.concatenate([old,spatial],axis=1),32)
    scaled=apply_standardizer(fixed[None],mask[None],mean,std)[0]
    return scaled,spatial,mask60


def wrist_center(spatial,mask):
    pts=[]
    for off in (0,30):
        sel=mask[:,off] & mask[:,off+1]
        if sel.any():pts.append(spatial[sel,off:off+2].mean(axis=0))
    return np.mean(pts,axis=0) if pts else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='artifacts/experiments/tcn_v3_spatial/model.keras')
    ap.add_argument('--trials',default='logs/feature_diagnostic')
    ap.add_argument('--output',default='logs/diagnose_v3_spatial.json')
    ap.add_argument('--threshold',type=float,default=.8)
    ap.add_argument('--min-shift-shoulders',type=float,default=1.0)
    ap.add_argument('--synthetic-shift',type=float,default=1.0,help='Test counterfactual 1 shoulder; never training evidence')
    a=ap.parse_args()
    def rp(q):
        p=Path(q);return p if p.is_absolute() else ROOT/p
    folder=rp(a.model).parent
    import tensorflow as tf
    model=tf.keras.models.load_model(rp(a.model))
    labels=json.loads((folder/'labels.json').read_text(encoding='utf8'))['labels']
    with np.load(folder/'preprocess_stats.npz',allow_pickle=False) as d:
        mean=np.asarray(d['mean'],np.float32);std=np.asarray(d['std'],np.float32)
    if mean.shape!=(513,) or std.shape!=(513,) or model.input_shape[-1]!=513:raise ValueError('Must use matching V3 513D artifacts')
    all_trials={}
    for meta in rp(a.trials).rglob('metadata.json'):
        m=json.loads(meta.read_text(encoding='utf8'))
        condition=str(m.get('condition',''))
        if condition not in ('correct','correct_repeat','wrong_position','wrong_handshape','wrong_orientation','partial'):continue
        sign=str(m.get('label') or m.get('sign') or meta.parents[2].name)
        subject=str(m.get('subject') or m.get('subject_id') or meta.parents[1].name)
        if sign not in labels:continue
        all_trials.setdefault((sign,subject),[]).append((condition,meta.parent))
    records=[]
    for (sign,subject),samples in sorted(all_trials.items()):
        refs=[p for condition,p in samples if condition in ('correct','correct_repeat')]
        if not refs:continue
        ref=refs[0]
        X0,s0,m0=candidate(ref,mean,std)
        conf0=model.predict(X0[None],verbose=0)[0]
        base_pos=wrist_center(s0,m0)
        for condition,p in samples:
            try:
                X,sp,ms=candidate(p,mean,std)
                prob=model.predict(X[None],verbose=0)[0]
                conf=float(prob[labels.index(sign)])
                pos=wrist_center(sp,ms)
                shift=float(np.linalg.norm(pos-base_pos)) if pos is not None and base_pos is not None else None
                rec=dict(sign=sign,subject=subject,condition=condition,trial=str(p),
                         prediction=labels[int(np.argmax(prob))],p_true_sign=conf,
                         baseline_p_true_sign=float(conf0[labels.index(sign)]),
                         drop=float(conf0[labels.index(sign)]-conf),
                         shoulder_shift=shift, 
                         is_large_shift=(shift is not None and shift>=a.min_shift_shoulders),
                         accepted_as_sign=(labels[int(np.argmax(prob))]==sign and conf>=a.threshold),
                         unknown_prediction=labels[int(np.argmax(prob))]=='UNKNOWN',
                         note='REAL_USER_ANNOTATED; requires sign-language annotation review')
                records.append(rec)
            except (ValueError,FileNotFoundError,KeyError) as ex:
                records.append(dict(sign=sign,subject=subject,condition=condition,trial=str(p),error=str(ex)))
        if a.synthetic_shift:
            try:
                sx,ss,sm=candidate(ref,mean,std,shift_shoulders=a.synthetic_shift)
                pr=model.predict(sx[None],verbose=0)[0]
                records.append(dict(sign=sign,subject=subject,condition='synthetic_shift',
                                    trial=str(ref),shoulder_shift=a.synthetic_shift,
                                    prediction=labels[int(np.argmax(pr))],
                                    p_true_sign=float(pr[labels.index(sign)]),
                                    baseline_p_true_sign=float(conf0[labels.index(sign)]),
                                    drop=float(conf0[labels.index(sign)]-pr[labels.index(sign)]),
                                    accepted_as_sign=(labels[int(np.argmax(pr))]==sign and pr[labels.index(sign)]>=a.threshold),
                                    unknown_prediction=labels[int(np.argmax(pr))]=='UNKNOWN',
                                    note='SYNTHETIC ONLY; outside frame possible. DO NOT report as real camera accuracy'))
            except (ValueError,FileNotFoundError,KeyError) as ex:
                records.append(dict(sign=sign,subject=subject,condition='synthetic_shift',error=str(ex)))
    by_condition={}
    for condition in sorted(set(r['condition'] for r in records)):
        vals=[r for r in records if r['condition']==condition and 'p_true_sign' in r]
        by_condition[condition]={
            'n':len(vals),
            'mean_p_true_sign':float(np.mean([r['p_true_sign'] for r in vals])) if vals else None,
            'accepted_as_true_sign':sum(r['accepted_as_sign'] for r in vals),
            'unknown':sum(r['unknown_prediction'] for r in vals),
            'mean_drop':float(np.mean([r['drop'] for r in vals])) if vals else None,
        }
    errors=[dict(sign=r.get('sign'),subject=r.get('subject'),condition=r.get('condition'),
                 error=r.get('error')) for r in records if 'error' in r]
    result=dict(mode='DIAGNOSTIC_NOT_BLIND_TEST',model=str(rp(a.model)),threshold=a.threshold,
                min_shift_shoulders=a.min_shift_shoulders,by_condition=by_condition,
                error_count=len(errors),errors=errors,records=records)
    out=rp(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(by_condition,ensure_ascii=False,indent=2));print('Saved:',out)
    if errors:
        print(f'WARNING: {len(errors)} diagnostic trials failed; inspect JSON errors before drawing conclusions.')
        for err in errors[:10]:print('  ',err)

if __name__=='__main__':main()
