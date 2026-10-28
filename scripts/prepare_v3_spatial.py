"""Merge signer-independent Phase07 513D known and REAL annotated negative trials.

Only diagnostic wrong_position, wrong_handshape, wrong_orientation, partial become UNKNOWN.
No synthetic samples are passed off as real. Scaler is FIT ON KNOWN TRAIN
by prepare_phase7_dataset, never fitted on VAL/TEST; no accidental V2 reuse.
"""
from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.feature_spatial import spatial60
from core.preprocess_phase7 import uniform_resample_or_pad,apply_standardizer

NEGATIVE_CONDITIONS={'wrong_position','wrong_handshape','wrong_orientation','partial'}
SPLITS=('train','val','test')


def signer_mapping(summary_path,extra=None):
    j=json.loads(Path(summary_path).read_text(encoding='utf-8'))
    m={}
    for split,users in j['subjects'].items():
        if split not in SPLITS:continue
        for user in users:
            u=str(user).upper()
            if u in m and m[u]!=split:raise ValueError(f'Signer leakage {u}')
            m[u]=split
    for split,users in (extra or {}).items():
        for user in users:
            u=user.upper()
            if u in m and m[u]!=split:raise ValueError(f'Conflicting signer split {u}')
            m[u]=split
    return m


def read_trial(path,mean,std):
    from core.feature_spatial import load_raw_npz
    from core.preprocess_phase7 import uniform_resample_or_pad
    folder=Path(path)
    with np.load(folder/'features.npz',allow_pickle=False) as d:
        raw453=np.asarray(d['features453'],np.float32)
    l,r,p,lp,rp,pp,ts=load_raw_npz(folder/'raw_landmarks.npz')
    feat=spatial60(l,r,p,lp,rp,pp,ts)
    if raw453.shape!=(len(feat),453):raise ValueError(f'Temporal feature mismatch: {folder}')
    combo=np.concatenate([raw453,feat],axis=1)
    fixed,mask,_=uniform_resample_or_pad(combo,32)
    scaled=apply_standardizer(fixed[None],mask[None],mean,std)[0]
    return scaled,mask,feat


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--known',default='dataset/processed/temporal_phase7_spatial513')
    ap.add_argument('--trials',default='logs/feature_diagnostic')
    ap.add_argument('--known-summary',default='dataset/processed/temporal_phase7/dataset_summary.json')
    ap.add_argument('--output',default='dataset/processed/temporal_v3_spatial')
    ap.add_argument('--mode',choices=('strict','dev'),default='strict')
    ap.add_argument('--new-train-signers',default='')
    ap.add_argument('--new-val-signers',default='')
    ap.add_argument('--new-test-signers',default='')
    a=ap.parse_args()
    def rp(x):
        p=Path(x);return p if p.is_absolute() else ROOT/p
    known=rp(a.known);out=rp(a.output);trials=rp(a.trials)
    extra={sp:[q.strip().upper() for q in getattr(a,f'new_{sp}_signers').split(',') if q.strip()] for sp in SPLITS}
    mapping=signer_mapping(rp(a.known_summary),extra)
    labels=json.loads((known/'labels.json').read_text(encoding='utf-8'))['labels']
    if 'UNKNOWN' in labels:raise ValueError('Known labels already include UNKNOWN; double counting is unsafe')
    labels=labels+['UNKNOWN'];unknown_id=len(labels)-1
    if 'REST' not in labels or len(labels)!=5:raise ValueError(f'Expect 4 KNOWN classes including REST, plus UNKNOWN: {labels}')
    with np.load(known/'preprocess_stats.npz',allow_pickle=False) as f:
        mean=np.asarray(f['mean'],np.float32);std=np.asarray(f['std'],np.float32)
        size=int(np.asarray(f['feature_dim']).flat[0]);target=int(np.asarray(f['target_frames']).flat[0])
    if (size,target)!=(513,32):raise ValueError('513D/32-frame scaler required')
    rows=[]; negative_by_split={k:[] for k in SPLITS}
    for mp in sorted(trials.rglob('metadata.json')):
        metadata=json.loads(mp.read_text(encoding='utf-8'))
        condition=str(metadata.get('condition',''))
        if condition not in NEGATIVE_CONDITIONS:continue
        subject=str(metadata.get('subject') or metadata.get('subject_id') or mp.parents[1].name).upper()
        if subject not in mapping:
            raise ValueError(f'No signer split for {subject} at {mp}; configure --new-*-signers')
        if not (mp.parent/'features.npz').exists() or not (mp.parent/'raw_landmarks.npz').exists():
            raise FileNotFoundError(f'Real landmarks/features required: {mp.parent}')
        X,M,_=read_trial(mp.parent,mean,std)
        sign=str(metadata.get('label') or metadata.get('sign') or mp.parents[2].name)
        if sign not in labels[:-1]:raise ValueError(f'Negative of unknown sign {sign} at {mp}')
        split=mapping[subject];uid=f'DIAG_{sign}_{subject}_{condition}_{mp.parent.name}'
        negative_by_split[split].append((uid,X,M,condition,subject,sign))
        rows.append(dict(id=uid,split=split,subject=subject,condition=condition,reference_sign=sign,raw=str(mp.parent)))
    if not negative_by_split['train']:
        raise RuntimeError('No REAL annotated hard negatives in TRAIN; collect via live_feature_diagnostic.py')
    if a.mode=='strict' and any(not negative_by_split[k] for k in ('val','test')):
        raise RuntimeError('STRICT requires real hard negatives from independent VAL and TEST signers. '
                           'Use --mode dev for exploration only; never report DEV scores as final.')
    out.mkdir(parents=True,exist_ok=True)
    counts={}; all_ids=set()
    for split in SPLITS:
        with np.load(known/f'{split}.npz',allow_pickle=False) as d:
            X=np.asarray(d['X'],np.float32); M=np.asarray(d['mask'],bool)
            y=np.asarray(d['y'],np.int64);ids=[str(v) for v in d['sample_ids']]
        if X.ndim!=3 or X.shape[1:]!=(32,513):raise ValueError(f'Known shape invalid: {X.shape}')
        neg=negative_by_split[split]
        if neg:
            X=np.concatenate([X,np.stack([z[1] for z in neg])]);M=np.concatenate([M,np.stack([z[2] for z in neg])]);
            y=np.concatenate([y,np.full(len(neg),unknown_id,np.int64)]);ids += [z[0] for z in neg]
        if any(q in all_ids for q in ids) or len(ids)!=len(set(ids)):
            raise ValueError(f'Duplicate IDs (including across splits) in {split}')
        all_ids.update(ids)
        np.savez_compressed(out/f'{split}.npz',X=X.astype(np.float32),mask=M.astype(bool),y=y,sample_ids=np.array(ids))
        counts[split]={'known':int(len(y)-len(neg)),'hard_negative':len(neg),'total':len(y),
                       'categories':dict(Counter(z[3] for z in neg))}
    import shutil
    shutil.copy2(known/'preprocess_stats.npz',out/'preprocess_stats.npz')
    (out/'labels.json').write_text(json.dumps({'labels':labels,'label_to_id':{s:i for i,s in enumerate(labels)}},ensure_ascii=False,indent=2),encoding='utf8')
    summary=dict(mode=a.mode,feature_dim=513,target_frames=32,labels=labels,counts=counts,
                 signer_assignment=mapping,hard_negatives=rows,
                 scaler='Phase07 Spatial 513D fitted on KNOWN TRAIN only; not re-fitted on test',
                 warning='DEV_ONLY. No independent negative VAL/TEST' if a.mode=='dev' else None)
    (out/'dataset_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'negative_manifest.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','split','subject','condition','reference_sign','raw']);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'mode':a.mode,'counts':counts,'signers':mapping},ensure_ascii=False,indent=2))
    print('READY TO TRAIN:',out)

if __name__=='__main__':main()
