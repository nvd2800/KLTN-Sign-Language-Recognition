"""Prepare a genuinely NEW signer-independent test set; NEVER fits a scaler.

Known positives are loaded from a Phase06 manifest and rebuilt as Phase06
453D + spatial60. Real user-annotated negatives come from diagnostic trials.
Explicit signer IDs MUST not occur in any old TRAIN/VAL/TEST data.

Example:
 python scripts/prepare_v3_blind_test.py --signers S11,S12 \
   --model-dir artifacts/experiments/tcn_v3_1_spatial
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.preprocess_phase7 import (load_feature_sequence,uniform_resample_or_pad,apply_standardizer)
from core.feature_spatial import spatial_from_row
from scripts.prepare_v3_spatial import signer_mapping,read_trial,NEGATIVE_CONDITIONS


def fresh_signers(requested,used_mapping):
    names=sorted({x.strip().upper() for x in requested.split(',') if x.strip()})
    if not names:raise ValueError('Provide --signers S11,S12')
    reuse=sorted(set(names)&set(used_mapping))
    if reuse:raise ValueError(f'Blind signer leakage: these signer(s) were already used in V3: {reuse}')
    return names


def make_dataset(known_rows,trials,signers,labels,mean,std):
    seen=set();samples=[];hard=[];known_n=0;neg_n=0
    for row in known_rows:
        if str(row.get('subject_id','')).upper() not in signers:continue
        if str(row.get('phase6_status','')).upper()!='PASS':continue
        lab=row.get('sign','')
        if lab not in labels or lab=='UNKNOWN':raise ValueError(f'Invalid known label {lab}')
        sid=str(row.get('sample_id',''))
        if not sid:raise ValueError('Blank sample_id in Phase06 blind manifest')
        f453=load_feature_sequence(row,ROOT,'hands_pose_face453')
        f60=spatial_from_row(row,ROOT)
        if len(f453)!=len(f60):raise ValueError(f'Length mismatch {sid}')
        combined=np.concatenate([f453,f60],axis=1)
        fixed,m,_=uniform_resample_or_pad(combined,32)
        scaled=apply_standardizer(fixed[None],m[None],mean,std)[0]
        if sid in seen:raise ValueError(f'Duplicate ID {sid}')
        seen.add(sid);samples.append((sid,scaled,m,labels.index(lab)))
        known_n+=1
    for mp in sorted(trials.rglob('metadata.json')):
        metadata=json.loads(mp.read_text(encoding='utf-8'))
        sub=str(metadata.get('subject') or metadata.get('subject_id') or '').upper()
        cond=str(metadata.get('condition',''))
        if sub not in signers or cond not in NEGATIVE_CONDITIONS:continue
        sign=str(metadata.get('label') or metadata.get('sign') or '')
        if sign not in labels or sign in ('UNKNOWN','REST'):
            raise ValueError(f'Invalid negative reference_sign {sign} at {mp}')
        sid=f'DIAG_{sign}_{sub}_{cond}_{mp.parent.name}'
        if sid in seen:raise ValueError(f'Duplicate ID {sid}')
        x,m,_=read_trial(mp.parent,mean,std)
        seen.add(sid);samples.append((sid,x,m,labels.index('UNKNOWN')))
        hard.append({'id':sid,'split':'test','subject':sub,'condition':cond,
                     'reference_sign':sign,'raw':str(mp.parent)})
        neg_n+=1
    if not known_n or not neg_n:
        raise RuntimeError(f'Blind test needs NEW known positives and NEW real hard negatives: known={known_n},negative={neg_n}')
    X=np.stack([z[1] for z in samples]).astype(np.float32)
    M=np.stack([z[2] for z in samples]).astype(bool)
    y=np.asarray([z[3] for z in samples],np.int64)
    ids=np.asarray([z[0] for z in samples])
    return X,M,y,ids,hard,known_n,neg_n


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--signers',required=True,help='NEW real people, e.g. S11,S12')
    ap.add_argument('--known-manifest',default='dataset/processed/features_phase6/feature_manifest.csv')
    ap.add_argument('--trials',default='logs/feature_diagnostic')
    ap.add_argument('--old-known-summary',default='dataset/processed/temporal_phase7/dataset_summary.json')
    ap.add_argument('--old-v3-summary',default='dataset/processed/temporal_v3_spatial/dataset_summary.json')
    ap.add_argument('--model-dir',default='artifacts/experiments/tcn_v3_spatial')
    ap.add_argument('--out',default='dataset/processed/temporal_v3_blind')
    a=ap.parse_args()
    def rp(x):
        p=Path(x);return p if p.is_absolute() else ROOT/p
    model=rp(a.model_dir)
    # Freeze signer history from the original model and baseline V3 dataset.
    # Do NOT consult a newly regenerated Phase07 summary: new blind signers
    # may already have been added there for feature extraction.
    history=[]
    for source in (rp(a.old_v3_summary),model/'dataset_summary.json'):
        if source.is_file():
            history.append(json.loads(source.read_text(encoding='utf-8')).get('signer_assignment',{}))
    if not history:
        history.append(signer_mapping(rp(a.old_known_summary)))
    used={}
    for assignment in history:
        for sub,sp in assignment.items():
            su=str(sub).upper()
            if su in used and used[su]!=sp:
                raise ValueError(f'Historical signer split mismatch {su}')
            used[su]=sp
    signers=fresh_signers(a.signers,used)
    labels=json.loads((model/'labels.json').read_text(encoding='utf-8'))['labels']
    with np.load(model/'preprocess_stats.npz',allow_pickle=False) as d:
        mean=np.asarray(d['mean'],np.float32);std=np.asarray(d['std'],np.float32)
    if mean.shape!=(513,) or std.shape!=(513,):raise ValueError('Model scaler must be 513D')
    with rp(a.known_manifest).open('r',encoding='utf-8-sig',newline='') as f:
        known_rows=list(csv.DictReader(f))
    X,M,y,ids,hard,known_n,neg_n=make_dataset(known_rows,rp(a.trials),signers,labels,mean,std)
    dst=rp(a.out)
    if dst.exists() and (dst/'test.npz').exists():
        raise FileExistsError(f'Blind test already exists, never overwrite silently: {dst}')
    dst.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dst/'test.npz',X=X,mask=M,y=y,sample_ids=ids)
    (dst/'dataset_summary.json').write_text(json.dumps({
        'mode':'blind_independent','feature_dim':513,'target_frames':32,
        'signer_assignment':{s:'test' for s in signers},
        'counts':{'test':{'known':known_n,'hard_negative':neg_n,'total':known_n+neg_n}},
        'hard_negatives':hard,
        'note':'NEW signer blind test using FROZEN model scaler; no fit; never included in TRAIN/VAL.'
    },ensure_ascii=False,indent=2),encoding='utf-8')
    (dst/'labels.json').write_text(json.dumps({'labels':labels},ensure_ascii=False,indent=2),encoding='utf-8')
    print('BLIND TEST CREATED:',dst,'new signers',signers,'known',known_n,'negative',neg_n)
    print('Evaluate with --split test --data',dst,'--model',model/'model.keras')

if __name__=='__main__':main()
