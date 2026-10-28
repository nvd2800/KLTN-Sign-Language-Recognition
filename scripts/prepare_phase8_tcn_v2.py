"""Create a five-class dataset using Phase07 known splits + Phase06 UNKNOWN inventory.
Does NOT modify the Phase07 or Phase08-V1 datasets/artifacts.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, sys
from collections import Counter
from pathlib import Path
import numpy as np, yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.tcn_v2_unknown import (read_csv,read_phase7_split,load_v1_scaler,
    assign_unknown_subject,subject_map_from_summary,get_extras,resample_and_scale)

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def prepare(root,cfg,mode,extra):
    def path(p):
        p=Path(p)
        return p if p.is_absolute() else root/p
    base=path(cfg['paths']['phase7_root']); out=path(cfg['paths']['v2_data_dir'])
    out.mkdir(parents=True,exist_ok=True)
    scalefile=base/'preprocess_stats.npz'
    mean,std=load_v1_scaler(scalefile)
    labels=json.loads((base/'labels.json').read_text(encoding='utf-8'))['labels']
    if len(labels)!=4 or 'UNKNOWN' in labels or len(set(labels))!=4:
        raise ValueError(f'Expected four unique known labels, got {labels}')
    labels_v2=list(labels)+['UNKNOWN']
    known_map=subject_map_from_summary(path(cfg['paths']['phase7_dataset_summary']))
    U=[];ids=set()
    inventory=path(cfg['paths']['unknown_inventory'])
    for r in read_csv(inventory):
        sid=(r.get('sample_id') or '').strip()
        subject=(r.get('subject_id') or '').strip().upper()
        category=(r.get('category') or '').strip()
        sp=assign_unknown_subject(subject,known_map,extra)
        featpath=path(r.get('feature_path') or '')
        if not sid or not category or sid in ids: raise ValueError(f'Invalid/duplicate UNKNOWN ID {sid!r}')
        ids.add(sid)
        if not featpath.is_file():raise FileNotFoundError(featpath)
        with np.load(featpath,allow_pickle=False) as d:
            feat=np.asarray(d['ablation_hands_pose_face453'],np.float32)
        x,mask=resample_and_scale(feat,mean,std)
        U.append({'split':sp,'sample_id':sid,'subject':subject,'category':category,'X':x,'mask':mask})
    if not U:raise ValueError('UNKNOWN inventory empty; run process_phase8_unknown.py')
    counts=Counter(u['split'] for u in U)
    if not counts['train']:
        raise ValueError('No UNKNOWN TRAIN sample. Current S01/S02 may have been assigned only VAL/TEST; '
                         'use Phase07 signer map (S01/S02 usually TRAIN), collect TRAIN samples.')
    if mode=='strict' and (not counts['val'] or not counts['test']):
        raise ValueError(f'STRICT protocol requires UNKNOWN VAL and TEST from held-out signers; got {dict(counts)}. '
                         'Collect additional unknown for signer IDs in Phase07 VAL/TEST (e.g. S08/S09), '
                         'or use --mode dev for an experimental 5-class model WITHOUT official UNKNOWN evaluation.')
    split_counts={}; split_categories={}
    for split in ('train','val','test'):
        known=read_phase7_split(base/f'{split}.npz')
        if known['y'].size and known['y'].max()>=len(labels):
            raise ValueError('Phase07 y out of range')
        z=[u for u in U if u['split']==split]
        if z:
            Ux=np.stack([u['X'] for u in z]); Um=np.stack([u['mask'] for u in z]);Uy=np.full(len(z),len(labels),np.int64)
            X=np.concatenate([known['X'],Ux]);M=np.concatenate([known['mask'],Um]);y=np.concatenate([known['y'],Uy]);
            names=known['ids']+[u['sample_id'] for u in z]
        else:
            X,M,y,names=known['X'],known['mask'],known['y'],known['ids']
        if len(set(names))!=len(names):raise ValueError(f'duplicate sample IDs within {split}')
        np.savez_compressed(out/f'{split}.npz',X=X.astype(np.float32),mask=M,y=y,sample_ids=np.asarray(names))
        split_counts[split]={'known':len(known['y']),'unknown':len(z),'total':len(y)}
        split_categories[split]=dict(Counter(u['category'] for u in z))
    # Copy V1 stats to make the bundled V2 model fully reproducible, no fit on test.
    shutil.copy2(scalefile,out/'preprocess_stats.npz')
    (out/'labels.json').write_text(json.dumps({'labels':labels_v2,
        'label_to_id':{s:i for i,s in enumerate(labels_v2)}},ensure_ascii=False,indent=2),encoding='utf-8')
    subject_splits={s:k for s,k in known_map.items()}
    for k,v in extra.items():subject_splits[k]=v
    protocol={
        'phase':'08-TCN-V2','mode':mode,'feature_shape':[32,453],
        'known_labels':labels,'labels':labels_v2,'split_counts':split_counts,
        'unknown_categories':split_categories,
        'unknown_subject_assignment':{s:subject_splits[s] for s in sorted({u['subject'] for u in U})},
        'scaler':'Frozen Phase07 TRAIN-only scaler (shared with TCN V1)',
        'scaler_sha256':file_hash(scalefile),
        'known_test_not_used_for_training':True,
        'warning':('EXPLORATORY ONLY: UNKNOWN VAL or TEST absent or previously viewed; '
                   'do not report signer-independent UNKNOWN performance.' if mode=='dev' else
                   'UNKNOWN TEST has been used in earlier phase8 threshold checks if reusing S02; '
                   'use newly untouched samples for a truly blind final evaluation.'),
        'missing_unknown_val':counts['val']==0,'missing_unknown_test':counts['test']==0,
    }
    (out/'dataset_summary.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nPHASE08 TCN V2 PREPARED')
    print('Mode:',mode,'| Labels:',labels_v2)
    for k,v in split_counts.items():print(k.upper(),v)
    print('UNKNOWN subjects:',protocol['unknown_subject_assignment'])
    print('Output:',out)
    if mode=='dev':print('WARNING: EXPERIMENTAL TRAINING ONLY. UNKNOWN validation/test may be absent.')
    return protocol

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/phase8_tcn_v2.yaml')
    ap.add_argument('--mode',choices=['dev','strict'],default='dev')
    ap.add_argument('--train-subjects',default='',help='New signers NOT in Phase07 known split')
    ap.add_argument('--val-subjects',default='')
    ap.add_argument('--test-subjects',default='')
    ap.add_argument('--root',default=None,help='Project root override for integration testing')
    a=ap.parse_args()
    root=Path(a.root).resolve() if a.root else ROOT
    conf=Path(a.config);conf=conf if conf.is_absolute() else root/conf
    cfg=yaml.safe_load(conf.read_text(encoding='utf-8'))
    prepare(root,cfg,a.mode,get_extras(a.train_subjects,a.val_subjects,a.test_subjects))
if __name__=='__main__':main()
