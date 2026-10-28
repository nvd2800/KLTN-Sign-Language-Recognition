"""Guard V3.1 rebuild against overlap/contamination with frozen V3 TEST.

Records SHA256 of original test NPZ, checks strict signer assignments and
ensures baseline TEST samples are NEVER merged into TRAIN or VAL.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def load_ids(path):
    with np.load(path,allow_pickle=False) as d:
        if 'sample_ids' not in d.files:raise ValueError(f'Missing IDs: {path}')
        ids={str(i) for i in d['sample_ids']}
    return ids


def check_dataset(baseline:Path,new:Path):
    base_test=load_ids(baseline/'test.npz')
    new_train=load_ids(new/'train.npz');new_val=load_ids(new/'val.npz')
    new_test=load_ids(new/'test.npz')
    bad_train=sorted(base_test&new_train);bad_val=sorted(base_test&new_val)
    intersect=sorted((new_train&new_val)|(new_train&new_test)|(new_val&new_test))
    issues=[]
    if bad_train:issues.append('Baseline TEST IDs moved into TRAIN: '+str(bad_train[:6]))
    if bad_val:issues.append('Baseline TEST IDs moved into VAL: '+str(bad_val[:6]))
    if intersect:issues.append('V3.1 split overlap: '+str(intersect[:6]))
    dataset=json.loads((new/'dataset_summary.json').read_text(encoding='utf-8'))
    if dataset.get('mode')!='strict':issues.append('V3.1 data must be STRICT')
    mapping=dataset.get('signer_assignment',{})
    for row in dataset.get('hard_negatives',[]):
        if mapping.get(str(row.get('subject','')).upper())!=row.get('split'):
            issues.append(f'Hard-negative signer split mismatch: {row.get("id")}')
    return {'pass':not issues,'issues':issues,'baseline_test_ids':len(base_test),
            'v3_1_train_ids':len(new_train),'v3_1_val_ids':len(new_val),
            'v3_1_test_ids':len(new_test),'baseline_test_reused_for_fitting':len(bad_train)+len(bad_val),
            'baseline_test_file_sha256':hashlib.sha256((baseline/'test.npz').read_bytes()).hexdigest(),
            'warning':'A new model informed by analysis of old TEST requires fresh blind TEST signers for official reporting.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',default='dataset/processed/temporal_v3_spatial')
    ap.add_argument('--candidate',default='dataset/processed/temporal_v3_1_spatial')
    ap.add_argument('--out',default='logs/v3_1_dataset_guard.json')
    a=ap.parse_args()
    def rp(p):
        p=Path(p);return p if p.is_absolute() else ROOT/p
    result=check_dataset(rp(a.baseline),rp(a.candidate))
    out=rp(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['pass']:raise SystemExit('Dataset guard FAILED. DO NOT TRAIN.')

if __name__=='__main__':main()
