"""Audit real, signer-safe V3 hard-negative recordings before V3.1 training.

Does not edit samples or create synthetic data. Signer split from known summary
is treated as immutable; no TEST/VAL clip is proposed for TRAIN.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.prepare_v3_spatial import signer_mapping,NEGATIVE_CONDITIONS
from core.v3_eval_trace import compute_landmark_qc


def audit(trials:Path, mapping, minimum:int=10):
    records=[]; hashes={}; duplicates=[];errors=[]
    for meta_path in sorted(trials.rglob('metadata.json')):
        try:
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            cond=str(meta.get('condition',''))
            if cond not in NEGATIVE_CONDITIONS:continue
            subject=str(meta.get('subject') or meta.get('subject_id') or '').upper().strip()
            sign=str(meta.get('label') or meta.get('sign') or '').strip()
            split=mapping.get(subject,'UNREGISTERED')
            raw=meta_path.parent/'raw_landmarks.npz';features=meta_path.parent/'features.npz'
            if not subject or not sign or split=='UNREGISTERED':
                errors.append(f'Missing/unknown metadata or signer in {meta_path}');continue
            if not raw.is_file() or not features.is_file():
                errors.append(f'Missing raw/features for {meta_path}');continue
            digest=hashlib.sha256(raw.read_bytes()).hexdigest()
            if digest in hashes:
                duplicates.append(dict(a=hashes[digest],b=str(raw),hash=digest))
            else:hashes[digest]=str(raw)
            qc=compute_landmark_qc(raw,sign)
            records.append(dict(subject=subject,split=split,sign=sign,condition=cond,
                                trial=str(meta_path.parent),sha256=digest,**qc))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            errors.append(f'Invalid metadata {meta_path}: {type(exc).__name__}: {exc}')
    train=[r for r in records if r['split']=='train']
    counts=collections.Counter((r['sign'],r['condition']) for r in train)
    signs=sorted({r['sign'] for r in train})
    gaps=[dict(sign=s,condition=c,available=counts[(s,c)],suggested_additional=max(0,minimum-counts[(s,c)]))
          for s in signs for c in sorted(NEGATIVE_CONDITIONS)]
    return {'total_real_trials':len(records),'train_trials':len(train),
            'val_trials':sum(r['split']=='val' for r in records),
            'test_trials':sum(r['split']=='test' for r in records),
            'counts_train':{f'{s}/{c}':n for (s,c),n in sorted(counts.items())},
            'gaps_train':gaps,'duplicates_exact_file':duplicates,
            'errors':errors,'qc_review_count':sum(r['qc_status']!='AVAILABLE' for r in records),
            'records':records,'recommendation':'Use only train split real negatives for fitting. Never copy existing TEST samples into TRAIN.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trials',default='logs/feature_diagnostic')
    ap.add_argument('--known-summary',default='dataset/processed/temporal_phase7/dataset_summary.json')
    ap.add_argument('--minimum-per-condition',type=int,default=10)
    ap.add_argument('--out',default='logs/v3_hardnegative_audit')
    a=ap.parse_args()
    def rp(p):
        p=Path(p);return p if p.is_absolute() else ROOT/p
    if a.minimum_per_condition<1:raise ValueError('--minimum-per-condition must be >=1')
    m=signer_mapping(rp(a.known_summary))
    result=audit(rp(a.trials),m,a.minimum_per_condition)
    target=rp(a.out);target.mkdir(parents=True,exist_ok=True)
    (target/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (target/'train_gaps.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['sign','condition','available','suggested_additional'])
        writer.writeheader();writer.writerows(result['gaps_train'])
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))
    print('Saved:',target)
    if result['errors'] or result['duplicates_exact_file']:
        raise SystemExit('AUDIT REVIEW REQUIRED: inspect audit.json; no files were removed')

if __name__=='__main__':main()
