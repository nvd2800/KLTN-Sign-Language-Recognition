"""Count paired diagnostic trials and validate feature shapes/coverage."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='logs/feature_diagnostic')
    p.add_argument('--label',default='xin_chao')
    p.add_argument('--subject',default='S01')
    args=p.parse_args()
    base=ROOT/args.input/args.label/args.subject
    rows=[];errors=[];counts=Counter()
    if not base.is_dir():raise FileNotFoundError(base)
    for meta_path in sorted(base.rglob('metadata.json')):
        try:
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            feat=meta_path.parent/'features.npz'
            with np.load(feat,allow_pickle=False) as d:
                f=d['features453'];c=d['candidate_v4_32'];m=d['candidate_v4_mask32']
            if f.ndim!=2 or f.shape[1]!=453 or c.shape!=(len(f),32) or m.shape!=(len(f),32):
                raise ValueError(f'Invalid feature shapes {f.shape} {c.shape} {m.shape}')
            if not np.isfinite(f).all() or not np.isfinite(c).all():
                raise ValueError('NaN/Inf features')
            condition=meta.get('condition','(unknown)')
            counts[condition]+=1
            rows.append({'condition':condition,'path':str(meta_path.parent),
                         'frames':len(f),'valid_candidate_ratio':float(m.mean()),
                         'model_prediction':meta.get('model_prediction','')})
        except (OSError,KeyError,ValueError,TypeError) as exc:
            errors.append(f'{meta_path}: {type(exc).__name__}: {exc}')
    result={'label':args.label,'subject':args.subject,'samples':len(rows),
            'counts':dict(counts),'errors':errors,'samples_detail':rows,
            'recommended':{'correct':3,'wrong_position':3,'wrong_handshape':3,'correct_repeat':2}}
    out=ROOT/args.input/'reports'/f'audit_{args.label}_{args.subject}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DIAGNOSTIC AUDIT |',args.label,args.subject)
    for condition in ('correct','wrong_position','wrong_handshape','wrong_orientation','partial','correct_repeat'):
        print(f'{condition:20}: {counts[condition]}')
    print('Errors:',len(errors),'| report:',out)
    if errors:
        print(*errors,sep='\n')
        raise SystemExit(1)

if __name__=='__main__':main()
