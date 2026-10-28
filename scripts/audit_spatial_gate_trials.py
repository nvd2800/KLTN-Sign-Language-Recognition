"""Evaluate spatial gate on separate diagnostic trials; reject seen signers by default."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.spatial_validation_gate import load_profile,trial_from_npz,evaluate
from scripts.build_spatial_gate_profile import collect,at_root


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',default='artifacts/spatial_gate_v1/profile.json')
    p.add_argument('--input',default='logs/feature_diagnostic')
    p.add_argument('--out',default='artifacts/spatial_gate_v1/independent_audit.json')
    p.add_argument('--dev-audit',action='store_true',help='Allow same recorded data used for calibration; not independent test')
    args=p.parse_args()
    prof=load_profile(at_root(args.profile))
    excluded=set(prof['gallery_subjects'])|set(prof['validation_subjects'])
    all_rows=[];by_condition=Counter()
    for label in prof['classes']:
        for r in collect(at_root(args.input),label):
            if not args.dev_audit and r['subject'] in excluded:continue
            ans=evaluate(prof,label,r['x'],r['m'])
            is_positive=r['condition'] in ('correct','correct_repeat')
            accepted=ans.decision=='PASS'
            all_rows.append(dict(label=label,condition=r['condition'],subject=r['subject'],
                 accept=accepted,expected_accept=is_positive,decision=ans.decision,
                 reason=ans.reason,source=r['path']))
            by_condition[(r['condition'],'accepted' if accepted else 'rejected')]+=1
    if not all_rows:
        raise RuntimeError('No eligible independent diagnostic trials. Use new subjects, or --dev-audit for a LEAKAGE-WARN exploratory check.')
    errors=sum(r['accept']!=r['expected_accept'] for r in all_rows)
    summary={"mode":"DEV_AUDIT_NOT_INDEPENDENT" if args.dev_audit else "INDEPENDENT_UNSEEN_SIGNER",
             "n":len(all_rows),"errors":errors,"counts":{f'{k[0]}_{k[1]}':v for k,v in by_condition.items()},
             "items":all_rows}
    out=at_root(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SPATIAL GATE AUDIT:',summary['mode'],'| samples',len(all_rows),'| errors',errors)
    print('Report:',out)
    if args.dev_audit:print('WARNING: This is NOT an independent TEST.')
if __name__=='__main__':main()
