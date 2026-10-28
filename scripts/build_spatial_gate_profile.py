"""Build per-label Spatial + Handshape gate profile from diagnostic clips.

DEV: 1 signer may be used for quick demonstration, NEVER final evidence.
Validated: disjoint gallery and validation signers, adequate sample count.
For independent test, use audit_spatial_gate_trials.py with a new unseen signer.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.spatial_validation_gate import (
    GROUPS, trial_from_npz, min_gallery_distances,
    threshold_on_validation, validate_profile, evaluate,
)


def at_root(s):
    p=Path(s)
    return p if p.is_absolute() else ROOT/p


def collect(root, label):
    base=root/label
    if not base.is_dir(): raise FileNotFoundError(f"Missing diagnostic label {label}: {base}")
    out=[]
    for mp in sorted(base.rglob('metadata.json')):
        try:
            meta=json.loads(mp.read_text(encoding='utf-8'))
            path=mp.parent/'features.npz'
            if not path.is_file(): continue
            condition=str(meta.get('condition',''))
            if condition not in ('correct','correct_repeat', 'wrong_position','wrong_handshape','wrong_orientation','partial'):
                continue
            subject=str(meta.get('subject') or mp.relative_to(base).parts[0]).upper()
            x,m=trial_from_npz(path)
            out.append(dict(subject=subject,condition=condition,path=str(path),x=x,m=m))
        except (ValueError,KeyError,TypeError,OSError) as exc:
            print('[SKIP BAD DIAGNOSTIC]',mp,type(exc).__name__,exc)
    return out


def parse_subjects(text):
    return {s.strip().upper() for s in text.split(',') if s.strip()}


def fit_label(rows, gallery_subjects, validation_subjects, dev):
    if not dev and (gallery_subjects & validation_subjects):
        raise RuntimeError('Gallery and validation signers overlap; use distinct signers or --dev')
    gallery = [r for r in rows if r['condition']=='correct' and r['subject'] in gallery_subjects]
    # correct_repeat validation to guarantee references themselves never count as positive validation
    positives = [r for r in rows if r['condition']=='correct_repeat' and r['subject'] in validation_subjects]
    if not gallery or not positives:
        raise RuntimeError('Need correct gallery and independent correct_repeat positive validation trials')
    if not dev and (gallery_subjects & validation_subjects):
        raise RuntimeError('Gallery and validation signers overlap; use distinct signers or --dev')
    refs=[(r['x'],r['m']) for r in gallery]
    spec={"references":[{"features":r['x'].tolist(),"mask":r['m'].astype(int).tolist(),"subject":r['subject'],"source":r['path']} for r in gallery],
          "thresholds":{},"validation":{}}
    diagnostics={}
    for group,(_,_,negative_condition) in GROUPS.items():
        negatives=[r for r in rows if r['condition']==negative_condition and r['subject'] in validation_subjects]
        valid_pos=[];valid_neg=[]
        for dest, items in ((valid_pos,positives),(valid_neg,negatives)):
            for r in items:
                s=min_gallery_distances(r['x'],r['m'],refs,[group])[group]
                if s['distance'] is not None:
                    dest.append(s['distance'])
        detail={"positive_n":len(valid_pos),"negative_n":len(valid_neg),
                "positive_distances":valid_pos,"negative_distances":valid_neg}
        if not valid_pos or not valid_neg:
            detail['usable']=False;detail['reason']='MISSING_VALID_CALIBRATION_DATA'
            if not dev: raise RuntimeError(f'{group}: insufficient valid calibration data')
            diagnostics[group]=detail;continue
        if not dev and (len(valid_pos)<5 or len(valid_neg)<5):
            raise RuntimeError(f'{group}: need >=5 valid positive and negative validation trials')
        threshold,metrics=threshold_on_validation(valid_pos,valid_neg)
        detail.update(metrics)
        detail['threshold']=threshold
        diagnostics[group]=detail
        if threshold is not None:
            spec['thresholds'][group]=threshold
            spec['validation'][group]={k:v for k,v in metrics.items() if k!='usable'}
    if not spec['thresholds']:
        raise RuntimeError('No discriminative groups in validation. Cannot build a useful gate.')
    if not dev and set(spec['thresholds'])!=set(GROUPS):
        raise RuntimeError('Validated profile requires all three gate groups')
    return spec,diagnostics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', default='logs/feature_diagnostic')
    p.add_argument('--labels',default='xin_chao',help='Comma-separated known classes')
    p.add_argument('--gallery-subjects',default='',help='Only correct trials build templates')
    p.add_argument('--val-subjects',default='',help='correct_repeat and wrong_* trials calibrate thresholds')
    p.add_argument('--out',default='artifacts/spatial_gate_v1/profile.json')
    p.add_argument('--dev',action='store_true',help='Exploratory mode for current one-signer data; NOT validated')
    args=p.parse_args()
    labels=[x.strip() for x in args.labels.split(',') if x.strip()]
    if not labels or any(x in ('REST','UNKNOWN') for x in labels):
        raise ValueError('Select one or more KNOWN labels only')
    root=at_root(args.input)
    datasets={label:collect(root,label) for label in labels}
    all_subjects={r['subject'] for rows in datasets.values() for r in rows}
    gallery_subjects=parse_subjects(args.gallery_subjects)
    val_subjects=parse_subjects(args.val_subjects)
    if args.dev:
        if not gallery_subjects: gallery_subjects=set(all_subjects)
        if not val_subjects: val_subjects=set(all_subjects)
    else:
        if not gallery_subjects or not val_subjects:
            raise ValueError('Validated mode requires explicit --gallery-subjects and --val-subjects')
        if gallery_subjects & val_subjects:
            raise ValueError('Signer overlap between gallery and validation')
    if not gallery_subjects or not val_subjects:
        raise ValueError('No gallery or validation signer available')
    classes={};diag={}
    for label,rows in datasets.items():
        classes[label],diag[label]=fit_label(rows,gallery_subjects,val_subjects,args.dev)
    profile={
        "schema_version":1,"calibration_mode":"DEV_ONLY" if args.dev else "VALIDATED",
        "created_at":datetime.now(timezone.utc).isoformat(),
        "model_family":"TCN_V2_453","candidate_version":"v4_candidate32",
        "min_overlap":0.55,"gallery_subjects":sorted(gallery_subjects),
        "validation_subjects":sorted(val_subjects),"classes":classes,
        "note":"Missing labels bypass with explicit warning. DEV thresholds are not scientific results."}
    validate_profile(profile)
    dest=at_root(args.out); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dest.with_name('calibration_report.json')
    report.write_text(json.dumps({"mode":profile['calibration_mode'],"diagnostics":diag,
                                  "gallery_subjects":sorted(gallery_subjects),
                                  "validation_subjects":sorted(val_subjects)},ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nSPATIAL + HANDSHAPE PROFILE CREATED')
    print('Mode:',profile['calibration_mode'],'(exploratory only)' if args.dev else '(signer-disjoint calibration)')
    for label,spec in classes.items():
        print(label,'references:',len(spec['references']))
        for group,detail in diag[label].items():
            print(f"  {group:25} threshold={detail.get('threshold')} pos={detail['positive_n']} neg={detail['negative_n']} {detail.get('reason','')}")
    print('Profile:',dest,'\nReport:',report)
    print('Next: run live script with --spatial-gate',dest)

if __name__=='__main__':main()
