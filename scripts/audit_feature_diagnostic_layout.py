"""Audit canonical diagnostic layout, manifest linkage and source videos.

Does not change samples or model data. Older samples remain reportable as
LEGACY_LAYOUT; run migrate_feature_diagnostic_layout.py --execute only after
backing up the recordings and reviewing its dry-run plan.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from core.feature_diagnostic_video import (
    check_trial_layout, DIAGNOSTIC_CONDITIONS,
)


def audit(root: Path) -> dict:
    root=Path(root)
    csv_path=root/'manifest.csv'
    manifest_rows=[]
    if csv_path.is_file():
        with csv_path.open(encoding='utf-8-sig',newline='') as f:
            manifest_rows=list(csv.DictReader(f))
    manifest_dirs=[str(Path(r.get('trial_dir','')).resolve()) for r in manifest_rows if r.get('trial_dir')]
    manifest_set=set(manifest_dirs)
    metas=sorted(set(root.glob('*/*/*/*/metadata.json')) |
                 set(root.glob('*/*/*/*/*/metadata.json')))
    records=[]
    for mp in metas:
        trial=mp.parent
        errors=[]
        try:
            metadata=json.loads(mp.read_text(encoding='utf-8'))
            label, subject, session, condition=check_trial_layout(trial,metadata,root)
        except (ValueError,TypeError,OSError) as exc:
            records.append({'trial_dir':str(trial),'status':'INVALID','errors':[str(exc)]})
            continue
        is_new=len(trial.relative_to(root).parts)==5
        for name in ('raw_landmarks.npz','features.npz'):
            if not (trial/name).is_file():errors.append(f'MISSING:{name}')
        source_video=metadata.get('video') or {}
        video_file=source_video.get('filename')
        if video_file and not (trial/video_file).is_file():
            errors.append(f'MISSING_VIDEO:{video_file}')
        if not video_file and not any((trial/n).is_file() for n in ('source_video.avi','source_video.mp4')):
            errors.append('VIDEO_NOT_PRESENT')
        if str(trial.resolve()) not in manifest_set:
            errors.append('MISSING_FROM_MANIFEST')
        records.append({
            'trial_dir':str(trial), 'sample_id':trial.name,
            'label':label, 'subject':subject, 'session':session,
            'condition':condition,'status':('REVIEW' if errors else 'PASS') if is_new else 'LEGACY_LAYOUT',
            'errors':errors,
        })
    actual={str(mp.parent.resolve()) for mp in metas}
    orphans=[path for path in manifest_set if path not in actual]
    statuses=Counter(r['status'] for r in records)
    conditions=Counter(r.get('condition') for r in records if r.get('condition'))
    return {
        'total_samples':len(records),'canonical_samples':sum(len(Path(r['trial_dir']).relative_to(root).parts)==5 for r in records),
        'manifest_rows':len(manifest_rows),
        'manifest_duplicate_paths':len(manifest_dirs)-len(manifest_set),
        'manifest_orphans':orphans,'status_counts':dict(statuses),
        'counts_by_condition':dict(conditions), 'records':records,
        'passed': bool(records) and not orphans and len(manifest_dirs)==len(manifest_set)
                  and all(r['status']=='PASS' for r in records),
    }


def main(argv=None):
    p=argparse.ArgumentParser(description='Read-only audit of Feature Diagnostic folder layout')
    p.add_argument('--input',default='logs/feature_diagnostic')
    p.add_argument('--out',default='logs/feature_diagnostic/reports/layout_audit.json')
    args=p.parse_args(argv)
    root=Path(args.input)
    if not root.is_absolute():root=ROOT/root
    if not root.exists():raise FileNotFoundError(root)
    data=audit(root)
    output=Path(args.out)
    if not output.is_absolute():output=ROOT/output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k!='records'},ensure_ascii=False,indent=2))
    print('Report:',output)
    return data

if __name__=='__main__':main()
