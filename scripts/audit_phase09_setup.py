"""Preflight Phase09 live inputs without opening camera or requiring TensorFlow.

Use: python scripts/audit_phase09_setup.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]

def rp(p):
    path=Path(p)
    return path if path.is_absolute() else ROOT/path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/phase9_live.yaml')
    args=parser.parse_args()
    cfg=yaml.safe_load(rp(args.config).read_text(encoding='utf-8'))
    files={k:rp(v) for k,v in cfg['artifacts'].items()}
    errors=[]
    for name,p in files.items():
        if not p.is_file():errors.append(f'Missing {name}: {p}')
    expected=['configs/feature_phase4.yaml','configs/feature_phase5.yaml','configs/feature_phase6.yaml',
              'core/feature_phase4.py','core/feature_phase5.py','core/feature_phase6.py',
              'core/preprocess_phase7.py','core/phase8_live.py','core/continuous_phase9.py',
              'core/phase9_live_logic.py']
    for item in expected:
        if not rp(item).is_file():errors.append(f'Missing dependency: {item}')
    if files['labels'].is_file():
        obj=json.loads(files['labels'].read_text(encoding='utf-8'))
        labels=obj['labels'] if isinstance(obj,dict) else obj
        if not isinstance(labels,list) or len(labels)!=5 or 'UNKNOWN' not in labels or 'REST' not in labels:
            errors.append(f'Bad five-class labels: {labels}')
        else:
            print('Labels:',labels)
    if files['stats'].is_file():
        with np.load(files['stats'],allow_pickle=False) as d:
            wanted={'mean','std','target_frames','feature_dim'}
            missing=wanted-set(d.files)
            if missing: errors.append(f'Stats keys missing: {sorted(missing)}')
            else:
                mean=np.asarray(d['mean']);std=np.asarray(d['std']); t=int(np.asarray(d['target_frames']).reshape(-1)[0]); feat=int(np.asarray(d['feature_dim']).reshape(-1)[0]);
                print('Scaler:',t,'x',feat)
                if mean.shape!=(453,) or std.shape!=(453,) or t!=32 or feat!=453:
                    errors.append('Scaler not Phase09 expected 32x453')
                if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
                    errors.append('Scaler contains nonfinite/nonpositive values')
    print('\nPHASE09 PREFLIGHT')
    print('Errors:',len(errors))
    for e in errors:print('  -',e)
    print('PHASE09 SETUP:', 'PASS (paths/scaler/labels only)' if not errors else 'FAIL')
    if not errors:print('Next: python scripts/live_phase09_tcn_v2.py --fullscreen')
    raise SystemExit(0 if not errors else 1)

if __name__=='__main__':main()
