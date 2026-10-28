"""Static preflight audit; does not require TensorFlow or open the webcam."""
import json,sys,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['features','dataset','model'],default='features')
    a=p.parse_args()
    needed=['core/feature_spatial.py','core/phase8_live.py','core/preprocess_phase7.py',
            'core/feature_phase4.py','core/feature_phase5.py','core/feature_phase6.py',
            'core/continuous_phase9.py','core/phase9_live_logic.py',
            'core/models_phase7.py',
            'configs/feature_phase4.yaml','configs/feature_phase5.yaml','configs/feature_phase6.yaml',
            'configs/phase7_spatial.yaml','configs/phase9_v3_spatial.yaml',
            'dataset/processed/features_phase6/feature_manifest.csv']
    if a.stage in ('dataset','model'):
        needed += ['dataset/processed/temporal_phase7_spatial513/preprocess_stats.npz',
                   'dataset/processed/temporal_v3_spatial/train.npz',
                   'dataset/processed/temporal_v3_spatial/val.npz',
                   'dataset/processed/temporal_v3_spatial/labels.json']
    if a.stage=='model':
        needed += ['artifacts/experiments/tcn_v3_spatial/model.keras',
                   'artifacts/experiments/tcn_v3_spatial/preprocess_stats.npz']
    missing=[]
    for n in needed:
        found=(ROOT/n).is_file()
        print('OK ' if found else 'MISS',n)
        if not found:missing.append(n)
    if a.stage in ('dataset','model') and not missing:
        f=ROOT/'dataset/processed/temporal_v3_spatial/train.npz'
        with np.load(f,allow_pickle=False) as d:
            if d['X'].ndim!=3 or d['X'].shape[1:]!=(32,513):raise RuntimeError(f'Invalid X shape: {d["X"].shape}')
        with np.load(ROOT/'dataset/processed/temporal_phase7_spatial513/preprocess_stats.npz',allow_pickle=False) as d:
            if d['mean'].shape!=(513,):raise RuntimeError('Invalid scaler513')
        with (ROOT/'dataset/processed/temporal_v3_spatial/labels.json').open(encoding='utf8') as fh:
            labels=json.load(fh)['labels']
            if len(labels)!=5 or 'UNKNOWN' not in labels:raise RuntimeError('Expected 5-class labels')
        print('PASS shape/scaler/labels')
    if missing:
        print(f'AUDIT MISSING {len(missing)} file(s)')
        raise SystemExit(1)
    print('AUDIT PASS:',a.stage)

if __name__=='__main__':main()
