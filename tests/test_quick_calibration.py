"""Integration test for guarded DEV calibration without TensorFlow dependency."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]


def test_quick_calibration_no_official_outputs(tmp_path):
    root = tmp_path / 'project'
    for d in ['scripts', 'core', 'configs', 'dataset/processed/temporal_phase7',
              'dataset/processed/unknown_phase8', 'dataset/unknown_phase8', 'artifacts/tcn']:
        (root/d).mkdir(parents=True, exist_ok=True)
    for name in ['calibrate_phase8_dev.py', 'calibrate_phase8_threshold.py',
                 'evaluate_phase8_unknown.py', 'audit_phase8.py']:
        shutil.copy2(PACKAGE / 'scripts' / name, root / 'scripts' / name)
    shutil.copy2(PACKAGE / 'tests/fixtures_core/unknown_phase8.py',root / 'core/unknown_phase8.py')
    shutil.copy2(PACKAGE / 'tests/fixtures_config/phase8_unknown.yaml',root / 'configs/phase8_unknown.yaml')
    (root / 'artifacts/tcn/model.keras').write_bytes(b'dummy')
    (root / 'dataset/unknown_phase8/unknown_manifest_mode.json').write_text(json.dumps({
        'mode':'exploratory_val_only', 'categories':['holding_phone','touching_face','scratching_head'],
        'subjects':['S01'], 'test_samples':0, 'val_samples':3,
    }))
    np.savez_compressed(root / 'dataset/processed/temporal_phase7/val.npz',
                        X=np.ones((3,32,453), dtype=np.float32))
    np.savez_compressed(root / 'dataset/processed/unknown_phase8/val_unknown.npz',
                        X=np.zeros((3,32,453), dtype=np.float32))
    (root / 'tensorflow.py').write_text('''import numpy as np
class Model:
    def predict(self, x, verbose=0):
        if x[0,0,0]>0:
            return np.asarray([[.95,.03,.01,.01],[.80,.10,.05,.05],[.90,.04,.03,.03]])
        return np.asarray([[.60,.2,.1,.1],[.4,.3,.2,.1],[.70,.1,.1,.1]])
class M:
    @staticmethod
    def load_model(p): return Model()
class K:
    models=M()
keras=K()
''')
    r=subprocess.run([sys.executable,'scripts/calibrate_phase8_dev.py'],cwd=root,capture_output=True,text=True)
    assert r.returncode == 0, r.stdout+r.stderr
    assert (root/'artifacts/phase8_unknown_dev/threshold_dev_only.json').exists()
    assert not (root/'artifacts/phase8_unknown/calibrated_threshold.json').exists()
    dev=json.loads((root/'artifacts/phase8_unknown_dev/threshold_dev_only.json').read_text())
    assert dev['status']=='EXPLORATORY_NOT_VALIDATED'
    assert dev['unknown_dev_count']==3
    r=subprocess.run([sys.executable,'scripts/calibrate_phase8_threshold.py'],cwd=root,capture_output=True,text=True)
    assert r.returncode!=0
    assert 'QUICK TEST' in (r.stdout+r.stderr)
    r=subprocess.run([sys.executable,'scripts/evaluate_phase8_unknown.py'],cwd=root,capture_output=True,text=True)
    assert r.returncode!=0
    assert 'QUICK TEST' in (r.stdout+r.stderr)
    r=subprocess.run([sys.executable,'scripts/audit_phase8.py'],cwd=root,capture_output=True,text=True)
    assert r.returncode!=0
    assert 'EXPLORATORY MODE' in (r.stdout+r.stderr)
