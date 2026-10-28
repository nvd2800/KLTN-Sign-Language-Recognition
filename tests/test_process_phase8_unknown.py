"""Regression tests. No camera, TensorFlow, or trained model required."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / 'scripts' / 'process_phase8_unknown.py'

spec = importlib.util.spec_from_file_location('phase8_unknown_pipeline_test', SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
# Loading script dependencies occurs on import; core modules are provided by project in actual usage.
# Our test imports occur only after temporary project construction via subprocess.


def create_project(tmp_path: Path):
    root = tmp_path / 'project'
    for dirname in ('core', 'scripts', 'configs'):
        (root / dirname).mkdir(parents=True)
    (root / 'scripts' / 'process_phase8_unknown.py').write_bytes(SOURCE.read_bytes())
    source_dir = Path(__file__).resolve().parent / 'fixtures_core'
    for name in ('feature_phase4.py', 'feature_phase5.py', 'feature_phase6.py'):
        shutil.copy2(source_dir / name, root / 'core' / name)
    for name in ('feature_phase4', 'feature_phase5', 'feature_phase6'):
        (root / 'configs' / f'{name}.yaml').write_text('{}\n', encoding='utf-8')
    return root


def create_sample(root: Path, category: str, subject: str, session='SES01', index=1, T=12):
    sample_id = f'UNKNOWN_{subject}_{session}_{index:03d}'  # deliberate duplicate across categories
    p = root / 'dataset' / 'raw' / 'UNKNOWN' / category / subject / session / sample_id
    p.mkdir(parents=True, exist_ok=True)
    left = np.zeros((T, 21, 3), np.float32)
    right = np.zeros_like(left)
    # Real wrist->middle MCP scale is nonzero.
    left[:, 9, 0] = .05
    right[:, 9, 0] = .05
    left[:, 4, 0] = .015
    left[:, 8, 0] = .030
    pose = np.zeros((T, 33, 3), np.float32)
    pose[:, 11, 0] = .4
    pose[:, 12, 0] = .6
    pose[:, 0, 1] = .35
    face = np.zeros((T, 468, 3), np.float32)
    face[:, 33, 0] = .4
    face[:, 263, 0] = .6
    arrays = {
        'timestamps': np.arange(T, dtype=float) / 30.,
        'left_hand': left, 'right_hand': right,
        'left_present': np.ones(T, bool),
        'right_present': np.zeros(T, bool),
        'pose': pose, 'face': face,
        'pose_present': np.ones(T, bool),
        'face_present': np.ones(T, bool),
    }
    np.savez_compressed(p / 'landmarks.npz', **arrays)
    (p / 'metadata.json').write_text(json.dumps({
        'sample_id': sample_id, 'subject_id': subject, 'session_id': session,
        'sign': 'UNKNOWN', 'category': category,
    }), encoding='utf-8')
    return p


def invoke(root, *flags):
    return subprocess.run([sys.executable, str(root / 'scripts' / 'process_phase8_unknown.py'), *flags],
                          cwd=root, text=True, capture_output=True)


def test_pipeline_453_and_category_unique_ids(tmp_path):
    root = create_project(tmp_path)
    create_sample(root, 'touching_face', 'S11')
    create_sample(root, 'scratching_head', 'S12')
    r = invoke(root, '--val-subjects', 'S11', '--test-subjects', 'S12')
    assert r.returncode == 0, r.stdout + r.stderr
    for c, s in [('touching_face','S11'), ('scratching_head','S12')]:
        sid=f'UNKNOWN_{c}_{s}_SES01_001'
        dirpath = root / 'dataset' / 'processed' / 'features_phase6_unknown' / c / s / 'SES01' / sid
        with np.load(dirpath / 'features_v3.npz', allow_pickle=False) as f:
            assert f['ablation_hands_pose_face453'].shape == (12,453)
            assert np.isfinite(f['ablation_hands_pose_face453']).all()
        with np.load(root / 'dataset' / 'processed' / 'features_phase5_unknown' / c / s / 'SES01' / sid / 'features_v2.npz') as f:
            assert f['full_features417'].shape == (12,417)
    import csv
    with (root / 'dataset' / 'unknown_phase8' / 'unknown_manifest.csv').open(encoding='utf-8-sig',newline='') as file:
        rows = list(csv.DictReader(file))
    assert {r['sample_id'] for r in rows} == {
        'UNKNOWN_touching_face_S11_SES01_001',
        'UNKNOWN_scratching_head_S12_SES01_001',
    }
    assert {r['split'] for r in rows} == {'val', 'test'}
    assert (root / 'dataset' / 'raw' / 'UNKNOWN' / 'touching_face' / 'S11' / 'SES01' / 'UNKNOWN_S11_SES01_001' / 'metadata.json').is_file()


def test_one_signer_does_not_fabricate_split(tmp_path):
    root = create_project(tmp_path)
    create_sample(root, 'random_hand', 'S01')
    r = invoke(root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (root / 'dataset' / 'unknown_phase8' / 'unknown_manifest.csv').exists()
    assert (root / 'dataset' / 'processed' / 'features_phase6_unknown' / 'unknown_inventory.csv').exists()


def test_split_overlap_is_rejected(tmp_path):
    root = create_project(tmp_path)
    create_sample(root, 'waving', 'S01')
    r = invoke(root, '--val-subjects', 'S01', '--test-subjects', 'S01')
    assert r.returncode != 0
    assert 'overlap' in (r.stdout + r.stderr).lower()


def test_bad_raw_sample_cannot_create_manifest(tmp_path):
    root = create_project(tmp_path)
    p1=create_sample(root, 'waving', 'S11')
    create_sample(root, 'pointing', 'S12')
    (p1 / 'landmarks.npz').write_bytes(b'not_an_npz')
    r=invoke(root, '--val-subjects', 'S11', '--test-subjects', 'S12')
    assert r.returncode != 0
    assert not (root / 'dataset' / 'unknown_phase8' / 'unknown_manifest.csv').exists()


def test_existing_manifest_is_not_silently_overwritten(tmp_path):
    root=create_project(tmp_path)
    create_sample(root, 'waving', 'S11')
    create_sample(root, 'pointing', 'S12')
    m=root / 'dataset' / 'unknown_phase8' / 'unknown_manifest.csv'
    m.parent.mkdir(parents=True)
    m.write_text('DO NOT TOUCH\n',encoding='utf-8')
    r=invoke(root,'--val-subjects','S11','--test-subjects','S12')
    assert r.returncode != 0
    assert m.read_text(encoding='utf-8') == 'DO NOT TOUCH\n'


def test_no_hand_unknown_is_valid_structurally(tmp_path):
    root = create_project(tmp_path)
    p = create_sample(root, 'no_hand', 'S11')
    original = np.load(p / 'landmarks.npz', allow_pickle=False)
    arrays = {key: original[key] for key in original.files}
    original.close()
    arrays['left_present'][:] = False
    arrays['right_present'][:] = False
    arrays['left_hand'][:] = 0.0
    arrays['right_hand'][:] = 0.0
    np.savez_compressed(p / 'landmarks.npz', **arrays)
    r = invoke(root)
    assert r.returncode == 0, r.stdout + r.stderr
    sid = 'UNKNOWN_no_hand_S11_SES01_001'
    target = root / 'dataset' / 'processed' / 'features_phase6_unknown' / 'no_hand' / 'S11' / 'SES01' / sid / 'features_v3.npz'
    with np.load(target, allow_pickle=False) as d:
        assert d['ablation_hands_pose_face453'].shape == (12, 453)
        assert np.all(d['hands_full417'][:, :126] == 0)


def test_legacy_categoryless_is_reported_and_skipped(tmp_path):
    root = create_project(tmp_path)
    create_sample(root, 'touching_face', 'S11')
    create_sample(root, 'pointing', 'S12')
    legacy = root / 'dataset' / 'raw' / 'UNKNOWN' / 'S01' / 'SES01' / 'UNKNOWN_S01_SES01_001'
    legacy.mkdir(parents=True)
    (legacy / 'landmarks.npz').write_bytes(b'legacy ignored')
    r = invoke(root, '--val-subjects', 'S11', '--test-subjects', 'S12')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'SKIP_LEGACY=1' in r.stdout
    assert (root / 'dataset' / 'unknown_phase8' / 'unknown_manifest.csv').exists()


def test_quick_mode_only_three_categories_single_signer(tmp_path):
    root=create_project(tmp_path)
    create_sample(root,'holding_phone','S01')
    create_sample(root,'scratching_head','S01')
    create_sample(root,'touching_face','S01')
    p=invoke(root,'--quick-test')
    assert p.returncode == 0, p.stdout + p.stderr
    import csv
    with (root/'dataset/unknown_phase8/unknown_manifest.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==3
    assert {r['category'] for r in rows}=={'holding_phone','scratching_head','touching_face'}
    assert {r['sample_id'] for r in rows}=={
      'UNKNOWN_holding_phone_S01_SES01_001',
      'UNKNOWN_scratching_head_S01_SES01_001',
      'UNKNOWN_touching_face_S01_SES01_001',
    }
    assert {r['split'] for r in rows}=={'val'}
    marker=json.loads((root/'dataset/unknown_phase8/unknown_manifest_mode.json').read_text(encoding='utf-8'))
    assert marker['mode']=='exploratory_val_only'
    assert marker['val_samples']==3 and marker['test_samples']==0
    assert 'QUICK TEST ONLY' in p.stdout


def test_quick_mode_rejects_fake_signer_independent_split(tmp_path):
    root=create_project(tmp_path)
    create_sample(root, 'touching_face','S01')
    r=invoke(root,'--quick-test','--val-subjects','S01')
    assert r.returncode != 0
    assert 'cannot be combined' in (r.stdout+r.stderr)


def test_official_mode_marker_does_not_claim_exploratory(tmp_path):
    root=create_project(tmp_path)
    create_sample(root,'holding_phone','S01')
    create_sample(root,'touching_face','S02')
    r=invoke(root,'--val-subjects','S01','--test-subjects','S02')
    assert r.returncode == 0,r.stdout+r.stderr
    mode=json.loads((root/'dataset/unknown_phase8/unknown_manifest_mode.json').read_text(encoding='utf-8'))
    assert mode['mode']=='signer_independent_val_test'
    assert mode['val_samples']==1 and mode['test_samples']==1


def test_quick_mode_keeps_existing_manifest_untouched(tmp_path):
    root=create_project(tmp_path)
    create_sample(root,'holding_phone','S01')
    m=root/'dataset/unknown_phase8/unknown_manifest.csv'
    m.parent.mkdir(parents=True,exist_ok=True)
    m.write_text('existing!')
    r=invoke(root,'--quick-test')
    assert r.returncode!=0
    assert m.read_text()=='existing!'
