import json
import tempfile
import unittest
from pathlib import Path
import importlib.util

P = Path(__file__).resolve().parents[1] / 'scripts' / 'check_v3_hardnegative_split.py'
spec = importlib.util.spec_from_file_location('checker', P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class Tests(unittest.TestCase):
    def test_new_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'xin_chao'/'S08'/'SES01'/'wrong_position'/'wrong_position_S08_SES01_001'
            p.mkdir(parents=True)
            for f in ('raw_landmarks.npz','features.npz'):(p/f).touch()
            (p/'metadata.json').write_text(json.dumps({'label':'xin_chao','subject':'S08','session':'SES01','condition':'wrong_position'}),encoding='utf-8')
            r=mod.scan(root,{'S08':'val'})
            self.assertEqual(r[0]['status'],'ELIGIBLE')
    def test_missing_npz(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'cam_on'/'S09'/'SES01'/'partial'/'partial_S09_SES01_001'
            p.mkdir(parents=True)
            (p/'metadata.json').write_text(json.dumps({'label':'cam_on','subject':'S09','session':'SES01','condition':'partial'}),encoding='utf-8')
            r=mod.scan(root,{'S09':'test'})
            self.assertEqual(r[0]['status'],'INVALID')
            self.assertIn('features.npz',r[0]['reason'])
    def test_positive_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'xin_chao'/'S01'/'SES01'/'correct'/'correct_S01_SES01_001'
            p.mkdir(parents=True)
            (p/'metadata.json').write_text(json.dumps({'condition':'correct','subject':'S01','label':'xin_chao','session':'SES01'}),encoding='utf-8')
            self.assertEqual(mod.scan(root,{'S01':'train'})[0]['status'],'SKIP')
    def test_legacy_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'xin_chao'/'S01'/'wrong_handshape'/'20261028T000000Z'
            p.mkdir(parents=True)
            for f in ('raw_landmarks.npz','features.npz'):(p/f).touch()
            (p/'metadata.json').write_text(json.dumps({'condition':'wrong_handshape','subject':'S01','label':'xin_chao'}),encoding='utf-8')
            self.assertEqual(mod.scan(root,{'S01':'train'})[0]['status'],'ELIGIBLE')
    def test_missing_condition_in_meta_is_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'xin_chao'/'S08'/'SES01'/'wrong_position'/'wrong_position_S08_SES01_001'
            p.mkdir(parents=True)
            for f in ('raw_landmarks.npz','features.npz'):(p/f).touch()
            (p/'metadata.json').write_text(json.dumps({'label':'xin_chao','subject':'S08','session':'SES01'}),encoding='utf-8')
            self.assertEqual(mod.scan(root,{'S08':'val'})[0]['status'],'INVALID')
    def test_bad_session(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'xin_chao'/'S08'/'SES01'/'wrong_position'/'wrong_position_S08_SES01_001'
            p.mkdir(parents=True)
            for f in ('raw_landmarks.npz','features.npz'):(p/f).touch()
            (p/'metadata.json').write_text(json.dumps({'label':'xin_chao','subject':'S08','session':'SES02','condition':'wrong_position'}),encoding='utf-8')
            self.assertEqual(mod.scan(root,{'S08':'val'})[0]['status'],'INVALID')

if __name__=='__main__': unittest.main()
