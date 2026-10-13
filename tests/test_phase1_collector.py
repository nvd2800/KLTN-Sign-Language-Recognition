from pathlib import Path
import json
import numpy as np

from core.collector import CollectionIdentity, CollectionSettings, build_metadata, atomic_save_sequence, verify_saved_sample
from core.types import FrameLandmarks, PresenceMask


def _frames(n=45):
    return [
        FrameLandmarks(
            frame_index=i,
            timestamp=i/30.0,
            left_hand=np.ones((21,3), dtype=np.float32),
            right_hand=np.zeros((21,3), dtype=np.float32),
            presence=PresenceMask(left_hand=True, pose=False, face=False),
        )
        for i in range(n)
    ]


def test_fixed_collection_roundtrip(tmp_path: Path):
    frames = _frames()
    identity = CollectionIdentity('xin_chao','S01','SES01')
    settings = CollectionSettings('v2.0-mini','1.1',0,(1280,720),30.0,'fixed')
    meta = build_metadata(frames=frames, sample_id='xin_chao_S01_SES01_001', identity=identity, settings=settings, actual_fps=30.0)
    out = tmp_path/'xin_chao_S01_SES01_001'
    atomic_save_sequence(out, frames, meta)
    report = verify_saved_sample(out)
    assert report['frame_count'] == 45
    data = json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    assert data['collection_mode'] == 'fixed'
    assert data['target_fps'] == 30.0


def test_video_import_mode_is_valid():
    CollectionSettings('v2.0-mini','1.1',-1,(1920,1080),30.0,'video_import').validate()
