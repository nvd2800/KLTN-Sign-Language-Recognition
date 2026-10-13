from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
from pathlib import Path
import numpy as np
from core.types import FrameLandmarks, PresenceMask
from core.schema import SampleMetadata, save_sequence, load_sequence


def main():
    frames=[]
    for i in range(5):
        frames.append(FrameLandmarks(
            frame_index=i,
            timestamp=i/30,
            left_hand=np.ones((21,3), dtype=np.float32)*i,
            right_hand=np.zeros((21,3), dtype=np.float32),
            presence=PresenceMask(left_hand=True),
        ))
    meta=SampleMetadata(
        dataset_version="v2.0", schema_version="1.0", sample_id="xin_chao_S01_SES01_001",
        sign="xin_chao", subject_id="S01", session_id="SES01", camera_id=0,
        fps=30.0, frame_count=5, resolution=[1280,720], left_hand_ratio=1.0,
        target_fps=30.0, duration_s=4/30, collection_mode="fixed",
    )
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"sample_001"
        save_sequence(p, frames, meta)
        arr, m=load_sequence(p)
        assert arr["left_hand"].shape == (5,21,3)
        assert arr["right_hand"].shape == (5,21,3)
        assert arr["left_present"].tolist() == [1]*5
        assert m["subject_id"] == "S01"
    print("SCHEMA PASS: save/load preserves Phase-0 dataset contract")


if __name__ == "__main__":
    main()
