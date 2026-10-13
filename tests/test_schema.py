from pathlib import Path
import numpy as np
from core.types import FrameLandmarks, PresenceMask
from core.schema import SampleMetadata, save_sequence, load_sequence


def test_save_load_roundtrip(tmp_path: Path):
    fs=[FrameLandmarks(i,i/30,np.ones((21,3),np.float32),np.zeros((21,3),np.float32),presence=PresenceMask(left_hand=True)) for i in range(4)]
    m=SampleMetadata("v2.0","1.0","xin_chao_S01_SES01_001","xin_chao","S01","SES01",0,30.0,4,[1280,720])
    d=tmp_path/"sample"
    save_sequence(d,fs,m)
    a,meta=load_sequence(d)
    assert a["left_hand"].shape==(4,21,3)
    assert a["right_hand"].shape==(4,21,3)
    assert a["left_present"].tolist()==[1,1,1,1]
    assert meta["sample_id"]==m.sample_id
