import sys,json
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_v3_spatial import signer_mapping

def test_signer_mapping_no_leak(tmp_path):
    p=tmp_path/'summary.json'
    p.write_text(json.dumps({'subjects':{'train':['S01','S02'],'val':['S03'],'test':['S04']}}))
    assert signer_mapping(p)['S01']=='train'
    assert signer_mapping(p,{'test':['S05']})['S05']=='test'
    with pytest.raises(ValueError):signer_mapping(p,{'test':['S01']})

from scripts.analyze_v3_log import analyze

def test_analyze_keeps_heuristic_separate():
    rows=[{'source':'TCN_V3_SPATIAL','label':'UNKNOWN','expected_label':'UNKNOWN',
           'test_condition':'wrong_position','trial_id':'1','emitted_token':''},
          {'source':'NO_HAND_RELEASE_HEURISTIC','label':'REST','expected_label':'',
           'test_condition':'REST','trial_id':'2','emitted_token':''}]
    x=analyze(rows)
    assert x['tc_model_windows']==1
    assert x['by_condition']['wrong_position']['agreement_rate']==1
    assert x['release_event_sources']['NO_HAND_RELEASE_HEURISTIC']==1

from scripts.evaluate_v3_spatial import metrics

def test_metrics_unknown_false_accept_rate():
    s=metrics(['REST','cam_on','xin_chao','xin_loi','UNKNOWN'],
              __import__('numpy').array([0,2,4,4]),__import__('numpy').array([0,2,2,4]))
    assert s['unknown_n']==2
    assert s['unknown_false_accept_rate']==.5
    assert s['known_accuracy']==1.0
