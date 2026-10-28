"""Integration test for evaluator; mocks Keras model, not actual model performance."""
import json,sys,types
import numpy as np
from scripts import evaluate_v3_spatial as ev


def test_evaluator_writes_per_sample_and_3_error_ids(tmp_path,monkeypatch):
    data=tmp_path/'dataset/processed/temporal_v3_spatial';data.mkdir(parents=True)
    model=tmp_path/'artifacts/experiments/tcn_v3_spatial';model.mkdir(parents=True)
    labels=['REST','cam_on','xin_chao','xin_loi','UNKNOWN']
    (model/'labels.json').write_text(json.dumps({'labels':labels}))
    (model/'model.keras').write_bytes(b'MOCK_DO_NOT_TRAIN')
    meta=[]
    for i in range(3):meta.append(dict(id=f'DIAG_TEST_{i}',split='test',subject='S09',
       condition=['wrong_position','wrong_handshape','partial'][i],reference_sign='xin_chao',raw=str(tmp_path/'nonexistent'/str(i))))
    (data/'dataset_summary.json').write_text(json.dumps({'mode':'strict','hard_negatives':meta}))
    np.savez_compressed(data/'test.npz',X=np.zeros((4,32,513),np.float32),y=np.array([4,4,4,2]),
                        sample_ids=['DIAG_TEST_0','DIAG_TEST_1','DIAG_TEST_2','known_x'])
    class MockModel:
        input_shape=(None,32,513)
        def predict(self, X, batch_size=32, verbose=0):
            return np.asarray([[.91,.02,.02,.02,.03],[.01,.94,.02,.02,.01],
                               [.01,.01,.01,.96,.01],[.01,.01,.95,.01,.02]],np.float32)
    fake=types.SimpleNamespace(keras=types.SimpleNamespace(models=types.SimpleNamespace(load_model=lambda p,compile:MockModel())))
    monkeypatch.setitem(sys.modules,'tensorflow',fake)
    monkeypatch.setattr(ev,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['evaluate_v3_spatial.py','--split','test'])
    ev.main()
    result=json.loads((model/'evaluation_test.json').read_text())
    assert result['n']==4 and len(result['error_cases'])==3
    assert (model/'predictions_test.csv').is_file()
    assert {x['sample_id'] for x in result['error_cases']}=={'DIAG_TEST_0','DIAG_TEST_1','DIAG_TEST_2'}
