from pathlib import Path
import json
import sys
import importlib.util
import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SRC))
from scripts.check_phase09_v3_1 import preflight
from scripts.live_phase09_v3_1 import launch


def make_project(tmp_path, *, counts=(140,31,69), baseline=False, std_dim=513):
    root=tmp_path
    modeldir=root/'artifacts'/'experiments'/('tcn_v3_spatial' if baseline else 'tcn_v3_1_spatial')
    modeldir.mkdir(parents=True)
    (modeldir/'model.keras').write_bytes(b'mock model for preflight only')
    (modeldir/'labels.json').write_text(json.dumps({'labels':['REST','cam_on','xin_chao','xin_loi','UNKNOWN']}))
    np.savez(modeldir/'preprocess_stats.npz',mean=np.zeros(std_dim),std=np.ones(std_dim),target_frames=32,feature_dim=std_dim)
    cnt={s:{'total':n} for s,n in zip(['train','val','test'],counts)}
    (modeldir/'dataset_summary.json').write_text(json.dumps({'mode':'strict','counts':cnt}))
    dataset=root/'dataset'/'processed'/'temporal_v3_1_spatial'
    dataset.mkdir(parents=True)
    (dataset/'dataset_summary.json').write_text(json.dumps({'mode':'strict','counts':{'train':{'total':140},'val':{'total':31},'test':{'total':69}}}))
    cfg=root/'configs'/'phase9_v3_1_spatial.yaml'
    cfg.parent.mkdir(parents=True)
    sub=modeldir.relative_to(root).as_posix()
    cfg.write_text('artifacts:\n' + f'  model: {sub}/model.keras\n  stats: {sub}/preprocess_stats.npz\n  labels: {sub}/labels.json\n' + 'logging:\n  predictions: logs/phase09_v3_1_spatial_predictions.csv\n')
    return root, cfg, dataset


def test_preflight_good(tmp_path):
    root,cfg,data=make_project(tmp_path)
    err,warn,info=preflight(root,cfg,dataset=data)
    assert err==[] and info['input_shape']==[32,513]


def test_preflight_reject_old_checkpoint(tmp_path):
    root,cfg,data=make_project(tmp_path,baseline=True)
    err,_,_=preflight(root,cfg,dataset=data)
    assert any('BASELINE' in msg for msg in err)


def test_preflight_reject_old_training_counts(tmp_path):
    root,cfg,data=make_project(tmp_path,counts=(126,26,49))
    err,_,_=preflight(root,cfg,dataset=data)
    assert any('counts do NOT match' in msg for msg in err)


def test_preflight_reject_bad_stats(tmp_path):
    root,cfg,data=make_project(tmp_path,std_dim=453)
    err,_,_=preflight(root,cfg,dataset=data)
    assert any('513' in msg for msg in err)


def test_launcher_dry_run(tmp_path):
    root,cfg,data=make_project(tmp_path)
    assert launch(['--config',str(cfg),'--dataset',str(data),'--dry-run'],root=root)==0


def test_launcher_bad_dry_run(tmp_path):
    root,cfg,data=make_project(tmp_path,counts=(126,26,49))
    assert launch(['--config',str(cfg),'--dataset',str(data),'--dry-run'],root=root)==2


def test_live_script_has_trial_flag():
    script=(SRC/'scripts'/'live_phase09_v3_spatial.py').read_text()
    assert "'--trial-isolation'" in script
    assert "feature_mode=\"spatial513\"" in script
    assert "spatial_last" in script
