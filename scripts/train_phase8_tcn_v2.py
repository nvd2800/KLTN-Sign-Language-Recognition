"""Train TCN V2 (5-way including UNKNOWN) from scratch, save best/last checkpoints."""
from __future__ import annotations
import argparse, json, random, shutil, sys
from pathlib import Path
import numpy as np, yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.tcn_v2_unknown import read_phase7_split, compute_class_weights
from core.models_phase7 import build_tcn_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/phase8_tcn_v2.yaml')
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--overwrite',action='store_true',help='Explicitly replace existing V2 model ONLY')
    a=p.parse_args()
    cp=Path(a.config);cp=cp if cp.is_absolute() else ROOT/cp
    cfg=yaml.safe_load(cp.read_text(encoding='utf-8'))
    data=ROOT/cfg['paths']['v2_data_dir'];out=ROOT/cfg['paths']['experiment_dir']
    info=json.loads((data/'dataset_summary.json').read_text(encoding='utf-8'))
    lbl=json.loads((data/'labels.json').read_text(encoding='utf-8'))['labels']
    if len(lbl)!=5 or 'UNKNOWN' not in lbl:raise ValueError('Five-class labels required')
    if (out/'model.keras').exists() and not a.overwrite:
        raise FileExistsError(f'Model already exists: {out}/model.keras; use --overwrite ONLY for intentional re-training')
    train=read_phase7_split(data/'train.npz');val=read_phase7_split(data/'val.npz')
    if not len(val['y']):raise ValueError('Validation set empty')
    weights=compute_class_weights(train['y'],len(lbl),float(cfg['training'].get('class_weight_cap',10)))
    seed=int(cfg['training'].get('seed',42));random.seed(seed);np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError as e:
        raise RuntimeError('TensorFlow missing. pip install -r requirements-phase7.txt') from e
    tf.random.set_seed(seed)
    m=cfg['model'];tr=cfg['training']
    model=build_tcn_model(32,453,len(lbl),
        filters=int(m['filters']),kernel_size=int(m['kernel_size']),
        dilations=tuple(m['dilations']),dense_units=int(m['dense_units']),
        dropout=float(m['dropout']),learning_rate=float(tr['learning_rate']))
    out.mkdir(parents=True,exist_ok=True)
    cb=[tf.keras.callbacks.ModelCheckpoint(str(out/'model.keras'),monitor='val_loss',mode='min',save_best_only=True,verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=int(tr['early_stopping_patience']),mode='min',restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=int(tr['reduce_lr_patience']),factor=0.5,min_lr=1e-6)]
    print('Training from scratch on combined known + UNKNOWN TRAIN')
    print('Training labels:',dict(zip(lbl,np.bincount(train['y'],minlength=len(lbl)).tolist())))
    print('Class weights:',weights)
    if info.get('missing_unknown_val'):
        print('WARNING: No UNKNOWN in VAL. Val loss cannot select for UNKNOWN recognition; DEV only.')
    hist=model.fit(train['X'],train['y'],validation_data=(val['X'],val['y']),
        epochs=int(a.epochs or tr['epochs']),batch_size=int(tr['batch_size']),
        shuffle=True,class_weight=weights,callbacks=cb,verbose=2)
    model.save(str(out/'last_model.keras'))
    shutil.copy2(data/'labels.json',out/'labels.json')
    shutil.copy2(data/'preprocess_stats.npz',out/'preprocess_stats.npz')
    shutil.copy2(data/'dataset_summary.json',out/'dataset_summary.json')
    (out/'training_config.json').write_text(json.dumps({'phase8_v2':cfg,'class_weights':weights,'mode':info['mode']},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'history.json').write_text(json.dumps({k:[float(v) for v in xs] for k,xs in hist.history.items()},indent=2),encoding='utf-8')
    print('\nTRAINED TCN V2:')
    print('  Best:',out/'model.keras')
    print('  Last:',out/'last_model.keras')
    print('  Scaler/Labels:',out/'preprocess_stats.npz',out/'labels.json')
    print('  WARNING: No final UNKNOWN generalization claim without independent UNKNOWN test data.')
if __name__=='__main__':main()
