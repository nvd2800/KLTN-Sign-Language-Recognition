"""TCN V3 513D/5-class, new checkpoints; never touches V2."""
from __future__ import annotations
import argparse,json,random,sys,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.models_phase7 import build_tcn_model


def load(path):
    with np.load(path,allow_pickle=False) as d:
        X=np.asarray(d['X'],np.float32);y=np.asarray(d['y'],np.int64)
    if X.ndim!=3 or X.shape[1:]!=(32,513) or len(X)!=len(y) or not np.isfinite(X).all():
        raise ValueError(f'Expected finite [N,32,513] in {path}, got {X.shape}')
    return X,y


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--data',default='dataset/processed/temporal_v3_spatial')
    a.add_argument('--out',default='artifacts/experiments/tcn_v3_spatial')
    a.add_argument('--epochs',type=int,default=120)
    a.add_argument('--batch-size',type=int,default=16)
    a.add_argument('--lr',type=float,default=0.001)
    a.add_argument('--overwrite',action='store_true')
    args=a.parse_args()
    def rp(q):
        p=Path(q);return p if p.is_absolute() else ROOT/p
    data=rp(args.data);out=rp(args.out)
    cfg=json.loads((data/'dataset_summary.json').read_text(encoding='utf8'))
    labels=json.loads((data/'labels.json').read_text(encoding='utf8'))['labels']
    if len(labels)!=5 or 'UNKNOWN' not in labels or cfg['feature_dim']!=513:raise ValueError('Wrong data/labels')
    if not cfg['counts']['train']['hard_negative']:raise RuntimeError('Need REAL hard negatives in TRAIN')
    if (out/'model.keras').exists() and not args.overwrite:
        raise FileExistsError('Existing V3 checkpoint; use --overwrite only after backup')
    X,y=load(data/'train.npz');XV,yv=load(data/'val.npz')
    if not len(yv):raise RuntimeError('VAL set empty')
    if len(set(y))!=5:raise RuntimeError(f'Every class including UNKNOWN must occur in TRAIN: {set(y)}')
    print('Mode:',cfg['mode'],'  counts:',cfg['counts'])
    if cfg['mode']!='strict':print('WARNING: DEV_ONLY; class UNKNOWN may not occur in independent VAL/TEST')
    import tensorflow as tf
    random.seed(42);np.random.seed(42);tf.random.set_seed(42)
    weights=len(y)/(5*np.bincount(y,minlength=5));weights={i:float(np.clip(v,.5,10.0)) for i,v in enumerate(weights)}
    out.mkdir(parents=True,exist_ok=True)
    model=build_tcn_model(32,513,5,filters=128,kernel_size=3,dilations=(1,2,4,8),dense_units=64,dropout=.30,learning_rate=args.lr)
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(out/'model.keras'),monitor='val_loss',save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=7,factor=.5,min_lr=1e-6),
    ]
    hist=model.fit(X,y,validation_data=(XV,yv),epochs=args.epochs,batch_size=args.batch_size,
                   class_weight=weights,callbacks=callbacks,verbose=2)
    model.save(str(out/'last_model.keras'))
    for file in ('labels.json','preprocess_stats.npz','dataset_summary.json'):
        shutil.copy2(data/file,out/file)
    (out/'history.json').write_text(json.dumps({k:[float(i) for i in v] for k,v in hist.history.items()},indent=2),encoding='utf8')
    print('Saved:',out/'model.keras')

if __name__=='__main__':main()
