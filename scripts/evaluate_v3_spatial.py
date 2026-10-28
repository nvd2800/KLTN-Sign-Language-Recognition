"""Evaluate held-out V3 model with exact frozen 513D scaler and 5 labels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.v3_eval_trace import (build_provenance_index, make_prediction_rows, write_csv, condition_summary)


def metrics(labels,y,pred,confidence=None):
    n=len(y);K=len(labels)
    cm=np.zeros((K,K),dtype=np.int64)
    for gt,pr in zip(y,pred):cm[int(gt),int(pr)]+=1
    per=[]
    for k,name in enumerate(labels):
        tp=int(cm[k,k]);support=int(cm[k].sum());fp=int(cm[:,k].sum()-tp)
        precision=tp/(tp+fp) if tp+fp else None
        recall=tp/support if support else None
        f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall>0 else None
        per.append(dict(label=name,support=support,precision=precision,recall=recall,f1=f1))
    known_idx=[i for i,l in enumerate(labels) if l!='UNKNOWN']
    known_mask=np.isin(y,known_idx); unk_mask=(y==labels.index('UNKNOWN'))
    macro=[z['f1'] for z in per if z['f1'] is not None]
    return dict(n=n,accuracy=float((y==pred).mean()) if n else None,
                macro_f1_supported=float(np.mean(macro)) if macro else None,
                known_n=int(known_mask.sum()),
                known_accuracy=float((y[known_mask]==pred[known_mask]).mean()) if known_mask.any() else None,
                unknown_n=int(unk_mask.sum()),
                unknown_recall=float((pred[unk_mask]==labels.index('UNKNOWN')).mean()) if unk_mask.any() else None,
                unknown_false_accept_rate=float((pred[unk_mask]!=labels.index('UNKNOWN')).mean()) if unk_mask.any() else None,
                confusion_matrix=cm.tolist(),per_class=per)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--split',choices=['train','val','test'],default='test')
    a.add_argument('--data',default='dataset/processed/temporal_v3_spatial')
    a.add_argument('--model',default='artifacts/experiments/tcn_v3_spatial/model.keras')
    a.add_argument('--out',default=None)
    a.add_argument('--predictions-out',default=None,help='Per-sequence trace CSV; default predictions_<split>.csv beside checkpoint')
    a.add_argument('--phase6-manifest',default='dataset/processed/features_phase6/feature_manifest.csv')
    a.add_argument('--no-qc',action='store_true',help='Skip raw-landmark QC to run faster')
    args=a.parse_args()
    def rp(x):
        p=Path(x);return p if p.is_absolute() else ROOT/p
    data=rp(args.data);mp=rp(args.model)
    labels=json.loads((mp.parent/'labels.json').read_text(encoding='utf8'))['labels']
    ds=json.loads((data/'dataset_summary.json').read_text(encoding='utf8'))
    with np.load(data/f'{args.split}.npz',allow_pickle=False) as d:
        X=np.asarray(d['X'],np.float32);y=np.asarray(d['y'],np.int64)
        if 'sample_ids' not in d.files:
            raise ValueError('Dataset lacks sample_ids: rebuild V3 dataset with provenance, do not fabricate IDs')
        sample_ids=[str(s) for s in d['sample_ids']]
    if X.ndim!=3 or X.shape[1:]!=(32,513):raise ValueError(f'Expected (N,32,513), got {X.shape}')
    if not len(y):raise RuntimeError('Split is empty')
    if not np.isfinite(X).all():raise ValueError('Nonfinite model inputs')
    dataset_labels=data/'labels.json'
    if dataset_labels.is_file() and json.loads(dataset_labels.read_text(encoding='utf-8'))['labels']!=labels:
        raise ValueError('Dataset/model label order mismatch; cannot evaluate safely')
    import tensorflow as tf
    model=tf.keras.models.load_model(mp,compile=False)
    if tuple(model.input_shape[1:])!=(32,513):raise ValueError('Model feature dimension mismatch')
    probs=np.asarray(model.predict(X,batch_size=32,verbose=0))
    if probs.shape!=(len(y),len(labels)):raise ValueError('Wrong model output')
    pred=probs.argmax(axis=1)
    results=metrics(labels,y,pred)
    meta_index,ds_for_trace=build_provenance_index(
        data,ROOT,rp(args.phase6_manifest) if rp(args.phase6_manifest).exists() else None)
    rows=make_prediction_rows(labels,y,probs,sample_ids,args.split,meta_index,ROOT,
                              with_qc=not args.no_qc)
    trace_path=rp(args.predictions_out) if args.predictions_out else mp.parent/f'predictions_{args.split}.csv'
    write_csv(trace_path,rows)
    results['trace_csv']=str(trace_path)
    results['error_cases']=[{k:r[k] for k in ('sample_id','subject','condition','true_label','predicted_label',
                 'confidence','error_type','qc_status','video_path')}
                 for r in rows if not r['prediction_correct']]
    results['by_condition']=condition_summary(rows)
    results['metadata_missing_count']=sum(r['source_type']=='UNRESOLVED' for r in rows)
    results.update(mode=ds['mode'],split=args.split,
                   warning=('DEV-only dataset / no genuine held-out UNKNOWN test'
                            if ds['mode'] not in ('strict','blind_independent') or results['unknown_n']==0 else None))
    output=rp(args.out) if args.out else mp.parent/f'evaluation_{args.split}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(results,ensure_ascii=False,indent=2));print('Saved:',output);print('Trace CSV:',trace_path)

if __name__=='__main__':main()
