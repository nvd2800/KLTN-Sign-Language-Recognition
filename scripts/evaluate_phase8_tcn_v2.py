"""Evaluate saved BEST TCN V2 checkpoint on known and, if present, held-out UNKNOWN TEST."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.tcn_v2_unknown import read_phase7_split,summarize_multiclass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/phase8_tcn_v2.yaml');a=ap.parse_args()
    c=Path(a.config);c=c if c.is_absolute() else ROOT/c
    cfg=yaml.safe_load(c.read_text(encoding='utf-8'))
    data=ROOT/cfg['paths']['v2_data_dir'];out=ROOT/cfg['paths']['experiment_dir']
    labels=json.loads((out/'labels.json').read_text(encoding='utf-8'))['labels']
    if len(labels)!=5:raise ValueError('Not a 5-class checkpoint')
    meta=json.loads((out/'dataset_summary.json').read_text(encoding='utf-8'))
    import tensorflow as tf
    model=tf.keras.models.load_model(out/'model.keras',compile=False)
    if tuple(model.input_shape[1:])!=(32,453) or model.output_shape[-1]!=len(labels):
        raise ValueError('Model, feature shape and labels mismatch')
    ds=read_phase7_split(data/'test.npz')
    p=model.predict(ds['X'],verbose=0)
    pred=p.argmax(axis=1).astype(np.int64)
    report=summarize_multiclass(ds['y'],pred,labels)
    unknown_idx=labels.index('UNKNOWN')
    um=ds['y']==unknown_idx;km=~um
    if um.any():
        report['unknown_test']={'n':int(um.sum()),
            'recall':float(np.mean(pred[um]==unknown_idx)),
            'false_accept_rate':float(np.mean(pred[um]!=unknown_idx))}
    else:
        report['unknown_test']={'n':0,'recall':None,'false_accept_rate':None,'status':'NOT_EVALUATED_NO_UNKNOWN_TEST'}
    report['known_test']={'n':int(km.sum()),
        'accuracy':float(np.mean(pred[km]==ds['y'][km])) if km.any() else None,
        'false_reject_rate':float(np.mean(pred[km]==unknown_idx)) if km.any() else None}
    report['model_path']=str(out/'model.keras')
    report['protocol_mode']=meta['mode']; report['limitations']=meta['warning']
    if not um.any():report['evaluation_status']='KNOWN_ONLY_DEV_INCOMPLETE'
    else:report['evaluation_status']='EXPLORATORY' if meta['mode']=='dev' else 'REVIEW_TEST_INDEPENDENCE'
    out.mkdir(parents=True,exist_ok=True)
    (out/'test_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'confusion_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['true\\pred']+labels)
        for name,row in zip(labels,report['confusion_matrix']):w.writerow([name]+row)
    with (out/'test_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        columns=['sample_id','true','pred','confidence','correct']+[f'p_{x}' for x in labels]
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for i,(real,guess) in enumerate(zip(ds['y'],pred)):
            row={'sample_id':ds['ids'][i],'true':labels[int(real)],'pred':labels[int(guess)],
                 'confidence':float(p[i,int(guess)]),'correct':int(real==guess)}
            row.update({f'p_{label}':float(prob) for label,prob in zip(labels,p[i])});w.writerow(row)
    print('\nPHASE08 TCN V2 TEST REPORT')
    print('Mode:           ',meta['mode'])
    print('Known n:        ',report['known_test']['n'],'accuracy:',report['known_test']['accuracy'])
    print('Unknown n:      ',report['unknown_test']['n'],'recall:',report['unknown_test']['recall'])
    print('Unknown FAR:    ',report['unknown_test']['false_accept_rate'])
    print('Known FRR:      ',report['known_test']['false_reject_rate'])
    print('Confusion matrix:',out/'confusion_matrix.csv')
    print('WARN:',report['limitations'])
if __name__=='__main__':main()
