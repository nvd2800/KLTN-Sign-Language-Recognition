"""Phase09 V3 per-trial analysis, separate predictions and confirmed tokens.

Window predictions overlap. Ground truth here is keyboard-marked, not gold
video segmentation. Distinguishes ML REST vs heuristic/manual release.
"""
from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path

PRED_SOURCES={'TCN_V3_SPATIAL','TCN_V3_SPATIAL_GATE'}
RELEASE_SOURCES={'NO_HAND_RELEASE_HEURISTIC','MANUAL_REST'}

def score_trials(rows):
    trials={}
    for r in rows:
        ident=str(r.get('trial_id','')).strip()
        if not ident or ident=='0':continue
        t=trials.setdefault(ident,{
            'trial_id':ident,'condition':r.get('test_condition',''),
            'expected_label':r.get('expected_label',''),
            'windows':0,'prediction_counts':Counter(),
            'confirmed_tokens':[],'release_sources':Counter(),
            'marker_start':None,'marker_end':None,'fsm_states':Counter(),
        })
        source=r.get('source','')
        if source=='TRIAL_START':t['marker_start']=r.get('time_s')
        if source=='TRIAL_END':t['marker_end']=r.get('time_s')
        if source in PRED_SOURCES:
            t['windows']+=1;t['prediction_counts'][r.get('label','')]+=1
            t['fsm_states'][r.get('fsm','')]+=1
        if source in RELEASE_SOURCES:t['release_sources'][source]+=1
        if r.get('emitted_token') and source!='MANUAL_REST':
            t['confirmed_tokens'].append(r['emitted_token'])
    for t in trials.values():
        expected=t['expected_label']; tokens=t['confirmed_tokens']
        if expected=='UNKNOWN' or expected=='REST':
            t['token_pass']=len(tokens)==0
            t['error_type']='FALSE_SIGN_TOKEN' if tokens else 'NONE'
        elif expected:
            t['token_pass']=expected in tokens and len(tokens)==1
            t['error_type']='MISSED_OR_EXTRA_TOKEN' if not t['token_pass'] else 'NONE'
        else:t['token_pass']=None;t['error_type']='UNLABELED'
        t['prediction_counts']=dict(t['prediction_counts'])
        t['fsm_states']=dict(t['fsm_states'])
        t['release_sources']=dict(t['release_sources'])
        t['incomplete']=t['marker_start'] is None or t['marker_end'] is None
    valid=[t for t in trials.values() if t['token_pass'] is not None and not t['incomplete']]
    negative=[t for t in valid if t['expected_label']=='UNKNOWN']
    correct=[t for t in valid if t['condition']=='correct']
    return {
        'trials':list(trials.values()),'n_marked_trials':len(trials),
        'n_complete_labeled_trials':len(valid),
        'n_complete_negative_trials':len(negative),
        'false_token_negative_trials':sum(not t['token_pass'] for t in negative),
        'n_complete_correct_trials':len(correct),
        'correct_token_trials':sum(bool(t['token_pass']) for t in correct),
        'trial_false_token_rate':(sum(not t['token_pass'] for t in negative)/len(negative)) if negative else None,
        'trial_correct_token_rate':(sum(bool(t['token_pass']) for t in correct)/len(correct)) if correct else None,
        'release_source_counts':dict(Counter(r.get('source','') for r in rows if r.get('source','') in RELEASE_SOURCES)),
        'model_windows':sum(r.get('source','') in PRED_SOURCES for r in rows),
        'protocol_notes':['Use --trial-isolation for independently marked trials, without changing continuous-mode behavior.',
                          'For continuous sign recognition, annotate start/end and intended token sequence on video separately.',
                          'Only completed trials with both TRIAL_START and TRIAL_END count toward trial-level rates.',
                          'Manual REST is excluded as a token; heuristic REST is not model prediction.',
                          'These results do not certify held-out signer accuracy.'],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='logs/phase09_v3_spatial_predictions.csv')
    p.add_argument('--out',default='logs/phase09_v3_trial_report.json')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    def rp(x):
        x=Path(x);return x if x.is_absolute() else root/x
    with rp(a.input).open('r',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    result=score_trials(rows)
    dest=rp(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},ensure_ascii=False,indent=2))
    print('Saved:',dest)

if __name__=='__main__':main()
