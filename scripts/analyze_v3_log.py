"""Evaluate labelled realtime windows; keep heuristic releases separate from TCN.
NOT a replacement for independent, timestamped ground-truth segment annotation.
"""
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path


def analyze(rows):
    infer=[r for r in rows if r.get('source','').startswith('TCN_V3') and r.get('expected_label')]
    bycond=defaultdict(list)
    for row in infer:bycond[row['test_condition']].append(row)
    stats={}
    for cond,items in sorted(bycond.items()):
        n=len(items);exp=items[0]['expected_label'];result=Counter(r['label'] for r in items)
        correct=sum(r['label']==r['expected_label'] for r in items)
        stats[cond]={'n_windows':n,'expected_label':exp,'agreement_rate':correct/n if n else None,
                     'output_counts':dict(result),'positive_is_reject_rate':sum(r['label']=='UNKNOWN' for r in items)/n if n else None}
    rest=Counter(r.get('source','') for r in rows if 'RELEASE' in r.get('source','') or 'MANUAL_REST' in r.get('source',''))
    tokens=[r for r in rows if r.get('emitted_token')]
    trials={}
    for r in rows:
        k=str(r.get('trial_id',''))
        if not k or k=='0':continue
        trials.setdefault(k,{'condition':r.get('test_condition',''),'expected_label':r.get('expected_label',''),
                             'predicted_tokens':[],'inference_windows':0})
        if r.get('source','').startswith('TCN_V3'):trials[k]['inference_windows']+=1
        if r.get('emitted_token') and r.get('source','')!='MANUAL_REST':
            trials[k]['predicted_tokens'].append(r['emitted_token'])
    return {'tc_model_windows':len(infer),'by_condition':stats,'release_event_sources':dict(rest),
            'emitted_tokens':len(tokens),'trials':trials,
            'caveats':['Window-level predictions are strongly overlapping; percentages are NOT independent observations.',
                       'Keyboard-marked trial boundaries are approximate; video annotation is needed for official metrics.',
                       'NO_HAND_RELEASE_HEURISTIC and MANUAL_REST are not TCN model REST predictions.']}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='logs/phase09_v3_spatial_predictions.csv')
    ap.add_argument('--out',default='logs/phase09_v3_spatial_summary.json')
    a=ap.parse_args();root=Path(__file__).resolve().parents[1]
    def rp(p):
        p=Path(p);return p if p.is_absolute() else root/p
    with rp(a.input).open('r',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    summary=analyze(rows)
    p=rp(a.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'by_condition':summary['by_condition'],'release_sources':summary['release_event_sources'],
                      'emitted_tokens':summary['emitted_tokens']},ensure_ascii=False,indent=2))
    print('Saved:',p)

if __name__=='__main__':main()
