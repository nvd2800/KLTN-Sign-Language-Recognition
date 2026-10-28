"""Summarize Phase09 prediction/token logs. No training and no 'accuracy' claim without GT."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def rp(x):
    p=Path(x)
    return p if p.is_absolute() else ROOT/p


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',default='logs/phase09_live_predictions.csv')
    p.add_argument('--tokens',default='logs/phase09_live_tokens.jsonl')
    p.add_argument('--out',default='logs/phase09_live_summary.json')
    a=p.parse_args()
    f=rp(a.predictions);t=rp(a.tokens)
    predictions=[];tokens=[]
    if f.is_file():
        with f.open('r',encoding='utf-8-sig',newline='') as handle:
            predictions=list(csv.DictReader(handle))
    if t.is_file():
        with t.open('r',encoding='utf-8') as handle:
            tokens=[json.loads(line) for line in handle if line.strip()]
    if not predictions and not tokens:
        raise SystemExit('No Phase09 realtime data found. Run webcam first.')
    report={
        'prediction_count':len(predictions),
        'token_count':len(tokens),
        'prediction_labels':dict(Counter(r.get('label','') for r in predictions)),
        'prediction_sources':dict(Counter(r.get('source','') for r in predictions)),
        'emitted_tokens':dict(Counter(r.get('label','') for r in tokens)),
        'token_sequence':[r.get('label','') for r in tokens],
        'manual_rest_count':sum(r.get('source')=='MANUAL_REST' for r in predictions),
        'auto_release_count':sum(r.get('source')=='NO_HAND_RELEASE' for r in predictions),
        'note':'Diagnostic counts only. No ground-truth => precision/recall not computed.',
    }
    out=rp(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Saved:',out)

if __name__=='__main__':main()
