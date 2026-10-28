"""Analyze V3.1 trials while leaving V3 log analysis untouched."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from analyze_v3_realtime_trials import score_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='logs/phase09_v3_1_spatial_predictions.csv')
    ap.add_argument('--out', default='logs/phase09_v3_1_trial_report.json')
    args=ap.parse_args()
    a,b=(Path(x) for x in (args.input,args.out))
    if not a.is_absolute():a=ROOT/a
    if not b.is_absolute():b=ROOT/b
    if not a.is_file():raise FileNotFoundError(f'V3.1 inference log not found: {a}')
    with a.open('r',encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    report=score_trials(rows)
    b.parent.mkdir(parents=True,exist_ok=True)
    b.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},ensure_ascii=False,indent=2))
    print('Saved',b)

if __name__=='__main__':main()
