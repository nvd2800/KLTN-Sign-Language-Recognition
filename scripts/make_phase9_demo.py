"""Generate a synthetic decision stream; not a trained-model accuracy test."""
import csv
from pathlib import Path
out=Path('logs/phase9_demo_input.csv')
out.parent.mkdir(parents=True,exist_ok=True)
pattern=( [('REST',0.99)]*5 + [('xin_chao',.94)]*8 + [('UNKNOWN',.50)]*3 +
          [('REST',.99)]*4 + [('xin_chao',.95)]*6 + [('REST',.99)]*4 +
          [('cam_on',.96)]*6 )
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['timestamp_s','label','confidence'])
    for i,(label,confidence) in enumerate(pattern):
        w.writerow([round(i*.1,2),label,confidence])
print(out)
