from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import yaml
cfg=yaml.safe_load((ROOT/'configs/phase8_tcn_v2.yaml').read_text(encoding='utf-8'))
out=ROOT/cfg['paths']['experiment_dir'];data=ROOT/cfg['paths']['v2_data_dir']
required=['model.keras','last_model.keras','labels.json','preprocess_stats.npz','dataset_summary.json',
          'history.json','training_config.json','test_metrics.json','confusion_matrix.csv','test_predictions.csv']
missing=[f for f in required if not (out/f).is_file()]
print('TCN V2 ARTIFACT AUDIT')
print('Missing:',missing)
if not missing:
    v=json.loads((out/'test_metrics.json').read_text(encoding='utf-8'))
    print('KNOWN accuracy:',v['known_test']['accuracy'])
    print('UNKNOWN TEST samples:',v['unknown_test']['n'])
    print('STATUS:', 'EXPERIMENTAL (NO INDEPENDENT UNKNOWN TEST)' if v['unknown_test']['n']==0 else 'Artifacts complete; inspect metrics and test independence')
    print('ARTIFACTS: PASS')
else:
    print('ARTIFACTS: FAIL')
    raise SystemExit(1)
