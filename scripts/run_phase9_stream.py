"""Use Phase08 model decisions in CSV/JSONL to validate the Phase09 state machine.
CSV columns: timestamp_s,label,confidence; JSONL same keys.
Live integration: invoke ContinuousRecognizer.update() for every temporal
inference window after Phase08 rejection, not per video frame.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.continuous_phase9 import ContinuousRecognizer, Phase09Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV/JSONL Phase08 decisions')
    parser.add_argument('--config', default='configs/phase9.yaml')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    base = Path.cwd()
    cfg = yaml.safe_load((base / args.config).read_text(encoding='utf-8'))
    rec_cfg = cfg['recognition']
    calibrated = base / cfg['model']['phase8_threshold_json']
    if calibrated.is_file():
        threshold = float(json.loads(calibrated.read_text(encoding='utf-8'))['threshold'])
    else:
        threshold = float(rec_cfg['fallback_confidence_threshold'])
        print('WARNING: Phase08 calibration not present; using provisional threshold.')
    engine = ContinuousRecognizer(Phase09Config(
        confidence_threshold=threshold,
        min_consecutive=int(rec_cfg['min_consecutive']),
        release_consecutive=int(rec_cfg['release_consecutive']),
        min_token_interval_s=float(rec_cfg['min_token_interval_s']),
        max_gap_s=float(rec_cfg['max_gap_s']),
        rest_label=str(rec_cfg['rest_label']),
        unknown_label=str(rec_cfg['unknown_label']),
        unknown_counts_as_release=bool(rec_cfg['unknown_counts_as_release']),
    ))
    path = base / args.input
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
    elif path.suffix.lower() == '.jsonl':
        rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    else:
        raise ValueError('Input must be .csv or .jsonl')
    if not rows:
        raise ValueError('No inference decisions found')
    out = base / (args.out or cfg['output']['event_log'])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for r in rows:
            event = engine.update(str(r['label']), float(r['confidence']), float(r['timestamp_s']))
            if event:
                payload = {'event': 'TOKEN', 'label': event.label,
                           'timestamp_s': event.timestamp_s, 'confidence': event.confidence}
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
                print('TOKEN:', event.label, '@', event.timestamp_s)
    print('STATE:', engine.snapshot())
    print('EVENT LOG:', out)


if __name__ == '__main__':
    main()
