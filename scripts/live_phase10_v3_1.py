"""Run Phase09 V3.1 webcam with Phase10 sentence builder enabled.

No model duplication. Phase09 launcher enforces matching checkpoint/scaler.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.live_phase09_v3_1 import launch


def main(argv=None):
    p=argparse.ArgumentParser(description='Phase10 Sentence + offline Vietnamese TTS over Phase09 TCN V3.1')
    p.add_argument('--config',default='configs/phase9_v3_Final.yaml',help='Phase09 config for V3.1 model')
    p.add_argument('--dataset',default='dataset/processed/temporal_v3_1_spatial')
    p.add_argument('--phase10-config',default='configs/phase10.yaml')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--check-model',action='store_true')
    args,extras=p.parse_known_args(argv)
    forwarded=['--config',args.config,'--dataset',args.dataset]
    if args.dry_run:forwarded.append('--dry-run')
    if args.check_model:forwarded.append('--check-model')
    if not args.dry_run:
        cfg=Path(args.phase10_config)
        if not cfg.is_absolute():cfg=ROOT/cfg
        if not cfg.is_file():
            print('ERROR: phase10 config missing:',cfg)
            return 2
        forwarded += ['--phase10-config',str(cfg)]
    forwarded += extras
    return launch(forwarded,root=ROOT)

if __name__=='__main__':sys.exit(main())
