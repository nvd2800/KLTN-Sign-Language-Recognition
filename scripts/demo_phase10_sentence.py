"""Offline text-only demo; does not pretend new pending labels were classified by AI."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.phase10_sentence import SentenceBuilder


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/phase10.yaml')
    p.add_argument('--tokens',nargs='+',default=['xin_chao','dau_phay','cam_on','dau_cam'])
    p.add_argument('--demo-pending',action='store_true',help='Text only: simulate pending vocabulary; NOT camera recognition')
    a=p.parse_args(argv)
    config=Path(a.config)
    if not config.is_absolute():config=ROOT/config
    cfg=yaml.safe_load(config.read_text(encoding='utf-8'))
    phrases=dict(cfg['phrases'])
    if a.demo_pending:phrases.update(cfg.get('vocabulary_pending',{}))
    b=SentenceBuilder(phrases,punctuation_labels=cfg.get('punctuation_labels'))
    missing=[]
    for token in a.tokens:
        if not b.add(token):missing.append(token)
    print('PREVIEW:',b.preview)
    sent=b.finalize()
    print('FINAL:',sent.text if sent else '(empty)')
    if missing:print('NOT_RECOGNIZED_BY_VOCAB:',','.join(missing))
    if a.demo_pending:print('WARNING: DEMO_PENDING is synthetic text-only, no new TCN class was trained')
    return 0 if not missing else 3

if __name__=='__main__':sys.exit(main())
