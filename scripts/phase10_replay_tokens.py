"""Test Phase10 with existing Phase09 JSONL confirmed token log; no camera/TF needed.

python scripts/phase10_replay_tokens.py --input logs/phase09_v3_1_spatial_tokens.jsonl
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.phase10_sentence import SentenceBuilder
from core.phase10_store import Phase10Store


def replay(source: Path,config:dict,out:Path,*,finalize_every:int=0) -> dict:
    b=SentenceBuilder(config.get('phrases'),max_tokens=int(config.get('sentence',{}).get('max_tokens',40)),punctuation_labels=config.get('punctuation_labels'))
    s=Phase10Store(out)
    total=accepted=ignored=0
    with source.open(encoding='utf-8-sig') as f:
        for lineno,line in enumerate(f,1):
            if not line.strip():continue
            try:obj=json.loads(line)
            except json.JSONDecodeError as exc:raise ValueError(f'Invalid JSONL line {lineno}: {exc}') from exc
            if obj.get('event') not in (None,'TOKEN') or not isinstance(obj.get('label'),str):
                continue
            label=obj['label'];total+=1
            if b.add(label):
                accepted+=1;s.append('TOKEN_ACCEPTED',label=label,source_line=lineno)
                if finalize_every>0 and len(b.active)>=finalize_every:
                    sentence=b.finalize()
                    if sentence:s.sentence(sentence)
            else:
                ignored+=1;s.append('IGNORED_LABEL',label=label,source_line=lineno)
    sentence=b.finalize()
    if sentence:s.sentence(sentence)
    s.export(b)
    return {'source':str(source),'tokens_total':total,'accepted':accepted,'ignored':ignored,'sentences':len(b.sentences),'transcript':str(s.transcript_path),'session':s.session_id,'preview':b.all_text()}


def main(argv=None):
    a=argparse.ArgumentParser()
    a.add_argument('--input',required=True)
    a.add_argument('--config',default='configs/phase10.yaml')
    a.add_argument('--out',default='logs/phase10_sessions')
    a.add_argument('--finalize-every',type=int,default=0,help='Demo only; defaults to one sentence for the full log')
    x=a.parse_args(argv)
    src=Path(x.input);cfg=Path(x.config);out=Path(x.out)
    if not src.is_absolute():src=ROOT/src
    if not cfg.is_absolute():cfg=ROOT/cfg
    if not out.is_absolute():out=ROOT/out
    conf=yaml.safe_load(cfg.read_text(encoding='utf-8'))
    result=replay(src,conf,out,finalize_every=x.finalize_every)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':sys.exit(main())
