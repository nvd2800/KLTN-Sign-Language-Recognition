"""Manage active vs pending Vietnamese phrase mapping WITHOUT changing model labels.

New phrases are PENDING until a trained TCN checkpoint contains that class.
The script backs up YAML before any modification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone
import yaml

ROOT=Path(__file__).resolve().parents[1]


def read_labels(path:Path)->set[str]:
    obj=json.loads(path.read_text(encoding='utf-8'))
    seq=obj.get('labels',[]) if isinstance(obj,dict) else obj
    if not isinstance(seq,list):raise ValueError('Invalid labels.json')
    return set(seq)


def apply(cfg:dict,action:str,label:str='',value:str='',model_labels:set[str]|None=None):
    active=cfg.setdefault('phrases',{})
    pending=cfg.setdefault('vocabulary_pending',{})
    if action=='add':
        if not re.fullmatch(r'[a-z][a-z0-9_]*',label):raise ValueError('Label must be snake_case')
        if label.upper() in ('REST','UNKNOWN') or label in ('REST','UNKNOWN'):
            raise ValueError('Reserved label')
        if not value.strip():raise ValueError('Vietnamese phrase required')
        if label in active:raise ValueError('Already active; use --deactivate first')
        pending[label]=value.strip()
    elif action=='activate':
        if model_labels is None:raise ValueError('Need model labels.json for activation')
        if label not in model_labels:raise ValueError('UNTRAINED_LABEL: '+label+' not in trained TCN labels')
        if label not in pending:raise ValueError('Not found in pending vocabulary')
        active[label]=pending.pop(label)
    elif action=='deactivate':
        if label not in active:raise ValueError('Not active')
        pending[label]=active.pop(label)
    elif action=='list':pass
    else:raise ValueError('Unknown action')
    return cfg


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=ROOT/'configs/phase10.yaml')
    p.add_argument('--model-labels',type=Path,help='Required to activate labels')
    commands=p.add_mutually_exclusive_group()
    commands.add_argument('--add',nargs=2,metavar=('LABEL','VIETNAMESE_TEXT'))
    commands.add_argument('--activate')
    commands.add_argument('--deactivate')
    a=p.parse_args(argv)
    path=a.config if a.config.is_absolute() else ROOT/a.config
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict):raise ValueError('Invalid Phase10 YAML')
    if a.add: action,label,value='add',a.add[0],a.add[1]
    elif a.activate: action,label,value='activate',a.activate,''
    elif a.deactivate: action,label,value='deactivate',a.deactivate,''
    else:action,label,value='list','',''
    model_labels=None
    if a.model_labels:
        mp=a.model_labels if a.model_labels.is_absolute() else ROOT/a.model_labels
        model_labels=read_labels(mp)
    apply(cfg,action,label,value,model_labels)
    if action!='list':
        backup=path.with_name(path.name+'.bak_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        if backup.exists():raise FileExistsError('Backup already exists '+str(backup))
        shutil.copy2(path,backup)
        path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        print('BACKUP:',backup)
    print('ACTIVE:',json.dumps(cfg.get('phrases',{}),ensure_ascii=False))
    print('PENDING:',json.dumps(cfg.get('vocabulary_pending',{}),ensure_ascii=False))
    if model_labels is not None:
        print('MODEL_SUPPORTED_ACTIVE:',sorted(set(cfg.get('phrases',{})) & model_labels))
    return 0

if __name__=='__main__':sys.exit(main())
