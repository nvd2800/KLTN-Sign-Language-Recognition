"""One-time explicit install of local vi_VN Piper neural voice (no runtime network).

python scripts/install_phase10_piper_voice.py --download
or copy manually and run --verify-only.  Files from rhasspy/piper-voices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import os
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
NAME='vi_VN-vais1000-medium'
FILES={
    NAME+'.onnx': ('vi/vi_VN/vais1000/medium/'+NAME+'.onnx','5e42428c4f6131f75557cf156c9c1526'),
    NAME+'.onnx.json': ('vi/vi_VN/vais1000/medium/'+NAME+'.onnx.json','5ec9669253541d64ba97f60e422f1ad0'),
}
BASE='https://huggingface.co/rhasspy/piper-voices/resolve/main/'


def md5(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def install(dst:Path, *, download=False, source_dir:Path|None=None):
    dst.mkdir(parents=True,exist_ok=True)
    for name,(remote,expected) in FILES.items():
        path=dst/name
        if path.is_file() and md5(path)==expected:
            print('VERIFIED:',path);continue
        if source_dir is not None:
            origin=source_dir/name
            if not origin.is_file():raise FileNotFoundError('Missing offline source: '+str(origin))
            tmp=dst/(name+'.tmp');shutil.copy2(origin,tmp)
        elif download:
            tmp=dst/(name+'.tmp')
            try:
                print('DOWNLOADING:',name)
                with urlopen(BASE+remote,timeout=45) as resp,tmp.open('wb') as out:
                    shutil.copyfileobj(resp,out)
            except BaseException:
                tmp.unlink(missing_ok=True);raise
        else:
            raise FileNotFoundError(f'Missing/invalid local voice: {path}; run --download once or --from-dir')
        if md5(tmp)!=expected:
            tmp.unlink(missing_ok=True)
            raise ValueError('Voice checksum mismatch: '+name)
        os.replace(tmp,path)
        print('INSTALLED:',path)
    obj=json.loads((dst/(NAME+'.onnx.json')).read_text(encoding='utf-8'))
    lang=str(obj.get('language',{}).get('code',''))
    if lang not in ('vi_VN','vi-VN'):
        raise ValueError('Expected Vietnamese vi_VN voice configuration, got '+lang)
    print('VOICE READY - local inference does not use the Internet')


def main(argv=None):
    p=argparse.ArgumentParser(description='Install/verify official vi_VN neural Piper voice')
    p.add_argument('--download',action='store_true',help='Explicit one-time download over Internet')
    p.add_argument('--from-dir',type=Path,help='Copy already-downloaded ONNX+JSON offline')
    p.add_argument('--dest',type=Path,default=ROOT/'voices/piper')
    a=p.parse_args(argv)
    if a.download and a.from_dir: p.error('Use --download OR --from-dir')
    install(a.dest,download=a.download,source_dir=a.from_dir)
    return 0

if __name__=='__main__': sys.exit(main())
