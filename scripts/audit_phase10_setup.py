"""Phase10 installation audit; no camera required, no model loading unless requested."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import json
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.phase10_sentence import SentenceBuilder
from core.phase10_tts import VietnameseSpeaker
from scripts.check_phase09_v3_1 import preflight,resolve


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--phase10-config',default='configs/phase10.yaml')
    p.add_argument('--phase09-config',default='configs/phase9_v3_Final.yaml')
    p.add_argument('--dataset',default='dataset/processed/temporal_v3_1_spatial')
    p.add_argument('--check-model',action='store_true')
    p.add_argument('--tts-check',action='store_true')
    x=p.parse_args(argv)
    path=resolve(ROOT,x.phase10_config)
    issues=[]
    if not path.is_file():
        issues.append('Missing Phase10 config '+str(path))
        c={}
    else:
        try:
            c=yaml.safe_load(path.read_text(encoding='utf-8'))
            b=SentenceBuilder(c.get('phrases'),max_tokens=int(c.get('sentence',{}).get('max_tokens',40)), punctuation_labels=c.get('punctuation_labels'))
            print('PHASE10 LABELS:',b.phrases)
        except Exception as e:
            issues.append(f'Invalid Phase10 config: {e}');c={}
    missing=[x for x in ('core/phase10_sentence.py','core/phase10_store.py','core/phase10_tts.py','core/phase10_ui.py','scripts/live_phase09_v3_spatial.py','scripts/live_phase09_v3_1.py') if not (ROOT/x).is_file()]
    if missing:issues.append('Missing required code: '+str(missing))
    errors,warnings,info=preflight(ROOT,resolve(ROOT,x.phase09_config),check_model=x.check_model,dataset=resolve(ROOT,x.dataset))
    issues+=errors
    for w in warnings:print('WARNING:',w)
    for k in ('model','input_shape','checkpoint_dataset_counts','selected_dataset_counts'):
        if k in info: print(k,':',info[k])
    if x.tts_check:
        tts=c.get('tts',{})
        voice=VietnameseSpeaker(tts.get('backend','auto'),int(tts.get('rate',145)), model_path=resolve(ROOT,tts.get('model_path','voices/piper/vi_VN-vais1000-medium.onnx')), wav_output_dir=resolve(ROOT,tts.get('wav_output_dir','logs/phase10_audio')),play_audio=bool(tts.get('play_audio',True)))
        print('VIETNAMESE TTS:',voice.info)
        if voice.info.backend=='piper' and voice.info.available:print('PIPER LOCAL VOICE: PASS')
        elif tts.get('backend')=='piper':issues.append('Piper local voice missing; install piper-tts and voice ONNX')
    print('PHASE10 PREFLIGHT:', 'PASS' if not issues else 'BLOCKED')
    for e in issues:print('ERROR:',e)
    return 2 if issues else 0

if __name__=='__main__':sys.exit(main())
