"""Diagnose Phase10 TTS without camera or TensorFlow.

Shows EXACT full Vietnamese text and backend before optionally synthesizing WAV.
No network operations are performed here.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import wave
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.phase10_sentence import SentenceBuilder
from core.phase10_tts import VietnameseSpeaker, prepare_speech_text


def diagnose(config:Path,*, tokens:list[str], synthesize:bool, play:bool)->int:
    cfg=yaml.safe_load(config.read_text(encoding='utf-8'))
    p10=cfg.get('tts',{})
    builder=SentenceBuilder(cfg.get('phrases'),max_tokens=int(cfg.get('sentence',{}).get('max_tokens',40)),
                            punctuation_labels=cfg.get('punctuation_labels'))
    for token in tokens:
        if not builder.add(token):
            print('IGNORED_TOKEN:',repr(token))
    text=prepare_speech_text(builder.preview)
    def resolve(path):
        path=Path(path)
        return path if path.is_absolute() else ROOT/path
    model=resolve(p10.get('model_path','voices/piper/vi_VN-vais1000-medium.onnx'))
    output=resolve(p10.get('wav_output_dir','logs/phase10_audio'))
    voice=VietnameseSpeaker(p10.get('backend','vieneu'),model_path=model,wav_output_dir=output,play_audio=play,
                            vieneu_endpoint=p10.get('vieneu_endpoint','http://127.0.0.1:8770/synthesize'),
                            vieneu_voice=p10.get('vieneu_voice','Minh Đức'))
    print('TOKENS:',repr(tokens))
    print('PHRASE_MAPPING:',repr(builder.phrases))
    print('FULL_VIETNAMESE_TEXT:',repr(text))
    print('UTF8_BYTES:',text.encode('utf-8').hex())
    print('BACKEND:',voice.info.backend,'AVAILABLE:',voice.info.available)
    if voice.info.backend=='vieneu':
        print('LOCAL_TTS_ENDPOINT:',p10.get('vieneu_endpoint','http://127.0.0.1:8770/synthesize'))
        print('VOICE_NAME:',p10.get('vieneu_voice','Minh Đức'))
    elif voice.info.backend=='piper':
        print('PIPER_COMMAND:',shutil.which('piper'))
        print('MODEL:',model,'EXISTS:',model.is_file())
        print('MODEL_JSON:',str(model)+'.json','EXISTS:',Path(str(model)+'.json').is_file())
    print('VOICE:',voice.info.detail)
    if not voice.info.available:
        if voice.info.backend=='vieneu':
            print('FIX: start the VieNeu server using .venv_tts in another terminal')
            print('FIX: .\\.venv_tts\\Scripts\\python.exe scripts/phase10_vieneu_server.py')
        else:
            print('FIX: python -m pip install piper-tts')
            print('FIX: python scripts/install_phase10_piper_voice.py --download')
        return 2
    if not synthesize:
        print('DRY_RUN_ONLY: use --synthesize --play to hear the whole sentence')
        return 0
    ok,status=voice.speak(text)
    print('SYNTHESIS:',status)
    if voice.last_wav:
        print('WAV_PATH:',voice.last_wav)
        try:
            with wave.open(str(voice.last_wav),'rb') as wf:
                secs=wf.getnframes()/float(wf.getframerate())
                print('WAV_DETAILS:',json.dumps({'duration_s':round(secs,3),'sample_rate':wf.getframerate(),
                                                  'channels':wf.getnchannels()},ensure_ascii=False))
                if secs<.25:
                    print('WARNING: audio is suspiciously short for an entire Vietnamese phrase')
        except Exception as e:print('WARNING: could not inspect WAV',str(e))
    return 0 if ok else 3


def main(argv=None):
    p=argparse.ArgumentParser(description='Check whole Vietnamese text sent to actual Piper TTS')
    p.add_argument('--config',default='configs/phase10.yaml')
    p.add_argument('--tokens',nargs='+',default=['xin_chao','cam_on','xin_loi'])
    p.add_argument('--synthesize',action='store_true')
    p.add_argument('--play',action='store_true',help='Play WAV through local speakers after synthesizing')
    args=p.parse_args(argv)
    c=Path(args.config)
    if not c.is_absolute():c=ROOT/c
    if not c.is_file():print('CONFIG_MISSING:',c);return 2
    try:return diagnose(c,tokens=args.tokens,synthesize=args.synthesize,play=args.play)
    except ValueError as e:
        print('INPUT_ERROR:',str(e));return 4

if __name__=='__main__':sys.exit(main())
