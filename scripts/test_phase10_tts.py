"""Check real local Vietnamese TTS without opening webcam or TensorFlow."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.phase10_tts import VietnameseSpeaker, prepare_speech_text


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/phase10.yaml')
    ap.add_argument('--text',default='Xin chào. Cảm ơn bạn!')
    ap.add_argument('--no-play',action='store_true',help='Only generate WAV; do not play audio')
    ap.add_argument('--backend',choices=['auto','piper','vieneu','windows','espeak','none'])
    a=ap.parse_args(argv)
    p=Path(a.config)
    if not p.is_absolute():p=ROOT/p
    cfg=yaml.safe_load(p.read_text(encoding='utf-8'))
    t=cfg.get('tts',{})
    def abspath(path):
        path=Path(path)
        return path if path.is_absolute() else ROOT/path
    speaker=VietnameseSpeaker(a.backend or t.get('backend','auto'),rate=int(t.get('rate',145)),
                              model_path=abspath(t.get('model_path','voices/piper/vi_VN-vais1000-medium.onnx')),
                              wav_output_dir=abspath(t.get('wav_output_dir','logs/phase10_audio')),
                              play_audio=not a.no_play,
                              vieneu_endpoint=t.get('vieneu_endpoint','http://127.0.0.1:8770/synthesize'),
                              vieneu_voice=t.get('vieneu_voice','Minh Đức'))
    print('BACKEND:',speaker.info.backend,'AVAILABLE:',speaker.info.available)
    print('VOICE:',speaker.info.detail)
    try:
        spoken_text=prepare_speech_text(a.text)
    except ValueError as ex:
        print('TEXT ERROR:',ex);return 4
    print('FULL_TEXT_TO_SPEAK:',repr(spoken_text))
    if not speaker.info.available:return 2
    ok,status=speaker.speak(spoken_text)
    print('RESULT:',status)
    if speaker.last_wav:print('WAV:',speaker.last_wav)
    return 0 if ok else 3

if __name__=='__main__':sys.exit(main())
