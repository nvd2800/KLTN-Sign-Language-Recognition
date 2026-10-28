"""Offline Vietnamese TTS: neural Piper (preferred) -> Vietnamese SAPI -> eSpeak.

Runtime NEVER downloads voices. `install_phase10_piper_voice.py` is an explicit,
one-time online installer. The neural synthesizer writes WAVs which are saved
for the session; a single background worker keeps Phase09 inference responsive.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import urllib.error
import urllib.parse
import urllib.request
import wave
import queue
import shutil
import subprocess
import sys
import threading
import uuid
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


# Only guard the edge where unrendered Phase09 labels accidentally reach TTS.
# Normal Phase10 already passes a complete Vietnamese sentence from SentenceBuilder.
_LEGACY_LABELS = {'xin_chao':'xin chào', 'cam_on':'cảm ơn', 'xin_loi':'xin lỗi'}


def prepare_speech_text(text: str) -> str:
    """Produce one NFC Vietnamese utterance, never a list of letters or label IDs."""
    if not isinstance(text, str):
        raise ValueError('TTS_INPUT_NOT_STRING: pass SentenceBuilder.preview or Sentence.text')
    value = unicodedata.normalize('NFC', re.sub(r'\s+', ' ', text).strip())
    if not value:
        raise ValueError('EMPTY_TEXT')
    tokens = value.split(' ')
    if tokens and all(t in _LEGACY_LABELS for t in tokens):
        value = ' '.join(_LEGACY_LABELS[t] for t in tokens)
        value = value[0].upper() + value[1:] + '.'
    elif '_' in value:
        raise ValueError('TTS_UNRENDERED_LABEL: send Vietnamese phrase, not label_xxx')
    tokens = value.split(' ')
    if len(tokens) >= 4 and all(len(t) == 1 and t.isalpha() for t in tokens):
        raise ValueError('TTS_CHARACTER_SPLIT: pass the full phrase, not spaced letters')
    return value


@dataclass(frozen=True)
class VoiceInfo:
    backend: str
    available: bool
    detail: str


class VietnameseSpeaker:
    def __init__(self, backend: str='auto', rate: int=145, *,
                 model_path: str | Path='voices/piper/vi_VN-vais1000-medium.onnx',
                 wav_output_dir: str | Path='logs/phase10_audio', play_audio:bool=True,
                 vieneu_endpoint: str='http://127.0.0.1:8770/synthesize',
                 vieneu_voice: str='Minh Đức',
                 runner=None,platform:str|None=None,which=None,player=None):
        if backend not in ('auto','piper','vieneu','none','windows','espeak'):
            raise ValueError('Unsupported TTS backend')
        self.rate=max(80,min(220,int(rate)))
        self._run=runner or subprocess.run
        self._which=which or shutil.which
        self._platform=platform or sys.platform
        self.model_path=Path(model_path)
        self.wav_output_dir=Path(wav_output_dir)
        self.play_audio=bool(play_audio)
        self.vieneu_endpoint=str(vieneu_endpoint)
        self.vieneu_voice=str(vieneu_voice)
        self._check_local_endpoint()
        self._player=player
        self._voice_name=''
        self.last_wav:Path|None=None
        self.info=self._detect(backend)

    def _check_local_endpoint(self):
        # Keep all TTS text on this machine. Do not allow accidental cloud URLs.
        url=urllib.parse.urlsplit(self.vieneu_endpoint)
        if (url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost','::1')
                or url.path!='/synthesize' or url.username or url.password):
            raise ValueError('VIENEU_ENDPOINT_NOT_LOCAL: use http://127.0.0.1:8770/synthesize')

    def _has_vieneu(self):
        url=self.vieneu_endpoint.rsplit('/',1)[0]+'/health'
        try:
            with urllib.request.urlopen(url,timeout=2) as r:
                info=json.loads(r.read(2048).decode('utf-8'))
                return r.status==200 and info.get('ready') is True
        except (OSError,UnicodeError,ValueError,urllib.error.URLError):
            return False

    def _has_piper(self) -> bool:
        model=self.model_path
        config=Path(str(model)+'.json')
        return model.is_file() and config.is_file() and bool(self._which('piper'))

    @staticmethod
    def _ps_command():
        return "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Culture.Name + '|' + $_.VoiceInfo.Name }"

    def _get_windows_voice(self):
        if self._platform!='win32' or not self._which('powershell'):
            return ''
        try:
            r=self._run(['powershell','-NoProfile','-NonInteractive','-Command',self._ps_command()],capture_output=True,text=True,timeout=15)
            if r.returncode==0:
                for line in r.stdout.splitlines():
                    culture,sep,name=line.strip().partition('|')
                    if sep and (culture.lower()=='vi-vn' or culture.lower().startswith('vi-')):
                        return name.strip()
        except (OSError,subprocess.TimeoutExpired,UnicodeError):pass
        return ''

    def _has_espeak_vi(self):
        if not self._which('espeak-ng'):
            return False
        try:
            r=self._run(['espeak-ng','--voices=vi'],capture_output=True,text=True,timeout=10)
            return r.returncode==0 and any('vi' in l.lower() for l in r.stdout.splitlines()[1:])
        except (OSError,subprocess.TimeoutExpired):return False

    def _detect(self,backend):
        if backend=='none':return VoiceInfo('none',False,'TTS disabled')
        if backend=='vieneu':
            ok=self._has_vieneu()
            return VoiceInfo('vieneu',ok,('VieNeu offline TTS ready: '+self.vieneu_endpoint) if ok else 'VieNeu local service unavailable; start it using the separate .venv_tts environment')
        if backend in ('auto','piper'):
            if self._has_piper():
                return VoiceInfo('piper',True,'Offline Vietnamese neural Piper voice: '+str(self.model_path))
            if backend=='piper':
                return VoiceInfo('piper',False,'Piper executable and local .onnx/.onnx.json files required')
        # Never silently switch to robotic SAPI/eSpeak: explicit opt-in only.
        if backend == 'windows':
            self._voice_name=self._get_windows_voice()
            if self._voice_name:return VoiceInfo('windows',True,'Vietnamese SAPI: '+self._voice_name)
            if backend=='windows':return VoiceInfo('windows',False,'No vi-VN Windows SAPI voice installed')
        if backend == 'espeak' and self._has_espeak_vi():
            return VoiceInfo('espeak',True,'Vietnamese eSpeak NG voice (robotic fallback)')
        return VoiceInfo(backend,False,'Piper voice not ready. Install ONNX/JSON + piper-tts; fallback voices require explicit opt-in')

    def _play_wav(self,path:Path)->tuple[bool,str]:
        if not self.play_audio:return True,'WAV_CREATED'
        if self._player is not None:
            return (True,'SPOKEN') if self._player(path) else (False,'WAV_CREATED_PLAYBACK_FAILED')
        if self._platform=='win32':
            try:
                import winsound
                winsound.PlaySound(str(path),winsound.SND_FILENAME)
                return True,'SPOKEN'
            except (OSError,ImportError,RuntimeError) as ex:
                return False,'WAV_CREATED_PLAYBACK_FAILED: '+str(ex)[:100]
        for command in ('aplay','ffplay'):
            if self._which(command):
                args=[command,'-q',str(path)] if command=='aplay' else [command,'-nodisp','-autoexit','-loglevel','error',str(path)]
                r=self._run(args,capture_output=True,text=True,timeout=120)
                return (True,'SPOKEN') if r.returncode==0 else (False,'WAV_CREATED_PLAYBACK_FAILED')
        return False,'WAV_CREATED_NO_PLAYER'

    def _speak_piper(self,text:str)->tuple[bool,str]:
        self.wav_output_dir.mkdir(parents=True,exist_ok=True)
        wav=self.wav_output_dir/('phase10_'+uuid.uuid4().hex[:16]+'.wav')
        try:
            # piper stdin must be UTF-8. Use bytes to avoid Windows cp1252 console issues.
            r=self._run(['piper','--model',str(self.model_path),'--output_file',str(wav)],
                        input=(text+'\n').encode('utf-8'),capture_output=True,timeout=120)
            if r.returncode!=0:
                wav.unlink(missing_ok=True)
                return False,'PIPER_FAILED: '+str(getattr(r,'stderr',b''))[:190]
            if not wav.is_file() or wav.stat().st_size<44 or wav.open('rb').read(4)!=b'RIFF':
                wav.unlink(missing_ok=True)
                return False,'PIPER_INVALID_WAV'
            self.last_wav=wav
            return self._play_wav(wav)
        except (OSError,subprocess.TimeoutExpired) as ex:
            wav.unlink(missing_ok=True)
            return False,'PIPER_FAILED: '+str(ex)[:200]

    def _speak_vieneu(self,text:str)->tuple[bool,str]:
        """Send a complete Unicode sentence to a local-only TTS server; get a WAV."""
        self.wav_output_dir.mkdir(parents=True,exist_ok=True)
        wav=self.wav_output_dir/('phase10_vieneu_'+uuid.uuid4().hex[:16]+'.wav')
        payload=json.dumps({'text':text,'voice':self.vieneu_voice},ensure_ascii=False).encode('utf-8')
        req=urllib.request.Request(self.vieneu_endpoint,data=payload,
                                   headers={'Content-Type':'application/json; charset=utf-8'},
                                   method='POST')
        try:
            with urllib.request.urlopen(req,timeout=180) as res:
                data=res.read(30*1024*1024+1)
            if len(data)>30*1024*1024 or not data.startswith(b'RIFF') or data[8:12]!=b'WAVE':
                return False,'VIENEU_INVALID_WAV'
            with wave.open(io.BytesIO(data),'rb') as wf:
                if wf.getnframes()<100 or wf.getframerate()<8000:
                    return False,'VIENEU_SHORT_OR_INVALID_WAV'
            wav.write_bytes(data)
            self.last_wav=wav
            return self._play_wav(wav)
        except urllib.error.HTTPError as e:
            detail=e.read(512).decode('utf-8',errors='replace')
            return False,f'VIENEU_HTTP_{e.code}: '+detail
        except (OSError,ValueError,EOFError,wave.Error,urllib.error.URLError) as ex:
            return False,'VIENEU_FAILED: '+str(ex)[:200]

    def speak(self,text:str)->tuple[bool,str]:
        try:
            text = prepare_speech_text(text)
        except ValueError as ex:
            return False,str(ex)
        if not self.info.available:return False,'VOICE_UNAVAILABLE'
        if self.info.backend=='piper':return self._speak_piper(text)
        if self.info.backend=='vieneu':return self._speak_vieneu(text)
        try:
            if self.info.backend=='windows':
                t64=base64.b64encode(text.encode('utf-8')).decode('ascii')
                v64=base64.b64encode(self._voice_name.encode('utf-8')).decode('ascii')
                cmd=("[Console]::OutputEncoding=[Text.Encoding]::UTF8; Add-Type -AssemblyName System.Speech; "
                     "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                     "$s.SelectVoice([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"+v64+"'))); "
                     "$s.Speak([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"+t64+"')))" )
                r=self._run(['powershell','-NoProfile','-NonInteractive','-Command',cmd],capture_output=True,text=True,timeout=120)
            else:
                r=self._run(['espeak-ng','-v','vi','-s',str(self.rate),text],capture_output=True,text=True,timeout=120)
            if r.returncode !=0:return False,'TTS_FAILED: '+str(r.stderr)[:150]
            return True,'SPOKEN'
        except (OSError,subprocess.TimeoutExpired) as ex:
            return False,'TTS_FAILED: '+str(ex)[:180]


class SpeechQueue:
    """Sequential offline worker. Does not block the camera processing loop."""
    def __init__(self,speaker:VietnameseSpeaker,max_pending:int=3):
        self.speaker=speaker
        self._queue=queue.Queue(maxsize=max_pending)
        self.last_status=speaker.info.detail
        self.last_text=''
        self._thread:threading.Thread|None=None
        self._closed=False

    def say(self,text:str)->bool:
        try:
            full_text=prepare_speech_text(text)
        except ValueError as ex:
            self.last_status=str(ex);return False
        if self._closed or not self.speaker.info.available:
            self.last_status=('QUEUE_CLOSED' if self._closed else 'VOICE_UNAVAILABLE: '+self.speaker.info.detail)
            return False
        self.last_text=full_text
        if self._thread is None:
            self._thread=threading.Thread(target=self._work,daemon=True)
            self._thread.start()
        try:
            self._queue.put_nowait(full_text)
            self.last_status='QUEUED'
            return True
        except queue.Full:
            self.last_status='QUEUE_FULL';return False

    def _work(self):
        while True:
            item=self._queue.get()
            try:
                if item is None:return
                ok,status=self.speaker.speak(item)
                self.last_status=status
                print(f'[PHASE10 TTS RESULT] backend={self.speaker.info.backend} ok={ok} status={status} text={item!r}', flush=True)
            finally:self._queue.task_done()

    def close(self)->None:
        self._closed=True
        if self._thread is not None:
            try:self._queue.put_nowait(None)
            except queue.Full:pass # never stall webcam shutdown
