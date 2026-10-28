"""Phase10 local-only VieNeu ONNX TTS sidecar.

Start with its own Python environment (NOT the TensorFlow/MediaPipe env):
  .venv_tts/Scripts/python.exe scripts/phase10_vieneu_server.py --voice "Minh Đức"

First startup may download model weights once. Subsequent syntheses are offline.
Only listens on 127.0.0.1 by default. Server accepts at most 500 characters.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_INPUT_BYTES = 4096
MAX_TEXT_LENGTH = 500
MAX_OUTPUT_BYTES = 30 * 1024 * 1024


def handler_factory(engine, *, voice_name: str):
    lock = threading.Lock()  # model may not be safe for concurrent calls

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print('[VIENEU] ' + fmt % args, flush=True)

        def _json(self, code, item):
            raw = json.dumps(item, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != '/health':
                return self._json(404, {'error': 'NOT_FOUND'})
            return self._json(200, {'ready': True, 'backend': 'vieneu', 'voice': voice_name, 'local_only': True})

        def do_POST(self):
            if self.path != '/synthesize':
                return self._json(404, {'error': 'NOT_FOUND'})
            try:
                n = int(self.headers.get('Content-Length', '-1'))
                if n <= 0 or n > MAX_INPUT_BYTES:
                    return self._json(413, {'error': 'INPUT_TOO_LARGE'})
                raw = self.rfile.read(n)
                data = json.loads(raw.decode('utf-8'))
                if not isinstance(data, dict):
                    raise ValueError('JSON_OBJECT_REQUIRED')
                text = data.get('text')
                request_voice = data.get('voice', voice_name)
                if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_LENGTH:
                    raise ValueError('INVALID_TEXT')
                if request_voice != voice_name:
                    raise ValueError('VOICE_MISMATCH: server voice is '+voice_name)
                # No model infer from untrusted remote clients; server only binds loopback.
                temp = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        temp = Path(f.name)
                    with lock:
                        waveform = engine.infer(text, voice=voice_name)
                        engine.save(waveform, str(temp))
                    if temp.stat().st_size > MAX_OUTPUT_BYTES:
                        return self._json(413, {'error': 'AUDIO_TOO_LARGE'})
                    audio = temp.read_bytes()
                finally:
                    if temp is not None:
                        temp.unlink(missing_ok=True)
                if not audio.startswith(b'RIFF') or audio[8:12] != b'WAVE':
                    raise ValueError('INVALID_AUDIO_WAV')
                with wave.open(io.BytesIO(audio), 'rb') as reader:
                    if reader.getnframes() < 100 or reader.getframerate() < 8000:
                        raise ValueError('EMPTY_OR_INVALID_WAV')
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(audio)))
                self.end_headers()
                self.wfile.write(audio)
                print(f'[VIENEU OK] text={text!r}, bytes={len(audio)}', flush=True)
            except (ValueError, UnicodeError, json.JSONDecodeError) as e:
                return self._json(400, {'error': str(e)[:120]})
            except (OSError, RuntimeError, TypeError) as e:
                print('[VIENEU INFERENCE ERROR]', repr(e), flush=True)
                return self._json(500, {'error': 'INFERENCE_FAILED', 'detail': str(e)[:150]})

    return Handler


def create_server(engine, host='127.0.0.1', port=8770, *, voice_name='Minh Đức'):
    if host not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('LOCAL_ONLY: never expose TTS text/audio API publicly')
    return ThreadingHTTPServer((host, port), handler_factory(engine, voice_name=voice_name))


def main(argv=None):
    ap = argparse.ArgumentParser(description='VieNeu local Vietnamese neural TTS for Phase 10')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--voice', default='Minh Đức')
    ap.add_argument('--offline',action='store_true',help='Require all voice/model assets to be already cached locally')
    args = ap.parse_args(argv)
    if args.offline:
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
    print('[VIENEU] Loading ONNX CPU engine ('+('offline cache only' if args.offline else 'first run may download model once')+')...', flush=True)
    try:
        from vieneu import Vieneu
    except ImportError as e:
        print('DEPENDENCY_MISSING: Use separate .venv_tts environment: python -m pip install vieneu')
        raise SystemExit(2) from e
    engine = Vieneu(backend='onnx')
    voices = engine.list_preset_voices()
    names = {str(v[1]) for v in voices if isinstance(v, (tuple, list)) and len(v) >= 2}
    # Official API accepts a human-readable voice name, such as "Minh Đức".
    names.update(str(v[0]) for v in voices if isinstance(v, (tuple, list)) and len(v) >= 2)
    if names and args.voice not in names:
        print('VOICE_UNAVAILABLE:', repr(args.voice), 'Available:', ', '.join(sorted(names)[:50]))
        raise SystemExit(3)
    server = create_server(engine, port=args.port, voice_name=args.voice)
    print(f'[VIENEU READY] http://127.0.0.1:{args.port}/health voice={args.voice!r}', flush=True)
    print('Keep this terminal open while Phase 10 realtime is running. Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
