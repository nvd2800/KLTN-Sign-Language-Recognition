PHASE10 VIENEU LOCAL OFFLINE WHOLE-SENTENCE TTS - FIX 2026-10-29
=============================================================
Nguyên nhân: FULL_VIETNAMESE_TEXT đúng nhưng voice Piper đọc như đánh vần.
Không sửa TCN V3.1, MediaPipe, Phase09 state machine hoặc Sentence Builder.

INSTALL: giải nén ZIP vào root project; sao lưu file bị chép đè.

TERMINAL 1 - Create isolated environment for Vietnamese speech engine:
  py -m venv .venv_tts
  .\.venv_tts\Scripts\python.exe -m pip install --upgrade pip
  .\.venv_tts\Scripts\python.exe -m pip install vieneu
  .\.venv_tts\Scripts\python.exe scripts\phase10_vieneu_server.py --voice "Minh Đức"

Note: first startup may download VieNeu voice weights. After models cached, use --offline.

TERMINAL 2 - Existing project environment (TensorFlow, MediaPipe unchanged):
  conda activate venv
  python scripts\diagnose_phase10_tts.py --synthesize --play
  python scripts\test_phase10_tts.py --backend vieneu --text "Xin chào. Cảm ơn bạn. Xin lỗi!"
  python scripts\live_phase10_v3_1.py --fullscreen

If the spoken WAV still sounds spelled, provide the actual WAV file from
logs/phase10_audio and backend/version status for auditory analysis.

Local-only architecture: Phase10 -> http://127.0.0.1:8770 -> VieNeu ONNX -> WAV.
Server is bound to loopback and never uploads your text to any hosted API.

IMPORTANT: no voice model/weights are included in this small patch ZIP.
The TTS code and mock-backed integration were tested; actual VieNeu audio
cannot be auditioned without installing and downloading the voice on your PC.
