# Phase 10.2 — Sửa lỗi TTS chỉ đọc từng ký tự

## Chẩn đoán nguyên nhân

Trong source Phase 10.1, `SentenceBuilder` đã chuyển `xin_chao` thành “Xin chào” và gửi **toàn bộ câu** cho `VietnameseSpeaker`. Do đó, trước khi sửa text builder hoặc AI, cần xác định `backend` đang thực thi và **chính xác văn bản vào TTS**. Cấu hình cũ `backend: auto` có thể tự dùng eSpeak-NG hoặc Windows SAPI khi Piper chưa cài; giọng dự phòng tiếng Việt có thể phát âm máy móc như đánh vần.

## Quy trình cài

Cập nhật `core/phase10_tts.py`, `configs/phase10.yaml`, `scripts/live_phase09_v3_spatial.py`, `scripts/test_phase10_tts.py`. File `configs/phase10.yaml` đổi thành `backend: piper` — yêu cầu voice neural được cài thật, không tự đổi giọng. Sau đó chạy:

```powershell
conda activate venv
python -m pip install piper-tts PyYAML Pillow
python scripts/install_phase10_piper_voice.py --download
python scripts/diagnose_phase10_tts.py
python scripts/diagnose_phase10_tts.py --synthesize --play
python scripts/test_phase10_tts.py --backend piper --text "Xin chào. Cảm ơn bạn. Xin lỗi!"
python scripts/live_phase10_v3_1.py --fullscreen
```

**Lưu ý:** Giữ source `scripts/install_phase10_piper_voice.py` có sẵn từ bản Phase 10.1 + 10.2. Model ONNX và `.onnx.json` tải một lần. Khi đã cài, chạy hoàn toàn offline.

## Kết quả cần nhìn thấy

```
BACKEND: piper AVAILABLE: True
FULL_VIETNAMESE_TEXT: 'Xin chào cảm ơn xin lỗi'
SYNTHESIS: SPOKEN
WAV_PATH: .../logs/phase10_audio/phase10_....wav
```

Khi bấm **T** trong webcam, terminal sẽ có:

```
[PHASE10 TTS INPUT] backend=piper full_sentence='Xin chào cảm ơn xin lỗi'
[PHASE10 TTS RESULT] backend=piper ok=True status=SPOKEN text='Xin chào cảm ơn xin lỗi'
```

* Nếu `VOICE_UNAVAILABLE`: chưa có CLI `piper` trong `venv`, ONNX hoặc `.onnx.json`.
* Nếu `TTS_CHARACTER_SPLIT`: phần gọi TTS đã truyền từng chữ cái. Kiểm tra `SentenceBuilder.preview`; không gửi `'x i n c h a o'`.
* Nếu `TTS_UNRENDERED_LABEL`: gọi TTS bằng raw label thay vì cụm từ tiếng Việt. Đọc `configs/phase10.yaml`.
* Nếu WAV tạo và đầu vào đúng nguyên câu, nhưng nghe như đánh vần: kiểm tra chất lượng voice Piper, thử voice tiếng Việt ONNX tương thích khác; đừng sửa scaler/TCN.
* Nếu `SYNTHESIS: SPOKEN` nhưng không nghe: kiểm tra Windows output device/volume, test WAV riêng bằng trình phát.

## Dữ liệu và checkpoint

Không thay đổi dataset, checkpoint V3.1, 513D, Phase09 FSM hay bất kỳ dữ liệu train nào. Thay đổi chỉ ở bước Text-to-Speech và log. `REST` và `UNKNOWN` vẫn bị SentenceBuilder bỏ qua.

## Test

```powershell
python -m pytest -q tests/test_phase10_tts_fullphrase.py tests/test_phase10_1_2.py tests/test_phase10.py
```

Đã kiểm thử chuyển câu và stdin UTF-8 bằng mock, **chưa xác nhận chất lượng âm thanh từ voice cài trên máy Windows của bạn**.
