# Phase 10.2 – sửa TTS đọc đánh vần dù văn bản đúng

## Nguyên nhân và lựa chọn sửa

`FULL_VIETNAMESE_TEXT` đã đúng chứng tỏ pipeline **TCN V3.1 → Phase 09 → Sentence Builder → Text UTF-8** hoạt động. Còn lỗi phát âm thuộc backend/voice/phonemizer của TTS. Thay Piper `vi_VN-vais1000-medium` bằng **VieNeu-TTS** (ONNX CPU) để kiểm tra trực tiếp giọng Việt khác. Đây là phương án thay thế voice, chưa chứng minh bằng nghe rằng mọi từ sẽ chuẩn.

**Giữ TensorFlow/MediaPipe trong env `venv` cũ**, không `pip install vieneu` trực tiếp vào env đó vì xung đột `numpy`, `protobuf`, `onnxruntime`, v.v. có thể làm hỏng Phase 09. Chạy một dịch vụ TTS cục bộ trong `.venv_tts` riêng.

## Kiến trúc

```text
Camera → MediaPipe → TCN V3.1 (32×513) → Phase09 confirmed token
  → Phase10 SentenceBuilder → nguyên câu UTF-8
  → localhost:8770/synthesize → VieNeu ONNX CPU → WAV → loa
```

Dịch vụ lắng nghe trên `127.0.0.1`, không gọi cloud API. Lần đầu cần tải model/voice từ kho phân phối; lần sau chạy không Internet khi đã cache. Kiểm tra giấy phép và dung lượng model khi đưa lên Raspberry Pi.

## Cài đặt (Windows PowerShell / VS Code)

**Terminal 1 – môi trường TTS tách biệt**

```powershell
cd D:\KLTN_KTMT\KLTN_Sign\Sign-Language-Recognition-Phase0
py -m venv .venv_tts
.\.venv_tts\Scripts\python.exe -m pip install --upgrade pip
.\.venv_tts\Scripts\python.exe -m pip install vieneu
.\.venv_tts\Scripts\python.exe scripts\phase10_vieneu_server.py --voice "Minh Đức"
```

Có thể dùng Python 3.11 nếu `py -3.11` có trên máy; nếu không, chạy bằng `py` hiện tại. Tùy phiên bản VieNeu, cần kiểm tra tương thích Python và khả năng tải model. Khởi động lần đầu có thể tải model; kiểm tra log trước khi mở webcam. Khi thấy `VIENEU READY`, để terminal này mở.

Xác minh server:

```powershell
Invoke-RestMethod http://127.0.0.1:8770/health
```

**Terminal 2 – môi trường AI cũ**

```powershell
conda activate venv
python scripts\diagnose_phase10_tts.py --synthesize --play
python scripts\test_phase10_tts.py --backend vieneu --text "Xin chào. Cảm ơn bạn. Xin lỗi!"
python scripts\live_phase10_v3_1.py --fullscreen
```

Khi cài xong và voice đã ở cache, có thể khởi động ở chế độ cấm tải:

```powershell
.\.venv_tts\Scripts\python.exe scripts\phase10_vieneu_server.py --voice "Minh Đức" --offline
```

## File cấu hình

`configs/phase10.yaml`:

```yaml
tts:
  backend: vieneu
  vieneu_endpoint: "http://127.0.0.1:8770/synthesize"
  vieneu_voice: "Minh Đức"
  wav_output_dir: "logs/phase10_audio"
  play_audio: true
```

Giữ key `model_path` nếu muốn thử lại Piper về sau. Không tự động fallback sang voice khác.

## Kiểm tra lỗi

| Triệu chứng | Nguyên nhân và xử lý |
|---|---|
| `VOICE_UNAVAILABLE` | Chưa mở Terminal 1 hoặc server chưa `READY`; kiểm tra health |
| `No module named vieneu` | Đã cài nhầm env; cài vào `.venv_tts` |
| `VOICE_MISMATCH` | Voice trong YAML không khớp voice server; dùng `Minh Đức` cả hai |
| Server lỗi khi tải model | Cần Internet lần đầu hoặc cache chưa đầy đủ; không chạy `--offline` trước khi tải |
| WAV tạo được nhưng vẫn phát âm lạ | Nghe file `logs/phase10_audio/phase10_vieneu_*.wav`, kiểm tra mẫu giọng và nội dung gửi |
| Webcam giật khi đọc | TTS và MediaPipe tranh CPU; giảm độ phân giải camera, dùng giọng đã tạo sẵn cho cụm từ thường gặp hoặc tăng tài nguyên |
| `FileExistsError` checkpoint | Không liên quan TTS; không train lại vì lỗi giọng |

## Bảo vệ tài nguyên và dữ liệu

- Dịch vụ chỉ bind vào `localhost`, không expose trên mạng LAN.
- Giới hạn đầu vào 500 ký tự, tối đa 30 MiB WAV.
- Không tải thư viện hoặc model khi inference đã chạy ở chế độ `--offline`.
- WAV được lưu vào `logs/phase10_audio/`; có thể xóa các bản nháp sau khi nghe.
- Đừng cài VieNeu vào env chứa TensorFlow đang chạy ổn định.

## Điều kiện PASS

1. Câu tiếng Việt với dấu hiển thị đúng.
2. Dịch vụ TTS health READY và xuất file WAV hợp lệ.
3. Nghe thực tế và xác nhận từ/câu không bị đánh vần; bộ test tự động không đánh giá chất lượng phát âm.
4. Phase09 vẫn phát đúng token khi TTS hoạt động đồng thời.
5. Sau restart/offline, TTS phát được mà không cần Internet.

## Nguồn tham khảo

- VieNeu SDK: https://github.com/xuanhieu/vieneu-tts
- Piper voices vi: https://huggingface.co/rhasspy/piper-voices/tree/main/vi/vi_VN
