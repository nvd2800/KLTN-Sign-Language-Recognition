# KLTN — Phase 10.1 & Phase 10.2 (bản nâng cấp)

**Cấu trúc:** `Camera → MediaPipe → TCN V3.1 32×513 → Phase09 TokenEvent → Phase10.1 Sentence Builder → Phase10.2 Piper Vietnamese TTS → WAV + loa`.

Đây là **patch** áp lên source Phase09/10 đang chạy, không chứa lại model Keras hay dataset RAW. Giữ checkpoint và scaler hiện hành. Chép đè file sau khi sao lưu. Không mở rộng số class mạng TCN một cách tự động.

## 1. Chức năng

### Phase 10.1 — Câu và dấu câu

- Chỉ nhận nhãn **đã được Phase09 xác nhận**; không nhận prediction mỗi frame.
- Chuyển các lớp hiện hành `xin_chao → xin chào`, `cam_on → cảm ơn`, `xin_loi → xin lỗi`.
- Viết hoa đầu câu và sau `.`, `?`, `!`; khoảng trắng đúng quanh dấu `, . ? !`; mặc định chấm cuối câu.
- `REST/UNKNOWN` không biến thành chữ. Ký hiệu lặp chính xác vẫn được giữ (không xóa lặp tùy tiện).
- Hỗ trợ dấu câu qua bàn phím hoặc nhãn điều khiển `dau_phay/dau_cham/dau_hoi/dau_cam` nếu một model tương lai được huấn luyện để xuất các nhãn đó.
- Các nhãn mới (`toi`, `ban`, `uong_nuoc`, `di`, v.v.) **ở `vocabulary_pending`**: có mapping nghĩa nhưng **chưa được TCN V3.1 5 lớp nhận diện**. Muốn chạy camera thật phải thêm dữ liệu KNOWN, split signer độc lập, train model nhiều lớp, đánh giá UNKNOWN và cập nhật labels/scaler/model.
- Quản lý từ điển an toàn bằng `scripts/phase10_manage_vocabulary.py`. Khi kích hoạt một nhãn mới script bắt buộc kiểm tra trong `labels.json` của model mới.

### Phase 10.2 — Tiếng Việt neural, offline

- Ưu tiên backend **Piper** voice `vi_VN-vais1000-medium` (ONNX khoảng 63 MB); nếu không có thì `auto` thử Windows SAPI vi-VN rồi eSpeak NG vi; không đọc bằng tiếng Anh khi thiếu giọng Việt.
- Model giọng nói **KHÔNG được đóng gói ZIP**. Người dùng cần tải một lần riêng, sau đó chạy và tạo audio **offline**; trong lúc realtime hệ thống không gọi mạng.
- Chạy TTS trên worker riêng; tự lưu WAV dưới `logs/phase10_audio/` và phát qua loa (Windows `winsound`; Linux `aplay`/`ffplay` nếu đã cài).
- Piper lần đầu cần nạp voice vào bộ nhớ và mỗi lần gọi CLI có thể có độ trễ. Nó không làm block luồng xử lý webcam, nhưng người dùng vẫn cần đo latency đầu/cuối.

## 2. Cài đúng đường dẫn

Tại thư mục project Windows VS Code:

```powershell
cd D:\KLTN_KTMT\KLTN_Sign\Sign-Language-Recognition-Phase0
conda activate venv
python -m pip install PyYAML Pillow piper-tts
```

Trước khi chép patch, sao lưu `scripts/live_phase09_v3_spatial.py`, `core/phase10_sentence.py`, `core/phase10_tts.py` và `configs/phase10.yaml`. Giải nén ZIP vào **gốc project**, chép đè file cùng tên. Không chép đè model/scaler/labels, vì gói không chứa chúng.

### Cài tiếng Việt Piper một lần (cần Internet lúc cài)

```powershell
python scripts/install_phase10_piper_voice.py --download
```

Script lấy ONNX và JSON từ kho `rhasspy/piper-voices`, kiểm tra checksum MD5 theo danh mục công bố rồi đặt vào:

```text
voices/piper/vi_VN-vais1000-medium.onnx
voices/piper/vi_VN-vais1000-medium.onnx.json
```

Khi đã tải từ máy khác và chép file về Windows offline, chạy:

```powershell
python scripts/install_phase10_piper_voice.py --from-dir D:\DownloadedPiperVoice
```

Nếu đã có voice đúng checksum thì chỉ cần:

```powershell
python scripts/install_phase10_piper_voice.py
```

**Không cần Internet khi chạy webcam và TTS sau khi đã cài đủ package và model.** Lưu ý voice model có MODEL_CARD và điều khoản license riêng; phải xem trước khi phân phối lại.

### Test TTS không cần webcam

```powershell
python scripts/test_phase10_tts.py --backend piper --text "Xin chào. Cảm ơn bạn!"
python scripts/test_phase10_tts.py --backend piper --no-play --text "Xin lỗi."
```

Nếu đúng: `BACKEND: piper`, `AVAILABLE: True`, `RESULT: SPOKEN` hoặc `WAV_CREATED`, có đường dẫn WAV trong log. Nếu `VOICE_UNAVAILABLE`: kiểm tra `pip show piper-tts`, chạy `where.exe piper`, kiểm tra cặp ONNX/JSON trong `voices/piper`.

## 3. Test Phase 10.1 không cần webcam

```powershell
python scripts/demo_phase10_sentence.py --tokens xin_chao dau_phay cam_on dau_cam
```

Kết quả mong đợi: `Xin chào, cảm ơn!`.

Test mô phỏng các từ mới chưa có trong TCN:

```powershell
python scripts/demo_phase10_sentence.py --demo-pending --tokens toi muon uong_nuoc dau_cham
```

Kết quả text **minh họa**, không phải AI đã hiểu ký hiệu `toi`/`muon`/`uong_nuoc`.

### Quản lý nhãn mới

```powershell
python scripts/phase10_manage_vocabulary.py
python scripts/phase10_manage_vocabulary.py --add cho "chờ"
```

Khi đã train lại một model thực sự có class `cho`, mới kích hoạt:

```powershell
python scripts/phase10_manage_vocabulary.py --activate cho --model-labels artifacts/experiments/MODEL_MOI/labels.json
```

Script từ chối `UNTRAINED_LABEL` nếu class không thuộc output model. Mọi thay đổi `phase10.yaml` đều tạo `.bak_*` trước.

## 4. Realtime Phase10.1 + Phase10.2

Kiểm tra checkpoint V3.1 với cấu hình Phase09 thực của bạn:

```powershell
python scripts/live_phase10_v3_1.py --config configs/phase9_v3_Final.yaml --dataset dataset/processed/temporal_v3_1_spatial --dry-run --check-model
python scripts/audit_phase10_setup.py --tts-check --check-model
```

Nếu model đã train trên dataset `temporal_v3_1_xin_loi_20261028`, hãy đổi **`--config` và `--dataset`** đúng theo phiên bản đang dùng. Trong Phase09 config, ba file model/stats/labels phải cùng thuộc một checkpoint.

Mở webcam:

```powershell
python scripts/live_phase10_v3_1.py --config configs/phase9_v3_Final.yaml --phase10-config configs/phase10.yaml --fullscreen
```

Phím Phase10:

| Phím | Tác dụng |
|---|---|
| `N` hoặc `Space` | Chốt câu (thêm dấu chấm nếu chưa có dấu kết thúc) |
| `,` | Thêm dấu phẩy |
| `.` | Thêm dấu chấm |
| `?` | Thêm dấu hỏi |
| `!` | Thêm dấu cảm |
| `Z` | Hoàn tác token hoặc dấu cuối |
| `X` | Xóa câu đang ghép, không xóa câu đã chốt |
| `T` | Phát âm câu đang ghép hoặc câu gần nhất |
| `Y` | Bật/tắt tự đọc khi chốt câu |
| `E` | Xuất transcript UTF-8 |
| `C` | Reset token Phase09 và câu đang ghép |
| `Q` | Lưu phiên & thoát |

Vẫn giữ các điều khiển Phase09. Nếu dấu `?`/`!` không được OpenCV bắt trên bàn phím quốc tế, kiểm tra `cv2.waitKey` layout; dấu `,` và `.` dễ dùng hơn.

## 5. Dữ liệu đầu ra

```text
logs/phase10_sessions/<session_id>/
├── events.jsonl
├── sentences.jsonl
├── transcript.txt
└── session_snapshot.json
logs/phase10_audio/
└── phase10_<random>.wav
```

`REST` dùng phục vụ phân đoạn Phase09, không mặc định chốt câu; `UNKNOWN` không phát thành chữ. TTS thất bại sẽ hiển thị trạng thái nhưng **văn bản không mất**.

## 6. Kiểm thử hồi quy

```powershell
python -m pytest -q tests/test_phase10.py tests/test_phase10_replay.py tests/test_phase10_integration_contract.py tests/test_phase10_1_2.py
```

Kiểm tra thêm Phase09:

```powershell
python -m pytest -q tests/test_phase9.py tests/test_phase9_live_logic.py tests/test_phase09_v3_1_launcher.py
```

Nếu kiểm thử nào không tồn tại trong bản source bạn đang dùng, xem `tests/` và chạy các file có sẵn. Trong môi trường tạo patch, logic đã được kiểm thử bằng mocks; **chưa có giọng Piper tải về hoặc webcam thật trong container**, nên cần chạy hai bài kiểm tra TTS và camera ở trên trên Windows để nghiệm thu.

## 7. Tiêu chí nghiệm thu & phát triển tiếp

- TCN V3.1 checkpoint đúng `32×513`; xác nhận signer split, không có duplicate xuyên VAL/TEST.
- Phase09 xác nhận token đúng, `WAIT_RELEASE` không bị kẹt, không phát lặp.
- Phase10.1 chấm câu đúng, giữ nhãn lặp, không thêm chữ cho `UNKNOWN/REST`, undo không lỗi.
- Phase10.2 phát giọng Việt **hoàn toàn offline sau khi cài**, độ trễ đo thực tế và WAV được lưu.
- Mở rộng thêm class thì tạo quy trình data collection → train → eval và **không được giả vờ đã nhận ký hiệu chỉ vì thêm trong YAML**.
- Kế tiếp Phase11: Web UI qua IP Raspberry Pi, và Phase12: tối ưu TensorFlow Lite/ONNX trên Pi. Piper ONNX có thể phù hợp thiết bị nhúng nhưng cần đo CPU/RAM thực tế.

## Tham khảo bên ngoài

- Piper (maintained): https://github.com/OHF-Voice/piper1-gpl
- Voice `vi_VN-vais1000-medium` được liệt kê tại https://huggingface.co/rhasspy/piper-voices/blob/main/voices.json
- Kho voice trực tiếp: https://huggingface.co/rhasspy/piper-voices/tree/main/vi/vi_VN/vais1000/medium
