# PHASE 10 – Sentence Builder & Vietnamese Text-to-Speech

**Đề tài:** Nhận diện ngôn ngữ ký hiệu tích hợp AI trên Raspberry Pi  
**Đầu vào:** token đã được Phase09 xác nhận bằng `ContinuousRecognizer`, không phải frame prediction.  
**Checkpoint kế thừa:** `artifacts/experiments/tcn_v3_1_xin_loi_20261028/model.keras` (32 × 513, 5 nhãn).  
**Triển khai:** source patch cho project gốc, không thay checkpoint, scaler, labels, dataset.

## 1. Công việc đã triển khai và phạm vi

- Core `SentenceBuilder`: `xin_chao` → `xin chào`, `cam_on` → `cảm ơn`, `xin_loi` → `xin lỗi`; hỗ trợ nhiều token/câu, sửa token cuối, xóa câu đang ghép, chốt câu. Không tự suy đoán ngữ pháp và không biến token thành câu vượt quá vốn từ hiện tại.
- REST/UNKNOWN bỏ qua, không được thêm vào văn bản hoặc phát âm; giữ token lặp hợp lệ khi Phase09 thực sự phát sau lần release.
- Lưu `events.jsonl`, `sentences.jsonl`, `session_snapshot.json`, `transcript.txt` riêng theo từng phiên. Không chạm vào Phase09 predictions/token logs.
- TTS tiếng Việt **tùy chọn và không chặn camera**: dùng Windows SAPI với **giọng vi-VN đã cài đặt** hoặc eSpeak NG có `vi`; nếu thiếu giọng tương thích, vẫn lưu văn bản, không đọc bằng giọng tiếng Anh và không tải dịch vụ từ mạng.
- UI Phase10 hiển thị bản nháp, câu cuối, trạng thái TTS; nếu Pillow chưa có thì hiển thị chữ không dấu trong overlay OpenCV, bản ghi vẫn có đủ dấu.

> **Chưa làm**: sửa ngữ pháp tự động bằng LLM, nhận diện câu ký hiệu liên tục không cần ngắt, streaming web, Raspberry Pi deployment, công nhận PASS theo kết quả webcam thật. Phase10 dùng đúng vốn từ hiện tại 3 ký hiệu + REST/UNKNOWN.

## 2. Cài vào source hiện hành

Giải nén ZIP vào thư mục project `D:\KLTN_KTMT\KLTN_Sign\Sign-Language-Recognition-Phase0` sao cho `core/`, `scripts/`, `configs/`, `tests/`, `docs/` cùng cấp.  
**Sao lưu file `scripts/live_phase09_v3_spatial.py` trước khi chép đè**. Bản Phase09 cũ vẫn chạy độc lập nếu không truyền `--phase10-config`.

Từ PowerShell trong VS Code:

```powershell
conda activate venv
python -m pip install Pillow
python -m pytest -q tests/test_phase10.py tests/test_phase10_replay.py
python scripts/audit_phase10_setup.py --tts-check
```

`Pillow` chỉ hỗ trợ hiển thị dấu tiếng Việt trong webcam, không phải một engine TTS. TensorFlow, MediaPipe, OpenCV, PyYAML kế thừa từ Phase09.

## 3. Kiểm tra model, rồi chạy camera

```powershell
python scripts/live_phase10_v3_1.py --dry-run --check-model
python scripts/live_phase10_v3_1.py --fullscreen
```

Config V3.1 mặc định là `configs/phase9_v3_Final.yaml`, với checkpoint `tcn_v3_1_xin_loi_20261028` và bộ dataset `temporal_v3_1_spatial`. Kiểm tra `PREFLIGHT: PASS` trước khi mở webcam. Nếu checkpoint khác, sửa 3 đường dẫn tại `configs/phase9_v3_Final.yaml` hoặc truyền `--config` khác; không lấy scaler/labels từ model khác. 

Chạy thử từng gesture có nhãn:

```powershell
python scripts/live_phase10_v3_1.py --fullscreen --target-label xin_chao --trial-isolation
```

`--trial-isolation` chỉ dành cho test từng động tác, có thể khiến chuỗi token bị ngắt theo trial. Khi ghép câu liên tục, bỏ nó.

## 4. Phím điều khiển Phase10

| Phím | Chức năng |
|---|---|
| `N` hoặc `Space` | Chốt câu hiện tại, tự thêm dấu chấm |
| `Z` | Hoàn tác token cuối của câu đang ghép; không xóa lịch sử token Phase09 |
| `X` | Xóa câu đang ghép; giữ các câu đã chốt |
| `T` | Đọc câu đang ghép, hoặc câu cuối cùng nếu câu đang ghép rỗng |
| `Y` | Bật/tắt tự đọc mỗi khi chốt câu (mặc định OFF) |
| `E` | Xuất lại transcript/snapshot thủ công |
| `C` | Reset token Phase09 và xóa câu đang ghép (vẫn giữ câu đã chốt) |
| `Q` | Thoát; tự chốt câu chưa chốt và xuất transcript |

Phím Phase09 giữ nguyên: `F` fullscreen, `V` landmarks, `M` mirror, `P` pause, `R` reset buffer, `B` REST thủ công (chỉ debug), `A` auto release, `S` snapshot, `0–6` đánh dấu trial.

### Lưu ý ngữ nghĩa ký hiệu

`xin_chao` + `cam_on` → `Xin chào cảm ơn.` là phép **nối cụm từ theo thứ tự token**, không phải một bộ dịch ngữ pháp tiếng Việt tự nhiên. REST chỉ kết thúc một cử chỉ để chống lặp; không tự động đóng câu. Người dùng quyết định ranh giới câu bằng `N`/`Space`.

## 5. Test không cần webcam bằng Phase09 token log

```powershell
python scripts/phase10_replay_tokens.py --input logs/phase09_v3_1_spatial_tokens.jsonl
```

Tùy chọn `--finalize-every 2` là chức năng minh họa/test tự động gom mỗi 2 token, **không phải** quy tắc ngữ pháp. Dữ liệu replay không sửa log Phase09 gốc, kết quả nằm trong `logs/phase10_sessions/<timestamp>_xxxxxx/`.

## 6. Cấu trúc bản ghi

```text
logs/phase10_sessions/
└── 20261028T123456Z_ab12cd/
    ├── events.jsonl             # accepted/undo/clear/tts/finalized
    ├── sentences.jsonl          # từng câu được chốt và token nguồn
    ├── transcript.txt           # văn bản UTF-8 để copy/chạy TTS
    └── session_snapshot.json    # bản chụp các câu và token đang ghép
```

Mỗi lần chốt câu sẽ ghi ngay disk; lúc thoát, cả câu đang ghép cũng được chốt. TTS yêu cầu voice có sẵn, hiển thị rõ `VOICE_UNAVAILABLE` nếu không tìm thấy.

## 7. Vì sao có thể chưa phát tiếng Việt?

`python scripts/audit_phase10_setup.py --tts-check` thông báo có giọng Việt trong máy hay không. Windows Speech SAPI không mặc định có giọng Việt trên mọi thiết bị. Nếu thiếu, có thể dùng eSpeak NG đã cài với `vi`, hoặc cài thêm gói tiếng nói Vietnamese cho Windows (nếu có hỗ trợ SAPI). Chương trình này không tự tải voice, không gửi văn bản lên dịch vụ ngoài.

Giọng tự nhiên như Vbee/KhanhTTS chưa được tích hợp vào gói này: nếu muốn chất lượng giọng tốt và hoàn toàn offline, cần cài model + runtime và kiểm tra tài nguyên máy/Pi riêng. Không tuyên bố có giọng Vbee từ Windows SAPI.

## 8. Test Phase10 trước khi nghiệm thu

1. `xin_chao` → `N` → xuất `Xin chào.`
2. `xin_chao` → REST → `xin_chao` → `N`: giữ 2 token (nếu Phase09 phát đủ 2 lần).
3. Giữ tay sai vị trí/handshape: TCN ra `UNKNOWN` → không thêm văn bản.
4. `cam_on` → `xin_loi` → `Z` → `N`: chỉ có `Cảm ơn.`.
5. Dùng `T` không có voice: không crash webcam, `VOICE_UNAVAILABLE`, văn bản vẫn lưu.
6. `Q` khi câu đang ghép chưa chốt: vẫn có transcript.
7. Chạy lại Phase09 **không có** `--phase10-config` không thay đổi lệnh/khả năng cũ.

## 9. Hướng phát triển tiếp theo

- **Phase10.1**: thêm vocabulary, mapping từ điển ký hiệu, dấu câu/chuẩn hóa câu theo quy tắc rõ ràng; tiêu chí người dùng chốt câu.
- **Phase10.2**: TTS tiếng Việt tự nhiên chạy offline (cần model runtime riêng) và kiểm tra tốc độ trên Windows/Pi.
- **Phase11 Web UI**: server Flask/FastAPI nhận event token, hiển thị realtime trên trình duyệt cùng Wi-Fi; thêm transcript lịch sử và nút phát TTS. Web không tự làm inference trừ khi server xử lý MediaPipe/TCN.
- **Phase12 Embedded Deployment**: đo FPS/latency/memory trên Raspberry Pi 4/5; tối ưu MediaPipe hoặc thay detector nhẹ nếu CPU quá tải; benchmark và so sánh baseline laptop.

## 10. Phạm vi bàn giao

Source patch không chứa checkpoint model, dataset raw, video hay font cài đặt. Chỉ chép đè `scripts/live_phase09_v3_spatial.py` trong project đã có Phase09 V3.1. Các unittest/CLI chạy không cần TensorFlow, nhưng webcam+giọng Việt phải xác nhận thực tế trên máy Windows của bạn.
