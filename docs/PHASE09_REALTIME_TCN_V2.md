# PHASE 09 — Nhận diện ký hiệu liên tiếp với TCN V2 (5 class)

## 1. Mục tiêu

Tích hợp checkpoint đã train tại `artifacts/experiments/tcn_v2_unknown/model.keras` vào webcam **theo thời gian thực**, xác nhận token khi nhận diện nhất quán, chống lặp trong lúc người dùng giữ nguyên ký hiệu, cho phép lặp lại cùng ký hiệu sau khi có `REST`, đồng thời lưu nhật ký để phân tích lỗi.

Bản này là **tích hợp Phase 09 với model Phase 08 V2 đã train**. Không thay đổi checkpoint, không train lại, không tự tuyên bố UNKNOWN đã đủ chính xác.

## 2. Kiến trúc

```text
Camera (raw frame, KHÔNG flip trước inference)
  ↓
MediaPipe Hands / Pose / Face
  ↓
45-frame rolling window, cứ 8 frame infer một lần
  ↓
Phase04 (126) + Phase05 (417) + Phase06 (453)
  ↓
Phase07 resample 32 + scaler đóng băng của V2
  ↓
TCN V2 → [REST, cam_on, xin_chao, xin_loi, UNKNOWN]
  ↓
Decision gate (confidence, margin, hand presence)
  ↓
Phase09 State Machine:
  IDLE → CANDIDATE → STABLE → WAIT_RELEASE
  ↓
Confirmed token stream + JSONL log
```

**Quan trọng:** Mỗi cập nhật cho State Machine tương ứng **một inference temporal window**; không phải một frame camera. `min_consecutive=3` là ba cửa sổ dự đoán, không phải 3 frame.

## 3. Cài đặt

Giải nén ZIP vào root project, ví dụ:

`D:\KLTN_KTMT\KLTN_Sign\Sign-Language-Recognition-Phase0`

Các file trong ZIP:

```text
configs/phase9_live.yaml
core/continuous_phase9.py
core/phase9_live_logic.py
core/phase8_live.py             # helper Phase08 giống bản V2; có thể giữ bản cũ nếu giống
scripts/live_phase09_tcn_v2.py
scripts/audit_phase09_setup.py
scripts/analyze_phase09_live.py
scripts/make_phase9_demo.py
scripts/run_phase9_stream.py
tests/test_phase9.py
tests/test_phase9_live_logic.py
docs/PHASE09_REALTIME_TCN_V2.md
```

Cần có các dependency Phase04/05/06/07 hiện tại của bạn, `tensorflow`, `mediapipe==0.10.21`, `opencv-python`, `numpy`, `pyyaml` và `pytest`. Không cần copy lại source các phase cũ.

## 4. Chạy VS Code / PowerShell

**Bước 1 — kiểm tra model/scaler/labels 5 class:**

```powershell
python scripts/audit_phase09_setup.py
```

Cần thấy `PHASE09 SETUP: PASS`. Nếu thiếu `model.keras`, bạn chưa train V2 thành công; không chạy model V1 thay thế.

**Bước 2 — unit tests:**

```powershell
pytest -q tests/test_phase9.py tests/test_phase9_live_logic.py
```

**Bước 3 — webcam thường hoặc fullscreen:**

```powershell
python scripts/live_phase09_tcn_v2.py
python scripts/live_phase09_tcn_v2.py --fullscreen
```

Nếu camera không mở:

```powershell
python scripts/live_phase09_tcn_v2.py --camera 1
```

Nếu máy chậm:

```powershell
python scripts/live_phase09_tcn_v2.py --cam-width 640 --cam-height 480 --stride 12
```

Tuning gate thử nghiệm:

```powershell
python scripts/live_phase09_tcn_v2.py --min-confidence 0.90 --stride 6
```

**Cảnh báo**: 0.80 mặc định là ngưỡng thăm dò cho TCN V2 5-class, KHÔNG phải threshold Phase08 V1 cũ 0.90 và chưa phải ngưỡng được calibrate khoa học. Nếu model V2 cho softmax gần 1.0 với gesture lạ, threshold đơn lẻ vẫn không bảo đảm loại được UNKNOWN.

## 5. Phím điều khiển

| Phím | Chức năng |
|---|---|
| F | Fullscreen / cửa sổ bình thường |
| V | Bật/tắt landmark trái/phải/pose/face |
| M | Mirror preview (không thay dữ liệu model) |
| P | Pause/resume inference, xóa buffer chưa hoàn chỉnh |
| C | Xóa token trong bộ nhớ, reset FSM; không xóa log |
| R | Reset rolling buffer, giữ token cũ |
| B | **Thả REST thủ công để debug**, không dùng làm chỉ số đánh giá tự động |
| A | Bật/tắt gợi ý release nếu không thấy tay đủ lâu |
| S | Chụp screenshot giao diện |
| Q / ESC | Thoát |

## 6. Quy tắc token, release, UNKNOWN

- `REST` không được emit ra token, mà dùng mở khóa `WAIT_RELEASE` sau 3 inference windows có REST.
- `UNKNOWN` không được emit và **không tự coi là REST**.
- Nếu không phát hiện bất kỳ bàn tay nào đủ `12` camera frames, hệ thống có thể phát tín hiệu release heuristic; dòng log có `source=NO_HAND_RELEASE`, không phải dự đoán chắc chắn của model. Điều này giúp lặp lại ký hiệu khi tay thả khỏi khung nhưng có nguy cơ false release khi tracking lỗi; bấm A để tắt khi kiểm định nghiêm ngặt.
- Tín hiệu `B` thả thủ công chỉ để gỡ lỗi. Không sử dụng số liệu thử nghiệm có can thiệp B để báo cáo mô hình độc lập.
- 3 lần dự đoán giống nhau không chứng minh ký hiệu đã đủ chuyển động. Sliding window có thể xác nhận **partial gesture** nếu model học sai. Để tốt hơn, cần Phase07 segment start/end hoặc data negative bổ sung.

## 7. Cách test realtime đúng

| Test | Thao tác | Kỳ vọng |
|---|---|---|
| A | Đứng yên / tay hạ | Không tự emit token |
| B | Làm trọn `xin_chao` | Chỉ emit đúng 1 `xin_chao` |
| C | Giữ nguyên tay sau sign | Không lặp liên tục |
| D | `xin_chao` → REST (thả tay) → `xin_chao` | Emit 2 token `xin_chao` |
| E | `xin_chao` → REST → `cam_on` | Emit 2 token theo đúng thứ tự |
| F | Chạm tóc / cầm điện thoại / chuyển động lạ | Không emit known token khi đó là UNKNOWN |
| G | Làm 30–50% một sign | Không emit token (cần kiểm tra, không được giả định) |

**Đánh giá khách quan:** Thu ít nhất vài chục trial/phép thử, ghi nhãn/truth và timestamps rồi tính false token rate, duplicate rate, recall known, unknown false accept. Mặc định log dự đoán và token **không có nhãn thật**, vì vậy script không tự in ra accuracy giả.

## 8. File đầu ra

```text
logs/phase09_live_predictions.csv   # Mỗi temporal inference và release heuristic
logs/phase09_live_tokens.jsonl      # Token đã CONFIRMED
logs/phase09_live_snapshots/        # Ảnh chụp (S)
```

Dùng:

```powershell
python scripts/analyze_phase09_live.py
```

Sẽ tạo `logs/phase09_live_summary.json`: số dự đoán, nhãn, token, release thủ công, release do không thấy tay. Không tự tính accuracy khi không có ground truth.

## 9. Lỗi thường gặp

1. **Missing model.keras / stats / labels** → chạy train TCN V2 và kiểm tra `artifacts/experiments/tcn_v2_unknown/`.
2. **MediaPipe không có `solutions`** → dùng env tương thích (trước đó bạn có MediaPipe `0.10.21`), chạy `python -m pip check`.
3. **Cửa sổ không mở** → thử `--camera 1`, đóng Teams/Zoom đang giữ webcam.
4. **Tay giữ nguyên nhưng dự đoán lặp** → kiểm tra FSM `WAIT_RELEASE`, log `NO_HAND_RELEASE`, tăng thời gian release nếu MediaPipe mất tracking.
5. **Không có token** → kiểm tra `buffer`, confidence/margin, `min_consecutive`, FPS, thử giảm `--min-confidence 0.75` để chẩn đoán; không báo Accuracy dựa trên chế độ này.
6. **UNKNOWN vẫn thành `xin_chao`** → do V2 chỉ học rất ít UNKNOWN; cần cải thiện data và đánh giá độc lập, không chỉ tăng threshold.

## 10. Trạng thái hoàn thành

- Mã tích hợp Phase09: đã viết và unit test.
- Webcam Windows: **người dùng cần chạy**.
- TCN V2 5-class: đã được người dùng train, nhưng dữ liệu UNKNOWN còn ít, tổng quát hóa chưa được chứng minh.
- Phase 09 COMPLETE: chỉ khi test camera thực tế, có log và số liệu chống lặp, false tokens, timing.

Sau khi PASS, mới tiếp tục Phase10 Sentence Builder + TTS (không lấy `REST` hoặc `UNKNOWN` làm từ trong câu).
