# PHASE 07 REALTIME DECISION ENGINE V2

## Vấn đề đang gặp

TCN offline đạt kết quả tốt, nhưng realtime trước đây dùng rolling window liên tục:

```text
32 frame -> predict
frame mới -> predict lại
frame mới -> predict lại
...
```

Model Phase 07 chỉ biết các class known nên mọi đoạn chuyển động, kể cả chưa hoàn thành ký hiệu, vẫn bị ép thành một class.

Ví dụ:

```text
REST -> giơ tay -> mới làm 30% ký hiệu
```

vẫn có thể bị dự đoán `xin_chao`, `cam_on`, ...

## Fix mới

Pipeline mới:

```text
Camera
  ↓
MediaPipe Hands + Pose + Face
  ↓
Motion segmentation
  ↓
REST
  ↓
START
  ↓
ACTIVE
  ↓
END
  ↓
Phase04 -> Phase05 -> Phase06
  ↓
Temporal resample/pad 32 frames
  ↓
Frozen TRAIN scaler
  ↓
TCN V1
  ↓
Confidence gate
  ↓
Top1-Top2 margin gate
  ↓
4/5 temporal preview stabilizer
  ↓
KNOWN / UNKNOWN
  ↓
COOLDOWN
  ↓
REST
```

## 1. Motion segmentation

Không dùng Phase04 normalized hand motion để detect START/END vì Phase04 đã trừ wrist origin, làm mất chuyển động tịnh tiến của bàn tay.

Decision Engine dùng raw normalized-image XY coordinates:

```text
motion =
max(
  mean hand-landmark displacement,
  upper-body pose displacement
)
```

Threshold ban đầu:

```text
START >= 0.012
END   <= 0.004
```

START phải xuất hiện liên tục:

```text
3 frames
```

END cần đứng yên:

```text
6 frames
```

## 2. Pre-roll

Giữ 4 frame trước START để không cắt mất onset của ký hiệu.

```text
pre_roll_frames = 4
```

## 3. Confidence gate

Model chỉ được xem là known sign khi:

```text
top1 >= 0.85
```

Nếu thấp hơn:

```text
UNKNOWN
```

## 4. Margin gate

Ngoài confidence còn yêu cầu:

```text
top1 - top2 >= 0.20
```

Ví dụ:

```text
xin_chao = 0.91
cam_on   = 0.04
margin   = 0.87
PASS
```

nhưng:

```text
xin_chao = 0.49
cam_on   = 0.40
margin   = 0.09
UNKNOWN
```

## 5. Stabilizer

Trong ACTIVE, script chạy preview prediction theo chu kỳ:

```text
12, 16, 20, 24, 28, ...
```

Chỉ xác nhận nếu ít nhất:

```text
4/5 preview gần nhất
```

cùng với final prediction.

Điều này giảm trường hợp một cửa sổ tạm thời nhảy class.

## 6. Hand-presence ratio

Segment phải có tay ở đủ số frame:

```text
min_hand_presence_ratio = 0.45
```

Mục đích tránh classify các đoạn chỉ có người di chuyển thân/đầu.

## 7. Cooldown

Sau khi phát một sign:

```text
cooldown = 0.9 s
```

và phải quay về REST/stillness trước khi sign tiếp theo được phép emit.

Điều này ngăn:

```text
xin_chao
xin_chao
xin_chao
xin_chao
```

bị phát liên tục chỉ vì người vẫn giữ nguyên tư thế.

## 8. UNKNOWN hiện tại

`UNKNOWN` trong V2 là decision-layer UNKNOWN:

```text
confidence fail
OR
margin fail
OR
stability fail
OR
hand presence fail
```

Đây chưa phải Phase08 UNKNOWN model hoàn chỉnh, nhưng rất phù hợp để ngăn Phase07 ép mọi gesture thành known class.

## 9. Cài file

Copy:

```text
configs/realtime_phase7.yaml
core/realtime_decision_phase7.py
scripts/live_tcn_decision_v2.py
tests/test_realtime_decision_phase7.py
```

vào project.

## 10. Test unit

```powershell
pytest -q tests/test_realtime_decision_phase7.py
```

Kỳ vọng:

```text
9 passed
```

## 11. Chạy realtime

```powershell
python scripts/live_tcn_decision_v2.py
```

Fullscreen:

```powershell
python scripts/live_tcn_decision_v2.py --fullscreen
```

## 12. Màn hình

Hiển thị:

```text
STATE: REST / ACTIVE / COOLDOWN

Motion
segment frame count
FPS

CANDIDATE
confidence
margin

CONFIRMED
known sign / UNKNOWN

L-HAND
R-HAND
POSE
FACE
```

## 13. Test đúng thứ tự

### Test A — không làm gì

```text
STATE = REST
CONFIRMED không tự nhảy known sign
```

### Test B — chỉ giơ tay rồi dừng

Kỳ vọng:

```text
START/ACTIVE có thể xảy ra
nhưng confidence/margin/stability phải hạn chế false emit
```

Nếu vẫn emit known sign, lưu false case để Phase08/data V2 xử lý.

### Test C — làm 30–50% ký hiệu rồi dừng

Kỳ vọng ưu tiên:

```text
UNKNOWN
```

### Test D — làm full known sign

Kỳ vọng:

```text
REST
-> START
-> ACTIVE
-> END
-> CONFIRMED known sign
-> COOLDOWN
-> REST
```

### Test E — giữ nguyên ký hiệu sau khi nhận

Không được phát cùng sign liên tục.

### Test F — sign thứ hai

Phải trở lại REST trước rồi mới START lại.

## 14. Threshold tuning

Các giá trị hiện tại là điểm khởi đầu:

```yaml
min_confidence: 0.85
min_top1_margin: 0.20
start_motion_threshold: 0.012
end_motion_threshold: 0.004
stable_required: 4
stable_window_count: 5
cooldown_seconds: 0.90
```

Không tune bằng locked TEST.

Tune bằng:
- live trials;
- TRAIN/VAL subjects;
- false-prediction log.

## 15. Khi nào mới train TCN V2?

Chỉ sau khi decision engine này đã được test.

Nếu lỗi còn lặp lại rõ theo class, ví dụ:

```text
partial cam_on -> xin_loi
```

thì thu targeted negatives/positives rồi train V2.

Không nên train V2 chỉ để che lỗi rolling-window decision layer.
