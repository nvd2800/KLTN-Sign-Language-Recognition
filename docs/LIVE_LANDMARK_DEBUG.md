# LIVE LANDMARK DEBUG — Phase 04/05/06

## Mục tiêu

Script này dùng webcam để xác nhận hệ thống thực sự nhận đúng:

- 21 landmark tay trái/phải
- Pose upper-body
- Face Mesh
- Phase 04 hand normalization
- Phase 05 angle / distance / velocity / acceleration
- Phase 06 pose / face compact features
- missing detection
- FPS
- mirror preview không làm đảo semantic Left/Right

## Cài thư viện

Trong virtual environment:

```powershell
pip install opencv-python mediapipe numpy
```

## Chạy

```powershell
python scripts/live_landmark_debug.py
```

Nếu camera mặc định không đúng:

```powershell
python scripts/live_landmark_debug.py --camera 1
```

Có thể chọn resolution:

```powershell
python scripts/live_landmark_debug.py --width 1280 --height 720
```

## Phím điều khiển

```text
1 = Hands only
2 = Hands + Pose
3 = Hands + Pose + Face
4 = bật/tắt bảng feature values
M = bật/tắt mirror preview
S = lưu ảnh + JSON metrics
R = reset history velocity/acceleration
Q hoặc ESC = thoát
```

## Test Phase 04

### Test dịch vị trí

Giữ nguyên handshape và di chuyển tay sang trái/phải.

Kỳ vọng:
- skeleton vẫn bám đúng;
- hand scale hợp lệ;
- angle_mean và thumb-index normalized không đổi quá mạnh.

### Test gần/xa camera

Giữ handshape, tiến tay gần camera rồi đưa xa.

Kỳ vọng:
- raw landmark thay đổi;
- normalized geometry tương đối ổn định.

### Test Left/Right

Đưa riêng từng tay.

Kỳ vọng:
- tay trái thật -> LEFT HAND = YES;
- tay phải thật -> RIGHT HAND = YES.

Mirror chỉ áp dụng cho preview sau inference.

## Test Phase 05

### Angle

Co / duỗi các ngón.

Kỳ vọng:

```text
angle_mean thay đổi
```

### Thumb-index distance

Chụm ngón cái và ngón trỏ:

```text
thumb-index giảm
```

Tách hai ngón:

```text
thumb-index tăng
```

### Velocity

Giữ tay đứng yên:

```text
velocity ≈ 0
```

Di chuyển tay:

```text
velocity tăng
```

### Acceleration

Di chuyển đều:

```text
acceleration tương đối nhỏ
```

Vung tay nhanh hoặc dừng đột ngột:

```text
acceleration thay đổi rõ
```

### Missing frame

Che tay hoặc đưa tay ra khỏi khung.

Kỳ vọng:
- HAND = NO;
- history không nối velocity xuyên qua missing frame;
- khi tay xuất hiện lại cần tối thiểu frame kế tiếp để velocity hợp lệ.

## Test Phase 06 — Pose

Mode:

```text
2 hoặc 3
```

Kỳ vọng skeleton:
- nose đúng đầu;
- shoulders đúng hai vai;
- elbows đúng khuỷu;
- wrists đúng cổ tay.

Test vị trí:
- đưa tay lên gần mặt -> wrist-to-nose giảm;
- đưa tay về ngực -> wrist-to-chest giảm;
- đưa tay ra xa thân -> wrist-to-chest tăng.

Đứng gần/xa camera:
- normalized pose relation nên ổn định tương đối.

## Test Phase 06 — Face

Mode:

```text
3
```

### Há miệng

Kỳ vọng:

```text
mouth_open tăng
```

### Cười / kéo ngang miệng

```text
mouth_width tăng
```

### Nhắm mắt

```text
eye_open giảm
```

### Nhướng mày

```text
brow_height tăng
```

### Quay đầu trái/phải

```text
yaw_proxy thay đổi
```

### Cúi/ngẩng

```text
pitch_proxy thay đổi
```

### Nghiêng đầu

```text
roll_proxy thay đổi
```

Lưu ý: yaw/pitch/roll là lightweight proxies, không phải head pose solvePnP chính xác.

## Stress Test

Nên thử:

```text
một tay
hai tay
hai tay chồng nhau
tay ra ngoài frame
tay che mặt
quay mặt
cúi đầu
ánh sáng yếu
đứng gần camera
đứng xa camera
di chuyển nhanh
```

Không được:
- crash
- NaN
- Inf
- Left/Right đảo semantic
- velocity nối qua missing hand

## Snapshot

Nhấn:

```text
S
```

sẽ tạo:

```text
debug_snapshots/
debug_YYYYMMDD_HHMMSS.jpg
debug_YYYYMMDD_HHMMSS.json
```

JSON ghi các giá trị feature tại thời điểm test để dùng làm bằng chứng debug / báo cáo.

## Tiêu chí PASS trước Phase 07

### Hands

```text
21 landmarks bám đúng tay
Left/Right đúng
normalization hợp lý
```

### Phase 05

```text
angle phản ứng với co/duỗi
thumb-index phản ứng với chụm/tách
velocity gần 0 khi đứng yên
velocity tăng khi di chuyển
acceleration phản ứng khi thay đổi tốc độ
```

### Phase 06

```text
Pose skeleton bám đúng
Face mesh bám đúng
mouth/eye/brow phản ứng đúng
yaw/pitch/roll proxy phản ứng đúng
```

### Stability

```text
không crash
không NaN/Inf
missing được phát hiện
FPS chấp nhận được
```

Chỉ sau khi audit offline PASS và live webcam test hợp lý mới nên bắt đầu Phase 07 Temporal Model.
