# PHASE 06 — Upper-body Pose + Face Features

## 1. Mục tiêu

Phase 06 bổ sung thông tin mà hand landmarks đơn thuần không biểu diễn tốt:

- vị trí tay so với đầu, vai và thân người;
- tư thế upper-body;
- độ mở miệng;
- độ mở mắt;
- độ cao lông mày;
- proxy hướng đầu.

Roadmap yêu cầu không đưa toàn bộ hàng trăm face landmark vào model. Thay vào đó dùng compact meaningful features.

## 2. Pipeline

```text
Phase 05 hands feature 417
           +
Raw Pose landmarks
           +
Raw Face landmarks
           ↓
Compact Pose 27
Compact Face 9
           ↓
Ablation sets
Hands
Hands + Pose
Hands + Pose + Face
```

## 3. Pose feature — 27 chiều

7 upper-body points:

```text
nose
left shoulder
right shoulder
left elbow
right elbow
left wrist
right wrist
```

Mỗi point có XYZ:

```text
7 × 3 = 21
```

Thêm 6 relation distances:

```text
left/right wrist -> nose
left/right wrist -> corresponding shoulder
left/right wrist -> chest center
```

Tổng:

```text
21 + 6 = 27 pose features
```

## 4. Pose normalization

Gốc:

```text
chest = (left_shoulder + right_shoulder) / 2
```

Scale:

```text
shoulder_width =
||left_shoulder - right_shoulder||_2
```

Với point p:

```text
p_norm =
(p - chest) / shoulder_width
```

Nhờ đó pose giảm phụ thuộc vào vị trí người trong ảnh và khoảng cách camera.

## 5. Compact Face — 9 chiều

```text
mouth_open
mouth_width
left_eye_open
right_eye_open
left_brow_height
right_brow_height
head_yaw_proxy
head_pitch_proxy
head_roll_proxy
```

Face scale:

```text
face_scale =
||left_eye_outer - right_eye_outer||_2
```

Các distance expression được chia cho face_scale.

Ví dụ:

```text
mouth_open =
||upper_lip - lower_lip|| / face_scale
```

```text
mouth_width =
||mouth_left - mouth_right|| / face_scale
```

## 6. Head orientation proxy

Đây là proxy nhẹ, không phải head pose PnP đầy đủ.

```text
eye_mid =
(left_eye_outer + right_eye_outer) / 2
```

Yaw proxy:

```text
yaw =
(nose_x - eye_mid_x) / face_scale
```

Pitch proxy:

```text
pitch =
(nose_y - eye_mid_y) / face_scale
```

Roll proxy:

```text
roll =
atan2(
  right_eye_y - left_eye_y,
  right_eye_x - left_eye_x
) / pi
```

Nếu sau này cần head orientation chính xác hơn có thể thay bằng solvePnP, nhưng bản này phù hợp cho compact edge feature.

## 7. Missing modality

Nếu Pose hoặc Face không tồn tại:

```text
feature = 0
mask = false
```

Không tạo NaN/Inf và không làm mất sample hands.

Điều này đặc biệt quan trọng vì dataset hiện tại có thể có `pose_present` / `face_present` nhưng chưa chắc raw landmark arrays đã được lưu đầy đủ.

## 8. Feature dimensions

Input từ Phase 05:

```text
Hands FULL = 417
```

Pose:

```text
27
```

Face:

```text
9
```

Ablation:

```text
Hands only
= 417

Hands + Pose
= 417 + 27
= 444

Hands + Pose + Face
= 417 + 27 + 9
= 453
```

## 9. Ablation

Các array:

```text
ablation_hands417
ablation_hands_pose444
ablation_hands_pose_face453
```

Sau này dùng cùng model và cùng split để so sánh:

```text
Hands
vs
Hands + Pose
vs
Hands + Pose + Face
```

Đây là phần có giá trị nghiên cứu vì chứng minh Pose/Face có thực sự cải thiện recognition hay không.

## 10. Raw array aliases hỗ trợ

Pose:

```text
pose
pose_landmarks
pose_xyz
```

Face:

```text
face
face_landmarks
face_xyz
```

Mask:

```text
pose_present
face_present
```

## 11. Output

```text
dataset/processed/features_phase6/
├── feature_manifest.csv
├── feature_summary.json
├── feature_audit.json
└── <sample_id>/
    ├── features_v3.npz
    └── feature_metadata.json
```

## 12. Coverage

Audit tách hai khái niệm:

### Structural PASS

Feature file đúng shape, không NaN/Inf, không missing sample.

### Modality coverage

```text
Pose coverage
Face coverage
```

Nếu raw Pose/Face chưa được lưu thì coverage có thể là 0 mặc dù structural pipeline PASS.

Không nên tuyên bố Pose/Face đã có giá trị nghiên cứu nếu coverage thực tế bằng 0.

## 13. Chạy

```powershell
pytest -q tests/test_phase6_features.py
python scripts/run_phase6_features.py
python scripts/audit_phase6.py
```

## 14. Exit Gate

Phase 06 PASS về pipeline khi:

- Phase-5 PASS samples = Phase-6 rows
- FAIL = 0
- Missing = 0
- Extra = 0
- Split mismatch = 0
- Feature errors = 0
- không NaN/Inf
- hands/pose/face ablation arrays đúng shape

Đồng thời phải xem:

```text
Pose coverage = ?
Face coverage = ?
```

Nếu coverage quá thấp, cần quay lại collector/dataset để lưu raw Pose/Face tốt hơn trước khi làm experiment cuối.

## 15. Sau Phase 06

Theo roadmap:

```text
Phase 07
Temporal Model

GRU chính
TCN so sánh
sequence classification thật sự
```
