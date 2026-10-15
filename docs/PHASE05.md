# PHASE 05 — Geometric + Motion Features

## 1. Tóm tắt ngắn gọn

Phase 04 mới mô tả hình dạng bàn tay ở từng frame. Phase 05 bổ sung thông tin hình học và chuyển động để phục vụ dynamic sign recognition.

```text
Phase-4 normalized XYZ
        ↓
Joint angles
Distances
Velocity
Acceleration
        ↓
Ablation-ready feature sets
        ↓
Feature extractor V2
```

## 2. Joint angle

Với ba điểm A-B-C, góc tại B:

```text
u = A - B
v = C - B

theta = arccos(
    (u · v) / (||u|| ||v||)
)
```

Code clip cosine về [-1,1] trước arccos để tránh lỗi floating point.

Phase 05 dùng 15 góc / tay:
- thumb: 3
- index: 3
- middle: 3
- ring: 3
- pinky: 3

Hai tay:

```text
15 × 2 = 30 angle features / frame
```

Mặc định góc được chia cho pi để nằm gần [0,1].

## 3. Distance

9 distance features:

```text
0 left thumb tip ↔ index tip
1 right thumb tip ↔ index tip
2 left wrist ↔ right wrist
3 left hand ↔ face
4 right hand ↔ face
5 left hand ↔ left shoulder
6 right hand ↔ right shoulder
7 left hand ↔ chest center
8 right hand ↔ chest center
```

Hai thumb-index distances được tính từ normalized hand coordinates.

Left-right wrist không thể tính từ Phase-4 local wrist coordinates vì cả hai wrist đã trở về 0. Vì vậy dùng raw wrist positions và chia cho mean hand scale.

Các body-relative distances cần Pose:
- face proxy = pose nose
- shoulder = MediaPipe Pose 11/12
- chest = midpoint(left shoulder, right shoulder)
- scale body = shoulder width

Nếu raw pose chưa có, 6 body distances giữ zero và `distance_mask=false`. Không tạo NaN/Inf. Phần pose/face đầy đủ vẫn được phát triển tiếp ở Phase 06.

## 4. Velocity

Theo roadmap:

```text
v_t = X_t - X_(t-1)
```

X là `normalized_xyz126`.

Quan trọng: velocity của một tay chỉ hợp lệ khi tay đó tồn tại ở cả frame `t-1` và `t`. Không nối chuyển động xuyên qua missing frame.

Frame đầu:

```text
v_0 = 0
velocity_valid = false
```

## 5. Acceleration

```text
a_t = v_t - v_(t-1)
```

Acceleration chỉ hợp lệ khi hai bước velocity liên tiếp đều hợp lệ.

```text
a_0 = 0
a_1 = 0
```

## 6. Clipping

Detector jump có thể sinh giá trị cực lớn. Config:

```yaml
velocity_clip_abs: 5.0
acceleration_clip_abs: 8.0
```

Clipping không thay thế QC; nó chỉ bảo vệ feature extractor khỏi outlier số học.

## 7. Feature dimensions

```text
Position      = 126
Angle         = 30
Distance      = 9
Velocity      = 126
Acceleration  = 126
--------------------------------
FULL          = 417 / frame
```

## 8. Ablation sets

Roadmap yêu cầu chuẩn bị ablation:

```text
P       = Position
P+V     = Position + Velocity
P+V+A   = Position + Velocity + Acceleration
FULL    = Position + Angle + Distance + Velocity + Acceleration
```

Shape:

```text
P       : (T,126)
P+V     : (T,252)
P+V+A   : (T,378)
FULL    : (T,417)
```

Các array được lưu:
- `ablation_position126`
- `ablation_position_velocity252`
- `ablation_position_velocity_acceleration378`
- `full_features417`

## 9. Output

```text
dataset/processed/features_phase5/<sample_id>/
├── features_v2.npz
└── feature_metadata.json
```

Global:
- `feature_manifest.csv`
- `feature_summary.json`
- `feature_audit.json`

## 10. Chạy Phase 05

```powershell
pytest -q tests/test_phase5_features.py
python scripts/run_phase5_features.py
python scripts/audit_phase5.py
```

## 11. Exit Gate

PASS khi:
- Phase-4 PASS samples = Phase-5 rows
- FAIL = 0
- Missing = 0
- Extra = 0
- Split mismatch = 0
- Feature errors = 0
- all required arrays đúng shape
- không NaN/Inf
- velocity không bridge qua missing hand
- có ablation-ready feature sets
- group feature có thể bật/tắt bằng config metadata

## 12. Lưu ý về Phase 06

Roadmap có hand-face, hand-shoulder, hand-chest ở Phase 05 nhưng Phase 06 mới là phase chuyên xử lý Pose + Face. Vì vậy implementation này hỗ trợ body-distance ngay nếu raw pose tồn tại; nếu chưa có, nó dùng zero + mask false và không làm Phase 05 fail. Phase 06 sẽ bổ sung Pose/Face feature đầy đủ.
