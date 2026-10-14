# PHASE 04 — Feature Engineering & Normalization

## Tóm gọn chương 4

Mục tiêu của Phase 04 là biến landmark bàn tay thô thành feature ổn định hơn trước khi học temporal model.

Pipeline:

```text
landmarks.npz
   ↓
XYZ 21 điểm / tay
   ↓
giữ LEFT / RIGHT cố định
   ↓
Translation normalization
wrist (landmark 0) → gốc tọa độ
   ↓
Scale normalization
||middle MCP (9) - wrist (0)|| → 1 đơn vị tham chiếu
   ↓
Missing/invalid hand → zero + presence mask
   ↓
LEFT 63 + RIGHT 63
   ↓
normalized_xyz126: (T, 126)
```

Phase 04 KHÔNG thêm velocity, acceleration, joint-angle hay distance nâng cao; các phần đó thuộc Phase 05.

## Công thức

Với landmark i:

```text
p'_i = p_i - p_wrist
```

Scale:

```text
s = ||p_middle_MCP - p_wrist||_2
```

Chuẩn hóa:

```text
p''_i = p'_i / s
```

Nếu `s <= scale_epsilon`, frame-hand đó không được chia; feature giữ zero và mask được đặt false.

## Input

Phase 04 mặc định nhận các sample Phase 03 có:

```text
ACCEPT
REVIEW
```

và loại:

```text
REJECT
```

Đây là policy hiện tại của project để 160 sample hiện tại có thể đi tiếp trong khi vẫn giữ nguyên `phase3_status`.

## Output mỗi sample

```text
dataset/processed/features_phase4/<sample_id>/
├── features.npz
└── feature_metadata.json
```

Các array chính:

```text
normalized_xyz126      (T,126)
left_normalized_xyz    (T,21,3)
right_normalized_xyz   (T,21,3)
hand_presence2         (T,2)
left_present           (T,)
right_present          (T,)
scale2                 (T,2)
scale_valid2           (T,2)
timestamps             (T,)
pose_present           (T,)
face_present           (T,)
```

Quy ước feature:

```text
0..62    = LEFT
63..125  = RIGHT
```

Không đảo left/right do preview mirror.

## Chạy

```powershell
pytest -q tests/test_phase4_normalization.py
python scripts/run_phase4_features.py
python scripts/audit_phase4.py
```

## Exit Gate

Phase 04 PASS khi:

- selected Phase-3 samples = Phase-4 rows
- FAIL rows = 0
- Missing = 0
- Extra = 0
- Split mismatch = 0
- Feature errors = 0
- mọi feature hữu hạn
- shape mỗi frame là 126
- wrist sau translation gần 0
- missing landmark luôn đi kèm mask
- LEFT 63 / RIGHT 63 không bị đảo

## Sau Phase 04

Tiếp tục Phase 05:
- joint angles
- inter-landmark distances
- hand-to-body / hand-to-hand relations
- velocity
- acceleration
