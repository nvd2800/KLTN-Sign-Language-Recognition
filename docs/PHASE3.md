# PHASE 03 — Dataset Quality Control

## 1. Mục tiêu

Phase 03 kiểm tra chất lượng của từng sequence sau khi Phase 02 đã cố định signer split.

Phase 03 KHÔNG:
- train model,
- thay đổi S01–S07 / S08 / S09–S10,
- sửa trực tiếp `dataset/raw`.

Phase 03 tạo một lớp QC độc lập để quyết định sample nào:
- `ACCEPT`: được phép đi tiếp,
- `REVIEW`: cần xem lại thủ công, mặc định không train,
- `REJECT`: lỗi rõ ràng, không được train.

## 2. Vì sao cần Phase 03?

Một sequence vẫn có thể có label đúng nhưng chất lượng xấu:
- mất tay quá nhiều frame,
- tay sát/cắt khỏi mép ảnh,
- landmark nhảy bất thường,
- video quá mờ,
- timestamp lỗi,
- sequence quá ngắn/dài,
- landmark chứa NaN/Inf,
- chuyển động gần như không có do quay sai.

Nếu các sample này âm thầm vào train, model sẽ học noise.

## 3. Quy tắc hiện tại

### Hard reject
- frame count ngoài 30–60,
- timestamp không tăng,
- hand landmark sai shape `(T,21,3)`,
- presence mask bị thiếu/sai chiều dài,
- landmark present có NaN/Inf,
- metadata frame count không khớp NPZ,
- sign thường có `any_hand_ratio < 0.90`,
- sign được khai báo hai tay có `both_hand_ratio < 0.85`.

### Ngoại lệ
`REST` và `UNKNOWN` không bị reject chỉ vì không có tay.
Điều này cần thiết vì REST là no-sign và UNKNOWN có thể chứa negative/no-hand.

### REVIEW
Các heuristic chưa đủ chắc để auto-reject:
- crop ratio cao,
- landmark jump cao,
- blur cao,
- motion quá thấp.

REVIEW mặc định `train_eligible = 0`.

## 4. Presence mask

Phase 01 đã lưu:
- `left_present`,
- `right_present`,
- `pose_present`,
- `face_present`.

Phase 03 chuẩn hóa thành:
- `left_hand_present`,
- `right_hand_present`,
- `any_hand_present`,
- `both_hands_present`,
- `pose_present`,
- `face_present`,
- `crop_flag`.

Không dùng zero vector mà không có mask.

## 5. Blur

Blur được tính từ `source_video.mp4` bằng variance of Laplacian.
Đây là heuristic phụ thuộc camera/resolution/ánh sáng nên mặc định chỉ đưa sample vào REVIEW.

Không nên biến ngưỡng blur thành hard reject trước khi quan sát histogram trên dataset thật.

## 6. Landmark jump

Với hai frame liên tiếp có cùng bàn tay present:

```text
d_t = mean_i || landmark_xy[t,i] - landmark_xy[t-1,i] ||
```

Nếu `d_t > jump_distance_threshold` thì pair đó bị đánh dấu jump.

`jump_ratio` = số pair jump / số pair hợp lệ.

Đây cũng là heuristic REVIEW vì ký hiệu nhanh có thể có chuyển động lớn hợp lệ.

## 7. Crop

Landmark normalized MediaPipe nằm gần:
- x = 0/1,
- y = 0/1

có khả năng bàn tay đang sát hoặc ra khỏi khung hình.

Nếu tỷ lệ frame crop cao hơn ngưỡng, sample vào REVIEW.

## 8. Artifact

```text
dataset/processed/quality/
├── quality_manifest.csv
├── quality_summary.json
├── quality_audit.json
└── <sample_id>/
    ├── quality.json
    └── quality_masks.npz
```

Raw dataset và Phase-2 split không bị thay đổi.

## 9. Chạy

Từ project root:

```powershell
python scripts/run_quality_control.py
python scripts/audit_quality.py
```

Sau khi đã xử lý hết REVIEW:

```powershell
python scripts/audit_quality.py --require-no-review
```

Unit tests:

```powershell
pytest -q tests/test_phase3_quality.py
```

## 10. Cách đọc kết quả

Ví dụ:

```text
ACCEPT : 140
REVIEW : 20
REJECT : 8
```

- 140 sample có thể đi tiếp.
- 20 sample phải xem video/quality.json.
- 8 sample bị loại khỏi pipeline train.

Không chuyển REJECT sang subject/split khác.
Split của Phase 02 phải giữ nguyên.

## 11. Exit Gate

Phase 03 hoàn thành khi:
- 100% sample Phase 02 có QC record,
- không duplicate QC,
- REJECT không train-eligible,
- REVIEW không train-eligible trước khi resolve,
- missing landmark luôn có mask,
- mọi REVIEW đã được xem lại hoặc xử lý,
- `audit_quality.py --require-no-review` PASS,
- có thống kê QC theo split/class/subject.

Sau đó mới sang Phase 04 Normalization.
