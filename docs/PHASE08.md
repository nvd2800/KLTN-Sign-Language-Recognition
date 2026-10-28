# PHASE 08 — UNKNOWN / Out-of-Vocabulary Detection

## Mục tiêu

Phase 08 giải quyết việc Softmax luôn ép một sequence thành class known. Hệ thống phải có quyền trả về `UNKNOWN`.

Roadmap yêu cầu:

- Thu UNKNOWN đa dạng: scratching_head, holding_phone, touching_face, random_hand, waving, pointing, transition, no_hand, ...
- Không thu UNKNOWN bằng một gesture lặp đi lặp lại.
- Dùng max softmax probability làm confidence.
- Thử các threshold: 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90.
- Chọn threshold bằng validation trade-off false accept / false reject.
- Không chọn threshold cảm tính.

## Thiết kế

Phase 08 V1 KHÔNG thêm UNKNOWN thành class thứ 5 trong classifier.

Dùng TCN V1 frozen:

```text
Sequence 32x453
→ TCN V1
→ Softmax known classes
→ max probability
→ calibrated threshold
→ known class / UNKNOWN
```

Core rule:

```python
confidence = max(prob)

if confidence < threshold:
    label = "UNKNOWN"
else:
    label = classes[argmax(prob)]
```

## Dataset UNKNOWN

Khuyến nghị cấu trúc raw:

```text
dataset/unknown_phase8/
├── scratching_head/
├── holding_phone/
├── touching_face/
├── random_hand/
├── waving/
├── pointing/
├── transition/
├── no_hand/
└── other_gesture/
```

Không dùng một động tác UNKNOWN duy nhất lặp hàng chục lần.

UNKNOWN phải có VAL và TEST riêng. Không dùng UNKNOWN TEST để chọn threshold.

## Feature pipeline

UNKNOWN phải đi qua CÙNG preprocessing với known:

```text
raw
→ Phase04 normalization
→ Phase05 geometric/motion
→ Phase06 hands+pose+face 453
→ Phase07 temporal resample/pad 32
→ frozen TRAIN scaler của TCN V1
```

Không fit scaler mới bằng UNKNOWN.

## Manifest

Copy:

```text
dataset_templates/unknown_manifest_template.csv
```

thành:

```text
dataset/unknown_phase8/unknown_manifest.csv
```

Các cột:

```text
sample_id
subject_id
category
split
feature_path
```

`split` chỉ được là `val` hoặc `test`.

## Chuẩn bị UNKNOWN temporal dataset

```powershell
python scripts/prepare_phase8_unknown.py
```

Output:

```text
dataset/processed/unknown_phase8/
├── val_unknown.npz
└── test_unknown.npz
```

## Calibrate threshold

```powershell
python scripts/calibrate_phase8_threshold.py
```

Script chạy đúng candidate thresholds:

```text
0.50
0.60
0.70
0.75
0.80
0.85
0.90
```

Mỗi threshold tính:

```text
Known Accept Rate
False Reject Rate (FRR)
UNKNOWN Reject Rate
False Accept Rate (FAR)
Balanced Accept/Reject score
```

### Định nghĩa

```text
Known Accept Rate =
known accepted / known validation samples

FRR =
known incorrectly rejected as UNKNOWN / known samples

UNKNOWN Reject Rate =
unknown correctly rejected / unknown samples

FAR =
unknown incorrectly accepted as known / unknown samples

Balanced =
(Known Accept Rate + UNKNOWN Reject Rate) / 2
```

Selection:

1. Balanced score cao nhất.
2. Nếu hòa: FAR thấp hơn.
3. Nếu vẫn hòa: FRR thấp hơn.
4. Nếu vẫn hòa: threshold cao hơn.

Output:

```text
artifacts/phase8_unknown/
├── threshold_sweep.csv
├── calibrated_threshold.json
└── validation_report.json
```

## Locked TEST evaluation

Sau khi threshold đã khóa:

```powershell
python scripts/evaluate_phase8_unknown.py
```

TEST chỉ dùng để báo cáo cuối.

Output:

```text
artifacts/phase8_unknown/test_report.json
```

Report có:

- known accept rate
- known FRR
- accuracy trên known samples được accept
- unknown rejection rate
- unknown FAR
- balanced score

## Audit

```powershell
python scripts/audit_phase8.py
```

PASS khi:

- UNKNOWN VAL có sample
- UNKNOWN TEST có sample
- threshold được calibrate
- threshold thuộc candidate list
- validation report tồn tại
- test report dùng đúng locked threshold

## Unit test

```powershell
pytest -q tests/test_phase8_unknown.py
```

## Quan trọng

Phase08 hiện chưa thể COMPLETE nếu chưa có UNKNOWN dataset thật.

Không được:
- lấy known TEST làm validation threshold;
- nhìn TEST rồi đổi threshold;
- dùng chỉ một kiểu random_hand làm toàn bộ UNKNOWN;
- fit scaler bằng UNKNOWN;
- tự chọn 0.85 vì realtime "trông có vẻ tốt".

## Sau Phase08

Phase09 mới làm continuous recognition state machine hoàn chỉnh:

```text
IDLE
→ CANDIDATE
→ STABLE
→ WAIT_RELEASE
→ IDLE
```

Decision Engine thử nghiệm ở Phase07 có thể được tái sử dụng, nhưng state machine chính thức thuộc Phase09.
