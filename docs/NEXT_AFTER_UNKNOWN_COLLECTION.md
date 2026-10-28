# PHASE 08 — tiếp tục sau khi đã collect UNKNOWN

Bạn đã hoàn thành bước thu UNKNOWN raw. Tiếp theo:

1. Audit UNKNOWN raw.
2. Chạy UNKNOWN qua đúng Phase04 -> Phase05 -> Phase06.
3. Tạo UNKNOWN VAL/TEST signer-independent.
4. Build unknown_manifest.csv.
5. Prepare temporal UNKNOWN.
6. Calibrate threshold trên VAL.
7. Khóa threshold.
8. Evaluate TEST.
9. Audit Phase08.

## 1. Audit raw UNKNOWN

```powershell
python scripts/audit_phase8_unknown_collection.py --root dataset/unknown_phase8
```

## 2. Process UNKNOWN qua Phase04 -> Phase06

UNKNOWN phải tạo được:

```text
features_v3.npz
```

với shape:

```text
[T, 453]
```

Đích đề xuất:

```text
dataset/processed/features_phase6_unknown/
  <category>/
    <subject>/
      <sample_id>/
        features_v3.npz
        metadata.json
```

Không fit scaler bằng UNKNOWN.

## 3. Chia UNKNOWN VAL / TEST theo signer

Ví dụ:

```text
UNKNOWN VAL  = S11,S12
UNKNOWN TEST = S13,S14
```

Không để signer overlap.

## 4. Build manifest

```powershell
python scripts/build_phase8_unknown_manifest.py `
  --features-root dataset/processed/features_phase6_unknown `
  --val-subjects S11,S12 `
  --test-subjects S13,S14
```

Output:

```text
dataset/unknown_phase8/unknown_manifest.csv
```

## 5. Prepare temporal UNKNOWN

Từ package Phase08 chính:

```powershell
python scripts/prepare_phase8_unknown.py
```

Output:

```text
dataset/processed/unknown_phase8/
  val_unknown.npz
  test_unknown.npz
```

## 6. Unit test

```powershell
pytest -q tests/test_phase8_unknown.py
```

## 7. Calibrate threshold trên VAL

```powershell
python scripts/calibrate_phase8_threshold.py
```

Thử:
0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90.

## 8. Evaluate TEST

```powershell
python scripts/evaluate_phase8_unknown.py
```

## 9. Audit

```powershell
python scripts/audit_phase8.py
```

Kỳ vọng cuối:

```text
Errors : 0
PHASE 8 : PASS
```

## Lưu ý quan trọng

- UNKNOWN TEST không dùng chọn threshold.
- Không thay scaler TCN V1.
- Không trộn UNKNOWN VAL/TEST theo sample nếu cùng signer.
- Không train lại TCN chỉ để làm Phase08 v1.
- Phase08 v1 dùng TCN frozen + calibrated rejection threshold.
