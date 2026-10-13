# PHASE 02 — Signer-independent Train / Validation / Test Split

## Mục tiêu

Phase 02 không train model. Mục tiêu là chia dữ liệu theo signer/subject:

- TRAIN: S01–S07
- VALIDATION: S08
- TEST: S09–S10

Điều kiện bắt buộc:

```text
TRAIN ∩ VAL  = ∅
TRAIN ∩ TEST = ∅
VAL   ∩ TEST = ∅
```

Toàn bộ session của cùng một subject phải nằm trong cùng một split.

## Vấn đề được giải quyết

Nếu cùng signer xuất hiện cả train và test, model có thể học đặc trưng cá nhân
thay vì bản chất ký hiệu. Kết quả Accuracy/F1 có thể bị thổi phồng.

Signer-independent split làm cho S09–S10 trở thành unseen signers.

## Vai trò

### TRAIN — S01–S07
Dùng để cập nhật trọng số model.

### VALIDATION — S08
Dùng cho hyperparameter, early stopping, best checkpoint và threshold.

### TEST — S09–S10
Chỉ dùng đánh giá cuối cùng. Không dùng test để chỉnh model.

## Artifact

```text
dataset/splits/
├── split.csv
├── split_summary.json
└── audit_report.json
```

Không copy landmark/video sang các thư mục train/val/test.
`split.csv` chỉ ánh xạ sample → split.

## Chạy

```powershell
python scripts/make_splits.py
python scripts/audit_dataset.py
pytest -q tests/test_phase2_split.py
```

## Exit gate

Phase 02 chỉ PASS khi:

- signer leakage = 0,
- mỗi subject thuộc đúng một split,
- mọi session của một subject cùng split,
- không có subject chưa được gán,
- không duplicate sample_id/path,
- S01–S10 đều hiện diện,
- Train/Val/Test đều đủ expected classes,
- split.csv bao phủ toàn bộ dataset/raw,
- S09–S10 chưa từng được dùng để chỉnh model.
