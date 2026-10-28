# PHASE 08 — Tự động xử lý UNKNOWN RAW → 453 features → manifest

## Nguồn đã kiểm tra

Bộ script này dùng lại **nguyên các hàm** từ 3 file `core/feature_phase4.py`, `core/feature_phase5.py`, `core/feature_phase6.py` người dùng đã cung cấp. Không sửa hệ thống KNOWN và không cần chạy Phase 02 trên thư mục UNKNOWN.

**Điểm khác biệt:** các script Phase 04–06 cũ đọc manifest KNOWN và giả định đường dẫn `raw/<sign>/<subject>/<session>/<sample>`. Trong Phase 08, raw UNKNOWN được thu theo đường dẫn có thêm cấp `category`, do đó dùng script độc lập này.

## 1. Cài

Giải nén, chép file:

```
scripts/process_phase8_unknown.py
```

vào thư mục project tương ứng. Không chép đè hoặc đổi các file core hiện tại.

Yêu cầu Python packages: `numpy`, `PyYAML` (đã có trong environment Phase 04–06). Các config sau cần tồn tại:

```
configs/feature_phase4.yaml
configs/feature_phase5.yaml
configs/feature_phase6.yaml
```

## 2. Cấu trúc đầu vào

```
dataset/raw/UNKNOWN/
  scratching_head/
    S01/
      SES01/
        UNKNOWN_S01_SES01_001/
          landmarks.npz
          metadata.json
          source_video.mp4   # optional, không đọc lại video
  touching_face/
    S01/
      SES01/
        UNKNOWN_S01_SES01_001/
          landmarks.npz
          metadata.json
```

Dù các sample RAW ở những category khác nhau có cùng tên, script sẽ tạo ID mới tại **đầu ra processed**, ví dụ:

```
UNKNOWN_scratching_head_S01_SES01_001
UNKNOWN_touching_face_S01_SES01_001
```

**Không thay đổi `metadata.json` gốc, không đổi tên raw folder và không xóa video gốc.**

Các RAW đời cũ dưới `UNKNOWN/S01/SES01/...` thiếu category sẽ được **SKIP_LEGACY** (ghi trong báo cáo), chứ không tự gán nhãn đoán mò.

## 3. Chạy lần đầu (chỉ xử lý features)

Trong thư mục gốc của project:

```powershell
python scripts/process_phase8_unknown.py
```

Lệnh sẽ:

1. Tìm toàn bộ `landmarks.npz` theo cấu trúc category/subject/session/sample.
2. Đọc `left_hand`, `right_hand`, timestamps và masks (cùng schema collector).
3. Gọi Phase04 normalization: 126 chiều.
4. Gọi Phase05 geometric + motion: 417 chiều.
5. Gọi Phase06 pose + face: 453 chiều.
6. Kiểm tra shapes, finite values, feature audits.
7. Xuất features ở **các thư mục riêng UNKNOWN** và danh sách inventory.

Các đầu ra:

```
dataset/processed/
  features_phase4_unknown/<category>/<subject>/<session>/<sample_id>/features.npz
  features_phase5_unknown/<category>/<subject>/<session>/<sample_id>/features_v2.npz
  features_phase6_unknown/<category>/<subject>/<session>/<sample_id>/features_v3.npz
  features_phase6_unknown/unknown_inventory.csv
  features_phase6_unknown/unknown_processing_audit.json
```

File `features_v3.npz` chứa đúng key `ablation_hands_pose_face453` với shape `[T,453]`, tương thích với Phase08 `prepare_phase8_unknown.py` hiện tại.

**Lưu ý:** Chưa chia VAL/TEST thì script **chưa tạo `unknown_manifest.csv`**, để tránh chọn nhầm split.

## 4. Chia UNKNOWN VAL/TEST theo người

Xem các subject đang có:

```powershell
Get-ChildItem .\dataset\raw\UNKNOWN -Directory | ForEach-Object {
  Get-ChildItem $_.FullName -Directory | Select-Object -ExpandProperty Name
} | Sort-Object -Unique
```

Ví dụ nếu thực sự đã thu S11 và S12:

```powershell
python scripts/process_phase8_unknown.py `
  --val-subjects S11 `
  --test-subjects S12
```

Nếu có S11,S12,S13,S14:

```powershell
python scripts/process_phase8_unknown.py `
  --val-subjects S11,S12 `
  --test-subjects S13,S14
```

Dùng tên subject **thật** của dataset, ví dụ S01/S02 nếu bạn có 2 người. Khi chỉ có S01, có thể xử lý features nhưng chưa đánh giá signer-independent giữa UNKNOWN VAL và UNKNOWN TEST.

Script ngăn trùng signer giữa VAL và TEST, không tạo manifest nếu một bên rỗng hay có RAW bị lỗi. Đầu ra:

```
dataset/unknown_phase8/unknown_manifest.csv
```

Cột đúng format Phase08:

```
sample_id,subject_id,category,split,feature_path
```

Nếu đã có manifest, script từ chối ghi đè. Bạn có thể kiểm tra split, sau đó chạy lại với `--overwrite-manifest` khi chủ động muốn thay.

## 5. Chuẩn bị temporal UNKNOWN và chạy Phase 08

Chỉ sau khi manifest đã có VAL/TEST hợp lệ:

```powershell
python scripts/prepare_phase8_unknown.py
```

Output:

```
dataset/processed/unknown_phase8/val_unknown.npz
dataset/processed/unknown_phase8/test_unknown.npz
```

Đảm bảo `X` shape `(N,32,453)` và N>0 ở cả hai tập:

```powershell
python -c "import numpy as np; v=np.load('dataset/processed/unknown_phase8/val_unknown.npz'); t=np.load('dataset/processed/unknown_phase8/test_unknown.npz'); print('VAL',v['X'].shape,'TEST',t['X'].shape)"
```

Phase08 `prepare` sẽ áp dụng **mean/std của TRAIN Phase07 frozen**, không được fit lại scaler bằng UNKNOWN.

Cuối cùng:

```powershell
pytest -q tests/test_phase8_unknown.py
python scripts/calibrate_phase8_threshold.py
python scripts/evaluate_phase8_unknown.py
python scripts/audit_phase8.py
```

**Phase 08 v1: chỉ hiệu chỉnh threshold của TCN V1 đã đóng băng; không train lại model.**

## 6. Đọc kết quả xử lý

Ví dụ:

```
[1/4] PASS UNKNOWN_scratching_head_S11_SES01_001 (45x453, split=val)
[2/4] PASS UNKNOWN_touching_face_S12_SES01_001 (45x453, split=test)
...
TOTAL=4 PASS=4 FAIL=0 SKIP_LEGACY=0
UNKNOWN manifest: ... (VAL=2, TEST=2)
```

Nếu có lỗi, xem `dataset/processed/features_phase6_unknown/unknown_processing_audit.json`.

Đối với `no_hand`, **không áp dụng điều kiện luôn phải có tay**, vì đây là loại negative hợp lệ. Script kiểm tra lỗi cấu trúc/feature; việc một sample có đúng semantic category không vẫn cần người thu đánh giá thủ công.

## 7. Bảo vệ dữ liệu / giới hạn

- Không chạy Phase 02 audit cũ trên UNKNOWN category-folder vì audit cũ chưa hỗ trợ cấu trúc này.
- Không thêm UNKNOWN vào known Phase02 `split.csv`.
- Không sửa `dataset/raw` hoặc `artifacts/tcn/model.keras`.
- Không thay đổi `dataset/processed/features_phase4`, `features_phase5`, `features_phase6` của KNOWN.
- Các feature được tính lại từ đầu trên UNKNOWN theo core cùng version; nếu bạn thay đổi core/config giữa train V1 và Phase08, cần xác minh chúng vẫn đúng version trước khi so sánh.
- Không đánh dấu Phase08 PASS cho đến khi UNKNOWN có VAL/TEST hợp lệ và báo cáo FAR/FRR thực tế.
