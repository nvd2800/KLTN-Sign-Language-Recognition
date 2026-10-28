# Phase 08 — TCN V2 UNKNOWN (5 class): TRAIN → EXPORT → TEST

## Vì sao cần bước này

TCN V1 chỉ có các class `REST`, `cam_on`, `xin_chao`, `xin_loi` và softmax của mô hình cũ thường gán UNKNOWN thành `xin_chao` với confidence khoảng 99.9%. Phiên bản V2 dưới đây huấn luyện lại **mô hình 5 class** bằng tập KNOWN Phase07 và UNKNOWN đã xử lý thành `features_v3.npz`.

**TCN V2 là một thí nghiệm closed-set 5 class**, chưa giải quyết trọn vẹn open-set OOV. Những loại UNKNOWN chưa từng học vẫn có thể bị phân loại nhầm. Cần đo trên loại động tác lạ chưa thấy ở train, ngoài kiểm thử khác signer.

## Nguồn dữ liệu

- KNOWN: `dataset/processed/temporal_phase7/{train,val,test}.npz` (32×453, đã scale bằng Phase07 TRAIN scaler)
- UNKNOWN: `dataset/processed/features_phase6_unknown/unknown_inventory.csv` được tạo bởi `process_phase8_unknown.py`, mỗi sample `features_v3.npz` có `ablation_hands_pose_face453`.
- Scaler: tái sử dụng đúng `dataset/processed/temporal_phase7/preprocess_stats.npz`. Không fit lại scaler từ VAL/TEST.
- Model: cùng kiến trúc TCN Phase07, train mới từ đầu, đầu ra 5 class.

### Rất quan trọng với dữ liệu bạn đang có

Bạn đang có 6 mẫu UNKNOWN từ S01, S02. Trong Phase07, nếu S01/S02 nằm trong KNOWN TRAIN thì **UNKNOWN của họ cũng phải là TRAIN**, không thể chuyển S02 thành UNKNOWN TEST, vì cùng người đã có trong KNOWN TRAIN. Script tra cứu map signer trong `dataset/processed/temporal_phase7/dataset_summary.json`, không tin split cũ trong UNKNOWN manifest.

Với 6 UNKNOWN (2 mỗi category) chỉ nên tập chạy kiểm thử kỹ thuật. Mỗi category nên có nhiều sample và người thực hiện; không cần thu cả 9 category. Để đánh giá chính thức, nên thu UNKNOWN thuộc subject độc lập ở nhóm Phase07 VAL và TEST; ví dụ S08 (VAL), S09/S10 (TEST), với điều kiện ID signer phản ánh đúng người thực.

## Cài đặt

Giải nén ZIP **tại thư mục gốc project**, chỉ thêm các file dưới đây; KHÔNG đè `core/models_phase7.py`, `core/preprocess_phase7.py`, `dataset` hoặc TCN V1.

```text
configs/phase8_tcn_v2.yaml
core/tcn_v2_unknown.py
core/phase8_live.py            (helper webcam đã có ở Phase08 Live Tester)
scripts/prepare_phase8_tcn_v2.py
scripts/train_phase8_tcn_v2.py
scripts/evaluate_phase8_tcn_v2.py
scripts/audit_phase8_tcn_v2.py
scripts/live_phase8_tcn_v2.py
tests/test_tcn_v2_unknown.py
```

Môi trường: dùng venv TensorFlow/MediaPipe đã chạy Phase07/08 trước đó.

## A. Train nhanh bản DEV trên dữ liệu hiện có

Chạy tại root project:

```powershell
pytest -q tests/test_tcn_v2_unknown.py
python scripts/prepare_phase8_tcn_v2.py --mode dev
python scripts/train_phase8_tcn_v2.py
python scripts/evaluate_phase8_tcn_v2.py
python scripts/audit_phase8_tcn_v2.py
```

**DEV** có nghĩa là có thể không có UNKNOWN VAL/TEST độc lập. Script train sử dụng KNOWN VAL để chọn checkpoint; vì vậy `val_loss` không đo khả năng UNKNOWN và model có thể overfit 6 sample. Không được báo cáo UNKNOWN recall hay FAR từ TRAIN.

Muốn train thử 30 epoch trước:

```powershell
python scripts/train_phase8_tcn_v2.py --epochs 30
```

Nếu checkpoint đã tồn tại, script không tự ghi đè; hãy lưu backup rồi dùng `--overwrite` khi bạn chủ động train lại.

## B. Train/evaluate theo STRICT protocol

Để STRICT PASS, ngoài UNKNOWN TRAIN phải có UNKNOWN thuộc signer VAL và TEST khác nhau theo split KNOWN Phase07. Ví dụ **chỉ khi đã collect thật** ở S08 và S09:

```powershell
python scripts/process_phase8_unknown.py
python scripts/prepare_phase8_tcn_v2.py --mode strict
python scripts/train_phase8_tcn_v2.py
python scripts/evaluate_phase8_tcn_v2.py
```

Nếu signer UNKNOWN mới hoàn toàn (ví dụ S11, S12 chưa có trong KNOWN), chỉ rõ:

```powershell
python scripts/prepare_phase8_tcn_v2.py --mode strict --val-subjects S11 --test-subjects S12
```

Tránh chia cùng signer ở KNOWN TRAIN và UNKNOWN TEST. Những UNKNOWN S02 đã từng được dùng kiểm thử/hiệu chỉnh Phase08 không còn là mẫu TEST mù đáng tin cậy; cần dữ liệu mới độc lập để báo cáo chính thức.

## C. Model được xuất ở đâu?

```text
artifacts/experiments/tcn_v2_unknown/
├── model.keras                 (BEST checkpoint theo val_loss)
├── last_model.keras            (last/early-stopped model)
├── labels.json                 (5 class)
├── preprocess_stats.npz        (copy frozen TRAIN scaler Phase07)
├── dataset_summary.json        (nguồn + split + cảnh báo đánh giá)
├── training_config.json
├── history.json
├── test_metrics.json           (sau evaluate)
├── confusion_matrix.csv        (sau evaluate)
└── test_predictions.csv        (sau evaluate)
```

Dataset hỗn hợp ở `dataset/processed/temporal_phase8_v2/`.

## D. Test camera TCN V2

```powershell
python scripts/live_phase8_tcn_v2.py --fullscreen
```

Phím: `0` thu trial UNKNOWN có ground truth; `1–4` trial known; `SPACE` trial không nhãn, `F` fullscreen, `V` overlay landmark, `P` rolling preview, `Q` thoát.

**Khác Phase08 V1:** đây là 5-class softmax **trực tiếp**, nên KHÔNG dùng threshold 0.90 cũ của TCN V1. TCN V2 chọn class UNKNOWN nếu class đó có xác suất cao nhất. Nếu cần nghiên cứu open-set thêm, sau này hiệu chỉnh một ngưỡng từ chối khác trên V2 VAL.

## E. Tiêu chí đánh giá

- Confusion matrix 5×5
- KNOWN TEST accuracy và false reject to UNKNOWN
- UNKNOWN TEST recall, false accept rate (chỉ khi có UNKNOWN TEST độc lập)
- Kiểm tra các thao tác lạ ngoài 3 category train để đánh giá open-set thật
- Realtime kiểm tra partial gesture, false accept, latency và chống phát lặp Phase09

### Cảnh báo dữ liệu

- 6 mẫu UNKNOWN S01/S02 không đủ để mô hình khái quát tốt.
- Nếu chưa có UNKNOWN TEST, file `test_metrics.json` sẽ để `unknown_test.recall = null` và `unknown_test.false_accept_rate = null` thay vì tự báo 0 hoặc 100.
- Kết quả TCN V1 và TCN V2 chỉ nên so sánh nếu dùng **cùng KNOWN locked TEST**. Với UNKNOWN phải có test blind độc lập.
