# TCN V1 → V2 Workflow

Mục tiêu của workflow này là giữ nguyên kết quả TCN V1, test realtime, ghi lỗi thực tế, thu data có mục tiêu, train TCN V2 từ đầu và so sánh công bằng trên cùng locked TEST set.

## 1. Freeze TCN V1

```powershell
python scripts/freeze_tcn_v1.py
```

Output:

```text
artifacts/experiments/tcn_v1_160samples/
├── model.keras
├── history.json
├── training_config.json
├── test_metrics.json
├── confusion_matrix.csv
├── preprocess_stats.npz
├── labels.json
├── dataset_summary.json
├── test.npz
├── locked_test_sample_ids.json
└── freeze_manifest.json
```

`locked_test_sample_ids.json` là tập TEST khóa lại để V1/V2 so sánh công bằng.

## 2. Live inference TCN V1

Yêu cầu project đang có:

```text
core/feature_phase5.py
core/feature_phase6.py
configs/feature_phase5.yaml
configs/feature_phase6.yaml
```

và MediaPipe legacy tương thích:

```powershell
pip install mediapipe==0.10.21 opencv-python tensorflow numpy
```

Chạy:

```powershell
python scripts/live_tcn_v1.py
```

Nếu camera khác:

```powershell
python scripts/live_tcn_v1.py --camera 1
```

Pipeline realtime dùng đúng logic Phase 04→05→06:
- hand normalization;
- Phase 05 full_features417;
- Phase 06 Hands+Pose+Face453;
- scaler V1 đã freeze;
- rolling window 32 frame;
- TCN V1.

Không fit scaler mới khi webcam chạy.

## 3. Test 4 class realtime

Sau khi buffer đủ 32 frame, màn hình hiển thị:
- prediction;
- confidence;
- probabilities;
- inference milliseconds.

Phím số tương ứng với labels hiện tại được hiển thị ở footer.

Ví dụ nếu footer:

```text
1:REST | 2:cam_on | 3:xin_chao | 4:xin_loi
```

thì sau khi thực hiện đúng ký hiệu, nhấn phím ground truth tương ứng để ghi trial.

Nên test tối thiểu khoảng:
- 20–30 lần/class;
- nhiều khoảng cách camera;
- tốc độ nhanh/chậm;
- hai góc nhẹ khác nhau;
- các subject TRAIN mới nếu có.

Không dùng người thuộc locked TEST để fine-tune.

## 4. Ghi false predictions

Live log mặc định:

```text
logs/phase7_live_tcn_v1.csv
```

Mỗi row lưu:
- ground_truth;
- prediction;
- correct;
- confidence;
- inference_ms;
- probability từng class.

Phân tích:

```powershell
python scripts/analyze_live_errors.py
```

Output:

```text
logs/phase7_live_error_summary.json
```

Xem:
- class nào false nhiều nhất;
- confusion pair nào lặp lại;
- class nào nên bổ sung data.

## 5. Thu data có mục tiêu

Không thu ngẫu nhiên tất cả class bằng nhau nếu chỉ một vài class lỗi.

Ví dụ live log cho thấy:

```text
cam_on -> xin_loi: 6 lần
xin_chao -> cam_on: 2 lần
```

thì ưu tiên:
- thêm nhiều `cam_on`;
- thêm các biến thể `cam_on` dễ bị giống `xin_loi`;
- thêm một phần `xin_loi` đối chứng;
- thêm `xin_chao` ở các motion gây nhầm.

Data mới vẫn phải đi đúng pipeline:

```text
Raw sequence
→ Phase 03 QC
→ Phase 04
→ Phase 05
→ Phase 06
```

Giữ signer-independent split.

## 6. Chuẩn bị TCN V2

Sau khi Phase 06 được rebuild với data mới:

```powershell
python scripts/prepare_tcn_v2.py
```

Quy tắc:
- TRAIN/VAL được cập nhật theo split;
- TEST bị khóa theo exact sample IDs của V1;
- scaler V2 fit lại TRAIN mới;
- TEST tuyệt đối không tham gia scaler/training.

Output:

```text
dataset/processed/temporal_phase7_v2/
```

## 7. Train TCN V2

Train từ đầu:

```powershell
python scripts/train_tcn_v2.py
```

Không load weight TCN V1 làm mặc định.

Output:

```text
artifacts/experiments/tcn_v2_targeted/
```

bao gồm:
- model.keras;
- scaler V2;
- labels;
- dataset summary;
- history.

## 8. Evaluate V2

```powershell
python scripts/evaluate_tcn_v2.py
```

Đánh giá trên locked V1 TEST sample IDs.

## 9. So sánh V1 vs V2

```powershell
python scripts/compare_tcn_v1_v2.py
```

Kỳ vọng:

```text
TCN V1 vs V2 — SAME LOCKED TEST
V1 accuracy=...
V2 accuracy=...
Δ accuracy=...
Δ macro_f1=...
```

Chỉ khi dùng cùng locked TEST mới có ý nghĩa so sánh trực tiếp.

## 10. Nguyên tắc quan trọng

- Không ghi đè V1.
- Không train trên locked TEST.
- Không dùng TEST để quyết định cần thu class nào; dùng live trials/VAL.
- Data mới phải đi lại Phase 03→06.
- V2 scaler phải fit TRAIN mới.
- Train V2 từ scratch là baseline chính; fine-tune có thể làm thêm như experiment phụ.
- Nếu thêm class mới, phải rebuild labels/model output và không so trực tiếp như cùng bài toán 4-class.
