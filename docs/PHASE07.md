# PHASE 07 — Temporal Model GRU / TCN

## 1. Mục tiêu

Phase 07 là trung tâm AI của hệ thống: thay vì dự đoán từng frame riêng lẻ, mô hình nhận cả chuỗi 32 frame và dự đoán một ký hiệu hoàn chỉnh.

```text
F01
F02
...
F32
 ↓
Temporal Model
 ↓
Softmax
 ↓
SIGN
```

## 2. Input

Mặc định dùng output tốt nhất từ Phase 06:

```text
ablation_hands_pose_face453
shape = (T,453)
```

Config có thể chuyển sang:

```text
hands417
hands_pose444
hands_pose_face453
```

để làm ablation.

## 3. Chuẩn hóa độ dài sequence

Target:

```text
32 frames
```

Nếu T >= 32:
- lấy 32 frame theo chỉ số phân bố đều từ đầu tới cuối;
- giữ thứ tự thời gian;
- temporal mask = true toàn bộ.

Nếu T < 32:
- giữ nguyên T frame;
- pad zero ở cuối cho đủ 32;
- mask=false ở frame padding.

Ví dụ:

```text
T=40 -> resample -> 32
T=32 -> giữ nguyên
T=25 -> 25 real + 7 padded
```

Không random crop ở preprocessing mặc định để validation/test có tính xác định.

## 4. Standardization

Đây là yêu cầu chống data leakage quan trọng.

Chỉ dùng TRAIN để tính:

```text
mean_j = trung bình feature j trên tất cả valid TRAIN frames

std_j = độ lệch chuẩn feature j trên tất cả valid TRAIN frames
```

Sau đó:

```text
x'_j = (x_j - mean_j) / std_j
```

Cùng `mean/std` này được áp dụng cho TRAIN, VAL và TEST.

Không fit scaler lại trên VAL hoặc TEST.

Padding luôn được đưa trở lại đúng zero sau standardization để GRU Masking nhận ra timestep padding.

## 5. Signer-independent split

Phase 07 không chia lại dataset.

Phải giữ nguyên:

```text
TRAIN: subjects Phase 02 train
VAL  : subjects Phase 02 val
TEST : subjects Phase 02 test
```

Script prepare kiểm tra signer leakage trước khi sinh dataset temporal.

## 6. GRU chính

Architecture:

```text
Input (32,F)
 ↓
Masking
 ↓
GRU 128, return_sequences=True
 ↓
Dropout 0.30
 ↓
GRU 64
 ↓
Dense 64 ReLU
 ↓
Dropout 0.30
 ↓
Dense num_classes Softmax
```

GRU được chọn làm model chính vì nhẹ hơn LSTM tương đối và phù hợp mục tiêu triển khai edge/Raspberry Pi.

## 7. TCN so sánh

Architecture:

```text
Input
 ↓
Causal Conv1D dilation=1
 ↓
Residual
 ↓
Causal Conv1D dilation=2
 ↓
Residual
 ↓
Causal Conv1D dilation=4
 ↓
Residual
 ↓
Causal Conv1D dilation=8
 ↓
Global Max Pooling
 ↓
Dense 64
 ↓
Softmax
```

Dilated convolution mở rộng temporal receptive field mà không cần recurrent state.

## 8. Training policy

```text
loss      = sparse categorical crossentropy
optimizer = Adam
epochs    = 120 max
batch     = 16
```

Callbacks:

```text
ModelCheckpoint
EarlyStopping
ReduceLROnPlateau
```

Checkpoint theo:

```text
val_loss
```

Không dùng epoch cuối máy móc.

## 9. Evaluation

Model chỉ được tuning bằng TRAIN/VAL.

TEST là signer chưa thấy và nên dùng sau khi config đã chốt.

Metrics:

```text
Accuracy
Macro F1
Classification report
Confusion matrix
```

Macro F1 quan trọng vì đánh giá đều các class thay vì chỉ nhìn accuracy tổng.

## 10. Output

Prepared dataset:

```text
dataset/processed/temporal_phase7/
├── train.npz
├── val.npz
├── test.npz
├── labels.json
├── preprocess_stats.npz
└── dataset_summary.json
```

GRU:

```text
artifacts/gru/
├── model.keras
├── history.json
├── training_config.json
├── test_metrics.json
└── confusion_matrix.csv
```

TCN tương tự:

```text
artifacts/tcn/
...
```

## 11. Quy trình chạy đúng

### Bước 1 — cài dependency

```powershell
pip install -r requirements-phase7.txt
```

### Bước 2 — unit test preprocessing

```powershell
pytest -q tests/test_phase7_preprocessing.py
```

Kỳ vọng:

```text
8 passed
```

### Bước 3 — tạo sequence dataset 32 frame

```powershell
python scripts/prepare_phase7_dataset.py
```

Kỳ vọng:

```text
Signer leakage : 0
PREPARE        : PASS
```

### Bước 4 — audit dataset temporal

```powershell
python scripts/audit_phase7.py
```

Kỳ vọng:

```text
PHASE 7 DATA : PASS
```

### Bước 5 — train GRU

```powershell
python scripts/train_phase7.py --model gru
```

### Bước 6 — train TCN

```powershell
python scripts/train_phase7.py --model tcn
```

### Bước 7 — evaluate GRU trên unseen signer TEST

```powershell
python scripts/evaluate_phase7.py --model gru
```

### Bước 8 — evaluate TCN

```powershell
python scripts/evaluate_phase7.py --model tcn
```

### Bước 9 — so sánh

```powershell
python scripts/compare_phase7_models.py
```

### Bước 10 — audit artifacts cuối

```powershell
python scripts/audit_phase7_models.py
```

## 12. Cách đọc training curve

Bình thường:

```text
train loss giảm
val loss giảm rồi ổn định
train accuracy tăng
val accuracy tăng hoặc dao động nhẹ
```

Overfit:

```text
train loss tiếp tục giảm
val loss tăng
```

EarlyStopping sẽ dừng và giữ best checkpoint.

Underfit:

```text
train accuracy thấp
val accuracy thấp
cả train/val loss còn cao
```

## 13. Dataset nhỏ — lưu ý

Dataset hiện tại chỉ khoảng 160 sequence và vài class, nên accuracy có thể dao động mạnh, đặc biệt VAL chỉ có ít sample.

Không nên:
- tăng model vô hạn;
- chỉnh hyperparameter dựa trên TEST;
- báo một lần accuracy duy nhất mà không có confusion matrix;
- bỏ qua unseen-signer split.

Nên:
- giữ architecture vừa phải;
- dùng EarlyStopping;
- báo Accuracy + Macro F1;
- sau này tăng số subject và số sequence/class.

## 14. Feature ablation

Sau pipeline đầu tiên với:

```text
hands_pose_face453
```

có thể đổi config:

```yaml
feature:
  active: "hands417"
```

rồi prepare/train lại.

Các thí nghiệm:

```text
Hands 417
Hands + Pose 444
Hands + Pose + Face 453
```

Nên dùng cùng:
- split
- sequence length
- model architecture
- seed
- training policy

để so sánh công bằng.

## 15. Exit Gate Phase 07

Phase 07 chỉ COMPLETE khi:

```text
[PASS] preprocessing unit tests
[PASS] signer leakage = 0
[PASS] temporal dataset audit
[PASS] scaler fit TRAIN only
[PASS] 32-frame input
[PASS] GRU best model.keras tồn tại
[PASS] TCN best model.keras tồn tại
[PASS] history được lưu
[PASS] unseen-signer TEST metrics tồn tại
[PASS] confusion matrix tồn tại
[PASS] model artifact audit PASS
```

## 16. Sau Phase 07

```text
Phase 08 — UNKNOWN / Out-of-Vocabulary Detection
```

Không sang Phase 08 trước khi model temporal đã học được các known signs tương đối ổn định.
