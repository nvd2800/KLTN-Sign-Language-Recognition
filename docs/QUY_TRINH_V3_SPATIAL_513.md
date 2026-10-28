# Phase V3 Spatial 513 – Quy trình triển khai (VS Code, Windows)

## Kiến trúc

```
TRAIN: raw landmarks + Phase06 453D
       → core/feature_spatial.py (60D = 30D x 2 hands)
       → concat [453D | 60D] → resample32 → TRAIN-only scaler513
       → real labelled hard negatives/UNKNOWN → TCN V3 [32,513] → labels5
LIVE:  MediaPipe hands/pose/face → exactly same 453D + spatial60()
       → same resample32 + frozen scaler513 → TCN V3 → Phase09 FSM
       → token, CSV (trial_id / condition / prediction / release source)
```

**Backwards compatible:** `build_model_input(...,feature_mode='phase6_453')` remains available for TCN V2. V3 explicitly uses `feature_mode='spatial513'`. V2 model and scaler are NEVER overwritten.

## 60 chiều giải thích chính xác

Mỗi tay 30 chiều, sắp xếp theo LEFT rồi RIGHT:
- 0–1 wrist XY tương đối tâm hai vai / chiều rộng vai
- 2–3 tâm lòng tay XY tương đối tâm hai vai / chiều rộng vai
- 4–7 khoảng cách 2D wrist→mũi, vai cùng phía, tâm vai, miệng / chiều rộng vai
- 8–12 5 finger curl đo từ khớp ngón (0 duỗi, gần 1 co)
- 13–15 vector pháp tuyến lòng tay 3D (chỉ dùng Hand XYZ)
- 16–17 hướng từ wrist→middleMCP trong ảnh XY
- 18–22 khoảng cách từ 5 đầu ngón tới wrist / kích thước bàn tay
- 23–27 khoảng cách từ 5 đầu ngón tới tâm lòng tay / kích thước bàn tay
- 28–29 tốc độ vị trí tương đối của wrist theo khung hình, tính chênh frame

Các đặc trưng 0–7 và 28–29 có thể không hợp lệ nếu mất Pose, nhóm 8–27 vẫn hoạt động nếu có Hand. Các feature thiếu gán 0. Quan trọng: không lấy khoảng cách giữa `Pose.z` và `Hand.z`.

## 1. Cài gói

Giải nén ZIP vào root project `D:\KLTN_KTMT\KLTN_Sign\Sign-Language-Recognition-Phase0`. Sao lưu ba file cũ: `core/preprocess_phase7.py`, `core/phase8_live.py`, `scripts/prepare_phase7_dataset.py`. Các bản trong ZIP đã nối 513D nhưng giữ đường 453D tương thích. V3 realtime là script mới, không ghi đè Phase09 V2.

```
conda activate venv
python -m pytest -q tests/test_v3_spatial.py tests/test_v3_split.py
```

## 2. Chuẩn bị KNOWN 513D

Nguồn là `dataset/processed/features_phase6/feature_manifest.csv` và `dataset/raw/<sign>/<subject>/<session>/<sample>/landmarks.npz`. **Không thể tái tạo spatial từ chỉ 453D** – cần raw landmarks của mỗi sample, nếu thiếu script sẽ báo lỗi cụ thể.

```
python scripts/prepare_phase7_dataset.py --config configs/phase7_spatial.yaml
```

Output riêng tại `dataset/processed/temporal_phase7_spatial513/` gồm train/val/test, labels, scaler513, summary. Scaler chỉ fit trên KNOWN TRAIN (đủ bảo đảm không dùng VAL/TEST).

## 3. Thu hard negative THẬT bằng công cụ đã có

Dùng `scripts/live_feature_diagnostic.py`, cùng người làm mỗi class nhiều lần:

```
python scripts/live_feature_diagnostic.py --label xin_chao --subject S01 --frames 32 --save-video
python scripts/live_feature_diagnostic.py --label cam_on --subject S01 --frames 32 --save-video
python scripts/live_feature_diagnostic.py --label xin_loi --subject S01 --frames 32 --save-video
```

Phím `1=correct`, `2=wrong_position`, `3=wrong_handshape`, `4=wrong_orientation`, `5=partial`, `6=correct_repeat`.

**Tiêu chí thu**: mỗi signer × ký hiệu ưu tiên ≥10 correct và ≥10 mỗi loại wrong_position/wrong_handshape. Thu ít nhất vài signer cho TRAIN và signer tách biệt cho VAL và TEST. Kiểm tra người làm ký hiệu theo quy ước, bảo đảm gesture sai không phải một ký hiệu hợp lệ khác. Không tự động coi dữ liệu test đã dùng nhiều lần là blind test.

**Lưu ý:** script V3 chỉ tự động nạp 4 điều kiện sai có nhãn, không tự động nạp các `correct` của Diagnostic vào TRAIN/VAL. Mẫu đúng dùng Phase07 KNOWN; để tăng mẫu đúng mới cần nhập qua collector KNOWN chuẩn và xử lý lại Phase04–06.

## 4. Ghép dataset 5 class

Nếu hiện chỉ có S01 (TRAIN) thì CHỈ chạy DEV:

```
python scripts/prepare_v3_spatial.py --mode dev
```

Khi có real negatives của S03 (VAL) và S04 (TEST) *và các signer tương ứng nằm ở đó trong Phase07 summary*:

```
python scripts/prepare_v3_spatial.py --mode strict
```

Nếu là signer mới chưa nằm trong Phase07 summary, gán split **nhất quán và trước đánh giá** bằng `--new-train-signers`, `--new-val-signers`, `--new-test-signers`. Không đẩy S01 vào TEST. TEST được giữ khóa.

Kết quả tại `dataset/processed/temporal_v3_spatial` gồm `train.npz`, `val.npz`, `test.npz`, `labels.json`, `preprocess_stats.npz`, `dataset_summary.json`, `negative_manifest.csv`.

## 5. Train TCN V3

```
python scripts/train_v3_spatial.py --epochs 120 --batch-size 16
```

Model riêng: `artifacts/experiments/tcn_v3_spatial/model.keras`, `preprocess_stats.npz`, `labels.json` và history. TensorFlow cần được cài trên Windows. **Không báo cáo kết quả chính thức nếu mode=dev, UNKNOWN VAL/TEST thiếu hoặc signer trùng.**

## 6. Chẩn đoán vị trí: target ≥1.0 chiều rộng vai

```
python scripts/diagnose_position_sensitivity.py --min-shift-shoulders 1.0 --synthetic-shift 1.0
```

File JSON tại `logs/diagnose_v3_spatial.json` báo tỷ lệ nhận lại nhãn gốc, p_true_sign trước/sau, UNKNOWN, và độ dịch measured theo 2 vai của các mẫu **thật**. Synthetic shift 1 vai giúp phân tích độ nhạy, **không được coi là real-world accuracy** vì có thể làm bàn tay ra ngoài camera.

Mục tiêu thực nghiệm: với wrong_position có measured shift >= 1.0 vai và wrong_handshape được annotator xác nhận, tỷ lệ nhận lại nhãn cũ phải giảm đáng kể so với đúng; mục tiêu định lượng (ví dụ <=20% false accept) đặt bằng VAL và đo trên TEST khóa, không đảm bảo trước khi có dữ liệu.

## 7. Phase09 realtime V3 + release + log có nhãn

```
python scripts/live_phase09_v3_spatial.py --fullscreen --target-label xin_chao
```

`configs/phase9_v3_spatial.yaml` có:
- `window_frames: 32`, `stride_frames: 8`.
- Khi không thấy cả hai tay trong 12 frame liên tiếp, bắt đầu cấp lần lượt một tín hiệu release REST mỗi 2 frame thật cho FSM. Chúng được ghi `source=NO_HAND_RELEASE_HEURISTIC`, **không phải dự đoán REST của model**.
- `release_consecutive: 3` (3 tín hiệu REST khác frame để mở khóa), không đếm `UNKNOWN` thành REST.
- Mỗi trial ấn `0=UNLABELED`, `1=correct`, `2=wrong_position`, `3=wrong_handshape`, `4=wrong_orientation`, `5=partial`, `6=REST`. Ghi `trial_id`, `test_condition`, `target_sign`, `expected_label`. Khi thử nên cách giữa các trial bằng tay thả hoàn toàn để auto release.
- `B` thủ công REST chỉ dùng debug, `source=MANUAL_REST` phải loại khỏi báo cáo metrics chính thức.

Log riêng tại `logs/phase09_v3_spatial_predictions.csv`, tokens JSONL. Đừng trộn với Phase09 V2 log cũ. TCN vẫn cần 3 dự đoán ổn định và thả tay/REST mới phát token tiếp theo.

## 8. Điểm chưa đạt, cần làm trên thiết bị người dùng

- Chưa có bộ hard negative đa signer thật trong container, nên CHƯA huấn luyện được model V3 chính thức.
- Chưa chạy TensorFlow webcam trên Windows hay Raspberry Pi.
- Không được kết luận rằng nắm tay hoặc dịch 1 vai đã được UNKNOWN cho đến khi có file `diagnose_v3_spatial.json` từ model thực tế.
- 60D mô tả vị trí/hình dáng theo toán học, chưa thay thế việc đánh giá từ vựng VSL và quy ước diễn đạt.

## 9. Phân tích log camera có nhãn

``
python scripts/analyze_v3_log.py
``

Xuất `logs/phase09_v3_spatial_summary.json` với tỷ lệ ở mức inference window theo điều kiện, tách số lần release từ **model REST**, **NO_HAND_RELEASE_HEURISTIC**, **MANUAL_REST**. Các window chồng lấn nhau nên không coi là các mẫu độc lập. Để đánh giá chính thức phải có nhãn clip với đầu/cuối và signer TEST chưa từng dùng hiệu chỉnh.

## 10. Đánh giá V3 trên signer TEST (không hiệu chỉnh sau khi xem TEST)

```
python scripts/evaluate_v3_spatial.py --split val
python scripts/evaluate_v3_spatial.py --split test
```

Kết quả `artifacts/experiments/tcn_v3_spatial/evaluation_test.json`: confusion matrix, accuracy/macro F1, Known Accuracy, UNKNOWN Recall, UNKNOWN False Accept Rate. Nếu split TEST không có UNKNOWN thì chỉ kiểm tra được các known labels; **không thể kết luận UNKNOWN FAR**. Không dùng TEST để điều chỉnh cấu hình hoặc ngưỡng confidence. So sánh với V2 khi cả hai cùng có bộ mẫu test định danh và cùng ground truth; V2 và V3 có input kích thước khác nhau nên không được nạp nhầm scaler/model.
