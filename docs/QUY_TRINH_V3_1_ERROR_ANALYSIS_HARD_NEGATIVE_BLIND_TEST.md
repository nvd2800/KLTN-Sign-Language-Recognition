# KLTN Sign Language – V3.1: truy vết lỗi TEST, hard negatives và Phase09

Phiên bản: 2026-10-28. **BẢN VÁ PHỤ TRỢ** dành cho project đã cài Phase V3 Spatial 513D, không thay checkpoint `tcn_v3_spatial/model.keras` và không bắt buộc train lại.

## Kết quả đã biết từ dữ liệu người dùng gửi

- `evaluation_test.json`: 49 sequence TEST, 46 đúng, Accuracy 93,88%, Macro F1 94,60%.
- 34/34 known (gồm REST) được nhận đúng; UNKNOWN 12/15 đúng.
- 3 lỗi UNKNOWN: 1 → REST, 1 → `cam_on`, 1 → `xin_loi`. **Chưa biết sample_id/subject/condition cụ thể** do báo cáo gốc không có prediction từng mẫu.
- `diagnose_v3_spatial(1).json`: synthetic shift đã chạy 1/1 và không còn lỗi, nhưng đó là thí nghiệm tổng hợp S01 chứ không phải blind test.

## Sơ đồ workflow

```text
Đóng băng V3 checkpoint/model.keras, labels, scaler 513D
    |
    v
Đánh giá TEST lần trước -> predictions_test.csv (49 dòng)
    |
    v
review_v3_test_errors.py -> ba lỗi và đường dẫn raw/video nếu tồn tại
    |
    v
Người thực hiện kiểm tra video + tracking; KHÔNG copy TEST vào TRAIN
    |
    v
Thu hard negatives MỚI của S01–S07 (TRAIN) / thêm POSITIVE TRAIN
    |
    v
Audit data -> prepare_v3_spatial --output temporal_v3_1_spatial
    |
    v
Guard chống test leakage -> train V3.1 (checkpoint RIÊNG)
    |
    v
Đánh giá VAL + Phase09 realtime / so sánh với V3
    |
    v
TEST MỚI từ S11,S12 hoặc signer hoàn toàn chưa dùng, scaler đã đóng băng
```

## A. Cài bản vá trong VS Code (Windows)

1. Giải nén ZIP **vào thư mục gốc project** `D:\KLTN_KTMT\KLTN_Sign\Sign-Language-Recognition-Phase0` (giữ thư mục `core/`, `scripts/`, `tests/`, `docs/`).
2. Sao lưu hai file cũ trước khi chép đè: `scripts/evaluate_v3_spatial.py`, `scripts/live_phase09_v3_spatial.py`.
3. Giữ nguyên `artifacts/experiments/tcn_v3_spatial/`; tuyệt đối không chép đè model, scaler, labels.
4. Sử dụng môi trường Python hiện tại của project, không cần cài package mới ngoài các thư viện đã dùng (`numpy`, `pytest`, `tensorflow`, `opencv-python`, `mediapipe`).

```powershell
conda activate venv
python -m pytest -q tests/test_v3_1_trace_review.py tests/test_v3_1_data_guard.py tests/test_v3_1_live_trials.py tests/test_v3_1_blind.py tests/test_v3_1_evaluate_cli.py
```

## B. Truy vết chính xác 3 lỗi UNKNOWN ở TEST (Bước 1 + 2)

```powershell
python scripts/evaluate_v3_spatial.py --split test
python scripts/review_v3_test_errors.py
```

Kết quả:

- `artifacts/experiments/tcn_v3_spatial/predictions_test.csv`: mọi sequence gồm `sample_id, subject, condition, true_label, predicted_label, confidence, true_label_probability, unknown_probability, QC ratios, raw_path, video_path`.
- `artifacts/experiments/tcn_v3_spatial/evaluation_test.json`: giữ các metrics cũ và bổ sung `error_cases`, `by_condition`.
- `logs/v3_test_error_review/error_review.csv`: chỉ các mẫu sai, có cột đánh giá con người để điền sau khi xem video.
- `logs/v3_test_error_review/README_review.md`: các trường hợp cần rà soát.
- `logs/v3_test_error_review/summary.json`: đếm lỗi, nhóm, metadata/nguồn chưa truy xuất được.

Tạo ảnh ghép 6 khung hình từ video nguồn (nếu lúc thu đã bật `--save-video`):

```powershell
python scripts/export_v3_error_video_contact_sheets.py
```

Ảnh được tạo tại `logs/v3_test_error_review/contact_sheets/`. Nếu một mẫu không có video, script báo `SKIP no video`; **không bịa video từ feature**.

**Ghi chú QC:** `main_hand_valid_ratio = max(left_hand_valid_ratio, right_hand_valid_ratio)` là proxy tay chính, chưa phải nhãn tay chính được gán bởi người thực hiện. Các mẫu tracking yếu gắn cờ REVIEW, không tự ý xóa để làm đẹp metric. Nếu `subject`, `condition` của known class thiếu trong manifest Phase06, ghi `UNRESOLVED`, không suy đoán.

### Diễn giải nhãn lỗi

- UNKNOWN → REST: **sai classification**, nhưng không trực tiếp tạo token ký hiệu; có thể tác động đến logic release.
- UNKNOWN → `cam_on` hoặc `xin_loi`: **nguy cơ phát token sai** nếu State Machine xác nhận sau nhiều cửa sổ.
- Đánh giá TEST đã được xem trong quá trình phát triển V3.1 nên không còn là *blind final test* cho V3.1.

## C. Thu hard negatives thật từ signer TRAIN (Bước 3)

Giữ split hiện tại: TRAIN S01–S07; VAL S08; TEST cũ S09–S10. **S11/S12 chỉ dùng TEST mới**, nếu thực sự là người mới.

Ví dụ S01 thuộc TRAIN:

```powershell
python scripts/collect_v3_train_negative.py --label xin_chao --subject S01 --frames 32
python scripts/collect_v3_train_negative.py --label cam_on --subject S02 --frames 32
python scripts/collect_v3_train_negative.py --label xin_loi --subject S03 --frames 32
```

Trong webcam thu phím `2` sai vị trí, `3` sai handshape, `4` sai hướng, `5` partial. Thu đủ mỗi nhóm và kiểm tra video + nhãn. Wrapper **sẽ từ chối `--subject S08`, `S09`, `S10`** vì không thuộc TRAIN.

Thống kê dữ liệu và cờ QC (mốc 10 mẫu/điều kiện chỉ là khởi đầu):

```powershell
python scripts/audit_v3_train_hardnegatives.py --minimum-per-condition 10
```

Output `logs/v3_hardnegative_audit/audit.json` và `train_gaps.csv`. Chú ý thu nhiều signer TRAIN, tránh lấy rất nhiều frame liên tiếp của cùng một video làm mẫu độc lập. Với nhãn sai tư thế nào trùng một ký hiệu hợp lệ khác thì phải gán nhãn thật, không gán UNKNOWN một cách máy móc.

**Tuyệt đối không dùng bất kỳ sequence S09/S10 hay mẫu 3 lỗi TEST hiện có để train hoặc tune threshold**, kể cả đổi tên file/subject.

## D. Rebuild V3.1 (Bước 4, chỉ khi đã có dữ liệu TRAIN mới)

Lưu bản gốc `dataset/processed/temporal_v3_spatial/` và `artifacts/experiments/tcn_v3_spatial/`.

```powershell
python scripts/prepare_v3_spatial.py --mode strict --output dataset/processed/temporal_v3_1_spatial
python scripts/guard_v3_1_dataset.py --baseline dataset/processed/temporal_v3_spatial --candidate dataset/processed/temporal_v3_1_spatial
```

Nếu guard báo `pass=false` thì **KHÔNG TRAIN**. File `logs/v3_1_dataset_guard.json` cho biết ID bị lẫn. Guard chỉ phát hiện lẫn ID và signer metadata; chưa thể bảo đảm hai mã signer là hai người thật khác nhau.

Khi guard PASS, mới có thể train V3.1 ở thư mục checkpoint riêng:

```powershell
python scripts/train_v3_spatial.py --data dataset/processed/temporal_v3_1_spatial --out artifacts/experiments/tcn_v3_1_spatial --epochs 120 --batch-size 16
python scripts/evaluate_v3_spatial.py --data dataset/processed/temporal_v3_1_spatial --model artifacts/experiments/tcn_v3_1_spatial/model.keras --split val --out artifacts/experiments/tcn_v3_1_spatial/evaluation_val.json --predictions-out artifacts/experiments/tcn_v3_1_spatial/predictions_val.csv
```

V3.1 chưa được train trong gói này. Ngay cả khi TEST cũ vẫn nằm trong dữ liệu `temporal_v3_1_spatial`, **không dùng kết quả TEST cũ để lựa chọn checkpoint hoặc ngưỡng mới**.

## E. BLIND TEST MỚI cho V3.1 (Bước 5)

Cần người thực sự mới, ví dụ S11,S12, thu cả ký hiệu **đúng** cho bốn lớp KNOWN (bao gồm REST theo protocol riêng) và các hard negatives ở webcam; các positive phải đi qua Phase01–06 để xuất `dataset/processed/features_phase6/feature_manifest.csv`, các hard negatives dùng `live_feature_diagnostic.py` (có `raw_landmarks.npz`, `features.npz`, `metadata.json`).

Sau khi có data và đã chốt V3.1 trên VAL:

```powershell
python scripts/prepare_v3_blind_test.py --signers S11,S12 --model-dir artifacts/experiments/tcn_v3_1_spatial --out dataset/processed/temporal_v3_1_blind
python scripts/evaluate_v3_spatial.py --data dataset/processed/temporal_v3_1_blind --model artifacts/experiments/tcn_v3_1_spatial/model.keras --split test --out artifacts/experiments/tcn_v3_1_spatial/evaluation_blind_test.json --predictions-out artifacts/experiments/tcn_v3_1_spatial/predictions_blind_test.csv
```

Script `prepare_v3_blind_test.py` **từ chối** signer đã tồn tại trong lịch sử V3/model trước đó, không fit scaler mới và không cho ghi đè bộ blind test. Khi tạo TEST mới từ S11/S12, cần bảo đảm Phase06 đã cập nhật manifest và các signer này không được dùng huấn luyện trước đó. Tách dữ liệu và kiểm chứng danh tính con người vẫn cần người quản lý dataset kiểm tra.

Nếu chưa có cả known positives và hard negatives của S11/S12, script sẽ dừng thay vì tạo TEST thiếu cân bằng loại mẫu.

## F. Đánh giá Phase09 realtime (Bước 5)

### Đo từng lượt riêng biệt (không đánh giá chuỗi liên tiếp)

```powershell
python scripts/live_phase09_v3_spatial.py --fullscreen --target-label xin_chao --trial-isolation
```

Phím `1=correct`, `2=wrong_position`, `3=wrong_handshape`, `4=wrong_orientation`, `5=partial`, `6=REST`, `0=unlabeled`. **Ghi một lượt xong nhấn phím lượt tiếp theo** để tạo marker `TRIAL_END`/`TRIAL_START`. Nhấn `Q` để kết thúc lượt cuối. Khi `--trial-isolation` bật, mỗi lượt được reset FSM; **không dùng chế độ này cho bài kiểm tra ký hiệu liên tiếp**.

```powershell
python scripts/analyze_v3_realtime_trials.py
```

Output `logs/phase09_v3_trial_report.json` báo tỷ lệ các lượt sai gây **token thực sự** và các lượt đúng phát token. Chỉ tính lượt có đủ `TRIAL_START` và `TRIAL_END`. Source `NO_HAND_RELEASE_HEURISTIC` và `MANUAL_REST` tách khỏi dự đoán TCN.

### Kiểm tra ký hiệu liên tiếp thật (bỏ isolation)

```powershell
python scripts/live_phase09_v3_spatial.py --fullscreen --target-label xin_chao
```

Thử `xin_chao → REST → cam_on → REST → xin_loi` và từng ký hiệu lặp `xin_chao → REST → xin_chao`. Đánh dấu thời điểm/ground truth bằng video ngoài; kiểm tra `WAIT_RELEASE`, `candidate_count`, `release_count`, `token jsonl`; không dùng độ chính xác trên rolling windows thay cho token accuracy. Tốt nhất không dùng phím B (MANUAL_REST) trong lượt đánh giá chính thức.

**Gợi ý:** lưu phiên log cũ trước khi chạy lượt mới; chương trình append vào `logs/phase09_v3_spatial_predictions.csv` nên tránh trộn các phiên. Ví dụ đổi tên file hiện tại trong Windows, rồi chạy lại.

## G. Trạng thái hoàn thành và giới hạn

- [x] Nâng cấp evaluator xuất từng sequence và QC (chạy unit tests + fake TensorFlow inference).
- [x] Truy vết nguồn và xuất checklist cho ba lỗi UNKNOWN nếu dataset thật có sẵn trên máy.
- [x] Công cụ lưu ảnh ghép video nguồn (cần file video thật).
- [x] Thu TRAIN-only và audit hard negative, chống ID leakage, guard V3.1.
- [x] Hỗ trợ TEST mới frozen scaler và kiểm tra signer chưa thấy.
- [x] Phase09 ghi marker trial và thống kê token theo từng lượt.
- [ ] Chưa truy xuất ID ba lỗi thật vì người dùng chưa gửi `test.npz`/negative manifest; sẽ xuất khi chạy evaluator trên Windows.
- [ ] Chưa xem video thực và chưa thu data mới từ camera.
- [ ] Chưa train checkpoint V3.1; chỉ có code train V3 từ gói trước.
- [ ] Chưa đánh giá blind TEST S11/S12 và realtime trên laptop/Raspberry Pi.

**Không có con số accuracy nào của V3.1 được suy diễn hoặc tạo giả.**
