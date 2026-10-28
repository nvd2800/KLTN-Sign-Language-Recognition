# FIX lưu Feature Diagnostic theo thư mục Session và Sample ID

## 1. Cấu trúc mới (đúng ảnh bạn gửi)

Ví dụ, với `--label cam_on --subject S07 --session SES01`, các mẫu được lưu như sau:

```text
logs/feature_diagnostic/
├── manifest.csv
└── cam_on/
    └── S07/
        ├── SES01/
        │   ├── cam_on_S07_SES01_001/
        │   │   ├── raw_landmarks.npz
        │   │   ├── features.npz
        │   │   ├── metadata.json
        │   │   └── source_video.avi
        │   ├── cam_on_S07_SES01_002/
        │   ├── cam_on_S07_SES01_003/
        │   └── cam_on_S07_SES01_006/
        └── SES02/
            └── cam_on_S07_SES02_001/
```

**Không có thư mục `condition`**: thay vào đó, `correct`, `wrong_position`, `wrong_handshape`, `wrong_orientation`, `partial` hay `correct_repeat` được lưu ở cả `metadata.json` và `manifest.csv`. Cách này giữ đúng cấu trúc `Signer / Session / Sample` như ảnh; các sample liên tiếp được tăng số bất kể loại điều kiện thu.

**Không dùng `--signer-alias`**. `subject` là mã signer, `session` là mã đợt thu. Để đúng với yêu cầu khóa luận, các mã signer khác nhau phải là các người khác nhau thật, không chỉ đổi mã.

## 2. Cài đặt

Sao lưu các file hiện tại rồi giải nén ZIP đè vào gốc dự án:

```text
core/feature_diagnostic_video.py
scripts/live_feature_diagnostic.py
scripts/collect_v3_train_negative.py
scripts/review_feature_diagnostic_videos.py
scripts/rebuild_feature_diagnostic_manifest.py
scripts/migrate_feature_diagnostic_layout.py
configs/feature_diagnostic.yaml
```

Gói chỉ bao gồm phần sửa, cần giữ nguyên các file `core/feature_phase4.py`, `feature_phase5.py`, `feature_phase6.py`, `phase8_live.py`, `spatial_handshape_diagnostic.py`, v.v. đã có trong project.

## 3. Thu video và feature mới

```powershell
python scripts/live_feature_diagnostic.py --label cam_on --subject S07 --session SES01 --frames 32 --fullscreen
```

Các phím vẫn như cũ:

- `1` correct
- `2` wrong_position
- `3` wrong_handshape
- `4` wrong_orientation
- `5` partial
- `6` correct_repeat
- `G` compare; `Q` quit.

Số mẫu tăng tự động: `cam_on_S07_SES01_001`, `_002`, `_003` ...; khi chuyển sang SES02, bắt đầu lại `_001`.

**Video được lưu tự động** từ frame camera gốc, cùng mẫu với landmark/feature. Nếu codec ghi lỗi, dữ liệu `.npz` vẫn được lưu và video status ghi `FAILED`, không báo `SAVED_VERIFIED` giả.

## 4. Thu hard negatives đúng TRAIN

```powershell
python scripts/collect_v3_train_negative.py --label cam_on --subject S07 --session SES01 --frames 32
```

Chương trình vẫn kiểm tra S07 thuộc TRAIN, không cho thu các signer VAL/TEST để đưa vào TRAIN.

## 5. Duyệt video và kiểm tra manifest

```powershell
python scripts/review_feature_diagnostic_videos.py --label cam_on --subject S07 --session SES01 --play
python scripts/review_feature_diagnostic_videos.py --label cam_on --subject S07 --session SES01 --contact-sheets
python scripts/rebuild_feature_diagnostic_manifest.py
```

`manifest.csv` tự cập nhật khi thu thành công, giữ các cột `sample_id`, `label`, `subject`, `session`, `condition`, `video_file`, `video_status`, `trial_dir` cùng thông tin QC. Script review và rebuild **đọc được cả cấu trúc cũ và mới**.

## 6. Nếu cần chuyển các sample Diagnostic cũ sang thư mục Session

Thư mục cũ là `label/subject/condition/timestamp`, được chương trình mới vẫn đọc được. Bạn **không bắt buộc di chuyển dữ liệu cũ**; hệ thống V3 chuẩn bị dataset bằng metadata và có thể đọc cả hai.

Nếu muốn hợp nhất cho đẹp và đồng bộ thư mục như ảnh, **sao lưu cả `logs/feature_diagnostic`** trước, rồi:

```powershell
# Chỉ xem trước, KHÔNG thay đổi file
python scripts/migrate_feature_diagnostic_layout.py

# Sau khi xác nhận danh sách chính xác
python scripts/migrate_feature_diagnostic_layout.py --execute
```

Nếu một sample cũ KHÔNG lưu `session` trong metadata, nó sẽ bị SKIP để tránh suy đoán nhầm. Chỉ dùng `--default-session SES01` nếu đã xác nhận chắc chắn các sample đó thuộc SES01.

Chế độ `--execute` di chuyển **nguyên thư mục sample** (không tái mã hóa video hoặc chỉnh sửa `.npz`), cập nhật `sample_id` và lưu `legacy_sample_id`/`legacy_trial_dir` trong metadata, sau đó rebuild manifest. Không chạy train đồng thời trong lúc migrate.

## 7. Tương thích pipeline TCN V3/V3.1

- `scripts/prepare_v3_spatial.py` đọc đệ quy `metadata.json`, nhận `condition`, `label`, `subject` từ metadata, nên không phụ thuộc thư mục condition nữa.
- `core/spatial_handshape_diagnostic.py` so sánh các trial bằng metadata tương tự.
- Không thay đổi `feature_dim=513`, số frame model 32, scaler hay checkpoint hiện tại.
- Nếu thêm dữ liệu mới để train TCN V3.1, **phải chạy lại `prepare_v3_spatial.py`** để file dataset tổng hợp chứa các sample mới. Sửa cấu trúc lưu tự nó không huấn luyện lại model.

## 8. Chạy kiểm thử

```powershell
python -m pytest -q tests/test_feature_diagnostic_session_folder.py tests/test_feature_diagnostic_session_manifest.py
```

Lưu ý: dữ liệu train, VAL, TEST vẫn cần phân chia theo đúng signer, không dùng mẫu TEST đã xem để huấn luyện lại.
