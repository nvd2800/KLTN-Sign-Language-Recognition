# FIX Feature Diagnostic — cấu trúc chuẩn theo ảnh (Session / Condition / Sample)

## 1. Cấu trúc cuối cùng

```text
logs/
└── feature_diagnostic/
    ├── manifest.csv
    ├── reports/
    └── xin_chao/
        └── S03/
            ├── SES01/
            │   ├── correct/
            │   │   ├── correct_S03_SES01_001/
            │   │   │   ├── raw_landmarks.npz
            │   │   │   ├── features.npz
            │   │   │   ├── metadata.json
            │   │   │   └── source_video.avi
            │   │   └── correct_S03_SES01_002/
            │   ├── wrong_position/
            │   │   └── wrong_position_S03_SES01_001/
            │   ├── wrong_handshape/
            │   │   └── wrong_handshape_S03_SES01_001/
            │   ├── wrong_orientation/
            │   │   └── wrong_orientation_S03_SES01_001/
            │   ├── partial/
            │   │   ├── partial_S03_SES01_001/
            │   │   └── partial_S03_SES01_002/
            │   └── correct_repeat/
            │       └── correct_repeat_S03_SES01_001/
            └── SES02/
                └── partial/
                    └── partial_S03_SES02_001/
```

Các phím 1–6 là sáu trạng thái **song song** trong thư mục Session, không phải sáu folder bên trong một `sample_id`.
Mỗi trạng thái có bộ đếm riêng, bắt đầu `001`; khi đổi Session bắt đầu lại từ `001`.

| Phím | Condition | Sample ID minh họa |
| --- | --- | --- |
| `1` | `correct` | `correct_S03_SES01_001` |
| `2` | `wrong_position` | `wrong_position_S03_SES01_001` |
| `3` | `wrong_handshape` | `wrong_handshape_S03_SES01_001` |
| `4` | `wrong_orientation` | `wrong_orientation_S03_SES01_001` |
| `5` | `partial` | `partial_S03_SES01_001` |
| `6` | `correct_repeat` | `correct_repeat_S03_SES01_001` |

## 2. Cài patch đúng

Giải nén ZIP vào thư mục gốc VS Code `Sign-Language-Recognition-Phase0`, giữ đúng vị trí file. Sao lưu `core/feature_diagnostic_video.py` và các scripts bị thay trước khi ghi đè. Không cần thay `core/spatial_handshape_diagnostic.py`, TCN V3 513D, scaler hoặc dataset TRAIN/VAL/TEST đã tạo.

```powershell
python -m pytest -q tests/test_feature_diagnostic_condition_layout.py
```

## 3. Thu mẫu mới (tự lưu video)

```powershell
python scripts/live_feature_diagnostic.py --label xin_chao --subject S03 --session SES01 --frames 32 --fullscreen
```
Trong cửa sổ webcam:
- Phím 2: Sai vị trí tay.
- Phím 3: Sai hình dạng tay (nắm/mở).
- Phím 4: Sai hướng lòng bàn tay.
- Phím 5: Ký hiệu chưa hoàn chỉnh.

Sử dụng phím 1–6. Mỗi lần lưu thành công có `raw_landmarks.npz`, `features.npz`, `metadata.json`, `source_video.avi` (nếu --video-format mp4 thì dùng `source_video.mp4` hoặc fallback AVI). `manifest.csv` ở gốc được cập nhật một dòng/mẫu. Video giữ nguyên chiều camera, không chèn landmark. Các trường `label`, `subject`, `session`, `condition`, `sample_id`, `video`, `qc` đều được ghi trong metadata.

Thu hard negative của signer TRAIN (vẫn kiểm tra split):

```powershell
python scripts/collect_v3_train_negative.py --label xin_chao --subject S03 --session SES01 --frames 32
```
thu tesst/val
python scripts/live_feature_diagnostic.py --label xin_chao --subject S03 --session SES01 --frames 32 --fullscreen
## 4. Sắp xếp lại dữ liệu cũ — KHÔNG thu lại

Có hai layout cũ đã tồn tại:

```text
A. label/subject/condition/timestamp/
B. label/subject/session/label_subject_session_001/
```

Bản mới tạo **layout C**: `label/subject/session/condition/condition_subject_session_001/`.

**Trước khi chuyển: sao lưu toàn bộ thư mục `logs/feature_diagnostic`.**

Xem trước (chưa thay đổi file):

```powershell
python scripts/migrate_feature_diagnostic_layout.py
```

Nếu mẫu A không có `session` trong `metadata.json`, tool mặc định **bỏ qua**, không tự gán Session. Chỉ khi bạn **biết chắc** Session ban đầu của các mẫu này, mới dùng `--default-session SES01`.

```powershell
python scripts/migrate_feature_diagnostic_layout.py --default-session SES01
```

Xem đường dẫn nguồn/đích trong `preview`, kiểm tra số mẫu `planned`, `skipped`. Sau khi đã sao lưu và xác nhận:

```powershell
python scripts/migrate_feature_diagnostic_layout.py --execute
```

Hoặc nếu dùng session mặc định đã xác minh:

```powershell
python scripts/migrate_feature_diagnostic_layout.py --default-session SES01 --execute
```

Tool di chuyển **nguyên thư mục**: video/NPZ không đổi byte; metadata cập nhật `sample_id` mới, lưu `legacy_sample_id` và `legacy_trial_dir`. `manifest.csv` được tạo lại; dữ liệu mới không bị đè. Nếu quá trình di chuyển phát sinh lỗi, tool cố khôi phục thư mục và metadata gốc; bản sao lưu bên ngoài vẫn là bắt buộc.

**Không tự đổi tên folder bằng File Explorer**, vì sẽ làm lệch metadata, manifest và tham chiếu của các bước sau. Không xóa bản ghi thiếu video.

## 5. Rebuild manifest / audit / review video

```powershell
python scripts/rebuild_feature_diagnostic_manifest.py
python scripts/audit_feature_diagnostic_layout.py
python scripts/review_feature_diagnostic_videos.py --label xin_chao --subject S03 --session SES01 --contact-sheets
python scripts/review_feature_diagnostic_videos.py --label xin_chao --subject S03 --session SES01 --play
```

Báo cáo audit: `logs/feature_diagnostic/reports/layout_audit.json`. Một mẫu chưa có video sẽ bị đánh dấu `REVIEW`, còn dữ liệu cũ chưa chuyển có trạng thái `LEGACY_LAYOUT`; không bị xóa.

## 6. Tương thích Phase V3 và V3.1

Các script `prepare_v3_spatial.py`, `audit_v3_train_hardnegatives.py`, `diagnose_position_sensitivity.py` và `prepare_v3_blind_test.py` tìm theo `rglob('metadata.json')` rồi đọc `label`, `subject`, `condition` từ metadata; vì vậy vẫn quét được layout C khi các metadata hiện diện đầy đủ. Script so sánh Diagnostic cũng dùng `rglob('metadata.json')`. Bản vá đồng bộ các script có dùng đường dẫn cố định, gồm reviewer và manifest rebuilder.

V3.1 phải giữ nguyên quy tắc phân tách signer TRAIN/VAL/TEST: thay đổi vị trí file không đồng nghĩa thay đổi split. Không train lại chỉ vì thay layout; chỉ `prepare_v3_spatial.py` lại khi chủ động muốn nạp thêm mẫu mới.

## 7. Kiểm soát chất lượng

- Chạy test Python, sau đó quay thử một mẫu `partial` bằng webcam Windows.
- Kiểm tra video đọc lại được và `frames` khớp metadata.
- Kiểm tra `partial_S03_SES01_001` chứa đủ NPZ/metadata/video.
- Kiểm tra manifest có `subject=S03`, `session=SES01`, `condition=partial`.
- Chạy `review_feature_diagnostic_videos.py --play` để đối chiếu video với nhãn.
- Khi đủ dữ liệu, chạy audit hard negatives và chuẩn bị lại V3.1 nếu thực sự cần.

*Lưu ý*: Các bài test được chạy trên dữ liệu tạo trong môi trường phát triển; chưa xác nhận codec trên webcam Windows của bạn.
