# Dataset V2 Contract — Phase 0

## 1. Nguyên tắc
- Một **sample** ở Phase 1 sẽ là **một lần thực hiện ký hiệu hoàn chỉnh**, không phải một frame.
- Raw landmarks được giữ nguyên để Phase 4 có thể thay đổi normalization mà không cần quay lại dữ liệu.
- Hand identity dùng **physical left/right của người ký**. Mirror chỉ dùng cho preview.
- Missing landmark phải có **presence mask**, không được suy luận chỉ từ vector zero.
- Tên người không đi vào dataset; dùng `S01`, `S02`, ...

## 2. Cấu trúc thư mục
```text
dataset/raw/<sign>/<subject>/<session>/<sample_id>/
    landmarks.npz
    metadata.json
    video.mp4        # optional, Phase 1
```

Ví dụ:
```text
dataset/raw/xin_chao/S01/SES01/xin_chao_S01_SES01_001/
```

## 3. `landmarks.npz`
- `timestamps`: `(T,)`, float64
- `left_hand`: `(T, 21, 3)`, float32, XYZ
- `right_hand`: `(T, 21, 3)`, float32, XYZ
- `pose`: `(T, P, 3)`, float32; P do MediaPipe version quyết định
- `face`: `(T, F, 3)`, float32; raw face optional ở preprocessing về sau
- `left_present`: `(T,)`, uint8
- `right_present`: `(T,)`, uint8
- `pose_present`: `(T,)`, uint8
- `face_present`: `(T,)`, uint8

Nếu modality missing trong một frame, slot dữ liệu được zero-fill nhưng presence mask = 0. Zero-fill chỉ là storage convention; mask mới là nguồn sự thật.

## 4. Metadata bắt buộc
`dataset_version`, `schema_version`, `sample_id`, `sign`, `subject_id`, `session_id`, `camera_id`, `fps`, `frame_count`, `resolution`.

Các field mô tả: `lighting`, `distance`, `left_hand_ratio`, `right_hand_ratio`, `pose_ratio`, `face_ratio`.

## 5. Quy tắc label
Mini-pipeline Phase 1 bắt đầu với:
- `xin_chao`
- `cam_on`
- `khong`
- `REST`: không thực hiện sign, tư thế nghỉ/trung tính.
- `UNKNOWN`: có gesture/chuyển động nhưng không thuộc vocabulary.

`REST != UNKNOWN`.

## 6. Những gì Phase 0 cố ý chưa làm
Không normalization, không velocity/acceleration, không GRU/TCN, không threshold UNKNOWN, không state machine, không sentence/TTS.
