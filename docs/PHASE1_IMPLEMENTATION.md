# PHASE 1 — Sequence Dataset V2

## 1. Mục tiêu
Phase 1 chuyển đơn vị dữ liệu từ **frame rời** sang **một lần thực hiện ký hiệu hoàn chỉnh = một sequence**.

Phase này **không train model**, **không normalization**, **không tự động reject theo QC**, và **không chia train/val/test**. Các việc đó thuộc Phase 2–4.

## 2. Dataset mini ban đầu
- `xin_chao`
- `cam_on`
- `khong`
- `REST`
- `UNKNOWN`

Mục tiêu mini-pipeline:
- 4 subject: `S01`–`S04`
- ~10 sequence/class/người
- >=2 session theo kế hoạch dữ liệu tổng thể
- sequence 30–60 frame
- fixed-mode mặc định: 45 frame

Tổng tối thiểu cho vòng mini đầu tiên: khoảng 200 sequence.

## 3. Định nghĩa một sample
Ví dụ:

```text
dataset/raw/xin_chao/S01/SES01/xin_chao_S01_SES01_001/
├── landmarks.npz
└── metadata.json
```

`landmarks.npz` chứa:
- `timestamps`: `(T,)`
- `left_hand`: `(T,21,3)`
- `right_hand`: `(T,21,3)`
- `pose`: `(T,P,3)`
- `face`: `(T,F,3)`
- `left_present`: `(T,)`
- `right_present`: `(T,)`
- `pose_present`: `(T,)`
- `face_present`: `(T,)`

Dữ liệu là **raw MediaPipe landmarks**. Không normalize ở Phase 1.

## 4. Fixed mode
Mặc định mỗi sample lấy 45 frame. Collector countdown 3 giây rồi ghi liên tục cho tới đủ số frame.

Ví dụ:

```powershell
python scripts/collect_sequence.py --sign xin_chao --subject S01 --session SES01 --samples 10 --mode fixed --frames 45
```

Fixed mode nên là chế độ chính của mini-dataset đầu tiên vì dễ kiểm soát và nhất quán.

## 5. Manual mode
Người thu quyết định kết thúc sample bằng SPACE. Mặc định chỉ cho phép kết thúc trong khoảng 30–60 frame. Khi đạt 60 frame collector tự kết thúc.

```powershell
python scripts/collect_sequence.py --sign xin_chao --subject S01 --session SES01 --samples 10 --mode manual
```

Manual mode phù hợp để kiểm tra ký hiệu dài/ngắn trước khi chuẩn hóa temporal length ở Phase 7.

## 6. REST và UNKNOWN
`REST`: trạng thái không chủ động thực hiện ký hiệu trong vocabulary. Không nên biến REST thành một gesture đặc biệt cố định.

`UNKNOWN`: gesture/chuyển động thật nhưng ngoài vocabulary. Cần đa dạng: sờ mặt, gãi đầu, chỉ tay, cầm vật, random hand movement, transition...

Không thu UNKNOWN bằng cách lặp duy nhất một gesture.

## 7. Subject và Session
- Subject: `S01`, `S02`...
- Session: `SES01`, `SES02`...
- Không dùng tên thật trong dataset path.
- Một session là một lần/buổi thu có điều kiện tương đối nhất quán.

Không chia train/test ở Phase 1. Phase 2 sẽ dùng subject/session metadata để thực hiện signer-independent split.

## 8. Metadata
Mỗi sample tự lưu:
- dataset/schema version
- sample/sign/subject/session
- camera ID
- actual sequence FPS
- target FPS
- frame count
- resolution
- lighting/distance
- left/right hand presence ratio
- pose/face presence ratio
- collection mode
- duration
- UTC creation time

Presence ratio ở Phase 1 chỉ dùng để **ghi nhận và audit**. Việc reject dựa trên ratio thuộc Phase 3.

## 9. An toàn dữ liệu
Collector:
- không overwrite sample đã tồn tại;
- tự tìm next sample index;
- ghi vào thư mục `.tmp` trước;
- chỉ rename thành sample chính khi NPZ + JSON ghi hoàn tất;
- sample abort bằng Q/ESC không được lưu;
- tự verify lại sample ngay sau save.

## 10. Kiểm tra dataset
Sau khi thu:

```powershell
python scripts/inspect_dataset.py --strict
```

Để kiểm tra exit gate cơ bản:

```powershell
python scripts/validate_phase1_dataset.py
```

Khi đã thu đủ mini dataset:

```powershell
python scripts/validate_phase1_dataset.py --require-mini-complete
```

## 11. Điều kiện đóng Phase 1
- 1 lần ký = 1 sequence.
- Mỗi sample có sign/subject/session/sample ID.
- Mọi sequence nằm trong range 30–60 frame của mini protocol.
- REST và UNKNOWN tồn tại.
- Không có sample hỏng cấu trúc.
- Collector không overwrite dữ liệu.
- Save/load/verify PASS.
- Mini dataset mục tiêu đủ 5 class × 4 subject × ~10 sample/class/subject.
- Chưa thực hiện train/val/test split trước Phase 2.
