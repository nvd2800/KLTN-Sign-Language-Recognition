# PHASE 09 — SPATIAL + HANDSHAPE VALIDATION GATE V1

## 1. Mục tiêu

Giữ **TCN V2** làm model nền `32 x 453` (không thay đổi trọng số, scaler, labels). Bổ sung tầng hậu kiểm sau TCN để giảm việc phát token khi **vị trí tay, hình dạng ngón hoặc hướng bàn tay không đúng** theo mẫu tham chiếu.

```text
Webcam -> MediaPipe -> 453 features -> TCN V2 -> confidence/margin gate
                                   \-> Candidate32 + explicit masks
TCN known label + Candidate32 -> Spatial Gate -> PASS / REJECT / INSUFFICIENT
                                              -> Phase09 state machine
REJECT/INSUFFICIENT -> UNKNOWN (không phát token)
```

**Chú ý:** Đây là phép thử hậu kiểm có điều kiện, KHÔNG PHẢI bộ luật ngôn ngữ ký hiệu đã được thẩm định. Những biến thể ký hiệu hợp lệ của nhiều người có thể bị từ chối nếu profile quá ít mẫu hoặc threshold quá chặt.

## 2. Kết quả Diagnostic hiện có — chỉ để định hướng

Từ `comparison_xin_chao_S01.json` (6 lượt, 1 signer), TCN V2 vẫn nhận `xin_chao` ở các điều kiện thử sai:

| Điều kiện | Confidence | RMS nhóm quan trọng | RMS lặp đúng |
|---|---:|---:|---:|
| đúng | 99.75% | — | — |
| sai vị trí | 99.61% | hand_body_position12 = 0.5543 | 0.2268 |
| sai handshape | 99.22% | finger_curl10 = 0.2890 | 0.0923 |
| sai hướng | 99.64% | palm_orientation6 = 1.0293 | 0.2561 |
| partial | 95.55% | wrist_relative_velocity4 = 0.0607 | 0.1091 |
| lặp đúng | 99.94% | — | — |

RMS không phải accuracy hoặc mức quan trọng của model. Có sự thay đổi đồng thời ở nhiều nhóm; tỷ lệ so sánh hợp lệ chỉ ~0.3 cho các nhóm Candidate32 (do mặt nạ, có thể là ký hiệu một tay). Vì vậy không sử dụng các giá trị trong bảng để hard-code ngưỡng chính thức.

## 3. Cấu trúc file cài đặt

Copy vào thư mục GỐC project và **giữ nguyên TCN V2 / Phase04-06**:

```text
core/
  spatial_validation_gate.py         # NEW
  spatial_handshape_diagnostic.py    # SAME diagnostic module, included for convenience
scripts/
  build_spatial_gate_profile.py      # NEW
  audit_spatial_gate_trials.py       # NEW
  live_phase09_tcn_v2.py            # REPLACE existing live script
configs/
  phase9_live.yaml                   # REPLACE or merge spatial_gate & logging
  
tests/
  test_spatial_validation_gate.py   # NEW
```

### Quan trọng: backup

```powershell
Copy-Item scripts/live_phase09_tcn_v2.py scripts/live_phase09_tcn_v2.backup.py
Copy-Item configs/phase9_live.yaml configs/phase9_live.backup.yaml
```

## 4. Chạy kiểm thử code

```powershell
pytest -q tests/test_spatial_validation_gate.py
python -m py_compile scripts/live_phase09_tcn_v2.py
```

## 5. Dùng ngay 6 lần thử `xin_chao/S01` ở chế độ DEV

Bạn đã có:

```text
logs/feature_diagnostic/xin_chao/S01/
  correct/...
  correct_repeat/...
  wrong_position/...
  wrong_handshape/...
  wrong_orientation/...
  partial/...
```

Tạo profile **DEV_ONLY** (mẫu đúng làm thư viện tham chiếu; `correct_repeat` và các `wrong_*` dùng chọn ngưỡng riêng từng nhóm):

```powershell
python scripts/build_spatial_gate_profile.py --labels xin_chao --dev
```

Đầu ra:

```text
artifacts/spatial_gate_v1/
  profile.json
  calibration_report.json
```

Profile chứa các mẫu Candidate32 + mask và ngưỡng riêng theo `hand_body_position12`, `finger_curl10`, `palm_orientation6` khi có khả năng phân tách. Nếu một nhóm không thể phân tách, DEV sẽ nêu rõ và bỏ nhóm đó; chế độ chính thức không chấp nhận thiếu nhóm.

**KHÔNG dùng dữ liệu `partial` làm negative để đặt ngưỡng Wrist Motion:** kết quả ban đầu cho thấy khoảng cách nhỏ hơn lần lặp đúng; cần cơ chế phân đoạn chuyển động độc lập.

### Cảnh báo nghiêm túc

DEV_ONLY chọn ngưỡng từ **chỉ một lần đúng lặp và một lần sai mỗi loại trên cùng S01**. Nó là bản demo kỹ thuật. KHÔNG dùng làm kết quả KLTN hoặc đưa ra kết luận mô hình cải thiện thực tế.

## 6. Test realtime Phase09 + TCN V2 + Spatial Gate

```powershell
python scripts/live_phase09_tcn_v2.py --fullscreen --spatial-gate artifacts/spatial_gate_v1/profile.json
```

Để chạy TCN V2 cũ **không gate**, so sánh:

```powershell
python scripts/live_phase09_tcn_v2.py --fullscreen
```

Thông tin trên màn hình:
- `RAW`: class từ TCN V2.
- `DECISION`: sau confidence/presence và Spatial Gate.
- `SPATIAL: PASS`: đạt cả các nhóm đang được hiệu chỉnh.
- `SPATIAL: REJECT`: sai một/một số nhóm; gửi `UNKNOWN` tới State Machine.
- `SPATIAL: INSUFFICIENT`: landmark/mask không đủ để so sánh, không emit known token.
- `SPATIAL: BYPASS_UNCONFIGURED`: class chưa có profile (ví dụ `cam_on`, `xin_loi`) nên **không được hậu kiểm**, TCN vẫn quyết định như trước.
- `SPATIAL: SKIP`: TCN dự đoán `REST`/`UNKNOWN`.

Log mới (không trộn header CSV cũ):

```text
logs/phase09_spatial_predictions.csv
logs/phase09_live_tokens.jsonl
```

Nếu bạn muốn phân biệt token v2 (không gate) và gate trong log, nên sao lưu/xóa riêng file token cũ trước mỗi phiên test; script không tự xóa log.

## 7. Quy tắc bảo đảm không phá model

- TCN V2 giữ nguyên `model.keras` 5-class, `preprocess_stats.npz` (453) và `labels.json`.
- Spatial Candidate32 chạy **song song** ngoài model, không cộng 32 vào input 453.
- `REST`, `UNKNOWN` không được đưa qua gate.
- Khi gate từ chối, dùng `UNKNOWN`, không tự tính đó là REST và không phát token.
- Dữ liệu thiếu landmark không được coi là tọa độ 0 hợp lệ; dùng mask và kiểm tra overlap tương đối với mẫu tham chiếu.
- Một tay không xuất hiện có thể là đúng với ký hiệu một tay, nên không ép tỷ lệ trên 32*group_dim.
- Chưa hiệu chỉnh class khác thì kết quả là `BYPASS_UNCONFIGURED`, KHÔNG giả vờ xác nhận đã kiểm tra hình dạng/vị trí các class đó.
- `palm_orientation6` từ 3D đơn camera là proxy, có thể nhiễu và không được xem là góc giải phẫu tuyệt đối.

## 8. Thu thêm mẫu để hiệu chỉnh đúng

Khuyến nghị tối thiểu để bắt đầu thử nghiêm túc: với **mỗi class**, có nhiều signer, gallery chứa nhiều lần `correct`, validation từ signer KHÁC gồm tối thiểu 5 lần `correct_repeat` và mỗi nhóm `wrong_position`, `wrong_handshape`, `wrong_orientation` ít nhất 5 lần hợp lệ. Mức này mới là điểm khởi đầu, chưa phải kết luận khoa học.

Ví dụ:

```text
S01, S02 -> gallery (correct)
S03, S04 -> validation (correct_repeat, wrong_*)
S05, S06 -> unseen TEST (không chọn threshold từ TEST)
```

Hiệu chỉnh trên signer khác:

```powershell
python scripts/build_spatial_gate_profile.py `
  --labels xin_chao,cam_on,xin_loi `
  --gallery-subjects S01,S02 `
  --val-subjects S03,S04
```

Sau đó test độc lập bằng signer khác:

```powershell
python scripts/audit_spatial_gate_trials.py `
  --input logs/feature_diagnostic `
  --profile artifacts/spatial_gate_v1/profile.json
```

Script chỉ dùng signer **không nằm** trong gallery/validation. Muốn tự kiểm tra replay hiện có (có leakage, không phải TEST):

```powershell
python scripts/audit_spatial_gate_trials.py --dev-audit
```

Để từ chối DEV_ONLY trong buổi nghiệm thu chạy:

```powershell
python scripts/live_phase09_tcn_v2.py --fullscreen `
  --spatial-gate artifacts/spatial_gate_v1/profile.json --require-validated-gate
```

DEV_ONLY sẽ bị từ chối đúng thiết kế.

## 9. Bước sau Gate V1

So sánh cùng thử nghiệm mới (đúng vị trí/sai vị trí, đúng handshape/sai handshape, sai hướng, partial) giữa:
- TCN V2 không gate.
- TCN V2 + gate V1.

Đo correct-known acceptance, false-reject-known, hard-negative rejection, delay, per-signer. Nếu gate phụ thuộc signer mạnh hay tỷ lệ false reject lớn: bổ sung positive, sửa tracking, hoặc đào tạo V3 485 chiều + hard negative với split độc lập. Không cần train V3 trước khi có bằng chứng.
