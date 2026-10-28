# PHASE 09 — Continuous Sign Recognition

## Mục tiêu

Từ luồng dự đoán Phase 08 (TCN + rejection) tạo ra **token** đã xác nhận, không phát lặp khi giữ nguyên động tác; hỗ trợ `A -> REST -> A` và nhật ký sự kiện. Đây là **phần decision/state-machine** có thể tích hợp vào live Phase07/08, chưa tự động tích hợp camera khi chưa đối chiếu toàn bộ code và feature pipeline trong máy người dùng.

## Các state

- `IDLE`: chờ một dự đoán known đạt threshold.
- `CANDIDATE`: các cửa sổ liên tiếp dự đoán cùng class; yêu cầu `min_consecutive=3` (đề xuất).
- `STABLE`: ứng viên đủ vững; chỉ phát **một** token.
- `WAIT_RELEASE`: khóa emission; cần quan sát `REST` đủ `release_consecutive=3` lần mới mở lại.

`UNKNOWN` mặc định **không** phải bằng chứng đã release. Nếu đồng nhất UNKNOWN với REST quá sớm, gesture lạ giữa các frame có thể mở khóa và phát lặp.

## Input

Mỗi lần Phase08 có dự đoán của **một temporal window**, truyền:

```python
engine.update(label='xin_chao', confidence=0.94, timestamp_s=1.2)
```

`label` đã phải qua Phase08 threshold. Nhãn `REST` là dự đoán lớp REST, còn `UNKNOWN` là rejected OOV. Timestamp phải tăng.

## Integration snippet

```python
from core.continuous_phase9 import ContinuousRecognizer, Phase09Config

engine = ContinuousRecognizer(Phase09Config(
    confidence_threshold=calibrated_threshold,  # đọc từ Phase08
    min_consecutive=3,
    release_consecutive=3,
    min_token_interval_s=0.4,
))

# Trong vòng lặp inference, KHÔNG phải trong vòng lặp từng frame:
# label = kết quả sau Phase08; conf = max_softmax
new_token = engine.update(label, conf, timestamp_s=time.perf_counter())
if new_token is not None:
    sentence_tokens.append(new_token.label)
    print('CONFIRMED TOKEN:', new_token.label)
```

Nếu segmentation Phase07 chỉ infer một lần sau END, không thể áp dụng trực tiếp 3-window voting; cần đưa các preview inference có mốc thời gian vào Phase09, hoặc điều chỉnh luật confirmation cho phù hợp. Không đếm 3 frame camera như 3 temporal predictions.

## Chạy smoke test với dữ liệu demo

Từ thư mục gốc project:

```powershell
pytest -q tests/test_phase9.py
python scripts/make_phase9_demo.py
python scripts/run_phase9_stream.py --input logs/phase9_demo_input.csv
```

File output:

```text
logs/phase9_events.jsonl
```

Nếu `artifacts/phase8_unknown/calibrated_threshold.json` chưa tồn tại, script báo warning và dùng threshold tạm `0.85`. **Không được gọi đó là threshold chính thức**.

## Bộ chỉ số cần kiểm tra bằng camera thật

1. Token precision / recall / F1 so với transcript tham chiếu và tolerance timestamp đã định trước.
2. Duplicate emission rate khi giữ một ký hiệu lâu.
3. Same-sign repeat: A -> REST -> A tạo ra hai tokens.
4. False tokens/min khi đứng yên, gesture lạ và transition.
5. Latency từ kết thúc ký hiệu đến emission.
6. UNKNOWN không bị emit thành từ.

## Điều kiện hoàn thành

- Phase08 threshold được calibrate đúng trên validation UNKNOWN.
- Test trên stream quay thực với nhãn thời gian, không chỉ unit-test.
- Một sign giữ lâu phát một token.
- Hai sign giống nhau cách nhau REST phát hai token.
- Log event và thống kê lỗi phát lặp/false positive.

## Giới hạn

Tập 4 class hiện tại (`REST`, `cam_on`, `xin_chao`, `xin_loi`) không đủ để chứng minh khả năng xây câu có ngữ nghĩa đa dạng. Ghép token thành câu/TTS thuộc các phase sau. Các tham số 3-window, 3-release, 0.4 s chỉ là giá trị khởi tạo; hiệu chỉnh bằng VAL/live recording, không sử dụng locked TEST để tune.
