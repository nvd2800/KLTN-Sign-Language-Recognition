# PHASE 09 — TEST REALTIME TCN V3.1 (513D)

## Cài đặt

Giải nén nội dung ZIP vào thư mục gốc `Sign-Language-Recognition-Phase0`. Đây là bản **bổ sung/patch**, không phải source đầy đủ. Yêu cầu đã có Phase04–06, Phase07, V3 Spatial 513D, Phase09 và các dependencies Python.

Sao lưu `scripts/live_phase09_v3_spatial.py` nếu muốn giữ bản cũ. ZIP bổ sung launcher, preflight, config riêng V3.1 và phiên bản script realtime có nhãn `--trial-isolation`.

## Chọn checkpoint

Mở `configs/phase9_v3_1_spatial.yaml`, thay cả 3 đường dẫn `model`, `stats`, `labels` cho đúng vị trí checkpoint **mới**. Mặc định: `artifacts/experiments/tcn_v3_1_spatial/`.

Nếu bạn train tại `artifacts/experiments/tcn_v3_1_spatial_verified/`, sửa cả ba sang thư mục này. Preflight sẽ báo BLOCKED nếu `dataset_summary.json` của checkpoint không khớp dataset mới `140 TRAIN / 31 VAL / 69 TEST` (khi dataset summary của máy cũng có số lượng này).

## Các lệnh VS Code

```powershell
python scripts/live_phase09_v3_1.py --dry-run
python scripts/live_phase09_v3_1.py --dry-run --check-model
python scripts/live_phase09_v3_1.py --fullscreen
python scripts/live_phase09_v3_1.py --fullscreen --target-label xin_chao --trial-isolation
python scripts/analyze_phase09_v3_1.py
```

Lệnh trực tiếp tương đương:

```powershell
python scripts/live_phase09_v3_spatial.py --config configs/phase9_v3_1_spatial.yaml --fullscreen
```

## Phím điều khiển

- `1`: đúng ký hiệu được chỉ định `--target-label`
- `2`: sai vị trí; `3`: sai handshape; `4`: sai hướng; `5`: partial
- `6`: REST, `0`: thử tự do; nhấn các phím này để gắn nhãn cho lần thử mới
- `F`: fullscreen; `V`: landmark; `M`: lật preview; `P`: pause
- `C`: xóa tokens/FSM; `R`: reset buffer; `B`: manual REST (CHỈ debug)
- `A`: bật/tắt no-hand release heuristic, `Q`: thoát

`--trial-isolation`: mỗi lần nhấn phím số (1–6) thì reset State Machine, phục vụ đo một động tác riêng biệt. **Không dùng khi kiểm tra chuỗi ký hiệu liên tiếp.**

Đợi 32 frame sau khi đổi tình huống để buffer đầy; sau đó mỗi 8 frame chương trình infer lại. Để đo phát token cần xác nhận một ký hiệu qua nhiều prediction, không chỉ xem RAW confidence.

## Log

- `logs/phase09_v3_1_spatial_predictions.csv`
- `logs/phase09_v3_1_spatial_tokens.jsonl`
- `logs/phase09_v3_1_spatial_snapshots/`

Ghi `TCN_V3_SPATIAL` là tên nhánh xử lý trong code, nhưng model thực tế được lấy từ cấu hình **V3.1**; xem terminal `Loading ...` và đường dẫn model để xác nhận.

## Lưu ý về đánh giá

- Realtime trên Windows chưa được chạy trong môi trường tạo ZIP; cần kiểm tra trên máy có webcam.
- Nếu chỉ có `history.json` trên 126/26 và `evaluation_val.json` có 26 mẫu, checkpoint có thể thuộc dataset cũ; phải xác thực trước khi tuyên bố kết quả V3.1.
- `--check-model` yêu cầu TensorFlow đã cài; `--dry-run` thông thường chỉ kiểm tra tệp, scaler, labels, provenance.
- Không dùng phím `B` khi thống kê release tự nhiên. No-hand release là heuristic, không phải REST do TCN dự đoán.
- Dùng signer thật độc lập cho test; dữ liệu TEST đã dùng để phát hiện lỗi V3 không còn blind đối với việc phát triển V3.1.
