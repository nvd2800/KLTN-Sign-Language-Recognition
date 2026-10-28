# Phase 08 — UNKNOWN subset / 1-signer quick test

**Không bắt buộc thu đủ 9 kiểu UNKNOWN.** Bạn có thể giữ ba category:
`holding_phone`, `scratching_head`, `touching_face` và phát triển thêm khi cần.

## Những gì đã sửa

- `process_phase8_unknown.py` tự quét toàn bộ category có thực, không đòi đủ 9 category.
- `--quick-test`: tạo manifest VAL tạm từ các UNKNOWN hiện có (kể cả chỉ S01), không bịa TEST.
- Dùng Phase04/05/06 gốc, feature [T,453], ID riêng theo category; không đụng dataset KNOWN.
- `calibrate_phase8_dev.py`: thử các confidence threshold của TCN V1, lưu vào `artifacts/phase8_unknown_dev/`.
- `calibrate_phase8_threshold.py`, `evaluate_phase8_unknown.py`, `audit_phase8.py` được chặn không công nhận quick test là đánh giá chính thức.

## Cài

Giải nén, copy file `.py` trong `scripts/` vào `scripts/` của project (backup các file cùng tên trước).
Giữ nguyên các core Phase04/05/06/07/08 và configs hiện tại.

## Chạy với 3 UNKNOWN S01 hiện có

```powershell
python scripts/process_phase8_unknown.py --quick-test
python scripts/prepare_phase8_unknown.py
python scripts/calibrate_phase8_dev.py
```

Nếu đã có `unknown_manifest.csv` từ lần trước, hãy kiểm tra file đó thật kỹ rồi mới dùng:

```powershell
python scripts/process_phase8_unknown.py --quick-test --overwrite-manifest
```

Expected: `TOTAL=3 PASS=3 FAIL=0`, UNKNOWN VAL=3, UNKNOWN TEST=0 (đúng với dev).

Đầu ra quick test:

```
dataset/unknown_phase8/unknown_manifest.csv
dataset/unknown_phase8/unknown_manifest_mode.json
dataset/processed/unknown_phase8/val_unknown.npz
dataset/processed/unknown_phase8/test_unknown.npz  # N=0
artifacts/phase8_unknown_dev/threshold_sweep_dev.csv
artifacts/phase8_unknown_dev/threshold_dev_only.json
```

**Lưu ý:** ba mẫu UNKNOWN từ một người chỉ đủ kiểm thử kỹ thuật đường ống. Không dùng để công bố tỷ lệ FAR/FRR đại diện, không coi Phase 08 PASS. Đây không phải train lại TCN; chỉ đánh giá confidence và thử threshold.

## Khi muốn đánh giá thật, không cần thêm loại gesture bắt buộc

Chỉ cần thu thêm **các mẫu ở bất kỳ category đã chọn**, nhưng ở **ít nhất hai signer khác nhau** cho UNKNOWN VAL và UNKNOWN TEST, tốt hơn thêm nhiều signer và đa dạng động tác trong các nhóm đang có.

Ví dụ với S01 và S02 (chỉ khi chúng là **hai người khác nhau**):

```powershell
python scripts/process_phase8_unknown.py --val-subjects S01 --test-subjects S02 --overwrite-manifest
python scripts/prepare_phase8_unknown.py
python scripts/calibrate_phase8_threshold.py
python scripts/evaluate_phase8_unknown.py
python scripts/audit_phase8.py
```

Nếu cần một protocol nghiêm ngặt hơn, UNKNOWN signer cho VAL và TEST nên độc lập, và các category/tình huống ngoài tập đã quan sát cũng nên được đánh giá thêm.

## Các giới hạn

- 3 mẫu từ 1 người: confidence threshold có thể bị overfit.
- Max-softmax threshold có thể không chặn được UNKNOWN rất giống known.
- `REST` là class KNOWN; không bắt buộc phải có category `no_hand` trong UNKNOWN.
- Nếu manifest cũ thuộc bộ dữ liệu khác, **không ghi đè bừa**; lưu backup trước.
