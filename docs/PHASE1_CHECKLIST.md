# PHASE 1 EXIT CHECKLIST

## Collector
- [ ] Fixed mode chạy ổn định.
- [ ] Manual mode chạy ổn định.
- [ ] Countdown hoạt động.
- [ ] Q/ESC abort không lưu sample dở.
- [ ] Sample index tự tăng.
- [ ] Không overwrite dữ liệu cũ.
- [ ] Save atomic `.tmp -> final` hoạt động.
- [ ] Auto verify sau save PASS.

## Dataset contract
- [ ] Một sample = một lần thực hiện sign.
- [ ] Mỗi sample có `landmarks.npz`.
- [ ] Mỗi sample có `metadata.json`.
- [ ] Hand shape = `(T,21,3)`.
- [ ] Có timestamps.
- [ ] Có presence masks.
- [ ] Metadata sign/subject/session/sample đúng.
- [ ] Frame count nằm trong 30–60.

## Mini dataset
- [ ] `xin_chao`
- [ ] `cam_on`
- [ ] `khong`
- [ ] `REST`
- [ ] `UNKNOWN`
- [ ] S01
- [ ] S02
- [ ] S03
- [ ] S04
- [ ] Khoảng 10 sequence/class/người.
- [ ] UNKNOWN đủ đa dạng.
- [ ] REST đúng nghĩa neutral/no sign.

## Audit
- [ ] `python scripts/inspect_dataset.py --strict` PASS.
- [ ] `python scripts/validate_phase1_dataset.py` PASS.
- [ ] Khi đủ mini dataset: `--require-mini-complete` PASS.
- [ ] Mở kiểm tra ngẫu nhiên 10–20 sample.
- [ ] Không có thư mục `.tmp` còn sót.

## Exit gate
- [ ] Collector PASS.
- [ ] Schema PASS.
- [ ] Mini dataset PASS.
- [ ] Dataset audit PASS.
- [ ] Chưa chia train/val/test trước Phase 2.
