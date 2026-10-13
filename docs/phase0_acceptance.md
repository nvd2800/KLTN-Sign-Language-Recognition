# Phase 0 Acceptance Checklist

## Gate A — Project architecture
- [x] SVM không còn được import trong core mới.
- [x] Code SVM V1 được cô lập trong `legacy/svm/` để truy vết lịch sử, không tham gia pipeline mới.
- [x] Có `core/`, `configs/`, `scripts/`, `tests/`, `docs/`, `dataset/`.

## Gate B — Camera/detector
- [ ] Camera thực tế mở ổn định 10–20 phút trên máy của nhóm.
- [ ] Hands phát hiện 0/1/2 tay mà không crash.
- [ ] Mất face/pose/hand không crash.
- [ ] Physical Left/Right được xác nhận bằng `scripts/check_landmarks.py` trên camera thực tế.

## Gate C — Data contract
- [x] Hand raw shape `(21,3)`.
- [x] Có timestamp.
- [x] Có presence mask.
- [x] Có schema subject/session/sample.
- [x] Có định nghĩa REST/UNKNOWN.
- [x] Save/load `.npz + metadata.json` có automated test.

## Gate D — Reproducibility
- [x] Không hard-code model path.
- [x] Camera/MediaPipe/dataset config tập trung trong YAML.
- [x] Có environment inspection script.
- [x] Có dataset/schema version.

## Kết luận
Code-side Phase 0 hoàn tất khi automated tests PASS. Hardware-side Phase 0 chỉ được đánh dấu PASS sau khi nhóm chạy Gate B trên webcam thật, đặc biệt test Left/Right.
