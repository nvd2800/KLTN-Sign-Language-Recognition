# PHASE 0 — Implementation Report

## Quyết định kiến trúc
Theo yêu cầu của đề tài, Phase 0 này **không giữ SVM làm baseline hoạt động**. Mã SVM V1 được cô lập trong `legacy/svm/` chỉ để truy vết lịch sử; pipeline mới không import hoặc phụ thuộc vào SVM.

## Pipeline Phase 0 đã triển khai
```text
RAW CAMERA FRAME
    -> CameraManager
    -> MediaPipe Hands + Pose + Face
    -> FrameLandmarks
       - physical left hand: (21,3)
       - physical right hand: (21,3)
       - pose: optional raw XYZ
       - face: optional raw XYZ
       - timestamp
       - presence mask
    -> Basic Quality Checker
    -> Dataset V2 Contract (.npz + metadata.json)
    -> Phase 1 Sequence Collector
```

## 10 bước đã thực hiện
1. Tách SVM khỏi core và đưa mã V1 vào `legacy/svm/`.
2. Tạo cấu trúc project V2: `core/configs/dataset/scripts/tests/docs/app`.
3. Tạo `CameraManager` duy nhất để open/read/release + thống kê FPS/frame fail.
4. Tạo `MediaPipeDetector` cho Hands/Pose/Face; mất modality không làm crash.
5. Tạo `FrameLandmarks` với XYZ, timestamp, presence mask.
6. Quy định canonical hand identity = **physical left/right của người ký**; mirror chỉ áp dụng ở preview.
7. Tạo `BasicQualityChecker` làm nền Phase 3.
8. Chốt Dataset V2 schema, metadata, subject/session/sample, REST/UNKNOWN.
9. Tạo preview/debug + camera check + landmark/left-right check + schema validation.
10. Tạo automated tests và acceptance checklist.

## Automated verification trong môi trường build
- `python scripts/validate_schema.py` -> PASS.
- `pytest -q` -> **6 passed**.
- `python -m compileall app core scripts tests main.py` -> PASS.

## Những kiểm tra phải chạy trên máy Windows thật
Môi trường build không có webcam/MediaPipe runtime nên các mục sau cần nhóm xác nhận trên máy thu dữ liệu:
1. `python scripts/check_camera.py --seconds 10`
2. `python scripts/check_landmarks.py`
3. Chỉ tay trái vật lý -> Left=YES, Right=NO.
4. Chỉ tay phải vật lý -> Left=NO, Right=YES.
5. Bật/tắt mirror preview không làm đổi hand identity.
6. Đưa tay/mặt ra khỏi khung -> không crash.
7. Chạy `python main.py` 10–20 phút để kiểm tra ổn định.

## Vì sao chưa đánh dấu hardware PASS
Handedness của MediaPipe phụ thuộc cách frame được mirror. Code Phase 0 đã thiết kế để raw frame không mirror và ánh xạ về physical left/right, nhưng acceptance cuối cùng phải xác nhận bằng webcam thật trước khi thu Dataset V2. Đây là gate bắt buộc để tránh thu sai toàn bộ Phase 1.

## Không nằm trong Phase 0
- Không classifier/SVM/GRU/TCN.
- Không normalization.
- Không angle/distance/velocity/acceleration.
- Không train/val/test split.
- Không UNKNOWN confidence threshold.
- Không continuous state machine.
- Không sentence/TTS/Web/Raspberry Pi.

## Gate để chuyển Phase 1
Chỉ bắt đầu collector sequence chính thức khi `docs/phase0_acceptance.md` đạt toàn bộ Gate B trên webcam thật. Sau đó Phase 1 sẽ triển khai collector fixed/manual, một lần ký hiệu = một sequence, metadata hoàn chỉnh và mini dataset 5 class.
