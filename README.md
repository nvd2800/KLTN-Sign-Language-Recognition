# KLTN Sign Language Recognition — Phase 0

Đây là bản tái cấu trúc Phase 0 theo quyết định **không dùng SVM làm baseline**. Pipeline mới tập trung vào camera, MediaPipe landmarks, physical left/right, presence mask, quality foundation và Dataset V2 contract để chuẩn bị Phase 1.

## Kiến trúc
```text
Camera -> MediaPipe Hands/Pose/Face -> FrameLandmarks XYZ + Presence Mask
       -> Basic Quality -> Dataset V2 contract -> Phase 1 Sequence Collector
```

SVM cũ chỉ nằm trong `legacy/svm/` và **không được import** bởi code mới.

## Cài đặt (Windows / VS Code)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Khuyến nghị Python 3.10/3.11 nếu MediaPipe trên phiên bản Python quá mới không có wheel tương thích.

## Chạy acceptance test phần mềm
```powershell
python scripts/validate_schema.py
pytest -q
python scripts/inspect_environment.py
```

## Test camera
```powershell
python scripts/check_camera.py --seconds 10
```

## Test Physical Left/Right + missing landmark
```powershell
python scripts/check_landmarks.py
```

Kỳ vọng:
- chỉ giơ **tay trái vật lý** -> `Left=YES Right=NO`
- chỉ giơ **tay phải vật lý** -> `Left=NO Right=YES`
- mất hand/face/pose -> không crash
- bật mirror preview không làm đổi identity lưu trữ

## Chạy preview Phase 0
```powershell
python main.py
```

## Tài liệu cần đọc trước Phase 1
- `docs/dataset_schema.md`
- `docs/collection_protocol.md`
- `docs/phase0_acceptance.md`

## Phase 0 chưa làm
Không train classifier; không normalization; không GRU/TCN; không UNKNOWN threshold; không state machine; không sentence/TTS/Web/Pi.
