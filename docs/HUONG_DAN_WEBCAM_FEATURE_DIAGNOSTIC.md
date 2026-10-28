# Phase 09 – Webcam Feature Diagnostic: Spatial-Aware + Handshape-Aware

## 1. Mục tiêu

Công cụ này **chẩn đoán feature**, **không train TCN V3**, **không sửa model TCN V2**, **không ghi vào `dataset/raw`**.

Đo trực tiếp sự khác biệt giữa 3 trường hợp cùng **một nhãn ký hiệu** và **một người**:

1. **correct** – ký hiệu chuẩn: hình dạng, vị trí, hướng và chuyển động đúng.
2. **wrong_position** – giữ hình dạng và chuyển động, nhưng đặt tay sai vị trí theo quy ước ký hiệu.
3. **wrong_handshape** – giữ vùng tay, đổi nắm/mở hoặc sai cấu hình ngón.

Các trường hợp tùy chọn: **wrong_orientation**, **partial**, **correct_repeat**. Lưu ít nhất 2–3 lần mỗi tình huống. **correct_repeat** giúp ước tính nhiễu camera/tốc độ giữa hai lần diễn đạt đúng.

## 2. File trong gói

```
core/spatial_handshape_diagnostic.py      # Feature diagnostic 453 + candidate 32, compare/report
scripts/live_feature_diagnostic.py        # Webcam collector, overlay, (tùy chọn) TCN V2
scripts/compare_feature_diagnostic.py     # So sánh saved trials không cần camera
scripts/audit_feature_diagnostic.py       # Audit dữ liệu và danh sách điều kiện còn thiếu
configs/feature_diagnostic.yaml           # Ghi chú giá trị mặc định
Tests ...
```

Chỉ **chép thêm các file mới** vào project. Không thay `core/feature_phase4.py`, `core/feature_phase5.py`, `core/feature_phase6.py`, model, scaler hoặc lớp State Machine Phase09.

Gói dựa vào các file đã có trong project:

```
core/feature_phase4.py
core/feature_phase5.py
core/feature_phase6.py
core/phase8_live.py
core/preprocess_phase7.py        # chỉ khi dùng --with-model
configs/feature_phase4.yaml
configs/feature_phase5.yaml
configs/feature_phase6.yaml
```

## 3. Chạy trong Windows PowerShell (VS Code)

Tại thư mục gốc:

```powershell
conda activate venv
pytest -q tests/test_spatial_handshape_diagnostic.py
python scripts/live_feature_diagnostic.py --label xin_chao --subject S01 --frames 45
```

Nếu muốn **đối chiếu cả dự đoán của model TCN V2** (và model/scaler/labels tồn tại):

```powershell
python scripts/live_feature_diagnostic.py --label xin_chao --subject S01 --frames 45 --with-model
```

Fullscreen:

```powershell
python scripts/live_feature_diagnostic.py --label xin_chao --subject S01 --fullscreen --with-model
```

Nếu webcam là camera 1:

```powershell
python scripts/live_feature_diagnostic.py --label cam_on --subject S01 --camera 1
```

Nếu muốn lưu cả video để xem lại (tốn nhiều dung lượng): thêm `--save-video` (AVI/MJPG). Mặc định vẫn lưu raw landmarks, features và metadata, đủ cho phân tích.

Nếu chỉ muốn kiểm tra dependency/config không mở camera:

```powershell
python scripts/live_feature_diagnostic.py --dry-run
python scripts/live_feature_diagnostic.py --dry-run --with-model
```

## 4. Phím thao tác

| Phím | Chức năng |
|---|---|
| 1 | Thu đúng ký hiệu (correct) |
| 2 | Thu sai vị trí tay (wrong_position) |
| 3 | Thu sai hình dạng ngón (wrong_handshape) |
| 4 | Thu sai hướng lòng bàn tay (wrong_orientation) |
| 5 | Thu một phần động tác (partial) |
| 6 | Thu thêm ký hiệu chuẩn lần nữa (correct_repeat) |
| G | So sánh các lần thu có sẵn, sinh CSV/JSON |
| V | Bật/tắt landmark |
| M | Mirror chỉ giao diện, không sửa tọa độ dùng tính feature |
| F | Fullscreen |
| Q / ESC | Thoát |

Sau khi nhấn 1–6 có countdown mặc định 1.5 giây, sau đó thu 45 frame từ webcam thực, báo `SAVED` và đường dẫn trên terminal.

**Quan trọng:** Sử dụng 45 frame **raw**, và nếu bật model, hệ thống resample thành **32 frame** theo pipeline Phase07, không cần train lại.

## 5. Các nhóm feature đang so sánh

### Existing 453 (không đổi)

| Nhóm | Chỉ số [start:end) | Số chiều |
|---|---|---:|
| Local hand position | 0:126 | 126 |
| Finger angles | 126:156 | 30 |
| Legacy distances | 156:165 | 9 |
| Local velocity | 165:291 | 126 |
| Local acceleration | 291:417 | 126 |
| Pose | 417:444 | 27 |
| Face | 444:453 | 9 |

### Candidate Spatial + Handshape 32 chiều (CHỈ CHẨN ĐOÁN)

| Nhóm | Vị trí | Số chiều |
|---|---|---:|
| Vị trí tay so với cơ thể – mỗi tay 6 giá trị | 0:12 | 12 |
| Finger curl 5 ngón × 2 tay | 12:22 | 10 |
| Palm normal / orientation proxy × 2 tay | 22:28 | 6 |
| Vận tốc cổ tay XY tương đối tâm vai | 28:32 | 4 |

- Vị trí tay–cơ thể dùng cùng hệ **XY normalized image** và chia cho bề rộng hai vai. **Không trộn Hand Z với Pose Z**.
- Finger curl: từ góc hai khớp ngón, quy ước gần 0 duỗi, gần 1 co; là **proxy** chưa hiệu chuẩn.
- Palm normal: tính từ 3D single-camera MediaPipe, có thể không ổn định và đảo dấu; cần kiểm chứng khi xoay tay.
- Không có tay/pose/face ⇒ feature bằng 0, mask bằng `false`; `coverage` trong report cho biết tỷ lệ giá trị thực sự so sánh được.
- Candidate32 **chưa được ghép vào TCN V2**. Input model vẫn **32 × 453**.

## 6. Đầu ra

Ví dụ:

```
logs/feature_diagnostic/
  xin_chao/
    S01/
      correct/
        20261018T...Z/
          raw_landmarks.npz
          features.npz
          metadata.json
      wrong_position/...
      wrong_handshape/...
      correct_repeat/...
  reports/
    comparison_xin_chao_S01.csv
    comparison_xin_chao_S01.json
    audit_xin_chao_S01.json
```

Dữ liệu thu chẩn đoán **không tự động thêm vào TRAIN** hoặc gán UNKNOWN.

## 7. So sánh offline

Sau khi có `correct` và ít nhất một điều kiện còn lại:

```powershell
python scripts/compare_feature_diagnostic.py --label xin_chao --subject S01
python scripts/audit_feature_diagnostic.py --label xin_chao --subject S01
```

Report ghi RMS, MAE và coverage các nhóm feature, cộng dự đoán TCN (nếu được bật khi thu).

Diễn giải ví dụ:

- **wrong_position**: `hand_position126` RMS thấp, `hand_body_position12` RMS cao ⇒ Phase04 đã loại ảnh hưởng vị trí, cần feature Spatial-Aware.
- **wrong_handshape**: `finger_angles30` RMS cao, `finger_curl10` RMS cao nhưng TCN vẫn dự đoán đúng class ⇒ model chưa dùng đủ handshape hoặc data chưa dạy mẫu sai.
- **wrong_orientation**: palm proxy thay đổi nhưng mô hình vẫn không phân biệt ⇒ cần dữ liệu và features cho hướng lòng bàn tay.
- **correct_repeat**: nhóm feature cũng biến thiên đáng kể ⇒ khác biệt có thể chỉ do tracking, tốc độ, góc quay, đừng kết luận từ một lần thu.

Không có threshold RMS toàn cục; chỉ số phụ thuộc từng nhóm có scale khác nhau. Không kết luận RMS cao nghĩa là model chính xác. Khi so sánh hai sequence, công cụ resample đều về 32 frame; timing/dynamic time warping chưa được giải quyết, vì vậy tốc độ diễn đạt vẫn có thể ảnh hưởng khoảng cách.

## 8. Tiêu chí chuyển sang Feature V4 / TCN V3

1. Có ít nhất 3 mẫu đúng và 3 mẫu sai vị trí/sai handshape cho mỗi class ưu tiên, từ cùng người trong một session; thêm người khác khi có thể.
2. Feature legacy có thể thay đổi nhưng model vẫn nhận sai → thêm hard negatives và thí nghiệm train lại/fine-tune.
3. Feature legacy không nhạy với sai vị trí trong khi Candidate32 nhạy rõ → thử Feature V4 và train TCN V3 cùng scaler mới.
4. Dùng tập VALIDATION có signer khác để chọn cấu hình; dùng TEST độc lập cuối cùng, không tối ưu theo TEST.
5. Không mặc định coi mọi biến thể là sai: cần quy ước ngôn ngữ ký hiệu cho từng class, tránh từ chối cử chỉ hợp lệ.

## 9. Kiểm tra lỗi thường gặp

- `No module named core.phase8_live`: chưa copy Phase08/Phase09 live core hiện có.
- `FileNotFoundError configs/feature_phase4.yaml`: chưa chạy từ project root hoặc thiếu config.
- `mediapipe has no attribute solutions`: MediaPipe API không tương thích, kiểm tra venv phiên bản `0.10.21` trước khi thay môi trường.
- `No model artifact`: bỏ `--with-model` vẫn test feature; hoặc kiểm tra đường dẫn model V2 và stats.
- `Need at least one condition=correct`: phải thu clip chuẩn trước khi so sánh.
- `coverage=0`/RMS `N/A`: landmark bắt buộc không hợp lệ trong cả hai clip; kiểm tra camera, khoảng cách, lighting.

**Phạm vi kiểm thử:** Unit tests kiểm tra toán học/shape và lưu/report bằng landmark mô phỏng; việc kiểm chứng tracking và phân loại bằng webcam Windows vẫn do bạn chạy trực tiếp.
