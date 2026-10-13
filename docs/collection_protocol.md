# Data Collection Protocol — Chuẩn bị cho Phase 1

## Mục tiêu
Đảm bảo dữ liệu sequence thu từ nhiều người và nhiều session có điều kiện đủ nhất quán để đánh giá khoa học.

## Thiết lập camera
1. Camera cố định ngang tầm hoặc gần ngang phần thân trên.
2. Khung hình phải chứa mặt, vai, khuỷu và cả hai bàn tay trong vùng thao tác bình thường.
3. Không để tay thường xuyên chạm/cắt mép ảnh.
4. Ghi resolution, FPS, camera ID, khoảng cách tương đối và điều kiện sáng vào metadata.
5. Preview có thể mirror; raw frame đưa detector không mirror.

## Subject/session
- Dùng ID ẩn danh `S01`, `S02`, ...
- Mỗi buổi thu là `SES01`, `SES02`, ...
- Không dùng tên thật trong tên file/thư mục.
- Phase 1 mini: 4 subject, ~10 sequence/class/người, tối thiểu 2 session khi có thể.

## Một sample
- Một sample = một lần thực hiện ký hiệu hoàn chỉnh.
- Mục tiêu ban đầu 30–60 frame/sample.
- Không cắt mỗi frame thành sample độc lập.
- Giữ timestamp để còn mô hình hóa tốc độ/thời gian.

## REST
Tư thế nghỉ hoặc không thực hiện ký hiệu. Cần nhiều biến thể tự nhiên, không chỉ một pose đứng yên duy nhất.

## UNKNOWN
Gesture ngoài vocabulary: gãi đầu, sờ mặt, cầm điện thoại, chỉ tay, vẫy tùy ngữ cảnh, random hand movement, transition... Không lặp duy nhất một gesture UNKNOWN.

## Kiểm tra trước khi thu thật
- `python scripts/check_camera.py --seconds 10`
- `python scripts/check_landmarks.py`
- Chỉ giơ tay trái: Left=YES, Right=NO.
- Chỉ giơ tay phải: Left=NO, Right=YES.
- Mirror ON/OFF không được đổi identity lưu trữ.
- Đưa tay/mặt ra khỏi khung: không crash.
- `python scripts/validate_schema.py` phải PASS.

## Quy tắc dữ liệu gốc
Không normalize vĩnh viễn tại collector. Phase 1 lưu raw landmark; Phase 3/4 tạo processed data sau.
