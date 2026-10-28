# Frozen V3 baseline – kết quả TEST đã được cung cấp (28/10/2026)

Dữ liệu nguồn `evaluation_test.json` do người dùng cung cấp, không phải kết quả chạy lại tại đây:

- TEST N=49; Accuracy=46/49=93.877551%; Macro F1=94.596295%.
- KNOWN 34/34 đúng (gồm REST 10/10, cam_on 8/8, xin_chao 7/7, xin_loi 9/9).
- UNKNOWN 12/15 đúng; 3/15 UNKNOWN sai: 1→REST, 1→cam_on, 1→xin_loi.
- Hai lỗi UNKNOWN→ký hiệu có nghĩa có nguy cơ tạo token nếu Phase09 xác nhận. UNKNOWN→REST có thể ảnh hưởng release nhưng không phát token ký hiệu trực tiếp.
- Các báo cáo tổng hợp ban đầu **không có sample_id, signer, condition, confidence theo từng mẫu**. Không thể xác định ba trường hợp lỗi cụ thể từ file này.
- Diagnostic `xin_chao/S01` có kết quả rất tốt nhưng S01 không phải unseen signer; synthetic shift 1.0 vai là 1 mẫu tổng hợp.
- TEST hiện tại đã được xem để phân tích nên sau khi cải tiến từ các quan sát đó phải dùng blind TEST mới cho kết luận V3.1.

Chạy `python scripts/evaluate_v3_spatial.py --split test` rồi `python scripts/review_v3_test_errors.py` để truy được ba sample_id thật từ dữ liệu giữ trên máy Windows.
