KLTN – V3.1 TEST TRACE / HARD NEGATIVE / REALTIME PATCH
=======================================================

Đây là bản PATCH cho source đã cài V3 Spatial 513D, không phải project đứng riêng.

KHÔNG CHÉP ĐÈ MODEL:
    artifacts/experiments/tcn_v3_spatial/model.keras
    artifacts/experiments/tcn_v3_spatial/preprocess_stats.npz
    artifacts/experiments/tcn_v3_spatial/labels.json

File sẽ được cập nhật (HÃY BACKUP):
    scripts/evaluate_v3_spatial.py
    scripts/live_phase09_v3_spatial.py

Các file mới nằm trong core/, scripts/, tests/, docs/.

CHẠY NHANH:
    python -m pytest -q tests/test_v3_1_trace_review.py tests/test_v3_1_data_guard.py tests/test_v3_1_live_trials.py tests/test_v3_1_blind.py tests/test_v3_1_evaluate_cli.py
    python scripts/evaluate_v3_spatial.py --split test
    python scripts/review_v3_test_errors.py
    python scripts/export_v3_error_video_contact_sheets.py

XEM HƯỚNG DẪN CHI TIẾT:
    docs/QUY_TRINH_V3_1_ERROR_ANALYSIS_HARD_NEGATIVE_BLIND_TEST.md

Thông tin thực tế từ TEST cũ: 49 sequence, 46 đúng, ba UNKNOWN sai.
Ba ID lỗi CHƯA BIẾT cho tới khi chạy evaluator trên dataset thật.
Không dùng TEST cũ để train hay chọn model V3.1.
