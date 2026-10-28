PHASE 03 QC FIX V3

Replace:
- configs/quality_phase3.yaml
- core/quality_phase3.py

Add:
- tests/test_phase3_thresholds_v3.py
- docs/PHASE3_THRESHOLD_FIX_V3.md

Rule:
>= 0.70       -> PASS
0.55 - <0.70 -> REVIEW
< 0.55        -> REJECT

REST:
no-hand + idle -> ACCEPT
hand detected / motion -> REVIEW
structural/data error -> REJECT
