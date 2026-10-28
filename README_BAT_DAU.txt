PHASE 09 | SPATIAL + HANDSHAPE WEBCAM FEATURE DIAGNOSTIC

Copy core/, scripts/, tests/ and docs/ into EXISTING project root.
Do not replace core/feature_phase4.py, core/feature_phase5.py,
core/feature_phase6.py, model.keras, scaler or labels.

1. Quick test:
   pytest -q tests/test_spatial_handshape_diagnostic.py

2. Open webcam (does not require model):
   python scripts/live_feature_diagnostic.py --label xin_chao --subject S01 --frames 45

3. Open webcam + optional frozen TCN V2:
   python scripts/live_feature_diagnostic.py --label xin_chao --subject S01 --frames 45 --with-model --fullscreen

4. Press:
   1 correct | 2 wrong_position | 3 wrong_handshape
   4 wrong_orientation | 5 partial | 6 correct_repeat
   G compare | V overlay | M mirror | F fullscreen | Q quit

5. After captures:
   python scripts/compare_feature_diagnostic.py --label xin_chao --subject S01
   python scripts/audit_feature_diagnostic.py --label xin_chao --subject S01

6. Inspect:
   logs/feature_diagnostic/reports/comparison_xin_chao_S01.csv

Scientific note: 453 features and frozen TCN V2 are unchanged.
The 32 added diagnostic dims are NOT automatically TCN-V3 model inputs.
This diagnostic does not prove that a 485-dim TCN will be more accurate.
