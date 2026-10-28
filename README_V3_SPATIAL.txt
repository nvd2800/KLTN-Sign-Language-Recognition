KLTN - V3 SPATIAL 513D + REAL HARD NEGATIVES + RELEASE FIX

DANH SACH THAY DOI:
  MOI: core/feature_spatial.py (60D), core/release_v3.py
  CAP NHAT TUONG THICH NGUOC: core/preprocess_phase7.py,
    core/phase8_live.py, scripts/prepare_phase7_dataset.py
  MOI: scripts/prepare_v3_spatial.py, train_v3_spatial.py,
    diagnose_position_sensitivity.py, live_phase09_v3_spatial.py,
    analyze_v3_log.py, audit_v3_spatial.py
  MOI: configs/phase7_spatial.yaml, configs/phase9_v3_spatial.yaml
  TEST: tests/test_v3_*.py

CACH CHAY TAI WINDOWS VS CODE - PROJECT ROOT:
  python scripts/audit_v3_spatial.py --stage features
  python -m pytest -q tests/test_v3_spatial.py tests/test_v3_split.py tests/test_v3_prepare.py
  python scripts/prepare_phase7_dataset.py --config configs/phase7_spatial.yaml

  # Thu hand-negatives bang Diagnostic DA CO, phim 2,3,4,5, nhieu signer:
  python scripts/live_feature_diagnostic.py --label xin_chao --subject S01 --frames 32

  python scripts/prepare_v3_spatial.py --mode dev
  python scripts/audit_v3_spatial.py --stage dataset
  python scripts/train_v3_spatial.py --epochs 120
  python scripts/evaluate_v3_spatial.py --split test
  python scripts/diagnose_position_sensitivity.py --synthetic-shift 1.0
  python scripts/live_phase09_v3_spatial.py --fullscreen --target-label xin_chao
  python scripts/analyze_v3_log.py

Strict only after independent VAL and TEST signers each have REAL labelled negatives:
  python scripts/prepare_v3_spatial.py --mode strict

Prerequisites from your existing project: Phase04–06 modules+configs,
core/models_phase7.py, core/continuous_phase9.py, core/phase9_live_logic.py,
scripts/live_feature_diagnostic.py + original dataset Phase06/raw.

IMPORTANT:
- 513D model requires new TRAIN-only scaler. Never run V3 with V2 stats.
- Old V2 checkpoint/artifacts are left untouched.
- No real TensorFlow training or webcam tests were performed in the creator
  environment. It lacks your raw dataset/model/GPU/camera, and TensorFlow.
- 14/14 offline unit/synthetic dataset tests passed locally. End-to-end model
  quality requires actual additional real negatives and signer-independent TEST.
- See docs/QUY_TRINH_V3_SPATIAL_513.md for every step and detailed caveats.
