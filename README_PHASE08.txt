PHASE 08 UNKNOWN / OOV DETECTION

1. Collect diverse UNKNOWN sequences.
2. Process them through the SAME Phase04->05->06 feature pipeline.
3. Create:
   dataset/unknown_phase8/unknown_manifest.csv
4. Prepare:
   python scripts/prepare_phase8_unknown.py
5. Test core:
   pytest -q tests/test_phase8_unknown.py
6. Calibrate threshold on VAL only:
   python scripts/calibrate_phase8_threshold.py
7. Evaluate locked TEST:
   python scripts/evaluate_phase8_unknown.py
8. Audit:
   python scripts/audit_phase8.py

IMPORTANT:
Phase08 is not complete until real UNKNOWN VAL/TEST data exists and threshold calibration has been run.
