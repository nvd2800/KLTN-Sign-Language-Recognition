PHASE 06 - UPPER BODY POSE + FACE

Files:
configs/feature_phase6.yaml
core/feature_phase6.py
scripts/run_phase6_features.py
scripts/audit_phase6.py
tests/test_phase6_features.py
docs/PHASE06.md

Feature dimensions:
Hands             = 417
Pose              = 27
Face              = 9
Hands + Pose      = 444
Hands + Pose+Face = 453

Run:
pytest -q tests/test_phase6_features.py
python scripts/run_phase6_features.py
python scripts/audit_phase6.py

IMPORTANT:
Inspect Pose coverage and Face coverage.
Structural PASS with coverage=0 means the pipeline works but raw Pose/Face data are unavailable.
