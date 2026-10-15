PHASE 05 - GEOMETRIC + MOTION FEATURES

Install:
configs/feature_phase5.yaml
core/feature_phase5.py
scripts/run_phase5_features.py
scripts/audit_phase5.py
tests/test_phase5_features.py
docs/PHASE05.md

Run:
pytest -q tests/test_phase5_features.py
python scripts/run_phase5_features.py
python scripts/audit_phase5.py

Outputs:
Position 126
Angles 30
Distances 9
Velocity 126
Acceleration 126
FULL 417

Ablation:
P=126
P+V=252
P+V+A=378
FULL=417
