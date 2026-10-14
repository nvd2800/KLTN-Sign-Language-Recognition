PHASE 04 FEATURE ENGINEERING + NORMALIZATION

Install these files into project root:
configs/feature_phase4.yaml
core/feature_phase4.py
scripts/run_phase4_features.py
scripts/audit_phase4.py
tests/test_phase4_normalization.py
docs/PHASE04.md

Commands:
pytest -q tests/test_phase4_normalization.py
python scripts/run_phase4_features.py
python scripts/audit_phase4.py

Feature:
21 XYZ x 2 hands = 126 features per frame.
Translation origin = wrist landmark 0.
Scale reference = wrist -> middle MCP landmark 9.
Missing/invalid hand = zeros WITH explicit presence mask.
