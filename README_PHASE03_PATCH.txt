PHASE 03 PATCH

Copy:
- configs/quality_phase3.yaml
- core/quality_phase3.py
- scripts/run_quality_control.py
- scripts/audit_quality.py
- tests/test_phase3_quality.py
- docs/PHASE3.md

Run:
pytest -q tests/test_phase3_quality.py
python scripts/run_quality_control.py
python scripts/audit_quality.py

Final exit gate after resolving REVIEW:
python scripts/audit_quality.py --require-no-review

Important:
- Phase-2 split remains unchanged.
- dataset/raw is never modified.
- Only ACCEPT samples are train-eligible automatically.
