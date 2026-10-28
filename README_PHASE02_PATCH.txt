PHASE 02 PATCH

Copy vào project:
- configs/split_phase2.yaml
- core/dataset_index.py
- core/split_utils.py
- scripts/make_splits.py
- scripts/audit_dataset.py
- tests/test_phase2_split.py
- docs/PHASE2.md

Chạy:
python scripts/make_splits.py
python scripts/audit_dataset.py
pytest -q tests/test_phase2_split.py

Split:
TRAIN = S01-S07
VAL   = S08
TEST  = S09-S10
