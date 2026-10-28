TCN V1 -> V2 WORKFLOW

1) Freeze:
python scripts/freeze_tcn_v1.py

2) Live:
python scripts/live_tcn_v1.py

3) Perform each known sign and press its displayed numeric ground-truth key.

4) Analyze errors:
python scripts/analyze_live_errors.py

5) Collect targeted data and rebuild Phase03->Phase06.

6) Prepare V2:
python scripts/prepare_tcn_v2.py

7) Train V2 from scratch:
python scripts/train_tcn_v2.py

8) Evaluate:
python scripts/evaluate_tcn_v2.py

9) Compare:
python scripts/compare_tcn_v1_v2.py

IMPORTANT:
Locked TEST sample IDs from V1 are reused for fair comparison.
