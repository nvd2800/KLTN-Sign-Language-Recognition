PHASE07 REALTIME DECISION ENGINE FIX

Files:
configs/realtime_phase7.yaml
core/realtime_decision_phase7.py
scripts/live_tcn_decision_v2.py
tests/test_realtime_decision_phase7.py
docs/PHASE07_REALTIME_FIX.md

Install into project root.

Test:
pytest -q tests/test_realtime_decision_phase7.py

Run:
python scripts/live_tcn_decision_v2.py

Fullscreen:
python scripts/live_tcn_decision_v2.py --fullscreen

Main fixes:
- REST -> ACTIVE -> END segmentation
- raw XY motion detection
- pre-roll
- confidence >= 0.85
- top1-top2 margin >= 0.20
- 4/5 prediction stability
- hand-presence gate
- UNKNOWN fallback
- cooldown + REST before repeat
