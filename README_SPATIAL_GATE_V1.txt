PHASE09 - SPATIAL + HANDSHAPE VALIDATION GATE V1

1. Backup current files:
   scripts/live_phase09_tcn_v2.py
   configs/phase9_live.yaml

2. Copy ZIP contents into your project root. Keep TCN V2 model, Phase04-06 core files.

3. Run tests:
   pytest -q tests/test_spatial_validation_gate.py

4. Build DEV profile with existing xin_chao/S01 diagnostics:
   python scripts/build_spatial_gate_profile.py --labels xin_chao --dev

5. Test live with gate:
   python scripts/live_phase09_tcn_v2.py --fullscreen --spatial-gate artifacts/spatial_gate_v1/profile.json

6. Compare without gate:
   python scripts/live_phase09_tcn_v2.py --fullscreen

7. Log:
   logs/phase09_spatial_predictions.csv

IMPORTANT: 6 diagnostic trials/1 signer -> DEV_ONLY; no scientifically validated thresholds.
IMPORTANT: Only xin_chao is protected by this example; other known labels show BYPASS_UNCONFIGURED.
IMPORTANT: Gate uses candidate32 outside TCN (model input remains 32x453).

Full instructions: docs/PHASE09_SPATIAL_HANDSHAPE_GATE_V1.md
