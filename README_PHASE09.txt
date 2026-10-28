PHASE09 REALTIME TCN V2 INTEGRATION

This extends the existing Phase09 state machine with real webcam TCN V2 inference.
DO NOT overwrite model.keras or train Phase08 again just to run this tester.

Requirements: Existing core Phase04/05/06, Phase07 preprocessing, trained
  artifacts/experiments/tcn_v2_unknown/model.keras
  artifacts/experiments/tcn_v2_unknown/preprocess_stats.npz
  artifacts/experiments/tcn_v2_unknown/labels.json

Copy contents of this ZIP to the project root.

RUN:
  python scripts/audit_phase09_setup.py
  pytest -q tests/test_phase9.py tests/test_phase9_live_logic.py
  python scripts/live_phase09_tcn_v2.py --fullscreen
  python scripts/analyze_phase09_live.py

WINDOW: F fullscreen, V landmarks, M mirror, P pause, C clear, R reset,
  B debug REST release, A toggle auto-release, S snapshot, Q quit.

Model V2 only trained on a few UNKNOWN samples, so an UNKNOWN gesture may still
be recognized as a known sign. This tool logs behavior; it does not claim
reliable OOV detection without testing new people and new movements.

See docs/PHASE09_REALTIME_TCN_V2.md for full instructions.
