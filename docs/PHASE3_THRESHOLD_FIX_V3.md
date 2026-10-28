# Phase 03 Semantic QC Fix v3

## Normal sign / one-hand sign

```text
any_hand_ratio >= 0.70
    -> PASS hand condition

0.55 <= any_hand_ratio < 0.70
    -> REVIEW

any_hand_ratio < 0.55
    -> REJECT
```

`both_hand_ratio` is ignored unless the class is explicitly listed in `two_hand_signs`.

## REST

REST = no-sign / sitting idle.

ACCEPT:
- structure is valid,
- almost no detected hand,
- low hand motion,
- blur is not suspicious.

REVIEW:
- hand is detected / raised,
- hand motion is too high,
- blur is suspicious.

REJECT:
- frame count invalid,
- timestamp invalid,
- NPZ/mask schema invalid,
- NaN/Inf,
- landmarks file missing.

REST is never rejected only because no hand is visible.

## Install

Replace:

```text
configs/quality_phase3.yaml
core/quality_phase3.py
```

Add:

```text
tests/test_phase3_thresholds_v3.py
```

Run:

```powershell
pytest -q tests/test_phase3_thresholds_v3.py
python scripts/run_quality_control.py
python scripts/audit_quality.py
```
