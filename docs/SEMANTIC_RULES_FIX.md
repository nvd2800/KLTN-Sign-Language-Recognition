# Phase 03 Semantic QC Fix

## Rule 1 — Normal known sign / one-hand sign

A normal sign does not require both hands.

ACCEPT when:
- structural checks pass,
- `any_hand_ratio >= min_any_hand_ratio`,
- no review heuristic is triggered.

REJECT when:
- too many frames have no detected hand,
- frame count/timestamp/schema/NaN/Inf is invalid.

`both_hand_ratio` is ignored unless the class appears in `two_hand_signs`.

## Rule 2 — Two-hand sign

Only classes explicitly listed under:

```yaml
two_hand_signs:
```

must satisfy:

```text
both_hand_ratio >= min_both_hand_ratio
```

## Rule 3 — REST

REST means no active sign / sitting idle.

ACCEPT:
- valid structure,
- almost no detected hand,
- low hand motion,
- blur not suspicious.

REVIEW:
- a hand is detected in more than the tolerated detector-flicker ratio,
- hand motion is too high,
- blur is suspicious.

REJECT:
- frame count invalid,
- timestamps invalid,
- schema/mask invalid,
- NaN/Inf,
- missing landmark file.

REST is NEVER rejected merely because no hand is visible.

## Recommended workflow

```powershell
pytest -q tests/test_phase3_semantic_rules.py
python scripts/run_quality_control.py
python scripts/audit_quality.py
```

Then inspect `hard_fail_reasons` and `review_reasons`.
