PATCH V3 SPATIAL DIAGNOSTIC – FIX SYNTHETIC SHIFT

Required replacement:
  scripts/diagnose_position_sensitivity.py

Optional:
  tests/test_diagnose_shift_fix.py
  docs/HUONG_DAN_FIX_SYNTHETIC_SHIFT.md

Works with original V3 Spatial 513D project and its existing core/configs.
Does NOT overwrite training, model or scaler.

After copying:
  python -m pytest -q tests/test_diagnose_shift_fix.py
  python scripts/diagnose_position_sensitivity.py --min-shift-shoulders 1.0 --synthetic-shift 1.0

Verify synthetic_shift n > 0 and error_count==0.
Validated locally: 6/6 tests PASS (including real Phase04-06 feature recompute with synthetic fixtures); real TensorFlow checkpoint not available in assistant runtime.
