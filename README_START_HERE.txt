KLTN SIGN-LANGUAGE | FEATURE DIAGNOSTIC CANONICAL LAYOUT PATCH
===============================================================

FINAL HIERARCHY (requested by user):
logs/feature_diagnostic/<label>/<subject>/<session>/<condition>/<condition>_<subject>_<session>_<NNN>/

Example:
logs/feature_diagnostic/xin_chao/S03/SES01/partial/partial_S03_SES01_001/

Six condition folders: correct, wrong_position, wrong_handshape,
wrong_orientation, partial, correct_repeat.

Every recorded sample:
  metadata.json
  raw_landmarks.npz
  features.npz
  source_video.avi  (or source_video.mp4 if --video-format mp4)

Root manifest: logs/feature_diagnostic/manifest.csv
No signer_alias. Session is mandatory in the new path.

INSTALL: Back up the project files and logs/feature_diagnostic, then extract
this ZIP into your project root and overwrite the listed source files.

TEST:
  python -m pytest -q tests/test_feature_diagnostic_condition_layout.py

COLLECT:
  python scripts/live_feature_diagnostic.py --label xin_chao --subject S03 --session SES01 --frames 32 --fullscreen

MIGRATE PREVIEW ONLY (no files changed):
  python scripts/migrate_feature_diagnostic_layout.py
MIGRATE (AFTER BACKUP AND PREVIEW):
  python scripts/migrate_feature_diagnostic_layout.py --execute

REBUILD MANIFEST:
  python scripts/rebuild_feature_diagnostic_manifest.py

REVIEW:
  python scripts/review_feature_diagnostic_videos.py --label xin_chao --subject S03 --session SES01 --contact-sheets
  python scripts/review_feature_diagnostic_videos.py --label xin_chao --subject S03 --session SES01 --play

AUDIT:
  python scripts/audit_feature_diagnostic_layout.py

IMPORTANT:
  Do NOT manually rename sample folders while leaving metadata.json unchanged.
  Read docs/HUONG_DAN_DIAGNOSTIC_LAYOUT_CHINH_XAC.md before migrating.
  Historical recordings without video cannot be retroactively recovered.
  All code tests are local simulated tests; real Windows webcam test is pending.
