# Phase 1 FIX
Replace the files in your project with the matching paths in this patch.

Important findings in the uploaded RAR:
- `core/collector.py` is 0 bytes, so Phase 1 cannot work.
- `core/schema.py` is still the old Phase-0 schema and does not match Phase-1 collector metadata.
- Phase-1 scripts need project-root bootstrap so `python scripts/...py` works reliably.
- `video_import` must be an accepted collection mode.
- Config must include the `collection:` section.

Run from project root:
```powershell
python scripts/audit_phase1_source.py
pytest -q
python scripts/validate_schema.py
python scripts/collect_sequence.py --sign xin_chao --subject S01 --session SES01 --samples 5 --mode fixed --frames 45
```
