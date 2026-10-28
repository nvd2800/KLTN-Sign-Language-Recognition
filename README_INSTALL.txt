KLTN SIGN LANGUAGE | PHASE10.1 + PHASE10.2
=========================================
Patch for existing project Phase09 V3.1 + Phase10 V1. Not a standalone training project.
1) BACKUP source files before overwrite.
2) Extract ZIP into project root; copy same relative core/, scripts/, configs/, tests/, docs/.
3) pip install PyYAML Pillow piper-tts
4) python scripts/install_phase10_piper_voice.py --download   (one-time network; explicit)
5) python scripts/test_phase10_tts.py --backend piper --text "Xin chào!"
6) python scripts/demo_phase10_sentence.py --tokens xin_chao dau_phay cam_on dau_cam
7) python scripts/live_phase10_v3_1.py --dry-run --check-model
8) python scripts/live_phase10_v3_1.py --fullscreen

Keep your own trained model+preprocess_stats+labels for 32x513, 5 classes.
Adding words to vocabulary_pending is NOT training new recognisable sign classes.
See docs/PHASE10_1_10_2_HUONG_DAN_FULL.md for all details.
