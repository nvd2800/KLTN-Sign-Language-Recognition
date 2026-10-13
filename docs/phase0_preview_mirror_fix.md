# Phase 0 Preview Mirror Fix

## Problem
The camera preview was mirrored after status text was rendered, causing FPS, Left/Right, Pose and Face text to appear horizontally reversed.

## Fix
The preview frame is now flipped first when `camera.mirror_preview: true`, then all status text is drawn onto the already-mirrored frame.

The detector still receives the raw, unmirrored camera frame, so the dataset handedness convention remains unchanged.

Affected files:
- `app/camera_preview.py`
- `scripts/check_landmarks.py`

`requirements.txt` also pins `mediapipe==0.10.21` because this Phase 0 implementation uses the `mp.solutions` API.
