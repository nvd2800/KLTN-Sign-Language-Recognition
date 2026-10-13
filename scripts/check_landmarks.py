from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import cv2
from core.config import load_config
from core.camera import CameraManager
from core.detector import MediaPipeDetector


def yn(v: bool) -> str:
    return "YES" if v else "NO"


def main():
    ap = argparse.ArgumentParser(description="Verify physical left/right and missing-landmark behavior")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["camera"]
    print("TEST INSTRUCTIONS:")
    print("1) Raise ONLY your physical LEFT hand: expect Left=YES Right=NO")
    print("2) Raise ONLY your physical RIGHT hand: expect Left=NO Right=YES")
    print("3) Raise both hands: expect YES/YES")
    print("4) Move hands/face out of view: program must not crash")
    print("Press Q to quit.")
    with CameraManager(c["id"], c["width"], c["height"], c["fps"]) as cam, MediaPipeDetector(cfg["mediapipe"]) as det:
        while True:
            cf = cam.read()
            if cf is None:
                continue
            lm = det.process(cf.image_bgr, cf.index, cf.timestamp)
            p = lm.presence
            # Mirror only the camera image. Draw status text AFTER mirroring so
            # FPS/Left/Right/Pose/Face remain readable (not horizontally reversed).
            view = cf.image_bgr.copy()
            if c.get("mirror_preview", True):
                view = cv2.flip(view, 1)

            lines = [
                f"FPS: {cam.stats.actual_fps:.1f}",
                f"Physical Left hand: {yn(p.left_hand)}",
                f"Physical Right hand: {yn(p.right_hand)}",
                f"Pose: {yn(p.pose)}",
                f"Face: {yn(p.face)}",
            ]
            for i, txt in enumerate(lines):
                cv2.putText(view, txt, (20, 35 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.imshow("Phase 0 - Landmark/Left-Right Check", view)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
