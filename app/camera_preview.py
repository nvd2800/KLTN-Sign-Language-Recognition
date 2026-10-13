from __future__ import annotations
import cv2
from core.camera import CameraManager
from core.detector import MediaPipeDetector
from core.quality import BasicQualityChecker


def run_preview(cfg: dict) -> None:
    c = cfg["camera"]
    checker = BasicQualityChecker()
    with CameraManager(c["id"], c["width"], c["height"], c["fps"]) as cam, MediaPipeDetector(cfg["mediapipe"]) as det:
        while True:
            cf = cam.read()
            if cf is None:
                continue
            lm = det.process(cf.image_bgr, cf.index, cf.timestamp)
            q = checker.inspect(lm)
            # Mirror only the CAMERA IMAGE first. Any overlay/text must be drawn
            # afterwards; otherwise cv2.flip() would also mirror the letters.
            view = cf.image_bgr.copy()
            if c.get("mirror_preview", True):
                view = cv2.flip(view, 1)

            lines = [
                f"Phase 0 | FPS {cam.stats.actual_fps:.1f}",
                f"Left={int(q.left_hand_present)} Right={int(q.right_hand_present)} Pose={int(q.pose_present)} Face={int(q.face_present)}",
                "Q/ESC: quit | Preview mirror does not alter stored handedness",
            ]
            for i, txt in enumerate(lines):
                cv2.putText(view, txt, (18, 34+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("KLTN Sign Language - Phase 0", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
