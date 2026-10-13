from __future__ import annotations
import time
from dataclasses import dataclass
import cv2
from .types import CameraFrame


@dataclass
class CameraStats:
    frames_ok: int = 0
    frames_failed: int = 0
    started_at: float = 0.0

    @property
    def elapsed(self) -> float:
        return max(time.perf_counter() - self.started_at, 1e-9)

    @property
    def actual_fps(self) -> float:
        return self.frames_ok / self.elapsed


class CameraManager:
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap: cv2.VideoCapture | None = None
        self.frame_index = 0
        self.stats = CameraStats()

    def open(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            return
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera id={self.camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.frame_index = 0
        self.stats = CameraStats(started_at=time.perf_counter())

    def read(self) -> CameraFrame | None:
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera is not open")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.stats.frames_failed += 1
            return None
        now = time.perf_counter()
        item = CameraFrame(
            index=self.frame_index,
            timestamp=now - self.stats.started_at,
            image_bgr=frame,
        )
        self.frame_index += 1
        self.stats.frames_ok += 1
        return item

    @property
    def resolution(self) -> tuple[int, int]:
        if self.cap is None:
            return self.width, self.height
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
