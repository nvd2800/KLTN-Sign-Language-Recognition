from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import time
from core.config import load_config
from core.camera import CameraManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["camera"]
    with CameraManager(c["id"], c["width"], c["height"], c["fps"]) as cam:
        start = time.perf_counter()
        while time.perf_counter() - start < args.seconds:
            cam.read()
        w, h = cam.resolution
        total = cam.stats.frames_ok + cam.stats.frames_failed
        success = 100.0 * cam.stats.frames_ok / max(total, 1)
        print(f"Camera ID: {c['id']}")
        print(f"Resolution: {w}x{h}")
        print(f"Target FPS: {c['fps']}")
        print(f"Actual FPS: {cam.stats.actual_fps:.2f}")
        print(f"Frame read success: {success:.2f}%")
        print(f"Dropped/failed frames: {cam.stats.frames_failed}")


if __name__ == "__main__":
    main()
