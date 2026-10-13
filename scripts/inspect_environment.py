from __future__ import annotations
import platform, sys
from importlib import metadata

PACKAGES = ["numpy", "opencv-python", "mediapipe", "PyYAML", "pytest"]
print("Python:", sys.version.replace("\n", " "))
print("OS:", platform.platform())
for p in PACKAGES:
    try:
        print(f"{p}: {metadata.version(p)}")
    except metadata.PackageNotFoundError:
        print(f"{p}: NOT INSTALLED")
