from core.config import load_config
from core.logging_utils import setup_logging
from app.camera_preview import run_preview


def main():
    cfg = load_config()
    setup_logging(cfg["logging"]["level"], cfg["logging"].get("file"))
    run_preview(cfg)


if __name__ == "__main__":
    main()
