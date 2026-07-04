"""Logger dùng chung, ghi ra console + file logs/app.log."""
import logging
from pathlib import Path

_BASE = Path(__file__).resolve().parents[2]
_LOG_DIR = _BASE / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
