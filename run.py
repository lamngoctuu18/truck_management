"""Điểm khởi động hệ thống.

Chạy:  venv\\Scripts\\python.exe run.py
Hoặc dùng file run.bat.
"""
import os

# Trỏ cache của các thư viện AI về ổ E (tránh ổ C đầy)
_BASE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_BASE, ".cache")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_CACHE, "ultralytics"))
os.environ.setdefault("HF_HOME", os.path.join(_CACHE, "huggingface"))
os.environ.setdefault("FAST_PLATE_OCR_HUB_HOME", os.path.join(_CACHE, "fast_plate_ocr"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE, "matplotlib"))

import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print("=" * 60)
    print("  HE THONG AI DEM XE CHO DAT/GACH VAO-RA")
    print("=" * 60)
    print(f"  Dashboard : http://localhost:{settings.WEB_PORT}")
    print(f"  Nguon video: {settings.VIDEO_SOURCE}  ({settings.CAMERA_TYPE})")
    print(f"  Database  : {settings.DB_NAME} @ {settings.DB_HOST}:{settings.DB_PORT}")
    print("=" * 60)
    uvicorn.run("app.main:app", host=settings.WEB_HOST, port=settings.WEB_PORT,
                reload=False, log_level="info")
