"""Cấu hình tập trung, đọc từ .env."""
import os
from pathlib import Path
from urllib.parse import quote_plus
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _abspath(rel: str) -> str:
    p = Path(rel)
    return str(p if p.is_absolute() else BASE_DIR / p)


def _bool(key: str, default: bool = False) -> bool:
    """Đọc biến môi trường boolean theo cách dễ dùng trong file .env."""
    raw = _get(key, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class Settings:
    # Database (PostgreSQL)
    DB_HOST = _get("DB_HOST", "localhost")
    DB_PORT = int(_get("DB_PORT", "5432"))
    DB_USER = _get("DB_USER", "postgres")
    DB_PASSWORD = _get("DB_PASSWORD", "123")
    DB_NAME = _get("DB_NAME", "vehicle_management")

    # Video
    VIDEO_SOURCE = _get("VIDEO_SOURCE", "0")
    CAMERA_TYPE = _get("CAMERA_TYPE", "GATE")
    CAMERA_NAME = _get("CAMERA_NAME", "Cong chinh")

    # AI
    FPS_PROCESS = float(_get("FPS_PROCESS", "10"))
    VEHICLE_MODEL = _abspath(_get("VEHICLE_MODEL", "models/yolov8n.pt"))
    PLATE_MODEL = _abspath(_get("PLATE_MODEL", "models/license_plate_detector.pt"))
    DEVICE = _get("DEVICE", "auto")
    VEHICLE_CONF = float(_get("VEHICLE_CONF", "0.35"))
    PLATE_CONF = float(_get("PLATE_CONF", "0.30"))
    OCR_MIN_CONF = float(_get("OCR_MIN_CONF", "0.40"))
    DEDUP_SECONDS = int(_get("DEDUP_SECONDS", "45"))
    # Đọc biển số tích luỹ theo track:
    # - chỉ đọc xe có bbox >= tỉ lệ này của khung hình (biển đủ nét)
    LPR_MIN_BBOX_RATIO = float(_get("LPR_MIN_BBOX_RATIO", "0.02"))
    # - conf đủ tốt thì ngừng đọc lại track đó (tiết kiệm GPU)
    LPR_GOOD_CONF = float(_get("LPR_GOOD_CONF", "0.85"))
    # - giới hạn/tần suất đọc và số phiếu đồng thuận trước khi chốt biển
    LPR_MAX_ATTEMPTS = int(_get("LPR_MAX_ATTEMPTS", "8"))
    LPR_INTERVAL_SEC = float(_get("LPR_INTERVAL_SEC", "0.35"))
    LPR_MIN_VOTES = int(_get("LPR_MIN_VOTES", "2"))
    LPR_TRACK_TTL_SEC = float(_get("LPR_TRACK_TTL_SEC", "6"))
    DWELL_THRESHOLD_SEC = int(_get("DWELL_THRESHOLD_SEC", "1800"))
    # Ngưỡng cảnh báo xe vào nhưng chưa ra (giờ)
    STUCK_IN_YARD_HOURS = float(_get("STUCK_IN_YARD_HOURS", "4"))
    # Ngưỡng đối soát thời gian AI vs phiếu cân (phút)
    RECONCILE_TIME_WINDOW_MIN = int(_get("RECONCILE_TIME_WINDOW_MIN", "30"))
    # Bật tự động thu thập ảnh crop để train sau này
    COLLECT_TRAINING_DATA = _get("COLLECT_TRAINING_DATA", "false").lower() == "true"

    # Demo đếm xe: các tính năng AI bổ sung có thể tắt để ưu tiên FPS/độ ổn định.
    DEMO_MODE = _bool("DEMO_MODE", True)
    VIDEO_LOOP = _bool("VIDEO_LOOP", False)
    PLAYBACK_REALTIME = _bool("PLAYBACK_REALTIME", True)
    ENABLE_PLATE_RECOGNITION = _bool("ENABLE_PLATE_RECOGNITION", True)
    ENABLE_LOAD_CLASSIFICATION = _bool("ENABLE_LOAD_CLASSIFICATION", False)
    ENABLE_PLATE_DEDUP = _bool("ENABLE_PLATE_DEDUP", False)
    ENABLE_EVIDENCE_CLIPS = _bool("ENABLE_EVIDENCE_CLIPS", False)
    ENABLE_ANALYTICS_ALERTS = _bool("ENABLE_ANALYTICS_ALERTS", False)
    COUNT_CLASSES = tuple(
        x.strip().lower()
        for x in _get("COUNT_CLASSES", "car,bus,truck").split(",")
        if x.strip()
    )
    LINE_HYSTERESIS_PX = float(_get("LINE_HYSTERESIS_PX", "6"))

    # Line
    LINE_X1 = float(_get("LINE_X1", "0.05"))
    LINE_Y1 = float(_get("LINE_Y1", "0.55"))
    LINE_X2 = float(_get("LINE_X2", "0.95"))
    LINE_Y2 = float(_get("LINE_Y2", "0.55"))
    IN_DIRECTION = _get("IN_DIRECTION", "down")

    # Web
    WEB_HOST = _get("WEB_HOST", "0.0.0.0")
    WEB_PORT = int(_get("WEB_PORT", "8000"))

    # Storage
    SNAPSHOT_DIR = _abspath(_get("SNAPSHOT_DIR", "data/snapshots"))
    CLIP_DIR = _abspath(_get("CLIP_DIR", "data/clips"))

    @property
    def db_url(self) -> str:
        pwd = quote_plus(self.DB_PASSWORD)
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{pwd}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def db_url_no_db(self) -> str:
        # Kết nối tới database hệ thống 'postgres' để tạo DB mới nếu chưa có.
        pwd = quote_plus(self.DB_PASSWORD)
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{pwd}"
            f"@{self.DB_HOST}:{self.DB_PORT}/postgres"
        )


settings = Settings()

# Tạo sẵn thư mục lưu trữ
for d in (settings.SNAPSHOT_DIR, settings.CLIP_DIR):
    Path(d).mkdir(parents=True, exist_ok=True)
