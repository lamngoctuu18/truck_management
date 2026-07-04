"""Đọc lại biển số cho sự kiện đã lưu snapshot nhưng OCR chưa có kết quả."""
from pathlib import Path

import cv2

from app.core.config import BASE_DIR
from app.core.logger import get_logger
from app.db.database import get_session
from app.db.models import VehicleEvent
from app.models_ai.evidence import save_snapshot
from app.models_ai.plate_recognizer import get_recognizer

log = get_logger("PlateBackfill")


def _snapshot_path(url: str):
    if not url or not url.startswith("/media/"):
        return None
    data_root = (BASE_DIR / "data").resolve()
    path = (data_root / url[len("/media/"):]).resolve()
    try:
        path.relative_to(data_root)
    except ValueError:
        return None
    return path


def reread_event_plate(event_id: int, recognizer=None):
    """Đọc lại một sự kiện, cập nhật DB và trả kết quả gọn cho API."""
    s = get_session()
    try:
        row = s.get(VehicleEvent, event_id)
        if row is None:
            return {"ok": False, "error": "event not found", "id": event_id}
        path = _snapshot_path(row.snapshot_url)
        if path is None or not Path(path).exists():
            row.plate_status = "SNAPSHOT_MISSING"
            s.commit()
            return {"ok": False, "error": "snapshot missing", "id": event_id}

        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            row.plate_status = "SNAPSHOT_MISSING"
            s.commit()
            return {"ok": False, "error": "snapshot unreadable", "id": event_id}

        reader = recognizer or get_recognizer()
        result = reader.read(image)
        if not result or not result.get("plate"):
            row.plate_status = "OCR_NOT_FOUND"
            row.plate_confidence = 0.0
            if not row.corrected_plate_number:
                row.plate_number = None
                row.plate_crop_url = None
            s.commit()
            return {"ok": False, "error": "plate not found", "id": event_id,
                    "plate_status": row.plate_status}

        row.plate_number = result["plate"]
        row.plate_confidence = float(result.get("confidence") or 0.0)
        row.plate_status = "OCR_OK"
        plate_img = result.get("plate_img")
        if plate_img is not None and plate_img.size:
            row.plate_crop_url = save_snapshot(
                plate_img, row.plate_number, tag="PLATE_RETRY")
        s.commit()
        log.info("Doc lai event=%s -> %s (%.3f)", row.id, row.plate_number,
                 row.plate_confidence)
        return {
            "ok": True, "id": row.id, "plate": row.plate_number,
            "plate_confidence": round(row.plate_confidence, 4),
            "plate_status": row.plate_status,
            "plate_crop_url": row.plate_crop_url,
        }
    except Exception as exc:  # noqa
        s.rollback()
        log.warning("Doc lai bien so event=%s loi: %s", event_id, exc)
        return {"ok": False, "error": str(exc), "id": event_id}
    finally:
        s.close()
