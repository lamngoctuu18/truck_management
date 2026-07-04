"""Lưu ảnh snapshot và video clip bằng chứng (4.5 / mục 2.2).

Trả về URL dạng "/media/snapshots/YYYYMMDD/xxx.jpg" để FastAPI serve trực tiếp
(mount StaticFiles("data") tại "/media").
"""
import os
from datetime import datetime
from collections import deque

import cv2
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("Evidence")


def _rel_url(abs_path: str) -> str:
    """Chuyển đường dẫn tuyệt đối trong data/ thành URL /media/..."""
    from app.core.config import BASE_DIR
    data_root = os.path.join(str(BASE_DIR), "data")
    rel = os.path.relpath(abs_path, data_root).replace("\\", "/")
    return f"/media/{rel}"


def save_snapshot(frame, plate: str = "", tag: str = "evt") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_plate = (plate or "NA").replace(" ", "")
    fname = f"{tag}_{safe_plate}_{ts}.jpg"
    day = datetime.now().strftime("%Y%m%d")
    day_dir = os.path.join(settings.SNAPSHOT_DIR, day)
    os.makedirs(day_dir, exist_ok=True)
    path = os.path.join(day_dir, fname)
    try:
        cv2.imwrite(path, frame)
    except Exception as e:  # noqa
        log.warning("Luu snapshot loi: %s", e)
        return ""
    return _rel_url(path)


class ClipBuffer:
    """Vòng đệm frame để xuất clip ngắn quanh thời điểm sự kiện (5-10s)."""

    def __init__(self, fps=10, seconds_before=4, seconds_after=4):
        self.fps = max(1, int(fps))
        self.maxlen = self.fps * (seconds_before + seconds_after)
        self.buf = deque(maxlen=self.maxlen)

    def push(self, frame):
        self.buf.append(frame.copy())

    def save_clip(self, plate: str = "", tag: str = "evt") -> str:
        if not self.buf:
            return ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_plate = (plate or "NA").replace(" ", "")
        fname = f"{tag}_{safe_plate}_{ts}.mp4"
        day = datetime.now().strftime("%Y%m%d")
        day_dir = os.path.join(settings.CLIP_DIR, day)
        os.makedirs(day_dir, exist_ok=True)
        path = os.path.join(day_dir, fname)
        h, w = self.buf[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        for f in list(self.buf):
            writer.write(f)
        writer.release()
        return _rel_url(path)
