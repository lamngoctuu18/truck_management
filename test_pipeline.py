"""Test pipeline offline trên 1 file video (không chạy web).

Chạy N giây, in số xe detect được, số lần cross vạch, số biển số đọc được.
"""
import os, time, sys
_BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_BASE, ".cache", "ultralytics"))
os.environ.setdefault("HF_HOME", os.path.join(_BASE, ".cache", "huggingface"))
os.environ.setdefault("FAST_PLATE_OCR_HUB_HOME", os.path.join(_BASE, ".cache", "fast_plate_ocr"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_BASE, ".cache", "matplotlib"))

import cv2
from app.core.config import settings
from app.db.database import init_db, get_session
from app.db.models import VehicleEvent
from app.models_ai.vehicle_detector import VehicleDetector
from app.models_ai.line_counter import LineCounter

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "data/videos/test_traffic.mp4"
MAX_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 120

init_db()
det = VehicleDetector()
cap = cv2.VideoCapture(VIDEO)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video {VIDEO} {w}x{h}")

# Vạch ngang giữa khung
counter = LineCounter((0, h*0.5, w, h*0.5), "down", dedup_seconds=2)

frames = 0; total_det = 0; crosses = 0
t0 = time.time()
while frames < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break
    frames += 1
    tracks = det.track(frame)
    total_det += len(tracks)
    evs = counter.update(tracks)
    crosses += len(evs)
    for e in evs:
        print(f"  CROSS frame={frames} track={e['track_id']} dir={e['direction']} {e['cls_name']}")
cap.release()
dt = time.time() - t0
print(f"\n== KET QUA ==")
print(f"Frames xu ly: {frames}  |  Thoi gian: {dt:.1f}s  |  FPS: {frames/dt:.1f}")
print(f"Tong detection: {total_det}  |  IN={counter.count_in} OUT={counter.count_out}  cross={crosses}")

# kiểm tra ghi DB
s = get_session()
n = s.query(VehicleEvent).count()
print(f"So su kien trong DB: {n}")
s.close()
