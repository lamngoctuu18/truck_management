"""Module cảnh báo bất thường (6.7) - phần chạy định kỳ nền.

Các cảnh báo phát hiện qua truy vấn DB theo chu kỳ:
  - STUCK_IN_YARD : xe đã vào (IN) nhưng sau X giờ chưa thấy lượt ra (OUT).

Các cảnh báo phát hiện tức thời (trong pipeline, không ở đây):
  - U_TURN          : xe quay đầu (line_counter phát hiện, pipeline ghi).
  - PLATE_DUP_LOC   : 1 biển số xuất hiện ở 2 camera/vị trí gần cùng lúc (pipeline).
  - OUT_WITHOUT_IN, NO_PLATE, LOW_PLATE_CONF, LONG_WAIT, CAMERA_OFFLINE (đã có).

Chạy trong 1 thread nền riêng, gọi callback raise_alert của pipeline.
"""
import time
import threading
from datetime import datetime, timedelta

from sqlalchemy import func
from app.core.config import settings
from app.core.logger import get_logger
from app.db.database import get_session
from app.db.models import VehicleEvent, AlertEvent

log = get_logger("Anomaly")


class AnomalyChecker:
    def __init__(self, raise_alert_cb, check_interval_sec=300):
        """raise_alert_cb(atype, plate, message, severity, evidence)"""
        self.raise_alert = raise_alert_cb
        self.check_interval = check_interval_sec
        self.stuck_hours = settings.STUCK_IN_YARD_HOURS
        self._running = False
        self._thread = None
        # tránh cảnh báo lặp cùng 1 biển số kẹt bãi
        self._alerted_stuck = set()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("AnomalyChecker started (interval=%ss, stuck>%sh)",
                 self.check_interval, self.stuck_hours)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        # chờ chút cho DB/pipeline sẵn sàng
        time.sleep(10)
        while self._running:
            try:
                self._check_stuck_in_yard()
            except Exception as e:  # noqa
                log.warning("Anomaly check loi: %s", e)
            # ngủ theo từng nhịp nhỏ để dừng nhanh
            slept = 0
            while self._running and slept < self.check_interval:
                time.sleep(2)
                slept += 2

    def _check_stuck_in_yard(self):
        """Xe có sự kiện IN cách đây > stuck_hours mà không có OUT sau đó."""
        s = get_session()
        try:
            cutoff = datetime.now() - timedelta(hours=self.stuck_hours)
            # Lấy sự kiện IN có biển số, cũ hơn cutoff
            in_events = (
                s.query(VehicleEvent)
                .filter(VehicleEvent.direction == "IN",
                        VehicleEvent.plate_number.isnot(None),
                        VehicleEvent.event_time <= cutoff)
                .all()
            )
            for ie in in_events:
                plate = ie.plate_number
                # Có OUT nào sau thời điểm IN này không?
                out_exists = (
                    s.query(VehicleEvent.id)
                    .filter(VehicleEvent.direction == "OUT",
                            VehicleEvent.plate_number == plate,
                            VehicleEvent.event_time > ie.event_time)
                    .first()
                )
                if out_exists:
                    continue
                key = f"{plate}_{ie.id}"
                if key in self._alerted_stuck:
                    continue
                # Đã cảnh báo trong DB chưa (tránh lặp qua nhiều lần khởi động)?
                dup = (
                    s.query(AlertEvent.id)
                    .filter(AlertEvent.alert_type == "STUCK_IN_YARD",
                            AlertEvent.plate_number == plate,
                            AlertEvent.created_at >= ie.event_time)
                    .first()
                )
                if dup:
                    self._alerted_stuck.add(key)
                    continue
                hrs = (datetime.now() - ie.event_time).total_seconds() / 3600.0
                self.raise_alert(
                    "STUCK_IN_YARD", plate,
                    f"Xe {plate} da vao luc {ie.event_time:%H:%M %d/%m} "
                    f"nhung sau {hrs:.1f} gio chua thay ra",
                    "WARN", ie.snapshot_url or "")
                self._alerted_stuck.add(key)
        finally:
            s.close()
