"""Module phát hiện xe chờ lâu (6.6).

Theo dõi từng track xe: nếu tâm xe nằm trong vùng chờ (ROI) và gần như đứng yên
trong khoảng thời gian vượt ngưỡng -> phát cảnh báo "xe chờ lâu".

- Vùng chờ (ROI) cấu hình theo tỉ lệ khung hình (x1,y1,x2,y2) 0..1.
- "Đứng yên" = tâm xe dịch chuyển < move_thresh (pixel) giữa các lần cập nhật.
- dwell_threshold_sec: ngưỡng thời gian chờ để cảnh báo (VD 30 phút).

Trả về trạng thái các xe đang chờ để dashboard hiển thị realtime.
"""
import time
from app.core.logger import get_logger

log = get_logger("DwellMonitor")


class DwellMonitor:
    def __init__(self, roi=(0.0, 0.0, 1.0, 1.0), dwell_threshold_sec=1800,
                 move_thresh_ratio=0.03):
        """roi: (x1,y1,x2,y2) tỉ lệ 0..1. move_thresh_ratio: ngưỡng dịch chuyển
        tính theo tỉ lệ chiều rộng khung hình để coi là 'đứng yên'."""
        self.set_roi(roi)
        self.dwell_threshold_sec = dwell_threshold_sec
        self.move_thresh_ratio = move_thresh_ratio
        # track_id -> {start, last_pos, last_seen, alerted, plate}
        self._state = {}
        self._frame_w = 1280

    def set_roi(self, roi, dwell_threshold_sec=None):
        self.x1, self.y1, self.x2, self.y2 = roi
        if dwell_threshold_sec is not None:
            self.dwell_threshold_sec = dwell_threshold_sec

    def reset(self):
        """Xoá trạng thái track của phiên hiện tại."""
        self._state.clear()

    def _in_roi(self, cx_ratio, cy_ratio):
        return (self.x1 <= cx_ratio <= self.x2 and
                self.y1 <= cy_ratio <= self.y2)

    def update(self, tracks, frame_w, frame_h):
        """Cập nhật với danh sách track. Trả về:
        - waiting: list xe đang chờ (kèm thời gian chờ)
        - new_alerts: list xe vừa vượt ngưỡng (để ghi cảnh báo 1 lần)
        """
        self._frame_w = frame_w
        now = time.time()
        move_thresh = self.move_thresh_ratio * frame_w
        seen_ids = set()
        new_alerts = []
        waiting = []

        for t in tracks:
            tid = t["track_id"]
            cx, cy = t["cx"], t["cy"]
            cxr, cyr = cx / frame_w, cy / frame_h
            if not self._in_roi(cxr, cyr):
                # Ra khỏi vùng chờ -> reset
                self._state.pop(tid, None)
                continue

            seen_ids.add(tid)
            st = self._state.get(tid)
            if st is None:
                self._state[tid] = {
                    "start": now, "last_pos": (cx, cy), "last_seen": now,
                    "alerted": False, "cls_name": t["cls_name"], "bbox": t["bbox"],
                }
                continue

            moved = ((cx - st["last_pos"][0]) ** 2 +
                     (cy - st["last_pos"][1]) ** 2) ** 0.5
            if moved > move_thresh:
                # Xe di chuyển đáng kể -> reset đồng hồ chờ
                st["start"] = now
                st["alerted"] = False
            st["last_pos"] = (cx, cy)
            st["last_seen"] = now
            st["bbox"] = t["bbox"]

            dwell = now - st["start"]
            waiting.append({
                "track_id": tid,
                "cls_name": st["cls_name"],
                "bbox": st["bbox"],
                "dwell_seconds": int(dwell),
                "over_threshold": dwell >= self.dwell_threshold_sec,
            })

            if dwell >= self.dwell_threshold_sec and not st["alerted"]:
                st["alerted"] = True
                new_alerts.append({
                    "track_id": tid,
                    "cls_name": st["cls_name"],
                    "bbox": st["bbox"],
                    "dwell_seconds": int(dwell),
                })
                log.info("XE CHO LAU track=%s dwell=%ds", tid, int(dwell))

        # Dọn track không còn thấy (đã rời khung) sau 5s
        for tid in list(self._state.keys()):
            if tid not in seen_ids and now - self._state[tid]["last_seen"] > 5:
                self._state.pop(tid, None)

        return waiting, new_alerts
